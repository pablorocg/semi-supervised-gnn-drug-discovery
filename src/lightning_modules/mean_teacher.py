import logging
import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import MetricCollection
from copy import deepcopy
from src.utils.utils import calculate_pos_weights_from_tensor

from src.utils.ogb_metrics import (
    MultiTaskAccuracy,
    MultiTaskAP,
    MultiTaskRMSE,
    MultiTaskROCAUC,
    SetF1Score,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MeanTeacherModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_outputs: int,
        ema_decay: float = 0.999,
        consistency_weight: float = 0.1,
        consistency_rampup_epochs: int = 50,
        learning_rate: float = 1e-4,
        warmup_epochs: int = 10,
        cosine_period_ratio: float = 1,
        compile: bool = False,
        weights: str = None,
        weight_decay: float = 1e-5,
        nesterov: bool = True,
        momentum: float = 0.9,
        task_type: str = "classification",
        train_dataset=None,
        train_idx=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "train_dataset", "train_idx"])

        # Student/Teacher models
        self.student_model = model
        self.teacher_model = deepcopy(model)
        for param in self.teacher_model.parameters():
            param.detach_()

        # Metrics
        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")
        self.test_metrics = self.configure_metrics("test")

        # Supervised loss setup
        self._setup_supervised_loss(train_dataset, train_idx)
        self.consistency_loss_fn = MSELoss()

        if self.hparams.weights is not None:
            self.load_weights(self.hparams.weights)

        if self.hparams.compile:
            self.student_model = torch.compile(self.student_model, dynamic=True)

    def _setup_supervised_loss(self, train_dataset, train_idx):
        """Set up pos_weight for BCE per task."""
        if train_dataset is not None and train_idx is not None:
            train_labels = train_dataset.data.y[train_idx].float()
            self.pos_weight = calculate_pos_weights_from_tensor(train_labels)
        else:
            self.pos_weight = None
        self.supervised_loss_fn = BCEWithLogitsLoss(reduction="none")

    def _masked_bce_loss(self, logits, labels, mask):
        """BCEWithLogitsLoss with per-task pos_weight and NaN masking."""
        device = logits.device
        
        if mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(device)
            loss_fn = BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
        else:
            loss_fn = BCEWithLogitsLoss(reduction="none")
        
        # Compute loss element-wise (shape: [batch_size, num_tasks])
        loss = loss_fn(logits, labels)
        
        # Apply mask and take mean only over valid elements
        masked_loss = loss[mask]
        
        return masked_loss.mean()


    def forward(self, x):
        return self.student_model(x)

    def _rampup_weight(self, current_epoch: int) -> float:
        """Sigmoid ramp-up for consistency weight."""
        if current_epoch > self.hparams.consistency_rampup_epochs:
            return self.hparams.consistency_weight
        p = torch.clamp(torch.tensor(current_epoch / self.hparams.consistency_rampup_epochs), 0.0, 1.0)
        rampup_factor = 1.0 / (1.0 + torch.exp(-5.0 * (p - 0.5))) * 2.0
        rampup_factor = torch.clamp(rampup_factor, 0.0, 1.0)
        return self.hparams.consistency_weight * rampup_factor.item()

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """EMA update for teacher model."""
        alpha = self.hparams.ema_decay
        for s_param, t_param in zip(self.student_model.parameters(), self.teacher_model.parameters()):
            t_param.data.mul_(alpha).add_(s_param.data, alpha=1 - alpha)
        for s_buffer, t_buffer in zip(self.student_model.buffers(), self.teacher_model.buffers()):
            if t_buffer.dtype.is_floating_point:
                t_buffer.data.mul_(alpha).add_(s_buffer.data, alpha=1 - alpha)

    def training_step(self, batch, batch_idx):
        labeled = batch["labeled"]
        unlabeled = batch["unlabeled"]
        device = self.device

        #  Supervised Loss 
        student_labeled_logits = self.student_model(labeled)
        labels = labeled.y.float().to(device)
        mask = ~torch.isnan(labels)


        supervised_loss = self._masked_bce_loss(student_labeled_logits, labels, mask)


        # Update metrics
        self.train_metrics.update(student_labeled_logits, labels)

        #  Consistency Loss 
        student_unlabeled_logits = self.student_model(unlabeled)
        with torch.no_grad():
            teacher_unlabeled_logits = self.teacher_model(unlabeled).detach()

        student_unlabeled_logits = student_unlabeled_logits.float()
        teacher_unlabeled_logits = teacher_unlabeled_logits.float()

        consistency_mask = ~torch.isnan(teacher_unlabeled_logits)
        if consistency_mask.sum() > 0:
            diff = (student_unlabeled_logits - teacher_unlabeled_logits) ** 2
            consistency_loss = (diff * consistency_mask.float()).sum() / consistency_mask.sum()
        else:
            consistency_loss = torch.tensor(0.0, device=device)
        

        #  Total Loss 
        rampup_weight = self._rampup_weight(self.current_epoch)
        total_loss = supervised_loss + rampup_weight * consistency_loss

        #  Logging 
        # Show only total loss in progress bar
        self.log(
            "train/loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=labeled.num_graphs,
        )

        # Log individual components (no duplication)
        self.log("train/supervised_loss", supervised_loss, on_step=True, on_epoch=True)
        self.log("train/consistency_loss", consistency_loss, on_step=True, on_epoch=True)
        self.log("train/consistency_weight", rampup_weight, on_step=True, on_epoch=True)

        return total_loss


    

    def validation_step(self, batch, batch_idx):
        with torch.inference_mode():
            logits = self.student_model(batch)
        labels = batch.y.float().to(self.device)

        mask = ~torch.isnan(labels)

        if mask.sum() == 0:
            loss = torch.tensor(0.0, device=self.device)
        else:
            valid_logits = logits.clone()
            valid_labels = labels.clone()
            valid_logits[~mask] = 0.0
            valid_labels[~mask] = 0.0
            loss = self.supervised_loss_fn(valid_logits, valid_labels)
            if loss.dim() > 0:  # ensure scalar
                loss = loss.mean()

        self.val_metrics.update(logits, labels)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        return loss



    def test_step(self, batch, batch_idx):
        with torch.inference_mode():
            logits = self.student_model(batch)
        labels = batch.y.float()
        mask = ~torch.isnan(labels)
        loss = self._masked_bce_loss(logits, labels, mask)
        self.test_metrics.update(logits, labels)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        return loss

    def configure_optimizers(self):
        from torch.optim import AdamW
        
        optimizer = AdamW(
            self.student_model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999)
        )
        
        steps_per_epoch = self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        total_steps = self.trainer.max_epochs * steps_per_epoch
        warmup_steps = self.hparams.warmup_epochs * steps_per_epoch

        cosine_steps = max(1, int(self.hparams.cosine_period_ratio * (total_steps - warmup_steps)))
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)

        if self.hparams.warmup_epochs > 0 and warmup_steps > 0:
            warmup_scheduler = LinearLR(optimizer, start_factor=1.0 / 1000, total_iters=warmup_steps)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = cosine_scheduler

        scheduler_config = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler_config]

    def configure_metrics(self, prefix: str):
        if self.hparams.task_type == "regression":
            return MetricCollection({f"{prefix}/rmse": MultiTaskRMSE(num_tasks=self.hparams.num_outputs)})
        elif self.hparams.task_type == "classification" and self.hparams.num_outputs == 1:
            return MetricCollection({
                f"{prefix}/pr_auc": MultiTaskAP(num_tasks=self.hparams.num_outputs),
                f"{prefix}/auroc": MultiTaskROCAUC(num_tasks=self.hparams.num_outputs),
                f"{prefix}/accuracy": MultiTaskAccuracy(num_tasks=self.hparams.num_outputs),
                f"{prefix}/set_f1": SetF1Score(),
            })
        elif self.hparams.task_type == "classification" and self.hparams.num_outputs > 1:
            return MetricCollection({
                f"{prefix}/pr_auc": MultiTaskAP(num_tasks=self.hparams.num_outputs),
                f"{prefix}/auroc": MultiTaskROCAUC(num_tasks=self.hparams.num_outputs),
                f"{prefix}/accuracy": MultiTaskAccuracy(num_tasks=self.hparams.num_outputs),
                f"{prefix}/set_f1": SetF1Score(),
            })
        else:
            raise ValueError(f"Unsupported task type: {self.hparams.task_type}")

    def on_train_epoch_end(self):
        if self.train_metrics:
            metrics = self.train_metrics.compute()
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
            self.train_metrics.reset()

    def on_validation_epoch_end(self):
        if self.val_metrics:
            try:
                metrics = self.val_metrics.compute()
                self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
            except RuntimeError as e:
                if "No positively labeled data" in str(e):
                    log.warning(f"Skipping metric computation at epoch {self.current_epoch}: {e}")
                    self.log("val/auroc", float('nan'), on_step=False, on_epoch=True)
                else:
                    raise
            finally:
                self.val_metrics.reset()

    def on_test_epoch_end(self):
        if self.test_metrics:
            metrics = self.test_metrics.compute()
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
            self.test_metrics.reset()

    def load_weights(self, weights_path: str):
        state_dict = torch.load(weights_path, map_location="cpu")
        self.student_model.load_state_dict(state_dict)
        self.teacher_model.load_state_dict(state_dict)
        log.info(f"Weights loaded into student and teacher from {weights_path}")




def main():
    import torch
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
    from src.models.gine import GINE
    from src.data.pcba import OgbgMolPcbaDataModule

    # GPUL40S optimization
    torch.set_float32_matmul_precision('medium')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    LR = 5e-5
    EMA_DECAY = 0.999
    CONSISTENCY_WEIGHT = 0.1
    RAMPUP_EPOCHS = 50
    
    NUM_EPOCHS = 300
    TRAIN_BATCH_SIZE = 128
    INFERENCE_BATCH_SIZE = 256
    NUM_WORKERS = 4
    
    # MolPCBA specifics
    NUM_TASKS = 128 
    
    # This sets up the OGB-MolPCBA dataset and the CombinedLoader for SSL
    data_module = OgbgMolPcbaDataModule(
        batch_size_train=TRAIN_BATCH_SIZE,
        batch_size_inference=INFERENCE_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        splits=[0.35, 0.35, 0.15, 0.15],  # Labeled, Unlabeled, Val, Test (I raised validation because of no positive labels error)
    )
    
    # Prepare data to get necessary objects for pos_weight calculation
    data_module.prepare_data()
    data_module.setup(stage="fit")
    num_outputs = data_module.num_tasks
    print(f"Using {num_outputs} output tasks")


    print(f"Dataset first sample y shape: {data_module.dataset[0].y.shape}")
    print(f"Dataset first sample y: {data_module.dataset[0].y}")
    print(f"Labeled dataset first sample y shape: {data_module.data_train_labeled[0].y.shape}")

    # Get dataset information needed for the MeanTeacherModule constructor
    # FIXED: Use the correct attribute names from your baseline code
    train_dataset = data_module.data_train_labeled  # Changed from train_dataset
    train_indices = data_module.train_idx  # This seems correct already

    # Instantiate the base GNN model (Student/Teacher)
    gnn_model = GINE(
        num_tasks=num_outputs,
        num_node_features=data_module.num_features,  
        num_edge_features=3,  
        hidden_dim=256,
        num_layers=5,
        dropout=0.5,
        use_residual=True,
        use_virtual_node=True,
        use_attention_pool=True,
        num_mlp_layers=2,
    )

    mean_teacher_model = MeanTeacherModule(
        model=gnn_model,
        num_outputs = data_module.num_tasks,
        
        # Mean Teacher Hyperparameters
        ema_decay=EMA_DECAY,
        consistency_weight=CONSISTENCY_WEIGHT,
        consistency_rampup_epochs=RAMPUP_EPOCHS,

        # Optimization & Stability
        learning_rate=LR,
        warmup_epochs=10,
        weight_decay=1e-5,
        task_type="classification",
        
        # Data needed for stability utilities (pos_weight calc)
        train_dataset=train_dataset,
        train_idx=train_indices,
    )

    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        # Save the best model based on validation PR-AUC (the correct metric for PCBA)
        ModelCheckpoint(
            monitor="val/pr_auc",
            mode="max",
            dirpath="checkpoints/",
            filename="best-mean-teacher-{epoch:02d}-{val/pr_auc:.4f}",
            save_top_k=1,
            verbose=True,
        ),
    ]

    # Logger
    wandb_logger = WandbLogger(
        project="semi-supervised-gnn-drug-discovery",
        name=f"MeanTeacher_C{CONSISTENCY_WEIGHT}",
    )

    trainer = L.Trainer(
        max_epochs=NUM_EPOCHS,
        accelerator="auto",
        devices="auto",
        # Use 16-mixed precision for speed/memory efficiency (recommended for GNNs)
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=1.0,
        accumulate_grad_batches=2,
    )

    # Start Training
    print("Starting Mean Teacher Training...")
    trainer.fit(mean_teacher_model, datamodule=data_module)
    print("Training Complete.")
    
    # Test the model
    trainer.test(mean_teacher_model, datamodule=data_module)


if __name__ == "__main__":
    main()