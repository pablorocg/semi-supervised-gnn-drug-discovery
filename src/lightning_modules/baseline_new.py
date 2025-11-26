import os
import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.optim import AdamW
from torchmetrics import MetricCollection, MeanMetric
from torch.nn import BCEWithLogitsLoss, MSELoss

from src.models.gine import GINE
from src.utils.utils import calculate_pos_weights_from_tensor
from src.utils.ogb_metrics import MultiTaskAccuracy, MultiTaskAP, MultiTaskROCAUC
from src.utils.path_utils import get_data_dir, get_logs_dir, get_models_dir, ensure_dir_exists

# Directories
data_dir = get_data_dir()
logs_dir = get_logs_dir()
models_dir = get_models_dir()

ensure_dir_exists(data_dir)
ensure_dir_exists(logs_dir)
ensure_dir_exists(models_dir)


class BaselineModule(L.LightningModule):
    def __init__(
        self,
        num_outputs: int,
        num_node_features: int,
        model: GINE = None,
        task_type: str = "classification",
        train_dataset=None,
        train_idx=None,
        learning_rate: float = 1e-4,
        dropout: float = 0.3,
        weight_decay: float = 1e-5,
        hidden_dim: int = 64,
        num_layers: int = 5,
        num_mlp_layers: int = 2,
        use_residual: bool = True,
        use_virtual_node: bool = True,
        use_attention_pool: bool = True,
        use_layer_norm: bool = True,
        weights: str = None,
        compile: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_dataset", "train_idx", "model"])

        self.model = model or GINE(
            num_tasks=num_outputs,
            num_node_features=num_node_features,
            num_edge_features=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_residual=use_residual,
            use_virtual_node=use_virtual_node,
            use_attention_pool=use_attention_pool,
            num_mlp_layers=num_mlp_layers,
            use_layer_norm=use_layer_norm,
        )

        if task_type == "classification" and num_outputs > 1 and train_idx is not None and train_dataset is not None:
            train_labels = train_dataset.data.y[train_idx].float()
            self.pos_weight = calculate_pos_weights_from_tensor(train_labels, max_weight=20.0)
        else:
            self.pos_weight = None

        self.supervised_loss_fn = self._setup_loss()
        
        # Metrics for classification performance
        self.train_metrics = self._configure_metrics("train")
        self.val_metrics = self._configure_metrics("val")
        self.test_metrics = self._configure_metrics("test")
        
        # Mean metrics for tracking average loss per epoch
        self.train_loss_epoch = MeanMetric()
        self.val_loss_epoch = MeanMetric()
        self.test_loss_epoch = MeanMetric()

        if weights is not None:
            self.load_weights(weights)

        if compile:
            self.model = torch.compile(self.model, dynamic=True)

    def _setup_loss(self):
        if self.hparams.task_type == "regression":
            return MSELoss(reduction='none')
        elif self.hparams.task_type == "classification":
            return BCEWithLogitsLoss(reduction='none')
        else:
            raise ValueError(f"Unsupported task type: {self.hparams.task_type}")

    def forward(self, x):
        return self.model(x)

    def _masked_loss(self, logits, labels, mask):
        valid_logits = logits[mask]
        valid_labels = labels[mask]
        
        if len(valid_logits) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        per_elem = self.supervised_loss_fn(valid_logits, valid_labels)
        
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(logits.device)
            task_indices = mask.nonzero(as_tuple=True)[1]
            task_weights = pos_weight[task_indices]
            weights = valid_labels * (task_weights - 1) + 1
            per_elem = per_elem * weights
        
        return per_elem.mean()

    def training_step(self, batch, batch_idx):
        labeled_batch = batch["labeled"]
        logits = self(labeled_batch)
        labels = labeled_batch.y.float()
        mask = ~torch.isnan(labels)
        
        if mask.sum() == 0:
            return None
        
        loss = self._masked_loss(logits, labels, mask)
        
        if torch.isnan(loss) or torch.isinf(loss):
            return None
        
        # Update metrics
        self.train_metrics.update(logits, labels)
        self.train_loss_epoch.update(loss)
        
        # Log step-level loss (for progress bar)
        self.log("train/loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch.get("labeled", batch) if isinstance(batch, dict) else batch
        logits = self(batch)
        labels = batch.y.float()
        mask = ~torch.isnan(labels)
        loss = self._masked_loss(logits, labels, mask)
        
        # Update metrics
        self.val_metrics.update(logits, labels)
        self.val_loss_epoch.update(loss)
        
        # Log validation loss
        self.log("val/loss_step", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss

    def test_step(self, batch, batch_idx):
        batch = batch.get("labeled", batch) if isinstance(batch, dict) else batch
        logits = self(batch)
        labels = batch.y.float()
        mask = ~torch.isnan(labels)
        loss = self._masked_loss(logits, labels, mask)
        
        # Update metrics
        self.test_metrics.update(logits, labels)
        self.test_loss_epoch.update(loss)
        
        return loss

    def _configure_metrics(self, prefix):
        return MetricCollection({
            f"{prefix}/pr_auc": MultiTaskAP(num_tasks=self.hparams.num_outputs),
            f"{prefix}/auroc": MultiTaskROCAUC(num_tasks=self.hparams.num_outputs),
            f"{prefix}/accuracy": MultiTaskAccuracy(num_tasks=self.hparams.num_outputs),
        })

    def on_train_epoch_end(self):
        # Compute and log metrics
        metrics = self.train_metrics.compute()
        avg_loss = self.train_loss_epoch.compute()
        
        # Add epoch loss to metrics dict
        metrics["train/loss_epoch"] = avg_loss
        
        # Log everything
        self.log_dict(metrics, on_epoch=True)
        
        # Print summary
        print(f"\nEpoch {self.current_epoch} Training Summary:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  AUROC: {metrics['train/auroc']:.4f}")
        print(f"  PR-AUC: {metrics['train/pr_auc']:.4f}")
        print(f"  Accuracy: {metrics['train/accuracy']:.4f}")
        
        # Reset metrics
        self.train_metrics.reset()
        self.train_loss_epoch.reset()

    def on_validation_epoch_end(self):
        # Compute and log metrics
        metrics = self.val_metrics.compute()
        avg_loss = self.val_loss_epoch.compute()
        
        # Add epoch loss to metrics dict
        metrics["val/loss_epoch"] = avg_loss
        
        # Log everything
        self.log_dict(metrics, on_epoch=True)
        
        # Print summary
        print(f"\nEpoch {self.current_epoch} Validation Summary:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  AUROC: {metrics['val/auroc']:.4f}")
        print(f"  PR-AUC: {metrics['val/pr_auc']:.4f}")
        print(f"  Accuracy: {metrics['val/accuracy']:.4f}\n")
        
        # Reset metrics
        self.val_metrics.reset()
        self.val_loss_epoch.reset()

    def on_test_epoch_end(self):
        # Compute and log metrics
        metrics = self.test_metrics.compute()
        avg_loss = self.test_loss_epoch.compute()
        
        # Add epoch loss to metrics dict
        metrics["test/loss_epoch"] = avg_loss
        
        # Log everything
        self.log_dict(metrics, on_epoch=True)
        
        # Print summary
        print(f"\nTest Results:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  AUROC: {metrics['test/auroc']:.4f}")
        print(f"  PR-AUC: {metrics['test/pr_auc']:.4f}")
        print(f"  Accuracy: {metrics['test/accuracy']:.4f}\n")
        
        # Reset metrics
        self.test_metrics.reset()
        self.test_loss_epoch.reset()

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)


if __name__ == "__main__":
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import WandbLogger
    from src.data.pcba import OgbgMolPcbaDataModule

    # DataModule
    data_module = OgbgMolPcbaDataModule(
        target=None,
        batch_size_train=64,
        batch_size_inference=126,
        num_workers=4,
        subset_size=218964
    )
    data_module.setup()

    print("=" * 60)
    print(f"Full dataset: {len(data_module.dataset)} graphs")
    print(f"Labeled dataset: {len(data_module.data_train_labeled)} graphs")
    print(f"First 10 labeled indices: {list(data_module.train_idx)[:10]}")
    print("=" * 60)

    # Baseline model
    baseline_model = BaselineModule(
        num_outputs=128,
        num_node_features=9,
        train_dataset=data_module.data_train_labeled,
        train_idx=data_module.train_idx,
        learning_rate=1e-4,
        weight_decay=1e-5,
        dropout=0.3,
        hidden_dim=256,
        num_layers=5,
        num_mlp_layers=2,
        use_residual=True,
        use_virtual_node=True,
        use_attention_pool=True,
        use_layer_norm=True,
        weights=None,
        compile=False
    )

    if baseline_model.pos_weight is not None:
        print(f"Pos weights - min: {baseline_model.pos_weight.min():.3f}, "
              f"max: {baseline_model.pos_weight.max():.3f}, "
              f"mean: {baseline_model.pos_weight.mean():.3f}")

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=models_dir,
            monitor='val/auroc',
            mode='max',
            save_top_k=3,
            filename='gine-{epoch:02d}-{val/auroc:.4f}',
            save_last=True,
        ),
        EarlyStopping(
            monitor='val/auroc',
            patience=30,
            mode='max',
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    # Trainer
    trainer = Trainer(
        max_epochs=150,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=WandbLogger(project="pcba-baseline", name="gine-run", save_dir=logs_dir, log_model=False),
        callbacks=callbacks,
        gradient_clip_val=1.0,
        val_check_interval=1.0,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    trainer.fit(model=baseline_model, datamodule=data_module)

    print("\n" + "=" * 60)
    print("Testing best model...")
    print("=" * 60 + "\n")
    
    trainer.test(model=baseline_model, datamodule=data_module, ckpt_path='best')