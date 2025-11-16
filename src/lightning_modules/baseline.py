import logging

import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaselineModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_outputs: int,
        learning_rate: float = 1e-3,
        warmup_epochs: int = 5,
        cosine_period_ratio: float = 1,
        compile: bool = True,
        weights: str = None,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        augmentations: bool = True,
        task_type: str = "regression",
    ):
        """
        Mean Teacher model for graph regression with semi-supervised learning.

        Args:
            model: Student model (teacher will be a copy with EMA updates)
            num_outputs: Number of output regression targets (e.g., 1 for QM9)
            ... (other args are the same) ...
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(ignore=["model"])

        # Metrics config
        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")
        self.test_metrics = self.configure_metrics("test")

        # Loss config
        self.loss_fn = self._setup_loss()

        # Model config
        self.model = model

        # Load weights
        if self.hparams.weights is not None:
            self.load_weights(self.hparams.weights)

        # Compile model
        if self.hparams.compile:
            self.model = torch.compile(self.model, dynamic=True)

    def _setup_loss(self):
        """Setup loss function based on task type."""

        task = self.hparams.task_type
        n_outputs = self.hparams.num_outputs

        if task == "regression":
            return MSELoss()
        elif task == "classification" and n_outputs == 1:
            return BCEWithLogitsLoss()
        elif task == "classification" and n_outputs > 1:
            return BCEWithLogitsLoss(reduction="none")
        else:
            raise ValueError(f"Unsupported task type: {task}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        labeled = batch["labeled"]
        logits = self(labeled)

        labels = labeled.y
        mask = ~torch.isnan(labels)

        if mask.sum() > 0:
            loss_unreduced = self.loss_fn(
                logits, labels.float()
            )
            loss = (loss_unreduced * mask.float()).sum() / mask.sum()

            
            self.train_metrics(logits[mask], labels[mask].int())
        else:
            loss = (logits * 0.0).sum()

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=labeled.num_graphs,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.inference_mode():
            logits = self(batch)  # [batch_size, num_tasks]

        labels = batch.y  # [batch_size, num_tasks]
        
        mask = ~torch.isnan(labels)  # [batch_size, num_tasks]
        
        if mask.sum() > 0:
            loss_unreduced = self.loss_fn(
                logits, 
                labels.float(), 
            )  # [batch_size, num_tasks]
            
            loss = (loss_unreduced * mask.float()).sum() / mask.sum()
            
            self.val_metrics(logits[mask], labels[mask].int())
        else:
            loss = (logits * 0.0).sum()
        
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
            logits = self(batch)  # [batch_size, num_tasks]

        labels = batch.y  # [batch_size, num_tasks]
        
        mask = ~torch.isnan(labels)  # [batch_size, num_tasks]
        
        if mask.sum() > 0:
            loss_unreduced = self.loss_fn(
                logits, 
                labels.float(), 
            )  # [batch_size, num_tasks]
            
            loss = (loss_unreduced * mask.float()).sum() / mask.sum()
            
            self.val_metrics(logits[mask], labels[mask].int())
        else:
            loss = (logits * 0.0).sum()
        
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        
        return loss

    def configure_optimizers(self):
        optimizer = SGD(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            momentum=self.hparams.momentum,
            nesterov=self.hparams.nesterov,
        )

        steps_per_epoch = self.trainer.estimated_stepping_batches
        total_steps = self.trainer.max_epochs * steps_per_epoch
        warmup_steps = self.hparams.warmup_epochs * steps_per_epoch

        cosine_steps = max(
            1, int(self.hparams.cosine_period_ratio * (total_steps - warmup_steps))
        )
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)

        if self.hparams.warmup_epochs > 0 and warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1.0 / 1000,
                total_iters=warmup_steps,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = cosine_scheduler

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler_config]

    def configure_metrics(self, prefix: str):
        """Configure metrics for training, validation, and testing."""
        if self.hparams.task_type == "regression":
            return MetricCollection(
                {
                    f"{prefix}/mae": MeanAbsoluteError(),
                    f"{prefix}/mse": MeanSquaredError(),
                    f"{prefix}/r2": R2Score(),
                }
            )
        elif (
            self.hparams.task_type == "classification" and self.hparams.num_outputs == 1
        ):
            # Binary classification - PR AUC is crucial for imbalanced data
            return MetricCollection(
                {
                    f"{prefix}/pr_auc": BinaryAveragePrecision(),  # PR AUC
                    f"{prefix}/auroc": BinaryAUROC(),
                    f"{prefix}/f1": BinaryF1Score(),
                    f"{prefix}/precision": BinaryPrecision(),
                    f"{prefix}/recall": BinaryRecall(),
                }
            )
        elif (
            self.hparams.task_type == "classification" and self.hparams.num_outputs > 1
        ):
            # Binary classification - PR AUC is crucial for imbalanced data
            return MetricCollection(
                {
                    f"{prefix}/pr_auc": BinaryAveragePrecision(),  # PR AUC
                    f"{prefix}/auroc": BinaryAUROC(),
                    f"{prefix}/f1": BinaryF1Score(),
                    f"{prefix}/precision": BinaryPrecision(),
                    f"{prefix}/recall": BinaryRecall(),
                }
            )
            # # Multiclass classification
            # return MetricCollection(
            #     {
            #         f"{prefix}/accuracy": MultilabelAccuracy(
            #             num_labels=self.hparams.num_outputs
            #         ),
            #         f"{prefix}/auroc": MultilabelAUROC(
            #             num_labels=self.hparams.num_outputs
            #         ),
            #         f"{prefix}/f1_macro": MultilabelF1Score(
            #             num_labels=self.hparams.num_outputs, average="macro"
            #         ),
            #         f"{prefix}/f1_weighted": MultilabelF1Score(
            #             num_labels=self.hparams.num_outputs, average="weighted"
            #         ),
            #         f"{prefix}/precision_macro": MultilabelPrecision(
            #             num_labels=self.hparams.num_outputs, average="macro"
            #         ),
            #         f"{prefix}/recall_macro": MultilabelRecall(
            #             num_labels=self.hparams.num_outputs, average="macro"
            #         ),
            #     }
            # )
        else:
            raise ValueError(f"Unsupported task type: {self.hparams.task_type}")

    def on_train_epoch_end(self):
        """Log training metrics at the end of training epoch."""
        if self.train_metrics:
            metrics = self.train_metrics.compute()
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
            self.train_metrics.reset()

    def on_validation_epoch_end(self):
        """Log validation metrics at the end of validation epoch."""
        if self.val_metrics:
            metrics = self.val_metrics.compute()
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
            self.val_metrics.reset()

    def on_test_epoch_end(self):
        """Log test metrics at the end of testing epoch."""
        if self.test_metrics:
            metrics = self.test_metrics.compute()
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
            self.test_metrics.reset()

    def load_weights(self, weights_path: str):
        """Load pretrained weights into both student and teacher models."""
        state_dict = torch.load(weights_path, map_location="cpu")
        self.student_model.load_state_dict(state_dict)
        log.info(f"Weights loaded from {weights_path}")


if __name__ == "__main__":
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger

    from src.data.qm9 import QM9DataModule
    from src.models.gcn import GCN
    from src.models.gin import GIN

    data_module = QM9DataModule(
        batch_size_train=128,
        batch_size_inference=256,
        num_workers=4,
        target=0,
    )

    model_name = "GIN"

    if model_name == "GIN":
        model = GIN(
            in_channels=11,
            hidden_channels=128,
            out_channels=1,
            num_layers=5,
        )
    else:
        model = GCN(
            num_node_features=11,
            hidden_channels=128,
        )

    # mean_teacher_model = MeanTeacherRegressionModel(
    #     model=model,
    #     num_outputs=1,
    #     learning_rate=0.01,
    #     augmentations=True,
    #     compile=True,
    # )

    trainer = Trainer(
        max_epochs=150,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=WandbLogger(project="mean-teacher-graph-regression"),
    )

    # trainer.fit(model=mean_teacher_model, datamodule=data_module)
