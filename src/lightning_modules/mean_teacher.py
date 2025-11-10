import logging
from copy import deepcopy

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import MetricCollection
from torchmetrics.regression import (  # <-- FIXED: Import regression metrics
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
)

# Set up logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MeanTeacherLoss(nn.Module):
    """
    Mean Teacher Loss combining supervised classification loss and consistency loss
    between student and teacher models.

    L_total = L_supervised(logits_student, labels) + consistency_weight(t) * L_consistency(logits_student, logits_teacher)

    Args:
        consistency_weight (float): Weight for the consistency loss term
            (can change with steps from lambda_t to lambda_T).

    Returns:
        total_loss (torch.Tensor): The combined loss value.
        supervised_loss (torch.Tensor): The supervised loss component.
        consistency_loss (torch.Tensor): The consistency loss component.
    """

    def __init__(self, consistency_weight: float = 1.0):
        super().__init__()

        self.supervised_loss_fn = nn.MSELoss()
        self.consistency_loss_fn = nn.MSELoss()
        self.consistency_weight = consistency_weight

    def forward(
        self,
        labeled_logits_student: torch.Tensor,
        unlabeled_logits_student: torch.Tensor,
        labeled_logits_teacher: torch.Tensor,
        unlabeled_logits_teacher: torch.Tensor,
        labels: torch.Tensor = None,
        consistency_weight: float = 1.0,
    ):
        """
        Compute the Mean Teacher loss.

        Args:
            logits_student: Logits from the student model (B, C)
            logits_teacher: Logits from the teacher model (B, C)
            labels: Ground truth labels (B,) for cross_entropy or (B, C) for bce/mse
                   Can be None for unlabeled data (only consistency loss computed)
            consistency_weight: Optional override for the consistency weight

        Returns:
            tuple: (total_loss, supervised_loss, consistency_loss)
        """
        # Use provided consistency weight or fall back to instance default
        weight = (
            consistency_weight
            if consistency_weight is not None
            else self.consistency_weight
        )

        # Compute supervised loss only if labels are provided
        if labels is not None:
            supervised_loss = self.supervised_loss_fn(labeled_logits_student, labels)
        else:
            supervised_loss = torch.tensor(0.0, device=labeled_logits_student.device)

        # Concatenate labeled and unlabeled logits for consistency loss
        logits_student = torch.cat(
            [labeled_logits_student, unlabeled_logits_student], dim=0
        )
        logits_teacher = torch.cat(
            [labeled_logits_teacher, unlabeled_logits_teacher], dim=0
        )

        # Use MSE for consistency loss (common practice in Mean Teacher)
        consistency_loss = F.mse_loss(logits_student, logits_teacher)

        # Combine losses
        total_loss = supervised_loss + weight * consistency_loss

        # Do I need to compute
        return total_loss, supervised_loss, consistency_loss

    def update_consistency_weight(self, new_weight: float):
        """
        Update the consistency weight (useful for ramping schedules).

        Args:
            new_weight: New consistency weight value
        """
        self.consistency_weight = new_weight


class MeanTeacherRegressionModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_outputs: int,  # Renamed from num_classes for clarity
        learning_rate: float = 1e-3,
        warmup_epochs: int = 5,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        weights: str = None,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        ema_decay: float = 0.999,
        consistency_weight: float = 1.0,
        consistency_rampup_epochs: int = 80,
        max_consistency_weight: float = 1.0,
    ):
        """
        Mean Teacher model for graph regression with semi-supervised learning.

        Args:
            model: Student model (teacher will be a copy with EMA updates)
            num_outputs: Number of output regression targets (e.g., 1 for QM9)
            ... (other args are the same) ...
        """
        super().__init__()

        self.loss_fn = MeanTeacherLoss(consistency_weight=consistency_weight)

        # Optimizer and scheduler params
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.cosine_period_ratio = cosine_period_ratio
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.momentum = momentum

        # Mean Teacher specific params
        self.ema_decay = ema_decay
        self.consistency_rampup_epochs = consistency_rampup_epochs
        self.max_consistency_weight = max_consistency_weight

        self.num_outputs = num_outputs  # Store number of regression targets

        # Config regression metrics
        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")
        self.test_metrics = self.configure_metrics("test")

        self.save_hyperparameters(ignore=["model"])

        self.student_model = model
        self.teacher_model = deepcopy(model)

        # Set the teacher model to evaluation mode and disable gradients
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        if weights is not None:
            log.info(f"Loading weights from {weights}")
            self.load_weights(weights)

        # Compile student model if requested
        if compile_mode is not None:
            log.info(f"Compiling student model with mode: {compile_mode}")
            self.student_model = torch.compile(self.student_model, mode=compile_mode)

    def get_consistency_weight(self, epoch: int) -> float:
        """
        Calculate consistency weight with sigmoid rampup.

        Args:
            epoch: Current epoch number

        Returns:
            Ramped consistency weight
        """
        if self.consistency_rampup_epochs == 0:
            return self.max_consistency_weight

        # Sigmoid rampup
        current = min(epoch, self.consistency_rampup_epochs)
        phase = 1.0 - current / self.consistency_rampup_epochs
        return float(
            self.max_consistency_weight * torch.exp(torch.tensor(-5.0 * phase * phase))
        )

    def ema_update_teacher(self):
        """
        Update the teacher model parameters using Exponential Moving Average (EMA)
        of the student model parameters.
        """
        with torch.inference_mode():
            for teacher_param, student_param in zip(
                self.teacher_model.parameters(), self.student_model.parameters()
            ):
                teacher_param.data.mul_(self.ema_decay).add_(
                    student_param.data, alpha=1 - self.ema_decay
                )

    def forward(self, x):
        return self.student_model(x)

    def training_step(self, batch, batch_idx):
        labeled = batch["labeled"]
        unlabeled = batch["unlabeled"]

        # Get student predictions
        labeled_preds_st = self(labeled)  # [BL, C]
        unlabeled_preds_st = self(unlabeled)  # [BU, C]

        # Get teacher predictions
        with torch.inference_mode():
            labeled_preds_teach = self.teacher_model(labeled)  # [BL, C]
            unlabeled_preds_teach = self.teacher_model(unlabeled)  # [BU, C]

        # Ensure labels are correct shape for MSE (e.g., [B, 1])
        labels = labeled.y
        if self.num_outputs == 1 and labels.dim() == 1:
            labels = labels.view(-1, 1)

        # Calculate loss
        total_loss, supervised_loss, consistency_loss = self.loss_fn(
            labeled_logits_student=labeled_preds_st,
            unlabeled_logits_student=unlabeled_preds_st,
            labeled_logits_teacher=labeled_preds_teach,
            unlabeled_logits_teacher=unlabeled_preds_teach,
            labels=labels,
            consistency_weight=self.get_consistency_weight(self.current_epoch),
        )

        # Log training metrics
        self.train_metrics(labeled_preds_st, labels)

        # Log losses
        self.log(
            "train/supervised_loss",
            supervised_loss,
            on_step=False,
            on_epoch=True,
            batch_size=labeled.num_graphs,  # Use num_graphs for GNNs
        )
        self.log(
            "train/consistency_loss",
            consistency_loss,
            on_step=False,
            on_epoch=True,
            batch_size=labeled.num_graphs + unlabeled.num_graphs,
        )
        self.log(
            "train/loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=labeled.num_graphs + unlabeled.num_graphs,
        )

        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update teacher model with EMA after each training batch."""
        self.ema_update_teacher()

    def validation_step(self, batch, batch_idx):
        # Use teacher model for validation (more stable)
        with torch.inference_mode():
            teacher_preds = self.teacher_model(batch)

        # Prepare labels
        labels = batch.y
        if self.num_outputs == 1 and labels.dim() == 1:
            labels = labels.view(-1, 1)  # Ensure [B, 1]

        # Compute supervised loss (MSE)
        supervised_loss = F.mse_loss(teacher_preds, labels)

        # Update metrics
        self.val_metrics(teacher_preds, labels)

        self.log(
            "val/loss",
            supervised_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        return supervised_loss

    def test_step(self, batch, batch_idx):
        # Use teacher model for testing (usually performs better)
        teacher_preds = self.teacher_model(batch)

        # Prepare labels
        labels = batch.y
        if self.num_outputs == 1 and labels.dim() == 1:
            labels = labels.view(-1, 1)  # Ensure [B, 1]

        # Compute regression loss (MSE)
        loss = F.mse_loss(teacher_preds, labels)

        # Update metrics
        self.test_metrics(teacher_preds, labels)

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
            self.student_model.parameters(),  # Only optimize student
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            nesterov=self.nesterov,
        )

        try:
            if self.trainer.datamodule:
                steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            else:
                steps_per_epoch = len(self.trainer.train_dataloader)
        except Exception:
            log.warning("Could not determine steps_per_epoch. Using fallback of 1000.")
            steps_per_epoch = 1000

        if steps_per_epoch == 0:
            log.warning("steps_per_epoch is zero. Using fallback of 1000.")
            steps_per_epoch = 1000

        total_steps = self.trainer.max_epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch
        cosine_steps = max(
            1, int(self.cosine_period_ratio * (total_steps - warmup_steps))
        )

        log.info(f"Optimizer: SGD, LR: {self.learning_rate}")
        log.info(
            f"Total steps: {total_steps}, Warmup steps: {warmup_steps}, Cosine steps: {cosine_steps}"
        )
        log.info(
            f"EMA decay: {self.ema_decay}, Max consistency weight: {self.max_consistency_weight}"
        )
        log.info(f"Consistency rampup epochs: {self.consistency_rampup_epochs}")

        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)

        if self.warmup_epochs > 0 and warmup_steps > 0:
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
        return MetricCollection(
            {
                f"{prefix}/mae": MeanAbsoluteError(),
                f"{prefix}/mse": MeanSquaredError(),
                f"{prefix}/r2": R2Score(),
            }
        )

    def on_train_epoch_end(self):
        if self.train_metrics:
            metrics = self.train_metrics.compute()
            self.log_dict(
                metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False
            )
            self.train_metrics.reset()

    def on_validation_epoch_end(self):
        if self.val_metrics:
            metrics = self.val_metrics.compute()
            self.log_dict(
                metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
            )
            self.val_metrics.reset()

    def on_test_epoch_end(self):
        if self.test_metrics:
            metrics = self.test_metrics.compute()
            self.log_dict(
                metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
            )
            self.test_metrics.reset()

    def load_weights(self, weights_path: str):
        """Load pretrained weights into both student and teacher models."""
        state_dict = torch.load(weights_path, map_location="cpu")
        self.student_model.load_state_dict(state_dict)
        self.teacher_model.load_state_dict(state_dict)
        log.info(f"Weights loaded from {weights_path}")


if __name__ == "__main__":
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import TQDMProgressBar
    from pytorch_lightning.loggers import WandbLogger

    from src.data.qm9 import QM9DataModule
    from src.models.gcn import GCN
    from src.utils.path_utils import get_logs_dir

    data_module = QM9DataModule(
        target=0,
        batch_size_train=32,
        batch_size_inference=32,
        num_workers=4,
    )

    # data_module.prepare_data()
    data_module.setup()

    baseline_module = MeanTeacherRegressionModel(
        model=GCN(num_node_features=data_module.num_features, hidden_channels=128),
        num_outputs=data_module.num_classes,
    )

    # Run trainer in debug mode fast_dev_run=True,
    trainer = Trainer(
        max_epochs=10,
        callbacks=[
            TQDMProgressBar(),
        ],
        logger=WandbLogger(name="drug_discovery_regression", save_dir=get_logs_dir()),
    )
    trainer.fit(baseline_module, datamodule=data_module)
