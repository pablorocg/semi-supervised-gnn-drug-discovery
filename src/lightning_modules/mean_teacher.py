import logging

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_ema import ExponentialMovingAverage
from torch_geometric.data import Data
from torch_geometric.utils.augmentation import mask_feature
from torchmetrics import MetricCollection
from torchmetrics.regression import (
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# class FeatureMasking(T.BaseTransform):
#     """
#     A PyG-compatible transform that wraps the 'mask_feature' function.

#     Args:
#         p (float): The masking ratio.
#         mode (str): The masking scheme ('row', 'col', 'all').
#         fill_value (float): The value for masked features.
#     """
#     def __init__(self, p: float = 0.5, mode: str = 'col', fill_value: float = 0.0):
#         self.p = p
#         self.mode = mode
#         self.fill_value = fill_value

#     def forward(self, data: Data) -> Data:
#         # self.training is a special attribute from BaseTransform.
#         # It's automatically set to True by model.train()
#         # and False by model.eval().
#         # This is the *only* way to call your function.

#         # We clone to avoid modifying the original data object in-place
#         data_clone = data.clone()

#         # Call your function, passing self.training to its training flag
#         masked_x, _ = mask_feature(
#             x=data_clone.x,
#             p=self.p,
#             mode=self.mode,
#             fill_value=self.fill_value,
#             training=self.training  # <-- This is the key!
#         )

#         data_clone.x = masked_x
#         return data_clone

#     def __repr__(self) -> str:
#         # This just makes it look nice when you print the transform
#         return f'{self.__class__.__name__}(p={self.p}, mode={self.mode})'


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
            supervised_loss = F.mse_loss(labeled_logits_student, labels)
        else:
            supervised_loss = torch.tensor(0.0, device=labeled_logits_student.device)

        # Concatenate labeled and unlabeled logits for consistency loss
        logits_student = torch.cat(
            [labeled_logits_student, unlabeled_logits_student], dim=0
        )
        logits_teacher = torch.cat(
            [labeled_logits_teacher, unlabeled_logits_teacher], dim=0
        )

        # Use MSE for consistency loss
        consistency_loss = F.mse_loss(logits_student, logits_teacher)

        # Combine losses
        total_loss = supervised_loss + weight * consistency_loss

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
        num_outputs: int,
        learning_rate: float = 1e-3,
        warmup_epochs: int = 5,
        cosine_period_ratio: float = 1,
        compile: bool = True,
        weights: str = None,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        ema_decay: float = 0.999,
        consistency_weight: float = 1e-1,
        consistency_rampup_epochs: int = 5,  # Constant cons weight
        max_consistency_weight: float = 10.0,
        augmentations: bool = True,
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

        # self.augmentations_transform = (
        #     FeatureMasking(p=0.1) if self.hparams.augmentations else None
        # )

        # Metrics config
        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")
        self.test_metrics = self.configure_metrics("test")

        # Loss config
        self.loss_fn = MeanTeacherLoss(self.hparams.consistency_weight)

        # Model config
        self.student_model = model

        # Load weights
        if self.hparams.weights is not None:
            self.load_weights(self.hparams.weights)

        # Compile model
        if self.hparams.compile:
            self.student_model = torch.compile(self.student_model, dynamic=True)
            log.info("Compiled student model with torch.compile()")
        
        # Config EMA for teacher model
        self.ema = ExponentialMovingAverage(
            self.student_model.parameters(), decay=self.hparams.ema_decay
        )
        
    def on_fit_start(self):
        """Move EMA to the correct device after Lightning has moved the model."""
        self.student_model.to(self.device)
        self.ema.to(self.device)

    def forward(self, x):
        return self.student_model(x)

    def training_step(self, batch, batch_idx):
        labeled = batch["labeled"]
        unlabeled = batch["unlabeled"]

        # # Apply augmentations if configured
        # if self.augmentations_transform is not None:
        #     labeled = self.augmentations_transform(labeled)
        #     unlabeled = self.augmentations_transform(unlabeled)

        # Get student predictions
        labeled_preds_st = self(labeled)  # [BL, C]
        unlabeled_preds_st = self(unlabeled)  # [BU, C]

        # Get teacher predictions
        with self.ema.average_parameters():
            labeled_preds_teach = self(labeled)  # [BL, C]
            unlabeled_preds_teach = self(unlabeled)  # [BU, C]

        # Ensure labels are correct shape for MSE (e.g., [BL, 1])
        labels = labeled.y
        if self.hparams.num_outputs == 1 and labels.dim() == 1:
            labels = labels.view(-1, 1)

        # Get current consistency weight based on ramp-up schedule
        consistency_weight = self._get_current_consistency_weight()

        # Calculate loss
        total_loss, supervised_loss, consistency_loss = self.loss_fn(
            labeled_logits_student=labeled_preds_st,  # [BL, C]
            unlabeled_logits_student=unlabeled_preds_st,  # [BU, C]
            labeled_logits_teacher=labeled_preds_teach,  # [BL, C]
            unlabeled_logits_teacher=unlabeled_preds_teach,  # [BU, C]
            labels=labels,  # [BL, C]
            consistency_weight=consistency_weight
        )

        # Log training metrics
        self.train_metrics(labeled_preds_st, labels)  # [BL, C]

        # Log losses
        self.log(
            "train/supervised_loss",
            supervised_loss,
            on_step=False,
            on_epoch=True,
            # batch_size=labeled.num_graphs,  # [BL]
        )
        self.log(
            "train/consistency_loss",
            consistency_loss,
            on_step=True,
            on_epoch=True,
            # batch_size=labeled.num_graphs + unlabeled.num_graphs,  # [BL + BU]
        )
        self.log(
            "train/loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            # batch_size=labeled.num_graphs + unlabeled.num_graphs,  # [BL + BU]
        )

        # Log consistency weight
        self.log(
            "train/consistency_weight",
            consistency_weight,
            on_step=False,
            on_epoch=True,
            prog_bar=False
        )

        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update teacher model with EMA after each training batch."""
        self.ema.update()

    def validation_step(self, batch, batch_idx):
        # Use teacher model for validation (usually performs better)
        with self.ema.average_parameters():
            teacher_preds = self(batch)

        # Prepare labels
        labels = batch.y
        if self.hparams.num_outputs == 1 and labels.dim() == 1:
            labels = labels.view(-1, 1)  # Ensure [B, 1]

        # Compute regression loss (MSE)
        supervised_loss = F.mse_loss(teacher_preds, labels)

        # Update metrics
        self.val_metrics(teacher_preds, labels)
        self.log(
            "val/loss",
            supervised_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )
        return supervised_loss

    def test_step(self, batch, batch_idx):
        # Use teacher model for testing
        with self.ema.average_parameters():
            teacher_preds = self(batch)

        # Prepare labels
        labels = batch.y
        if self.hparams.num_outputs == 1 and labels.dim() == 1:
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

        log.info(f"Optimizer: SGD, LR: {self.hparams.learning_rate}")
        log.info(
            f"Total steps: {total_steps}, Warmup steps: {warmup_steps}, Cosine steps: {cosine_steps}"
        )
        log.info(
            f"EMA decay: {self.hparams.ema_decay}, Max consistency weight: {self.hparams.max_consistency_weight}"
        )
        log.info(f"Consistency rampup epochs: {self.hparams.consistency_rampup_epochs}")

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
        """Configure regression metrics for training, validation, and testing."""
        return MetricCollection(
            {
                f"{prefix}/mae": MeanAbsoluteError(),
                f"{prefix}/mse": MeanSquaredError(),
                f"{prefix}/r2": R2Score(),
            }
        )

    def on_train_epoch_end(self):
        """Log training metrics at the end of training epoch."""
        if self.train_metrics:
            metrics = self.train_metrics.compute()
            self.log_dict(
                metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False
            )
            self.train_metrics.reset()

    def on_validation_epoch_end(self):
        """Log validation metrics at the end of validation epoch."""
        if self.val_metrics:
            metrics = self.val_metrics.compute()
            self.log_dict(
                metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
            )
            self.val_metrics.reset()

    def on_test_epoch_end(self):
        """Log test metrics at the end of testing epoch."""
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
        log.info(f"Weights loaded from {weights_path}")

    def _get_current_consistency_weight(self) -> torch.Tensor:
        """
        Calculates the consistency weight based on the ramp-up schedule
        from the Mean Teacher paper (arXiv:1703.01780v6).

        Returns:
            torch.Tensor: The consistency weight for the current step.
        """
        rampup_epochs = self.hparams.consistency_rampup_epochs
        
        # If ramp-up is 0, use the max weight immediately
        if rampup_epochs == 0 or self.total_rampup_steps == 0:
            return torch.tensor(self.hparams.max_consistency_weight, device=self.device)

        # self.global_step is tracked by Lightning
        current_step = torch.tensor(self.global_step, dtype=torch.float32, device=self.device)
        total_steps = torch.tensor(self.total_rampup_steps, dtype=torch.float32, device=self.device)

        # Calculate current ramp-up fraction, x \in [0, 1]
        x = torch.clamp(current_step / total_steps, 0.0, 1.0)

        # Calculate sigmoid-shaped ramp-up: e^(-5 * (1-x)^2)
        # This is the function described in the paper 
        ramp_value = torch.exp(-5.0 * torch.pow(1.0 - x, 2))

        current_weight = self.hparams.max_consistency_weight * ramp_value
        return current_weight


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

    model_name = "GCN"

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

    mean_teacher_model = MeanTeacherRegressionModel(
        model=model,
        num_outputs=1,
        learning_rate=0.01,
        augmentations=True,
        compile=True,
    )

    trainer = Trainer(
        max_epochs=150,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true" ,
        logger=WandbLogger(project="mean-teacher-graph-regression"),
    )

    trainer.fit(model = mean_teacher_model, datamodule=data_module)
