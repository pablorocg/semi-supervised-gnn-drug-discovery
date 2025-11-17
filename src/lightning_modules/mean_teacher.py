import logging
<<<<<<< HEAD
from typing import Callable, Optional
=======
>>>>>>> origin/main

import pytorch_lightning as L
import torch
import torch.nn as nn
<<<<<<< HEAD
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import MetricCollection

from ema_pytorch import EMA
import torch.nn.functional as F


from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAveragePrecision,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassAveragePrecision,
    MulticlassAUROC,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

# Set up logging
=======
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

>>>>>>> origin/main
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


<<<<<<< HEAD
class BaselineModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        task_type: str = "classification",
        learning_rate: float = 1e-3,
        warmup_epochs: int = 5,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
=======
class MeanTeacherLoss(nn.Module):
    """
    Calculates the Mean Teacher loss based on arXiv:1703.01780v6.

    This loss combines a supervised loss (sum of MSE) on labeled data and
    a consistency loss (mean of MSE) on all data (labeled + unlabeled).

    L_total = L_supervised_sum + consistency_weight(t) * L_consistency_mean

    The supervised loss uses reduction='sum' to correctly scale its gradient
    contribution by the number of labeled examples in the batch, as described
    in the paper's appendix (B.1).

    The consistency loss uses reduction='mean' (default).
    """

    def __init__(
        self,
        initial_weight: float = 0.0,
        max_weight: float = 1.0,
        steps: int = 5000,
    ):
        """
        Initializes the stateless Mean Teacher Loss module.
        """
        super().__init__()

        self.max_weight = max_weight
        self.initial_weight = initial_weight
        self.steps = steps

    def compute_consistency_weight(self, timestep: float) -> torch.Tensor:
        """
        Computes the consistency weight at a given timestep using a sigmoid ramp-up.

        Args:
            timestep: Current training step (0 to max_steps).
        Returns:
            torch.Tensor: The consistency weight at the current timestep.
        """
        if timestep >= self.steps:
            return torch.tensor(self.max_weight)
        else:
            phase = 1.0 - timestep / self.steps
            weight = self.max_weight * float(torch.exp(-5.0 * phase * phase))
            return torch.tensor(weight)

    def forward(
        self,
        labeled_logits_student: torch.Tensor,
        unlabeled_logits_student: torch.Tensor,
        labeled_logits_teacher: torch.Tensor,
        unlabeled_logits_teacher: torch.Tensor,
        labels: torch.Tensor,
        timestep: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the Mean Teacher loss.

        Args:
            labeled_logits_student: Student predictions for labeled data [BL, C].
            unlabeled_logits_student: Student predictions for unlabeled data [BU, C].
            labeled_logits_teacher: Teacher predictions for labeled data [BL, C].
                                     (Should be computed within torch.no_grad())
            unlabeled_logits_teacher: Teacher predictions for unlabeled data [BU, C].
                                      (Should be computed within torch.no_grad())
            labels: Ground truth for labeled data [BL, C].
            consistency_weight: The current, ramped-up consistency weight (w(t)).

        Returns:
            A tuple containing:
            - total_loss (torch.Tensor): The combined loss.
            - supervised_loss (torch.Tensor): The supervised component (sum).
            - consistency_loss (torch.Tensor): The consistency component (mean).
        """

        # Consistency Loss (L_consistency_mean)
        # Calculated on ALL data (labeled + unlabeled).
        # We concatenate the tensors for a single, efficient calculation.
        all_logits_student = torch.cat(
            [labeled_logits_student, unlabeled_logits_student], dim=0
        )
        all_logits_teacher = torch.cat(
            [labeled_logits_teacher, unlabeled_logits_teacher], dim=0
        )

        consistency_loss = self.compute_consistency_weight(timestep) * F.mse_loss(
            all_logits_student, all_logits_teacher
        )

        # Supervised Loss (L_supervised_sum)
        # Calculated ONLY on labeled data.
        supervised_loss = F.mse_loss(labeled_logits_student, labels, reduction="mean")

        # L_total = 100 * p_labeled * L_supervised_sum + w(t) * L_consistency_mean
        total_loss = supervised_loss + consistency_loss

        return total_loss, supervised_loss, consistency_loss


class MeanTeacherRegressionModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_outputs: int,
        learning_rate: float = 1e-3,
        warmup_epochs: int = 5,
        cosine_period_ratio: float = 1,
        compile: bool = True,
>>>>>>> origin/main
        weights: str = None,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
<<<<<<< HEAD
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
        lambda_consistency: float = 1.0, 
        ema_decay: float = 0.999,
    ):
        super().__init__()

        # Loss function for binary and multi-class classification
        self.loss_fn = (
            if task_type = "regression" 
            self.loss_fn = nn.mse_loss
            if num_classes == 1
            self.loss_fn = nn.BCEWithLogitsLoss() 
            else nn.mse_loss()
        )

        # Optimizer and scheduler params
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.cosine_period_ratio = cosine_period_ratio
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.momentum = momentum

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

        self.num_classes = num_classes

=======
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
>>>>>>> origin/main
        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")
        self.test_metrics = self.configure_metrics("test")

<<<<<<< HEAD
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.ema_decay = ema_decay
        self.lambda_consistency = lambda_consistency

        if weights is not None:
            log.info(f"Loading weights from {weights}")
            self.load_weights(weights)

        self.ema = EMA(self.model, beta=self.ema_decay)

        self.model = (
            torch.compile(model, mode=compile_mode)
            if compile_mode is not None
            else model
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # Graph classification step for training
        logits = self.model(batch)
        labels = batch.y.view(-1, self.num_classes).float()
        # Supervised loss
        supervised_loss = self.loss_fn(logits, labels)

        #Teacher forward pass (no grad)
        with torch.no_grad():
            teacher_logits = self.ema.ema_model(batch)
        # Consistency loss
        lambda_consistency = self.lambda_consistency # weight for consistency loss
        consistency_loss = F.mse_loss(logits, teacher_logits)

        # Total loss
        loss = supervised_loss + lambda_consistency * consistency_loss

        # Update metrics with student logits
        self.train_metrics(logits, labels)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.y.size(0),
        )
        return loss


    def validation_step(self, batch, batch_idx):
        logits = self.ema.ema_model(batch)
        labels = batch.y.view(-1, self.num_classes).float()
        loss = self.loss_fn(logits, labels)
        self.val_metrics(logits, labels)
        self.log(
            "val/loss",
=======
        # Loss config
        self.loss_fn = MeanTeacherLoss()

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

        self._config_consistency_weight()

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
            timestep=self.global_step,
        )

        # Log training metrics
        self.train_metrics(labeled_preds_st, labels)  # [BL, C]

        # Log losses
        self.log(
            "train/supervised_loss",
            supervised_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/consistency_loss",
            consistency_loss,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # Log consistency weight
        self.log(
            "train/consistency_weight",
            consistency_weight,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
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
>>>>>>> origin/main
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
<<<<<<< HEAD
            batch_size=batch.y.size(0),
=======
>>>>>>> origin/main
        )
        return loss

    def configure_optimizers(self):
        optimizer = SGD(
<<<<<<< HEAD
            self.model.parameters(),
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

        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)

        if self.warmup_epochs > 0 and warmup_steps > 0:
=======
            self.student_model.parameters(),
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
>>>>>>> origin/main
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
<<<<<<< HEAD
        # Binary classification metrics
        if self.num_classes == 1:
            return MetricCollection(
                {
                    f"{prefix}/avg_precision": BinaryAveragePrecision(),
                    f"{prefix}/accuracy": BinaryAccuracy(),
                    f"{prefix}/precision": BinaryPrecision(),
                    f"{prefix}/recall": BinaryRecall(),
                    f"{prefix}/f1": BinaryF1Score(),
                    f"{prefix}/rocauc": BinaryAUROC(),
                }
            )
        # Multi-class classification metrics
        elif self.num_classes > 1:
            return MetricCollection(
                {
                    f"{prefix}/avg_precision": MulticlassAveragePrecision(
                        num_classes=self.num_classes, average="macro"
                    ),
                    f"{prefix}/accuracy": MulticlassAccuracy(
                        num_classes=self.num_classes, average="macro"
                    ),
                    f"{prefix}/precision": MulticlassPrecision(
                        num_classes=self.num_classes, average="macro"
                    ),
                    f"{prefix}/recall": MulticlassRecall(
                        num_classes=self.num_classes, average="macro"
                    ),
                    f"{prefix}/f1": MulticlassF1Score(
                        num_classes=self.num_classes, average="macro"
                    ),
                    f"{prefix}/rocauc": MulticlassAUROC(
                        num_classes=self.num_classes, average="macro"
                    ),
                }
            )
        else:
            raise ValueError("num_classes must be at least 1.")

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Applies transforms to the batch *after* it has been moved to the
        target device (e.g., GPU).
        """
        if self.trainer.training and self.train_transforms is not None:
            batch = self.train_transforms(batch)
        elif (
            self.trainer.validating or self.trainer.sanity_checking
        ) and self.val_transforms is not None:
            batch = self.val_transforms(batch)
        elif (
            self.trainer.testing or self.trainer.predicting
        ) and self.test_transforms is not None:
            batch = self.test_transforms(batch)
        return batch

    def on_train_epoch_end(self):
=======
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
>>>>>>> origin/main
        if self.train_metrics:
            metrics = self.train_metrics.compute()
            self.log_dict(
                metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=False
            )
            self.train_metrics.reset()

<<<<<<< HEAD
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Update EMA weights after each training step
        self.ema.update()

    def on_validation_epoch_end(self):
=======
    def on_validation_epoch_end(self):
        """Log validation metrics at the end of validation epoch."""
>>>>>>> origin/main
        if self.val_metrics:
            metrics = self.val_metrics.compute()
            self.log_dict(
                metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
            )
            self.val_metrics.reset()

    def on_test_epoch_end(self):
<<<<<<< HEAD
=======
        """Log test metrics at the end of testing epoch."""
>>>>>>> origin/main
        if self.test_metrics:
            metrics = self.test_metrics.compute()
            self.log_dict(
                metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
            )
            self.test_metrics.reset()

    def load_weights(self, weights_path: str):
<<<<<<< HEAD
        state_dict = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict)

        log.info(f"Weights loaded from {weights_path}")


if __name__ == "__main__":
    from src.models.gcn import GCN
    from src.data.qm9 import QM9DataModule
    from src.data.moleculenet import MoleculeNetDataModule
    from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import WandbLogger
    import wandb

    # data_module = QM9DataModule(
    #     target=0,
    #     batch_size_train=32,
    #     batch_size_inference=32,
    #     num_workers=1,
    # )
    data_module = QM9DataModule( # if was to use another dataset, I would have to add the name parameter (moleculenet)
        target=4,  
        batch_size_train=256,
        batch_size_inference=256,
        num_workers=4,
        
    )
    data_module.prepare_data()
    data_module.setup()

    baseline_module = BaselineModel(
        model=GCN(num_node_features=data_module.num_features, hidden_channels=64),
        num_classes=data_module.num_classes,
        lambda_consistency=1.0,
        ema_decay=0.999,
    )

    wandb_logger = WandbLogger(
        project="semi-supervised-gnn",
        name="mean-teacher-qm9",
        log_model=True,
        save_dir="./logs",
    )

    # Watch the model to log gradients and parameters
    wandb_logger.watch(baseline_module.model, log="all", log_freq=100)

    # Add checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="./checkpoints",
        filename="mean-teacher-qm9-{epoch:02d}-{val/loss:.2f}",
        save_top_k=3,
        mode="min",
    )

    # Run trainer in debug mode fast_dev_run=True,
    trainer = Trainer(max_epochs=100,
     callbacks=[TQDMProgressBar(), checkpoint_callback],
      logger=wandb_logger,
      log_every_n_steps=10,
      enable_progress_bar=True,
      enable_model_summary=True,)
    trainer.fit(baseline_module, datamodule=data_module)
=======
        """Load pretrained weights into both student and teacher models."""
        state_dict = torch.load(weights_path, map_location="cpu")
        self.student_model.load_state_dict(state_dict)
        log.info(f"Weights loaded from {weights_path}")

    def _config_consistency_weight(self):
        """Configures the consistency weight schedule."""
        rampup_epochs = self.hparams.consistency_rampup_epochs
        self.total_rampup_steps = (
            rampup_epochs * self.trainer.estimated_stepping_batches
        )

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
        current_step = torch.tensor(
            self.global_step, dtype=torch.float32, device=self.device
        )
        total_steps = torch.tensor(
            self.total_rampup_steps, dtype=torch.float32, device=self.device
        )

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
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=WandbLogger(project="mean-teacher-graph-regression"),
    )

    trainer.fit(model=mean_teacher_model, datamodule=data_module)
>>>>>>> origin/main
