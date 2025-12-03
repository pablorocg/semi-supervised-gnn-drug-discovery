"""Mean Teacher semi-supervised learning module."""

import logging
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics import AUROC, AveragePrecision, MeanMetric, MetricCollection

from src.utils.graph_augmentation import GraphAugmentor

log = logging.getLogger(__name__)


class MeanTeacherModule(pl.LightningModule):
    """Mean Teacher semi-supervised learning module."""

    def __init__(
        self,
        model: nn.Module,
        num_outputs: int,
        learning_rate: float = 0.001,
        warmup_epochs: int = 5,
        cosine_period_ratio: float = 0.5,
        optimizer: str = "adamw",
        weight_decay: float = 0.0001,
        nesterov: bool = True,
        momentum: float = 0.9,
        compile: bool = False,
        weights: Optional[torch.Tensor] = None,
        loss_weights: Optional[torch.Tensor] = None,
        ema_decay: float = 0.999,
        consistency_rampup_epochs: int = 5,
        max_consistency_weight: float = 1.0,
        node_drop_rate: float = 0.1,
        edge_drop_rate: float = 0.1,
        feature_mask_rate: float = 0.1,
        feature_noise_std: float = 0.01,
        edge_attr_noise_std: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.student = model
        self.teacher = AveragedModel(
            model, avg_fn=lambda avg, new, _: ema_decay * avg + (1 - ema_decay) * new
        )

        for teacher_param, student_param in zip(
            self.teacher.parameters(), self.student.parameters()
        ):
            teacher_param.data.copy_(student_param.data)

        self.augmentor = GraphAugmentor(
            node_drop_rate=node_drop_rate,
            edge_drop_rate=edge_drop_rate,
            feature_mask_rate=feature_mask_rate,
            feature_noise_std=feature_noise_std,
            edge_attr_noise_std=edge_attr_noise_std,
        )
        log.info(f"Graph augmentation enabled: {self.augmentor}")

        self.num_outputs = num_outputs
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.cosine_period_ratio = cosine_period_ratio
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.momentum = momentum
        self.ema_decay = ema_decay
        self.consistency_rampup_epochs = consistency_rampup_epochs
        self.max_consistency_weight = max_consistency_weight

        self.register_buffer("weights", weights)
        self.register_buffer("loss_weights", loss_weights)

        self.train_metrics = self._configure_metrics("train")
        self.val_metrics = self._configure_metrics("val")
        self.test_metrics = self._configure_metrics("test")

        if compile:
            self.student = torch.compile(self.student)

    def _configure_metrics(self, prefix: str) -> MetricCollection:
        """Configure metrics for training/validation/test."""
        return MetricCollection(
            {
                "roc_auc": AUROC(task="multilabel", num_labels=self.num_outputs),
                "ap": AveragePrecision(task="multilabel", num_labels=self.num_outputs),
                "loss": MeanMetric(),
                "sup_loss": MeanMetric(),
                "cons_loss": MeanMetric(),
                "cons_weight": MeanMetric(),
            },
            prefix=f"{prefix}/",
        )

    def forward(self, data):
        """Forward pass through student model."""
        return self.student(data)

    def _compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute supervised loss with optional masking."""
        if mask is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits[mask], targets[mask], weight=self.weights, reduction="mean"
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, weight=self.weights, reduction="mean"
            )

        if self.loss_weights is not None:
            loss = loss * self.loss_weights

        return loss

    def _sigmoid_rampup(self, current_epoch: int) -> float:
        """Sigmoid ramp-up function for consistency weight."""
        if self.consistency_rampup_epochs == 0:
            return 1.0
        phase = 1.0 - current_epoch / self.consistency_rampup_epochs
        return float(torch.exp(torch.tensor(-5.0 * phase * phase)))

    def training_step(self, batch, batch_idx):
        """Training step with Mean Teacher."""
        labeled, unlabeled = batch

        labeled_aug1 = self.augmentor(labeled)
        unlabeled_aug1 = self.augmentor(unlabeled)
        labeled_aug2 = self.augmentor(labeled)
        unlabeled_aug2 = self.augmentor(unlabeled)

        pred_l_student = self.student(labeled_aug1)
        pred_u_student = self.student(unlabeled_aug1)

        with torch.no_grad():
            pred_l_teacher = self.teacher(labeled_aug2)
            pred_u_teacher = self.teacher(unlabeled_aug2)

        mask = ~torch.isnan(labeled.y)
        supervised_loss = self._compute_loss(pred_l_student, labeled.y, mask)

        consistency_loss_labeled = F.mse_loss(pred_l_student, pred_l_teacher)
        consistency_loss_unlabeled = F.mse_loss(pred_u_student, pred_u_teacher)
        consistency_loss = (consistency_loss_labeled + consistency_loss_unlabeled) / 2

        consistency_weight = (
            self._sigmoid_rampup(self.current_epoch) * self.max_consistency_weight
        )

        total_loss = supervised_loss + consistency_weight * consistency_loss

        self.log("train/loss_step", total_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/sup_loss_step", supervised_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/cons_loss_step", consistency_loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train/cons_weight_step", consistency_weight, on_step=True, on_epoch=False, prog_bar=True)

        self.train_metrics["loss"].update(total_loss)
        self.train_metrics["sup_loss"].update(supervised_loss)
        self.train_metrics["cons_loss"].update(consistency_loss)
        self.train_metrics["cons_weight"].update(consistency_weight)

        return total_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update teacher model with EMA after each batch."""
        self.teacher.update_parameters(self.student)

    def on_train_epoch_end(self):
        """Update batch normalization statistics for the teacher model."""
        self.log("train/loss_epoch", self.train_metrics["loss"].compute())
        self.log("train/sup_loss_epoch", self.train_metrics["sup_loss"].compute())
        self.log("train/cons_loss_epoch", self.train_metrics["cons_loss"].compute())
        self.log("train/cons_weight_epoch", self.train_metrics["cons_weight"].compute())

        self.train_metrics.reset()

        if (self.current_epoch + 1) % 10 == 0:
            log.info(f"Updating teacher batch norm at epoch {self.current_epoch}")

            try:
                device = next(self.student.parameters()).device
            except StopIteration:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if isinstance(device, str):
                device = torch.device(device)

            try:
                update_bn(self.trainer.train_dataloader, self.teacher.module, device=device)
            except Exception as e:
                log.warning(f"Failed to update batch norm: {e}")

    def validation_step(self, batch, batch_idx):
        """Validation step using teacher model."""
        pred = self.teacher(batch)
        mask = ~torch.isnan(batch.y)
        loss = self._compute_loss(pred, batch.y, mask)

        self.val_metrics["loss"].update(loss)

        pred_probs = torch.sigmoid(pred)
        target_clean = torch.where(torch.isnan(batch.y), torch.zeros_like(batch.y), batch.y).long()

        if mask.any():
            self.val_metrics["roc_auc"].update(pred_probs, target_clean)
            self.val_metrics["ap"].update(pred_probs, target_clean)

        return loss

    def on_validation_epoch_end(self):
        """Log validation metrics."""
        metrics = self.val_metrics.compute()
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        """Test step using teacher model."""
        pred = self.teacher(batch)
        mask = ~torch.isnan(batch.y)
        loss = self._compute_loss(pred, batch.y, mask)

        self.test_metrics["loss"].update(loss)

        pred_probs = torch.sigmoid(pred)
        target_clean = torch.where(torch.isnan(batch.y), torch.zeros_like(batch.y), batch.y).long()

        if mask.any():
            self.test_metrics["roc_auc"].update(pred_probs, target_clean)
            self.test_metrics["ap"].update(pred_probs, target_clean)

        return loss

    def on_test_epoch_end(self):
        """Log test metrics."""
        metrics = self.test_metrics.compute()
        for name, value in metrics.items():
            self.log(name, value)
        self.test_metrics.reset()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.student.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.student.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.student.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=self.nesterov,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return (epoch + 1) / self.warmup_epochs
            else:
                progress = (epoch - self.warmup_epochs) / (
                    self.trainer.max_epochs * self.cosine_period_ratio - self.warmup_epochs
                )
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }