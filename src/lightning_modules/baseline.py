import logging

import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import MetricCollection

from src.utils.ogb_metrics import (
    MultitaskAccuracy,
    MultitaskAveragePrecision,
    MultitaskF1,
    MultitaskPR_AUC,
    MultitaskROC_AUC,
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
        optimizer: str = "adamw",
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        loss_weights: torch.Tensor = None,
    ):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(ignore=["model"])

        # Metrics config
        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")
        self.test_metrics = self.configure_metrics("test")

        # Loss config
        pos_weights = (
            self.hparams.loss_weights if self.hparams.loss_weights is not None else None
        )
        self.loss_fn = BCEWithLogitsLoss(reduction="none", pos_weight=pos_weights)

        # Model config
        self.model = model

        # Load weights
        if self.hparams.weights is not None:
            self.load_weights(self.hparams.weights)

        # Compile model
        if self.hparams.compile:
            self.model = torch.compile(self.model, dynamic=True)

    def forward(self, x):
        return self.model(x)

    def _compute_masked_loss(self, logits, targets):
        """
        Computes loss while ignoring NaN values in targets.
        """
        is_labeled = ~torch.isnan(targets)
        targets_safe = torch.where(is_labeled, targets, torch.zeros_like(targets))
        loss_matrix = self.loss_fn(logits, targets_safe)
        masked_loss = loss_matrix * is_labeled.float()
        final_loss = masked_loss.sum() / (is_labeled.sum() + 1e-8)
        return final_loss

    def training_step(self, batch, batch_idx):
        data = batch["labeled"]

        logits = self(data)
        labels = data.y.float()
        loss = self._compute_masked_loss(logits, labels)

        self.train_metrics.update(logits.float(), labels)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=data.num_graphs,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        with torch.inference_mode():
            logits = self(batch)

        labels = batch.y.float()
        loss = self._compute_masked_loss(logits, labels)

        self.val_metrics.update(logits.float(), labels)

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
            logits = self(batch)

        labels = batch.y.float()
        loss = self._compute_masked_loss(logits, labels)

        self.test_metrics.update(logits.float(), labels)

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
        if self.hparams.optimizer == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=self.hparams.momentum,
                nesterov=self.hparams.nesterov,
            )
        else:
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                amsgrad=False,
                betas=(0.9, 0.98),
            )

        steps_per_epoch = (
            self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        )

        total_warmup_steps = int(self.hparams.warmup_epochs * steps_per_epoch)
        # cosine_half_period is from max to min
        cosine_steps = int(
            self.hparams.cosine_period_ratio
            * (self.trainer.max_epochs * steps_per_epoch - total_warmup_steps)
        )

        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1.0 / 1000,
            total_iters=total_warmup_steps,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[total_warmup_steps],
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,  # scheduler is updated after each batch
        }

        return [optimizer], [scheduler_config]

    def configure_metrics(self, prefix: str):
        kwargs = {"num_tasks": self.num_outputs}

        return MetricCollection(
            {
                f"{prefix}/roc_auc": MultitaskROC_AUC(**kwargs),
                f"{prefix}/ap": MultitaskAveragePrecision(**kwargs),
                f"{prefix}/pr_auc": MultitaskPR_AUC(**kwargs),
                f"{prefix}/acc": MultitaskAccuracy(**kwargs),
                f"{prefix}/f1": MultitaskF1(**kwargs),
            }
        )

    def on_train_epoch_end(self):
        if self.train_metrics:
            metrics = self.train_metrics.compute()
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
            self.train_metrics.reset()

    def on_validation_epoch_end(self):
        if self.val_metrics:
            metrics = self.val_metrics.compute()
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
            self.val_metrics.reset()

    def on_test_epoch_end(self):
        if self.test_metrics:
            metrics = self.test_metrics.compute()
            self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)
            self.test_metrics.reset()

    def load_weights(self, weights_path: str):
        device = next(self.model.parameters()).device

        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
        self.load_state_dict(checkpoint["state_dict"], strict=True)
        log.info(f"Loaded weights from {weights_path}")


if __name__ == "__main__":
    pass
