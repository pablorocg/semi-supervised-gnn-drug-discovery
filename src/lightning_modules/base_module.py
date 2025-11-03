import logging
from abc import abstractmethod
from typing import Callable, Optional

import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import MetricCollection

from src.utils.metrics import MultiTaskAP, MultiTaskRMSE, MultiTaskROCAUC

# Set up logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        warmup_epochs: int = 5,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        weights: str = None,
        optimizer: str = "SGD",
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_fn = None
        self.warmup_epochs = warmup_epochs
        self.cosine_period_ratio = cosine_period_ratio
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.momentum = momentum
        assert 0 < cosine_period_ratio <= 1

        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

        self.num_classes = num_classes
        self.save_hyperparameters(ignore=["model"])

        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")
        self.test_metrics = self.configure_metrics("test")

        self.model = model

        if weights is not None:
            # log.info(f"Loading weights from {weights}")
            # self.load_weights(weights)
            print("Loading weights is not implemented yet.")

        self.model = (
            torch.compile(model, mode=compile_mode)
            if compile_mode is not None
            else model
        )

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        if self.optimizer == "SGD":
            optimizer = SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                nesterov=self.nesterov,
            )
        elif self.optimizer == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                amsgrad=False,
                betas=(0.9, 0.98),
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

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

        log.info(f"Optimizer: {self.optimizer}, LR: {self.learning_rate}")
        log.info(
            f"Total steps: {total_steps}, Warmup steps: {warmup_steps}, Cosine steps: {cosine_steps}"
        )

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
                f"{prefix}/rocauc": MultiTaskROCAUC(),
                f"{prefix}/ap": MultiTaskAP(),
                f"{prefix}/rmse": MultiTaskRMSE(),
            }
        )

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
        if self.train_metrics:
            metrics = self.train_metrics.compute()
            self.log_dict(
                metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
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
