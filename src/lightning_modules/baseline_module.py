import logging
from typing import Callable, Optional

import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import MetricCollection

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
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaselineModel(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        warmup_epochs: int = 5,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        weights: str = None,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
    ):
        super().__init__()

        # Loss function for binary and multi-class classification
        self.loss_fn = (
            nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss()
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

        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")
        self.test_metrics = self.configure_metrics("test")

        self.save_hyperparameters(ignore=["model"])

        self.model = model

        if weights is not None:
            log.info(f"Loading weights from {weights}")
            self.load_weights(weights)

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
        loss = self.loss_fn(logits, labels)
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
        logits = self.model(batch)
        labels = batch.y.view(-1, self.num_classes).float()
        loss = self.loss_fn(logits, labels)
        self.val_metrics(logits, labels)
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.y.size(0),
        )
        return loss

    def configure_optimizers(self):
        optimizer = SGD(
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
        state_dict = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict)

        log.info(f"Weights loaded from {weights_path}")


if __name__ == "__main__":
    from src.models.gcn import GCN
    from src.data.qm9 import QM9DataModule
    from src.data.moleculenet import MoleculeNetDataModule
    from pytorch_lightning.callbacks import TQDMProgressBar
    from pytorch_lightning import Trainer

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
    )

    # Run trainer in debug mode fast_dev_run=True,
    trainer = Trainer(max_epochs=100, callbacks=[TQDMProgressBar()], logger=False)
    trainer.fit(baseline_module, datamodule=data_module)
