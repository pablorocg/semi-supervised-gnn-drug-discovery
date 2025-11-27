import logging
import copy
import random

import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics import MetricCollection

from src.utils.ogb_metrics import (
    MultitaskAveragePrecision,
    MultitaskF1,
    MultitaskPR_AUC,
    MultitaskROC_AUC,
)

log = logging.getLogger(__name__)


class MeanTeacherModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
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
        ema_decay: float = 0.999,
        consistency_rampup_epochs: int = 5,
        max_consistency_weight: float = 1.0,
        validate_features: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.num_outputs = num_outputs

        # Load weights if provided
        if weights:
            self.load_weights(weights)

        # Compilation
        if compile:
            self.student = torch.compile(model, dynamic=True)
            log.info("Student model compiled.")
        else:
            self.student = model

        ema_decay = self.hparams.ema_decay
        # EMA Setup
        self.teacher = AveragedModel(
            self.student,
            avg_fn=lambda avg_param, model_param, num_averaged: ema_decay * avg_param + (1 - ema_decay) * model_param
        )

        # Loss configuration
        self.register_buffer(
            "pos_weights",
            self.hparams.loss_weights
            if self.hparams.loss_weights is not None
            else None,
        )

        # Metrics
        self.train_metrics = self._configure_metrics("train")
        self.val_metrics = self._configure_metrics("val")
        self.test_metrics = self._configure_metrics("test")

    def on_fit_start(self):
        """Ensure EMA device placement."""
        # Get device of student model
        self.student.to(self.device)
        self.teacher.to(self.device)

    def forward(self, x):
        return self.student(x)

    @staticmethod
    def compute_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        consistency_weight: float,
        pos_weights: torch.Tensor = None,
    ):
        # Supervised loss
        is_labeled = ~torch.isnan(labels)
        safe_labels = torch.where(is_labeled, labels, torch.zeros_like(labels))

        loss_matrix = F.binary_cross_entropy_with_logits(
            student_logits[:labels.shape[0]],
            safe_labels,
            reduction="none",
            pos_weight=pos_weights,
        )
        supervised_loss = (loss_matrix * is_labeled.float()).sum() / (is_labeled.sum() + 1e-8)

        # Consistency loss - MSE on probabilities
        student_probs = torch.sigmoid(student_logits)
        teacher_probs = torch.sigmoid(teacher_logits).detach()
        cons_loss = F.mse_loss(student_probs, teacher_probs)
        
        weighted_cons_loss = consistency_weight * cons_loss
        total_loss = supervised_loss + weighted_cons_loss

        return total_loss, supervised_loss, weighted_cons_loss

    def augment_batch(self, batch):
        """Apply augmentation to a batch of graphs."""
        from torch_geometric.data import Batch
        if hasattr(batch, "batch"):
            data_list = batch.to_data_list()
            aug_list = [self._augment_single_graph(g) for g in data_list]
            return Batch.from_data_list(aug_list)
        else:
            return self._augment_single_graph(batch)

    @staticmethod
    def _augment_single_graph(data):
        """Augment a single graph"""
        data = copy.deepcopy(data)
        
        # Random edge dropout
        if random.random() < 0.3 and data.edge_index.size(1) > 0:
            edge_mask = torch.rand(data.edge_index.size(1)) > 0.2
            data.edge_index = data.edge_index[:, edge_mask]
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_attr = data.edge_attr[edge_mask]
        
        # Random node feature noise (only if features are float type)
        if random.random() < 0.3:
            if data.x.dtype in [torch.float, torch.float32, torch.float64]:
                noise = torch.randn_like(data.x) * 0.1
                data.x = data.x + noise
            # else:
            #     mask = torch.rand(data.x.shape) > 0.1
            #     data.x = data.x * mask.long()
        
        return data

    def training_step(self, batch, batch_idx):
        labeled = batch["labeled"]
        unlabeled = batch["unlabeled"]
        
        # Apply augmentations
        labeled_student = self.augment_batch(labeled)
        unlabeled_student = self.augment_batch(unlabeled)
        
        labeled_teacher = self.augment_batch(labeled)
        unlabeled_teacher = self.augment_batch(unlabeled)
        
        # Student Forward
        pred_l_student = self.student(labeled_student)
        pred_u_student = self.student(unlabeled_student)
        
        
        all_preds_student = torch.cat([pred_l_student, pred_u_student], dim=0)
        
        # Teacher Forward
        with torch.inference_mode():
            pred_l_teacher = self.teacher(labeled_teacher)
            pred_u_teacher = self.teacher(unlabeled_teacher)
            all_preds_teacher = torch.cat([pred_l_teacher, pred_u_teacher], dim=0)
        
        # Labels (only for labeled data)
        labels = labeled.y
        
        # Get Consistency Weight
        curr_cons_weight = self._get_consistency_weight()
        
        # Compute Loss
        loss, sup_loss, cons_loss = self.compute_loss(
            all_preds_student,
            all_preds_teacher,
            labels,
            curr_cons_weight,
            self.pos_weights,
        )
        
        batch_size = labeled.num_graphs

        self.train_metrics.update(pred_l_student.float(), labels.float())
        
        # Logging
        self.log_dict(
            {
                "train/loss": loss,
                "train/sup_loss": sup_loss,
                "train/cons_loss": cons_loss,
                "train/cons_weight": curr_cons_weight,
                "train/pred_disagreement": (all_preds_student - all_preds_teacher).abs().mean(),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size
        )
        
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Only update teacher every N steps for stability
        if self.global_step % 1 == 0:  # Can increase to 2-5 for more stability
            self.teacher.update_parameters(self.student)

    def on_train_epoch_end(self):
        self._log_metrics(self.train_metrics)
        
        """ # Update teacher BN stats every 5 epochs
        if self.current_epoch % 5 == 0:
            # Convert Lightning's device string to torch.device
            device = torch.device(self.device) if isinstance(self.device, str) else self.device
            
            update_bn(
                self.trainer.train_dataloader,
                self.teacher,
                device=device
            ) """
            

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, self.val_metrics, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, self.test_metrics, "test")

    def _shared_eval_step(self, batch, metric_collection, prefix):
        preds = self.teacher.module(batch)
        labels = batch.y


        is_labeled = ~torch.isnan(labels)
        safe_labels = torch.where(is_labeled, labels, torch.zeros_like(labels))

        loss_matrix = F.binary_cross_entropy_with_logits(
            preds, safe_labels, reduction="none", pos_weight=self.pos_weights
        )
        loss = (loss_matrix * is_labeled.float()).sum() / (is_labeled.sum() + 1e-8)

        batch_size = batch.num_graphs

        # Update metrics 
        metric_collection(preds.float(), labels.float())

        self.log(
            f"{prefix}/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size
        )
        return loss



    def _get_consistency_weight(self) -> float:
        """Linear ramp-up (more stable than sigmoid)."""
        ramp_epochs = self.hparams.consistency_rampup_epochs
        if ramp_epochs == 0:
            return float(self.hparams.max_consistency_weight)

        max_steps = ramp_epochs * max(1,
            self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        )

        if self.global_step >= max_steps:
            return float(self.hparams.max_consistency_weight)

        # Linear ramp-up
        return float(
            self.hparams.max_consistency_weight * (self.global_step / max_steps)
        )

    def configure_optimizers(self):
        params = self.student.parameters()

        if self.hparams.optimizer == "sgd":
            optimizer = SGD(
                params,
                lr=self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                nesterov=self.hparams.nesterov,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = AdamW(
                params,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                betas=(0.9, 0.999),
            )

        steps_per_epoch = (
            self.trainer.estimated_stepping_batches // self.trainer.max_epochs
        )
        warmup_steps = int(self.hparams.warmup_epochs * steps_per_epoch)
        total_steps = self.trainer.estimated_stepping_batches

        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_steps),
                CosineAnnealingLR(optimizer, T_max=max(1,(total_steps - warmup_steps))),
            ],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def _configure_metrics(self, prefix: str):
        kwargs = {"num_tasks": self.num_outputs}
        return MetricCollection(
            {
                f"{prefix}/roc_auc": MultitaskROC_AUC(**kwargs),
                f"{prefix}/ap": MultitaskAveragePrecision(**kwargs),
                f"{prefix}/pr_auc": MultitaskPR_AUC(**kwargs),
                f"{prefix}/f1": MultitaskF1(**kwargs),
            }
        )

    def load_weights(self, weights_path: str):
        state_dict = torch.load(weights_path, map_location="cpu")
        self.student.load_state_dict(state_dict)
        log.info(f"Loaded weights from {weights_path}")
        
    def on_validation_epoch_end(self):
        self._log_metrics(self.val_metrics)

    def on_test_epoch_end(self):
        self._log_metrics(self.test_metrics)

    def _log_metrics(self, metric_collection):
        output = metric_collection.compute()
        self.log_dict(output, on_step=False, on_epoch=True, sync_dist=True)
        metric_collection.reset()