import logging
import pytorch_lightning as L
import torch
import torch.nn.functional as F
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics import MetricCollection
from torch_geometric.data import Batch

from src.utils.ogb_metrics import (
    MultitaskAveragePrecision,
    MultitaskF1,
    MultitaskPR_AUC,
    MultitaskROC_AUC,
)

log = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch_geometric.utils import dropout_edge

import torch
import torch.nn as nn
from torch_geometric.utils import dropout_edge

class SafeDropEdge(nn.Module):
    def __init__(self, p=0.05, force_undirected=False):
        super().__init__()
        self.p = p
        self.force_undirected = force_undirected

    def forward(self, data):
        # 1. Si no estamos entrenando o p=0, pasamos
        if not self.training or self.p <= 0:
            return data
            
        # 2. dropout_edge devuelve (nuevo_indice, mascara_booleana)
        edge_index, edge_mask = dropout_edge(
            data.edge_index, 
            p=self.p, 
            force_undirected=self.force_undirected,
            training=self.training
        )
        
        # 3. Clonamos datos para no romper la referencia original
        data = data.clone()
        data.edge_index = edge_index
        
        # 4. CRÍTICO: Filtrar también edge_attr usando la máscara
        if data.edge_attr is not None:
            # La máscara nos dice qué aristas sobrevivieron
            data.edge_attr = data.edge_attr[edge_mask]
            
        return data

class FeatureMasking(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, data):
        if not self.training or self.p <= 0:
            return data

        x = data.x
        # Crear máscara en el mismo dispositivo (CPU/GPU) que x
        mask = torch.empty_like(x).bernoulli_(1 - self.p)
        
        data = data.clone()
        data.x = x * mask
        return data

class GraphAugmentation(nn.Module):
    """
    Contenedor para aplicar ambas aumentaciones secuencialmente
    dentro del modelo.
    """
    def __init__(self, drop_edge_p=0.05, mask_feat_p=0.1):
        super().__init__()
        self.drop_edge = SafeDropEdge(p=drop_edge_p)
        self.mask_feat = FeatureMasking(p=mask_feat_p)

    def forward(self, data):
        # Aplicar en cadena
        data = self.mask_feat(data)
        data = self.drop_edge(data)
        return data

# Usage
# train_dataset = Dataset(..., transform=augmentations)


class MeanTeacherModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        num_outputs: int,
        learning_rate: float = 1e-3,
        warmup_epochs: int = 5,
        compile: bool = True,
        weights: str = None,
        optimizer: str = "adamw",
        cosine_period_ratio: float = 1,
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        loss_weights: torch.Tensor = None,
        # EMA & Consistency
        ema_decay: float = 0.999,
        ema_update_freq: int = 1,  
        consistency_rampup_epochs: int = 5,
        max_consistency_weight: float = 1.0,
        rampup_schedule: str = "linear",  # Options: 'linear' or 'sigmoid'
        # Augmentation Params 
        aug_prob_edge: float = 0.3,
        aug_prob_node: float = 0.3,
        edge_drop_rate: float = 0.2,
        node_noise_std: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.num_outputs = num_outputs

        # Load weights
        if weights:
            self.load_weights(weights)

        
        if compile:
            self.student = torch.compile(model, dynamic=True)
            log.info("Student model compiled.")
        else:
            self.student = model

        # EMA Setup
        ema_decay = self.hparams.ema_decay
        self.teacher = AveragedModel(
            self.student,
            avg_fn=lambda avg, model, _: ema_decay * avg + (1 - ema_decay) * model
        )

        self.register_buffer(
            "pos_weights",
            self.hparams.loss_weights if self.hparams.loss_weights is not None else None,
        )

        self.train_metrics = self._configure_metrics("train")
        self.val_metrics = self._configure_metrics("val")
        self.test_metrics = self._configure_metrics("test")

        self.augmentations = GraphAugmentation()

    def on_fit_start(self):
        self.student.to(self.device)
        self.teacher.to(self.device)

    def forward(self, x):
        return self.student(x)

    # def augment_batch(self, batch: Batch) -> Batch:
    #     """
    #     Vectorized augmentation (File 1).
    #     Operates on GPU tensors without CPU roundtrips or deepcopy.
    #     """
    #     batch = batch.clone() 
        
    #     # 1. Feature Noise (Vectorized per-graph probability)
    #     if self.hparams.aug_prob_node > 0:
    #         if batch.x.dtype in [torch.float, torch.float32, torch.float64]:
    #             # Boolean mask per graph [num_graphs]
    #             graph_mask = torch.rand(batch.num_graphs, device=self.device) < self.hparams.aug_prob_node
    #             # Expand to nodes
    #             node_mask = graph_mask[batch.batch]
                
    #             if node_mask.any():
    #                 noise = torch.randn_like(batch.x) * self.hparams.node_noise_std
    #                 batch.x = batch.x + (noise * node_mask.unsqueeze(-1))

    #     # 2. Edge Dropout (Global approximation for speed)
    #     if self.hparams.aug_prob_edge > 0 and batch.edge_index.size(1) > 0:
    #         effective_drop_rate = self.hparams.aug_prob_edge * self.hparams.edge_drop_rate
    #         mask = torch.rand(batch.edge_index.size(1), device=self.device) > effective_drop_rate
            
    #         batch.edge_index = batch.edge_index[:, mask]
    #         if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
    #             batch.edge_attr = batch.edge_attr[mask]

    #     return batch

    def training_step(self, batch, batch_idx):
        labeled = batch["labeled"]
        unlabeled = batch["unlabeled"]
        
        # 1. FIX: Asymmetric Augmentation
        # Student gets NOISE (Strong Aug)
        labeled_student = self.augmentations(labeled)
        unlabeled_student = self.augmentations(unlabeled)
        
        # Teacher gets CLEAN/WEAK input (Standard Aug only, no extra noise)
        # Assuming 'labeled' and 'unlabeled' are raw data. 
        # If your dataloader applies geometric transforms, this is fine.
        # Do NOT apply self.augment_batch to teacher if it adds noise/drops edges.
        labeled_teacher = labeled 
        unlabeled_teacher = unlabeled
        
        # Student Forward
        all_student_data = Batch.from_data_list([labeled_student, unlabeled_student])
        all_preds_student = self.student(all_student_data)
        
        # Teacher Forward
        with torch.inference_mode():
            all_teacher_data = Batch.from_data_list([labeled_teacher, unlabeled_teacher])
            all_preds_teacher = self.teacher(all_teacher_data)
        
        # Loss Calculation
        num_labeled = labeled.num_graphs
        labels = labeled.y
        curr_cons_weight = self._get_consistency_weight()
        
        loss, sup_loss, cons_loss = self.compute_loss(
            all_preds_student,
            all_preds_teacher,
            labels,
            curr_cons_weight,
            self.pos_weights,
            num_labeled_graphs=num_labeled
        )
        
        # Metrics & Logging 
        self.train_metrics.update(all_preds_student[:num_labeled].float(), labels.float())
        
        self.log_dict(
            {
                "train/loss": loss,
                "train/sup_loss": sup_loss,
                "train/cons_loss": cons_loss,
                "train/cons_weight": curr_cons_weight,
                # File 1 Feature: Monitor collapse/divergence
                "train/pred_disagreement": (all_preds_student - all_preds_teacher).abs().mean(),
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=num_labeled
        )
        
        return loss

    @staticmethod
    def compute_loss(
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        consistency_weight: float,
        pos_weights: torch.Tensor = None,
        num_labeled_graphs: int = 0
    ):
        # 1. Supervised Loss (On labeled slice)
        student_labeled = student_logits[:num_labeled_graphs]
        
        is_labeled = ~torch.isnan(labels)
        safe_labels = torch.where(is_labeled, labels, torch.zeros_like(labels))

        loss_matrix = F.binary_cross_entropy_with_logits(
            student_labeled,
            safe_labels,
            reduction="none",
            pos_weight=pos_weights,
        )
        supervised_loss = (loss_matrix * is_labeled.float()).sum() / (is_labeled.sum() + 1e-8)

        # 2. Consistency Loss (On full batch)
        # Using Sigmoid + MSE (Standard for Mean Teacher)
        # student_probs = torch.sigmoid(student_logits)
        # teacher_probs = torch.sigmoid(teacher_logits).detach()
        # cons_loss = F.mse_loss(student_probs, teacher_probs)

        cons_loss = F.mse_loss(student_logits, teacher_logits.detach())
        

        weighted_cons_loss = consistency_weight * cons_loss

        return supervised_loss + weighted_cons_loss, supervised_loss, weighted_cons_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # File 1 Feature: Configurable update frequency
        if self.global_step % self.hparams.ema_update_freq == 0:
            self.teacher.update_parameters(self.student)

    def on_train_epoch_end(self):
        self._log_metrics(self.train_metrics)
        
        # 2. FIX: Re-enable BN updates
        # This is mandatory for the Teacher to track the Student's distribution
        if self.current_epoch % 1 == 0:
            # Handle CombinedLoader which yields dictionaries.
            # update_bn expects a simple iterable of inputs (Batch objects), not {key: Batch}.
            bn_loader = None
            
            # Strategy A: Get loader directly from DataModule (Cleanest)
            if self.trainer.datamodule is not None:
                # Use unlabeled data for better distribution statistics
                if hasattr(self.trainer.datamodule, "unsupervised_train_dataloader"):
                    bn_loader = self.trainer.datamodule.unsupervised_train_dataloader()
                elif hasattr(self.trainer.datamodule, "supervised_train_dataloader"):
                    bn_loader = self.trainer.datamodule.supervised_train_dataloader()
            
            # Strategy B: Extract from CombinedLoader
            if bn_loader is None:
                tl = self.trainer.train_dataloader
                if hasattr(tl, "loaders") and isinstance(tl.loaders, dict):
                    # Prioritize unlabeled
                    bn_loader = tl.loaders.get("unlabeled", tl.loaders.get("labeled"))
                else:
                    # Fallback for standard single-loader setups
                    bn_loader = tl

            if bn_loader is not None:
                update_bn(
                    bn_loader,
                    self.teacher,
                    device=self.device
                )

    def _get_consistency_weight(self) -> float:
        """Handles both Linear (File 1) and Sigmoid (File 2) schedules."""
        ramp_epochs = self.hparams.consistency_rampup_epochs
        if ramp_epochs == 0:
            return float(self.hparams.max_consistency_weight)

        # Common calculations
        total_steps = self.trainer.estimated_stepping_batches
        if self.trainer.max_epochs > 0:
            max_steps = ramp_epochs * (total_steps // self.trainer.max_epochs)
        else:
            max_steps = total_steps # Fallback

        if self.global_step >= max_steps:
            return float(self.hparams.max_consistency_weight)

        # Schedule Logic
        if self.hparams.rampup_schedule == "sigmoid":
            # File 2: Sigmoid Ramp-up
            phase = 1.0 - (self.global_step / max_steps)
            return float(
                self.hparams.max_consistency_weight
                * torch.exp(torch.tensor(-5.0 * phase * phase))
            )
        else:
            # File 1: Linear Ramp-up (Default/Optimized)
            return float(
                self.hparams.max_consistency_weight * (self.global_step / max_steps)
            )

    def validation_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, self.val_metrics, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_eval_step(batch, self.test_metrics, "test")

    def _shared_eval_step(self, batch, metric_collection, prefix):
        # Evaluation uses the Teacher (EMA)
        preds = self.teacher.module(batch)
        labels = batch.y

        is_labeled = ~torch.isnan(labels)
        safe_labels = torch.where(is_labeled, labels, torch.zeros_like(labels))

        loss_matrix = F.binary_cross_entropy_with_logits(
            preds, safe_labels, reduction="none", pos_weight=self.pos_weights
        )
        loss = (loss_matrix * is_labeled.float()).sum() / (is_labeled.sum() + 1e-8)

        metric_collection(preds.float(), labels.float())
        self.log(f"{prefix}/loss", loss, on_epoch=True, batch_size=batch.num_graphs)
        return loss

    def configure_optimizers(self):
        params = self.student.parameters()

        # 1. Optimizer Setup
        if self.hparams.optimizer.lower() == "sgd":
            optimizer = SGD(
                params,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                momentum=self.hparams.momentum,
                nesterov=self.hparams.nesterov,
            )
        else:
            optimizer = AdamW(
                params,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
                amsgrad=False,
                betas=(0.9, 0.999),
            )

        # 2. Robust Step Calculation
        # Lightning's estimated_stepping_batches accounts for max_epochs, 
        # accum_grad_batches, and the size of the *largest* dataloader (due to max_size_cycle)
        total_steps = self.trainer.estimated_stepping_batches
        
        # Safety fallback if estimation fails (returns inf or 0)
        if total_steps is None or total_steps == 0 or float(total_steps) == float('inf'):
            # Manual fallback: assuming you check val every epoch
            # You might need to adjust '600' based on your PCBA calculation from previous chats
            steps_per_epoch = len(self.trainer.train_dataloader) 
            total_steps = steps_per_epoch * self.trainer.max_epochs

        warmup_steps = int(self.hparams.warmup_epochs * (total_steps / self.trainer.max_epochs))
        
        # Ensure we have steps left for cosine
        cosine_steps = total_steps - warmup_steps

        print(f"--- Scheduler Config ---\nTotal Steps: {total_steps}\nWarmup Steps: {warmup_steps}\nCosine Steps: {cosine_steps}\n------------------------")

        # 3. Schedulers
        # Phase 1: Linear Warmup (0 -> lr_max)
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=1e-4, # Start near 0
            end_factor=1.0,
            total_iters=warmup_steps,
        )

        # Phase 2: Cosine Annealing (lr_max -> eta_min)
        # We use eta_min to prevent LR from dying completely
        cosine_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=cosine_steps, 
            eta_min=self.hparams.learning_rate * 0.001 # Decay to 1% of max LR
        )

        # Combine
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1, 
        }

        return [optimizer], [scheduler_config]

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
        # FIX: Added weights_only=False to handle Hydra/OmegaConf objects in checkpoint
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)
        
        # Handle cases where the state_dict might be nested under "state_dict" key
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
            
        # Clean up keys if loading a Lightning checkpoint into a raw model
        # (Removes "student." or "model." prefixes if necessary)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("student."):
                new_state_dict[k.replace("student.", "")] = v
            else:
                new_state_dict[k] = v
                
        # Load into student
        try:
            self.student.load_state_dict(new_state_dict, strict=False)
            log.info(f"Loaded weights from {weights_path}")
        except RuntimeError as e:
            log.warning(f"Strict loading failed, trying raw load. Error: {e}")
            self.student.load_state_dict(state_dict, strict=False)
        
    def on_validation_epoch_end(self):
        self._log_metrics(self.val_metrics)

    def on_test_epoch_end(self):
        self._log_metrics(self.test_metrics)

    def _log_metrics(self, metric_collection):
        output = metric_collection.compute()
        self.log_dict(output, on_step=False, on_epoch=True, sync_dist=True)
        metric_collection.reset()