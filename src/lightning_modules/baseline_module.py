# import logging

# import pytorch_lightning as L
# import torch

# from src.lightning_modules.base_module import BaseModule

# # Set up logging
# log = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


# class MyGCNModule(BaseModule):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.loss_fn = torch.nn.BCEWithLogitsLoss()

#     def _common_step(self, batch, batch_idx):
#         #  forward(x: Tensor, edge_index: Union[Tensor, SparseTensor], edge_weight: Optional[Tensor] = None, edge_attr: Optional[Tensor] = None, batch: Optional[Tensor] = None, batch_size: Optional[int] = None, num_sampled_nodes_per_hop: Optional[List[int]] = None, num_sampled_edges_per_hop: Optional[List[int]] = None) → Tensor
#         logits = self.model(batch.x, batch.edge_index)
#         logits = global_mean_pool(logits, batch.batch)
#         targets = batch.y.float()
#         mask = ~torch.isnan(targets)
#         loss = self.loss_fn(logits[mask], targets[mask])
#         targets_int = batch.y.clone().int()

#         return loss, logits, targets_int

#     def training_step(self, batch, batch_idx):
#         loss, logits, targets_int = self._common_step(batch, batch_idx)
#         self.train_metrics(logits, targets_int)
#         self.log(
#             "train/loss",
#             loss,
#             on_step=True,
#             on_epoch=True,
#             prog_bar=True,
#             sync_dist=True,
#             batch_size=batch.num_graphs,
#         )
#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss, logits, targets_int = self._common_step(batch, batch_idx)
#         self.val_metrics(logits, targets_int)
#         self.log(
#             "val/loss",
#             loss,
#             on_step=False,
#             on_epoch=True,
#             prog_bar=True,
#             sync_dist=True,
#             batch_size=batch.num_graphs,
#         )
#         return loss

#     def test_step(self, batch, batch_idx):
#         loss, logits, targets_int = self._common_step(batch, batch_idx)
#         self.test_metrics(logits, targets_int)
#         self.log(
#             "test/loss",
#             loss,
#             on_step=False,
#             on_epoch=True,
#             prog_bar=True,
#             sync_dist=True,
#         )
#         return loss



import logging
from typing import Callable, Optional

import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch_geometric.nn import global_mean_pool
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
    MetricCollection,
)
import torch_geometric

log = logging.getLogger(__name__)


class BaselineClassifier(L.LightningModule):
    """
    Baseline classifier for graph-level prediction on MoleculeNet datasets.
    
    Supports:
    - Any GNN architecture (GCN, GAT, GIN, etc.)
    - Multi-label classification with NaN handling
    - Configurable optimizer (SGD or AdamW)
    - Warmup + Cosine annealing scheduling
    - Comprehensive metrics (AUROC, AUPRC, F1, Accuracy)
    
    Usage:
        model = YourGNNModel(num_node_features, hidden_dim, num_classes)
        
        module = BaselineClassifier(
            model=model,
            num_classes=num_classes,
            learning_rate=0.001,
            optimizer="AdamW"
        )
        
        trainer = Trainer(max_epochs=200)
        trainer.fit(module, datamodule)
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        optimizer: str = "AdamW",
        weight_decay: float = 1e-5,
        momentum: float = 0.9,
        nesterov: bool = True,
        warmup_epochs: int = 5,
        cosine_period_ratio: float = 1.0,
        pooling: str = "mean",  # Options: "mean", "sum", "max", "add"
        compile_mode: Optional[str] = None,
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
    ):
        """
        Args:
            model: GNN model to use (e.g., GCN, GAT, GIN)
            num_classes: Number of output classes
            learning_rate: Initial learning rate
            optimizer: Optimizer to use ("SGD" or "AdamW")
            weight_decay: Weight decay for regularization
            momentum: Momentum for SGD optimizer
            nesterov: Whether to use Nesterov momentum
            warmup_epochs: Number of epochs for linear warmup
            cosine_period_ratio: Ratio of total epochs for cosine annealing (0, 1]
            pooling: Graph pooling method ("mean", "sum", "max", "add")
            compile_mode: PyTorch 2.0 compile mode (None, "default", "reduce-overhead", "max-autotune")
            train_transforms: Optional on-the-fly transforms for training
            val_transforms: Optional on-the-fly transforms for validation
            test_transforms: Optional on-the-fly transforms for testing
        """
        super().__init__()
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.nesterov = nesterov
        self.warmup_epochs = warmup_epochs
        self.cosine_period_ratio = cosine_period_ratio
        self.num_classes = num_classes
        self.pooling = pooling
        
        assert 0 < cosine_period_ratio <= 1.0, "cosine_period_ratio must be in (0, 1]"
        
        # Save hyperparameters (except model for serialization)
        self.save_hyperparameters(ignore=["model", "train_transforms", "val_transforms", "test_transforms"])
        
        # Transforms
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms
        
        # Loss function for multi-label classification with NaN handling
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        # Model
        self.model = model
        if compile_mode is not None:
            log.info(f"Compiling model with mode: {compile_mode}")
            self.model = torch.compile(model, mode=compile_mode)
        
        # Metrics
        self.train_metrics = self._configure_metrics("train")
        self.val_metrics = self._configure_metrics("val")
        self.test_metrics = self._configure_metrics("test")
        
        # Pooling function
        self._pooling_fn = self._get_pooling_function(pooling)
    
    def _get_pooling_function(self, pooling: str):
        """Get the appropriate pooling function"""
        pooling_functions = {
            "mean": global_mean_pool,
            "sum": lambda x, batch: torch_geometric.nn.global_add_pool(x, batch),
            "max": lambda x, batch: torch_geometric.nn.global_max_pool(x, batch),
            "add": lambda x, batch: torch_geometric.nn.global_add_pool(x, batch),
        }
        
        if pooling not in pooling_functions:
            raise ValueError(
                f"Unknown pooling method: {pooling}. "
                f"Available: {list(pooling_functions.keys())}"
            )
        
        return pooling_functions[pooling]
    
    def _configure_metrics(self, prefix: str) -> MetricCollection:
        """Configure metrics for multi-label classification"""
        task = "multilabel"
        avg_strategy = "macro"
        
        return MetricCollection(
            {
                f"{prefix}/AUROC": AUROC(
                    task=task,
                    num_labels=self.num_classes,
                    average=avg_strategy,
                ),
                f"{prefix}/AUPRC": AveragePrecision(
                    task=task,
                    num_labels=self.num_classes,
                    average=avg_strategy,
                ),
                f"{prefix}/F1": F1Score(
                    task=task,
                    num_labels=self.num_classes,
                    average=avg_strategy,
                ),
                f"{prefix}/Accuracy": Accuracy(
                    task=task,
                    num_labels=self.num_classes,
                    average=avg_strategy,
                ),
            },
        )
    
    def forward(self, batch):
        """
        Forward pass through the model.
        
        Args:
            batch: PyG Batch object with x, edge_index, batch, y
        
        Returns:
            Graph-level predictions (after pooling)
        """
        # Node-level embeddings
        node_embeddings = self.model(
            batch.x, 
            batch.edge_index,
            edge_attr=batch.edge_attr if hasattr(batch, 'edge_attr') else None,
        )
        
        # Graph-level pooling
        graph_embeddings = self._pooling_fn(node_embeddings, batch.batch)
        
        return graph_embeddings
    
    def _common_step(self, batch, batch_idx):
        """
        Common step for training, validation, and testing.
        Handles NaN values in targets (common in MoleculeNet datasets).
        
        Returns:
            loss: Computed loss (only on valid labels)
            logits: Model predictions
            targets_int: Integer targets for metrics
        """
        # Forward pass
        logits = self(batch)
        
        # Get targets and create mask for valid labels
        targets = batch.y.float()
        mask = ~torch.isnan(targets)
        
        # Compute loss only on valid labels
        if mask.any():
            loss = self.loss_fn(logits[mask], targets[mask])
        else:
            # No valid labels in this batch (rare)
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Convert targets to int for metrics
        targets_int = batch.y.clone()
        targets_int[~mask] = 0  # Fill NaN with 0 for metrics (will be ignored)
        targets_int = targets_int.int()
        
        return loss, logits, targets_int
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        loss, logits, targets_int = self._common_step(batch, batch_idx)
        
        # Update metrics
        self.train_metrics(logits, targets_int)
        
        # Log loss
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.num_graphs,
        )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        loss, logits, targets_int = self._common_step(batch, batch_idx)
        
        # Update metrics
        self.val_metrics(logits, targets_int)
        
        # Log loss
        self.log(
            "val/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.num_graphs,
        )
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        loss, logits, targets_int = self._common_step(batch, batch_idx)
        
        # Update metrics
        self.test_metrics(logits, targets_int)
        
        # Log loss
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch.num_graphs,
        )
        
        return loss
    
    def on_train_epoch_end(self):
        """Compute and log metrics at the end of training epoch"""
        if self.train_metrics:
            metrics = self.train_metrics.compute()
            self.log_dict(
                metrics, 
                on_step=False, 
                on_epoch=True, 
                sync_dist=True, 
                prog_bar=True
            )
            self.train_metrics.reset()
    
    def on_validation_epoch_end(self):
        """Compute and log metrics at the end of validation epoch"""
        if self.val_metrics:
            metrics = self.val_metrics.compute()
            self.log_dict(
                metrics, 
                on_step=False, 
                on_epoch=True, 
                sync_dist=True, 
                prog_bar=True
            )
            self.val_metrics.reset()
    
    def on_test_epoch_end(self):
        """Compute and log metrics at the end of test epoch"""
        if self.test_metrics:
            metrics = self.test_metrics.compute()
            self.log_dict(
                metrics, 
                on_step=False, 
                on_epoch=True, 
                sync_dist=True, 
                prog_bar=True
            )
            self.test_metrics.reset()
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""
        
        # Optimizer
        if self.optimizer_name == "SGD":
            optimizer = SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                nesterov=self.nesterov,
            )
            log.info(
                f"Using SGD optimizer: lr={self.learning_rate}, "
                f"weight_decay={self.weight_decay}, momentum={self.momentum}"
            )
        elif self.optimizer_name == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
            )
            log.info(
                f"Using AdamW optimizer: lr={self.learning_rate}, "
                f"weight_decay={self.weight_decay}"
            )
        else:
            raise ValueError(
                f"Unknown optimizer: {self.optimizer_name}. "
                f"Available: ['SGD', 'AdamW']"
            )
        
        # Learning rate scheduler
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
        
        log.info(
            f"Scheduler config: total_steps={total_steps}, "
            f"warmup_steps={warmup_steps}, cosine_steps={cosine_steps}"
        )
        
        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)
        
        # Add warmup if specified
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
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Apply transforms to the batch after it has been moved to the device.
        Useful for GPU-accelerated augmentations.
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
    


