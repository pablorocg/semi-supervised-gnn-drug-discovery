import os
import sys

print("=== Script started ===", flush=True)
print("DATA DIR:", os.getenv("SOURCE_DATA_DIR"), flush=True)
print("LOGS DIR:", os.getenv("LOGS_DIR"), flush=True)
print("MODELS DIR:", os.getenv("MODELS_DIR"), flush=True)

print("Importing torch...", flush=True)
import torch
import torch.nn as nn
import pytorch_lightning as L
from torch.optim import AdamW
from torchmetrics import MetricCollection
from torch.nn import BCEWithLogitsLoss, MSELoss
print("Torch imports done", flush=True)

print("Importing custom modules...", flush=True)
from src.models.gine import GINE
from src.utils.utils import calculate_pos_weights_from_tensor
from src.utils.ogb_metrics import (
    MultiTaskAccuracy,
    MultiTaskAP,
    MultiTaskROCAUC,
    SetF1Score,
)
from src.utils.path_utils import get_data_dir, get_logs_dir, get_models_dir, ensure_dir_exists
print("Custom imports done", flush=True)

print("Getting directories...", flush=True)
data_dir = get_data_dir()
logs_dir = get_logs_dir()
models_dir = get_models_dir()

print(f"data_dir: {data_dir}", flush=True)
print(f"logs_dir: {logs_dir}", flush=True)
print(f"models_dir: {models_dir}", flush=True)

print("Ensuring directories exist...", flush=True)
ensure_dir_exists(data_dir)
ensure_dir_exists(logs_dir)
ensure_dir_exists(models_dir)
print("Directories created", flush=True)


class BaselineModule(L.LightningModule):
    # ... your entire class stays the same ...
    def __init__(
        self,
        num_outputs: int,
        num_node_features: int,
        model: GINE = None,
        task_type: str = "classification",
        train_dataset=None,
        train_idx=None,
        learning_rate: float = 1e-4,
        dropout: float = 0.3,
        weight_decay: float = 1e-5,
        hidden_dim: int = 64,
        num_layers: int = 5,
        num_mlp_layers: int = 2,
        use_residual: bool = True,
        use_virtual_node: bool = True,
        use_attention_pool: bool = True,
        use_layer_norm: bool = True,
        weights: str = None,
        compile: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_dataset", "train_idx", "model"])

        if model is not None:
            self.model = model
        else:
            self.model = GINE(
                num_tasks=num_outputs,
                num_node_features=num_node_features,
                num_edge_features=3,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                use_residual=use_residual,
                use_virtual_node=use_virtual_node,
                use_attention_pool=use_attention_pool,
                num_mlp_layers=num_mlp_layers,
                use_layer_norm=use_layer_norm,
            )

        if task_type == "classification" and num_outputs > 1 and train_idx is not None:
            if train_dataset is not None:
                train_labels = train_dataset.data.y[train_idx].float()
                self.pos_weight = calculate_pos_weights_from_tensor(train_labels, max_weight=10.0)
            else:
                self.pos_weight = None
        else:
            self.pos_weight = None

        self.supervised_loss_fn = self._setup_loss()

        self.train_metrics = self._configure_metrics("train")
        self.val_metrics = self._configure_metrics("val")
        self.test_metrics = self._configure_metrics("test")

        if self.hparams.weights is not None:
            self.load_weights(self.hparams.weights)

        if self.hparams.compile:
            self.model = torch.compile(self.model, dynamic=True)

    def _setup_loss(self):
        task = self.hparams.task_type
        n_outputs = self.hparams.num_outputs

        if task == "regression":
            return MSELoss(reduction='none')
        elif task == "classification":
            # Don't use pos_weight in the loss function itself
            # We'll handle it manually in _masked_loss
            return BCEWithLogitsLoss(reduction='none')
        else:
            raise ValueError(f"Unsupported task type: {task}")

    def forward(self, x):
        return self.model(x)

    def _masked_loss(self, logits, labels, mask):
        # Only calculate loss on valid (non-NaN) entries
        valid_logits = logits[mask]
        valid_labels = labels[mask]
        
        if len(valid_logits) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # Calculate BCE loss only on valid entries
        per_elem = self.supervised_loss_fn(valid_logits, valid_labels)
        
        # Apply pos_weight manually if it exists
        if self.pos_weight is not None:
            pos_weight = self.pos_weight.to(logits.device)
            # Expand pos_weight to match the valid entries
            # We need to figure out which task each valid entry belongs to
            # Since mask has shape [batch, num_tasks], we need to apply weights per task
            
            # Get the task indices for each valid entry
            task_indices = mask.nonzero(as_tuple=True)[1]  # Get column indices
            task_weights = pos_weight[task_indices]
            
            # Apply weight: weight = label * (task_weight - 1) + 1
            weights = valid_labels * (task_weights - 1) + 1
            per_elem = per_elem * weights
        
        # Return mean over all valid elements
        return per_elem.mean()

    def training_step(self, batch, batch_idx):
        labeled_batch = batch["labeled"]
        
        # Check input data
        if batch_idx < 5:  # Only print for first few batches
            print(f"\n=== Batch {batch_idx} Debug ===", flush=True)
            print(f"Input x - shape: {labeled_batch.x.shape}, has NaN: {torch.isnan(labeled_batch.x).any()}", flush=True)
            print(f"Input x - range: [{labeled_batch.x.min():.3f}, {labeled_batch.x.max():.3f}]", flush=True)
            print(f"Labels y - shape: {labeled_batch.y.shape}, has NaN: {torch.isnan(labeled_batch.y).any()}", flush=True)
        
        # Forward pass
        logits = self(labeled_batch)
        
        if batch_idx < 5:
            print(f"Logits - shape: {logits.shape}, has NaN: {torch.isnan(logits).any()}", flush=True)
            print(f"Logits - range: [{logits.min():.3f}, {logits.max():.3f}]", flush=True)
        
        labels = labeled_batch.y.float()
        mask = ~torch.isnan(labels)
        
        if batch_idx < 5:
            print(f"Mask - valid labels: {mask.sum().item()} / {mask.numel()}", flush=True)
        
        # Check if we have any valid labels
        if mask.sum() == 0:
            print(f"WARNING: Batch {batch_idx} has no valid labels, skipping", flush=True)
            return None
        
        # Calculate loss
        loss = self._masked_loss(logits, labels, mask)
        
        if batch_idx < 5:
            print(f"Loss: {loss.item():.6f}", flush=True)
        
        # Check for NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n!!! NaN/Inf detected at batch {batch_idx} !!!", flush=True)
            print(f"Logits stats - min: {logits.min():.3f}, max: {logits.max():.3f}, mean: {logits.mean():.3f}", flush=True)
            print(f"Labels stats - min: {labels[mask].min():.3f}, max: {labels[mask].max():.3f}", flush=True)
            print(f"Mask sum: {mask.sum()}", flush=True)
            
            # Check model parameters
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    print(f"NaN in parameter: {name}", flush=True)
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN in gradient: {name}", flush=True)
            
            return None  # Skip this batch
        
        self.train_metrics.update(logits, labels)
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Validation likely doesn't use CombinedLoader, but let's be safe
        if isinstance(batch, dict):
            batch = batch.get("labeled", batch)
        
        logits = self(batch)
        labels = batch.y.float()
        mask = ~torch.isnan(labels)
        loss = self._masked_loss(logits, labels, mask)
        self.val_metrics.update(logits, labels)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Test likely doesn't use CombinedLoader, but let's be safe
        if isinstance(batch, dict):
            batch = batch.get("labeled", batch)
        
        logits = self(batch)
        labels = batch.y.float()
        mask = ~torch.isnan(labels)
        loss = self._masked_loss(logits, labels, mask)
        self.test_metrics.update(logits, labels)
        self.log("test/loss", loss, on_epoch=True)
        return loss

    def _configure_metrics(self, prefix):
        return MetricCollection(
            {
                f"{prefix}/pr_auc": MultiTaskAP(num_tasks=self.hparams.num_outputs),
                f"{prefix}/auroc": MultiTaskROCAUC(num_tasks=self.hparams.num_outputs),
                f"{prefix}/accuracy": MultiTaskAccuracy(num_tasks=self.hparams.num_outputs),
            }
        )

    def on_train_epoch_end(self):
        m = self.train_metrics.compute()
        self.log_dict(m, on_epoch=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        m = self.val_metrics.compute()
        self.log_dict(m, on_epoch=True)
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        m = self.test_metrics.compute()
        self.log_dict(m, on_epoch=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        return AdamW(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)


if __name__ == "__main__":
    print("\n=== ENTERING MAIN BLOCK ===", flush=True)
    
    print("Importing training modules...", flush=True)
    import torch
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import WandbLogger
    from src.data.pcba import OgbgMolPcbaDataModule
    print("Training imports done", flush=True)

    print("Creating DataModule...", flush=True)
    data_module = OgbgMolPcbaDataModule(
        batch_size_train=64,
        batch_size_inference=126,
        num_workers=4,
        target=None,
        subset_size=218964,
    )
    print("DataModule created", flush=True)
    
    print("Setting up data...", flush=True)
    data_module.setup()
    print("Data setup complete", flush=True)

    print("=" * 60, flush=True)
    print(f"Full dataset: {len(data_module.dataset)} graphs", flush=True)
    print(f"Labeled dataset: {len(data_module.data_train_labeled)} graphs", flush=True)
    print(f"First 10 labeled indices: {list(data_module.train_idx)[:10]}", flush=True)
    print("=" * 60, flush=True)

    print("Creating baseline model...", flush=True)
    baseline_model = BaselineModule(
        num_outputs=128,
        num_node_features=9,
        train_dataset=data_module.data_train_labeled,
        train_idx=data_module.train_idx,
        learning_rate=5e-5,
        weight_decay=1e-5,
        dropout=0.3,
        hidden_dim=64,
        num_layers=5,
        num_mlp_layers=2,
        use_residual=True,
        use_virtual_node=True,
        use_attention_pool=True,
        use_layer_norm=True,
        weights=None,
        compile=False
    )
    print("Baseline model created", flush=True)

    if baseline_model.pos_weight is not None:
        print(f"Pos weights - min: {baseline_model.pos_weight.min():.3f}, "
            f"max: {baseline_model.pos_weight.max():.3f}, "
            f"mean: {baseline_model.pos_weight.mean():.3f}", flush=True)
        print(f"Pos weights first 10 values: {baseline_model.pos_weight[:10]}", flush=True)  # Add this line

    print("Setting up callbacks...", flush=True)
    callbacks = [
        ModelCheckpoint(
            dirpath=models_dir,
            monitor='val/auroc',
            mode='max',
            save_top_k=3,
            filename='gine-{epoch:02d}-{val/auroc:.4f}',
            save_last=True,
        ),
        EarlyStopping(
            monitor='val/auroc',
            patience=30,
            mode='max',
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]
    print("Callbacks created", flush=True)

    print("Creating trainer...", flush=True)
    trainer = Trainer(
        max_epochs=150,
        accelerator="auto",
        devices="auto",
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=WandbLogger(project="pcba-baseline", name="gine-run", save_dir=logs_dir, log_model=False),
        callbacks=callbacks,
        gradient_clip_val=1.0,
        val_check_interval=1.0,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
    )
    print("Trainer created", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("Starting training...", flush=True)
    print("=" * 60 + "\n", flush=True)
    
    trainer.fit(model=baseline_model, datamodule=data_module)

    print("\n" + "=" * 60, flush=True)
    print("Testing best model...", flush=True)
    print("=" * 60 + "\n", flush=True)
    
    trainer.test(model=baseline_model, datamodule=data_module, ckpt_path='best')
    
    print("\n=== SCRIPT COMPLETED SUCCESSFULLY ===", flush=True)