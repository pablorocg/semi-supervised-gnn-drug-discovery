import logging

import pytorch_lightning as L
import torch

from src.lightning_modules.base_module import BaseModule

# Set up logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MyGCNModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def _common_step(self, batch, batch_idx):
        #  forward(x: Tensor, edge_index: Union[Tensor, SparseTensor], edge_weight: Optional[Tensor] = None, edge_attr: Optional[Tensor] = None, batch: Optional[Tensor] = None, batch_size: Optional[int] = None, num_sampled_nodes_per_hop: Optional[List[int]] = None, num_sampled_edges_per_hop: Optional[List[int]] = None) → Tensor
        logits = self.model(batch.x, batch.edge_index)
        logits = global_mean_pool(logits, batch.batch)
        targets = batch.y.float()
        mask = ~torch.isnan(targets)
        loss = self.loss_fn(logits[mask], targets[mask])
        targets_int = batch.y.clone().int()

        return loss, logits, targets_int

    def training_step(self, batch, batch_idx):
        loss, logits, targets_int = self._common_step(batch, batch_idx)
        self.train_metrics(logits, targets_int)
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
        loss, logits, targets_int = self._common_step(batch, batch_idx)
        self.val_metrics(logits, targets_int)
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
        loss, logits, targets_int = self._common_step(batch, batch_idx)
        self.test_metrics(logits, targets_int)
        self.log(
            "test/loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss


if __name__ == "__main__":
    from torch_geometric.transforms import AddSelfLoops, Compose, NormalizeFeatures
    from torch_geometric.nn import global_mean_pool
    from src.data.datamodules import MoleculeNetDataModule
    from torch_geometric.nn import GCN
    #  GCN(in_channels: int, hidden_channels: int, num_layers: int, out_channels: Optional[int] = None, dropout: float = 0.0, act: Optional[Union[str, Callable]] = 'relu', act_first: bool = False, act_kwargs: Optional[Dict[str, Any]] = None, norm: Optional[Union[str, Callable]] = None, norm_kwargs: Optional[Dict[str, Any]] = None, jk: Optional[str] = None, **kwargs)

    # --- Start of runnable test script ---
    log.info("--- Starting Training Test ---")

    class ToFloatTransform:
        """Converts node features `data.x` to float."""

        def __call__(self, data):
            data.x = data.x.float()
            return data

    # LOGICAL FIX: Define transforms for pre-processing
    graph_transforms = Compose(
        [
            ToFloatTransform(),  # <-- ADDED: Convert features to float here
            AddSelfLoops(),
            NormalizeFeatures(),  # Often good, but let's skip for simplicity
        ]
    )

    # 1. Setup DataModule
    data_module = MoleculeNetDataModule(
        name="SIDER",  
        batch_size=32,
        split_idx=0,
        k_splits=5,
        seed=42,
        test_split_ratio=0.2,
        pre_transform=None,  
    )

    # 2. Setup Model
    model = GCN(in_channels=9, hidden_channels=128, num_layers=3, out_channels=27, dropout=0.2,add_self_loops=False)

    # 3. Setup Lightning Module
    module = MyGCNModule(
        model=model,
        num_classes=27, 
        learning_rate=1e-3,
        warmup_epochs=1,
        cosine_period_ratio=0.9,
        compile_mode=None,  
        optimizer="SGD",
        weight_decay=1e-5,
        train_transforms=graph_transforms,
        val_transforms=graph_transforms,
        test_transforms=graph_transforms,
    )

    # 4. Setup Trainer
    trainer = L.Trainer(
        max_epochs=100,
        # fast_dev_run=True,  # Runs 1 train batch and 1 val batch to check for errors
        accelerator="auto",
        devices=1,
        log_every_n_steps=1,

    )

    # 5. Run Training Test
    try:
        trainer.fit(module, datamodule=data_module)
        log.info("--- Training Test Successful! ---")
    except Exception as e:
        log.error("--- Training Test FAILED ---")
        log.error(e)
        raise e
