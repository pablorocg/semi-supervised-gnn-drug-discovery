from src.data.ogbg_molpcba_datamodule import OgbgMolpcbaDataModule
from torch_geometric.nn import GCN
from src.utils.utils import get_logs_dir

# Read config from hydra
import pytorch_lightning as L
from src.lightning_modules.baseline_module import BaselineGNNModule
from torch_geometric.transforms import Compose, AddSelfLoops, NormalizeFeatures
from torch_geometric.nn import global_mean_pool

from src.utils.transforms import ToFloatTransform
import logging
import torch 
from pytorch_lightning.loggers import TensorBoardLogger


log = logging.getLogger(__name__)


graph_transforms = Compose(
    [
        ToFloatTransform(),
        AddSelfLoops(),
        NormalizeFeatures(),
    ]
)


class GCN2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.model = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=out_channels,
            dropout=0.2,
            add_self_loops=False
        )
    
    def forward(self, x, edge_index, batch):
        x = self.model(x, edge_index)
        x = global_mean_pool(x, batch)
        return x



data_module = OgbgMolpcbaDataModule(
    batch_size=128,
    num_workers=4,

)

data_module.setup(stage="fit")


# 3. Setup Lightning Module
module = BaselineGNNModule(
    model=GCN2(data_module.num_node_features, 128, data_module.num_classes, 3),
    num_classes=data_module.num_classes,
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
    max_epochs=10,
    # fast_dev_run=True,  # Runs 1 train batch and 1 val batch to check for errors
    accelerator="auto",
    devices=1,
    logger=TensorBoardLogger(save_dir=get_logs_dir(), name="baseline_gcn"),
    log_every_n_steps=1,
    enable_progress_bar=True,
    enable_model_summary=True,
    enable_checkpointing=False,
    # default_root_dir=get_logs_dir(),  # Specify a directory for logs and checkpoints
)

# 5. Run Training Test
try:
    trainer.fit(module, datamodule=data_module)
    log.info("--- Training Test Successful! ---")
except Exception as e:
    log.error("--- Training Test FAILED ---")
    log.error(e)
    raise e
