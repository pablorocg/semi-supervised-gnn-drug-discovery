from torch_geometric.transforms import AddSelfLoops, Compose, NormalizeFeatures
from torch_geometric.nn import global_mean_pool
from src.data.datamodules import MoleculeNetDataModule
from torch_geometric.nn import GCN
from src.utils.utils import get_logs_dir

# Read config from hydra 
import pytorch_lightning as L
from src.lightning_modules.gcn_module import MyGCNModule
from src.utils.transforms import ToFloatTransform
import logging




log = logging.getLogger(__name__)


graph_transforms = Compose(
    [
        ToFloatTransform(), 
        AddSelfLoops(),
        NormalizeFeatures()
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

    # This will trigger the file creation/loading
    dm.prepare_data()

    # This will load the splits into datasets
    dm.setup()

# 2. Setup Model
model = GCN(in_channels=9, hidden_channels=128, num_layers=3, out_channels=27, dropout=0.2, add_self_loops=False)

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

module.print_model_summary()



# 4. Setup Trainer
trainer = L.Trainer(
    max_epochs=100,
    # fast_dev_run=True,  # Runs 1 train batch and 1 val batch to check for errors
    accelerator="auto",
    devices=1,
    log_every_n_steps=1,
    enable_progress_bar=True,
    enable_model_summary=True,
    enable_checkpointing=False,
    default_root_dir=get_logs_dir(),  # Specify a directory for logs and checkpoints

)

# 5. Run Training Test
try:
    trainer.fit(module, datamodule=data_module)
    log.info("--- Training Test Successful! ---")
except Exception as e:
    log.error("--- Training Test FAILED ---")
    log.error(e)
    raise e