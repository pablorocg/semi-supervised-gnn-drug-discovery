from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger

from src.data.qm9 import QM9DataModule
from src.models.gcn import GCN
from src.utils.path_utils import get_logs_dir
from src.lightning_modules.mean_teacher import MeanTeacherRegressionModel


@hydra.main(config_path="../../config/dataset", config_name="qm9")
def main(cfg):



    data_module = QM9DataModule(
        target=0,
        batch_size_train=32,
        batch_size_inference=32,
        num_workers=4,
    )

# data_module.prepare_data()
data_module.setup()

baseline_module = MeanTeacherRegressionModel(
    model=GCN(num_node_features=data_module.num_features, hidden_channels=128),
    num_outputs=data_module.num_classes,
)

# Run trainer in debug mode fast_dev_run=True,
trainer = Trainer(
    max_epochs=10,
    callbacks=[
        TQDMProgressBar(),
    ],
    logger=WandbLogger(name="drug_discovery_regression", save_dir=get_logs_dir()),
)
trainer.fit(baseline_module, datamodule=data_module)