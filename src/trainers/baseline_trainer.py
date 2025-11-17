import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from src.data.moleculenet import MoleculeNetDataModule
from src.data.pcba import OgbgMolPcbaDataModule
from src.data.qm9 import QM9DataModule
from src.lightning_modules.baseline import BaselineModule
from src.utils.path_utils import get_configs_dir
from pytorch_lightning import seed_everything


@hydra.main(
    config_path=get_configs_dir(),
    config_name="baseline_config.yaml",  
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:
    """Main training pipeline."""
    # Print full config
    print(OmegaConf.to_yaml(cfg))
    
    # Set seed for reproducibility
    seed_everything(cfg.seed, workers=True)
    
    # Instantiate datamodule based on dataset name
    if cfg.dataset.name == "QM9":
        dm = instantiate(cfg.dataset.init, _target_=QM9DataModule)
    elif cfg.dataset.name == "ogbg-molpcba":
        dm = instantiate(cfg.dataset.init, _target_=OgbgMolPcbaDataModule)
    else:
        dm = instantiate(cfg.dataset.init, _target_=MoleculeNetDataModule)

    
    dm.setup('fit')

    # Get dataset properties
    n_outputs = dm.num_tasks
    task_type = dm.task_type
    in_channels = dm.num_features

    print(f"Number of input features: {in_channels}, type of task: {task_type}, number of outputs: {n_outputs}")
    
    # Instantiate model with dynamic in_channels and out_channels
    model = instantiate(
        cfg.model.init,
        # _recursive_=False,
        # in_channels=in_channels,
        # out_channels=n_outputs,
        num_tasks=n_outputs# 19
    )
    
    # Create lightning module
    lightning_module = instantiate(
        cfg.lightning_module.init,
        _target_=BaselineModule,
        model=model,
        num_outputs=n_outputs,
        task_type=task_type,
    )
    
    # Setup logger
    logger = instantiate(cfg.logger.wandb, _target_=WandbLogger)
    
    # Instantiate trainer
    trainer = instantiate(
        cfg.trainer.init,
        _target_=Trainer,
        logger=logger,
    )
    
    # Train and test
    trainer.fit(model=lightning_module, datamodule=dm)
    # trainer.test(model=lightning_module, datamodule=dm)


if __name__ == "__main__":
    main()