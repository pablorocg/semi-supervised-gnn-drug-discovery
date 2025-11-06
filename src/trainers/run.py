# src/pl_run.py

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from utils import seed_everything


@hydra.main(
    config_path="../configs/",
    config_name="run.yaml",
    version_base=None,
)
def main(cfg: DictConfig):
    # Print out the full config
    print(OmegaConf.to_yaml(cfg))

    # Seed everything for reproducibility
    # Use PL's seed_everything for full coverage (trainers, dataloaders)
    pl.seed_everything(cfg.seed, workers=True)

    # Instantiate the Lightning Logger (e.g., WandbLogger)
    # The config now points to pytorch_lightning.loggers.WandbLogger
    logger = hydra.utils.instantiate(cfg.logger)

    # Instantiate the LightningDataModule (from src/qm9.py)
    dm = hydra.utils.instantiate(cfg.dataset.init)

    # Instantiate the LightningModule (from src/pl_model.py)
    # We pass the relevant sub-configs to its constructor
    model = hydra.utils.instantiate(
        cfg.model.lightning_module,
        model_cfg=cfg.model.init,
        optim_cfg=cfg.trainer.init.optimizer,
        scheduler_cfg=cfg.trainer.init.scheduler,
        num_models=cfg.ensemble.num_models,
        _recursive_=False,  # Ensure hydra doesn't resolve inside the sub-configs
    )

    if cfg.compile_model:
        model = torch.compile(model)

    # Instantiate the Lightning Trainer
    # The config (configs/trainer/pl_trainer.yaml) now configures this
    trainer = hydra.utils.instantiate(
        cfg.trainer.pl_trainer, logger=logger, deterministic=cfg.force_deterministic
    )

    # Start training
    trainer.fit(model, datamodule=dm)

    # Run testing
    # Note: The README mentions test data wasn't implemented.
    # This will run the test loop if you want it.
    results = trainer.test(model, datamodule=dm)
    print("Test Results:", results)


if __name__ == "__main__":
    main()
