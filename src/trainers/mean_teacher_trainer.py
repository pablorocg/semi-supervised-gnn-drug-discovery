import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from src.utils.path_utils import get_configs_dir

log = logging.getLogger(__name__)


@hydra.main(
    config_path=get_configs_dir(),
    config_name="mean_teacher_config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Main training pipeline."""
    # Print full config
    log.info(OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    seed_everything(cfg.seed, workers=True)

    dm = instantiate(cfg.dataset.init)
    dm.setup("fit")

    # Get dataset properties
    task_type = dm.task_type
    num_node_features = dm.num_node_features
    num_edge_features = dm.num_edge_features
    num_tasks = dm.num_tasks

    log.info("Dataset properties:")
    log.info(f"Task type: {task_type}")
    log.info(f"Number of node features: {num_node_features}")
    log.info(f"Number of edge features: {num_edge_features}")
    log.info(f"Number of tasks: {num_tasks}")
    log.info(f"Model name: {cfg.model.name}")

    # Instantiate model
    model = instantiate(
        cfg.model.init,
        dataset=cfg.dataset.name,
        num_tasks=num_tasks,
    )

    # Create lightning module
    lightning_module = instantiate(
        cfg.lightning_module.init,
        model=model,
        num_outputs=num_tasks,
        loss_weights=dm.get_pos_weights(),
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

    trainer.test(model=lightning_module, datamodule=dm)

    # # Load best checkpoint before testing
    # best_model_path = trainer.checkpoint_callback.best_model_path
    # print(f"Loading best model from: {best_model_path}")

    # lightning_module = lightning_module.load_weights(best_model_path)

    # trainer.test(model=lightning_module, datamodule=dm)


if __name__ == "__main__":
    main()
