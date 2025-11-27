import logging
import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
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

    # Instantiate DataModule 
    dm = instantiate(cfg.dataset.init)
    dm.setup("fit")  # prepare splits

    # Dataset info
    log.info("Dataset properties:")
    log.info(f"Task type: {dm.task_type}")
    log.info(f"Number of node features: {dm.num_node_features}")
    log.info(f"Number of edge features: {dm.num_edge_features}")
    log.info(f"Number of tasks: {dm.num_tasks}")
    log.info(f"Model name: {cfg.model.name}")

    # Instantiate Backbone Model
    model = instantiate(
        cfg.model.init,
        num_tasks=dm.num_tasks,
    )

    # Instantiate Lightning Module
    lightning_module = instantiate(
        cfg.lightning_module.init,
        model=model,
        num_outputs=dm.num_tasks,
        loss_weights=dm.get_pos_weights(),
    )

    #  Instantiate Logger 
    logger = instantiate(cfg.logger.wandb)

    # Instantiate trainer
    trainer = instantiate(
        cfg.trainer.init,
        _target_=Trainer,
        logger=logger,
        gradient_clip_val=1.0,
    )

    #  Train 
    log.info("Starting training...")
    trainer.fit(model=lightning_module, datamodule=dm)

    # Test (loads best checkpoint automatically) 
    log.info("Starting testing with best model...")
    trainer.test(datamodule=dm, ckpt_path="best")


if __name__ == "__main__":
    main()
