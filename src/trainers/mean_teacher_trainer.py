import logging
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

    # 1. Instantiate DataModule
    dm = instantiate(cfg.dataset.init)
    # dm.prepare_data() # Ensure data is downloaded
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

    # 2. Instantiate Backbone Model
    # We pass num_tasks here as the backbone often needs output dim
    model = instantiate(
        cfg.model.init,
        num_tasks=num_tasks,
    )

    # 3. Instantiate Lightning Module
    # FIX: Point to .init so Hydra finds the _target_ class
    lightning_module = instantiate(
        cfg.lightning_module.init,
        model=model,
        num_outputs=num_tasks,
        loss_weights=dm.get_pos_weights(),
    )

    # Setup logger
    logger = instantiate(cfg.logger.wandb)

    # Instantiate trainer
    trainer = instantiate(
        cfg.trainer.init,
        logger=logger,
    )

    # 4. Train
    log.info("Starting training...")
    trainer.fit(model=lightning_module, datamodule=dm)
    
    # 5. Test
    # This automatically loads the best checkpoint tracked by ModelCheckpoint
    # No need for manual state_dict loading
    log.info("Starting testing with best model...")
    trainer.test(datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    main()