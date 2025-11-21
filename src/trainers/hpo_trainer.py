import logging

import hydra
import optuna
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from src.data.moleculenet import MoleculeNetDataModule
from src.data.pcba import OgbgMolPcbaDataModule
from src.lightning_modules.baseline import BaselineModule
from src.utils.path_utils import get_configs_dir

logging.basicConfig(level=logging.INFO)


@hydra.main(
    config_path=get_configs_dir(),
    config_name="hpo_config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Main training pipeline."""
    
    print(OmegaConf.to_yaml(cfg))
    
    seed_everything(cfg.seed, workers=True)

    if cfg.dataset.name == "ogbg-molpcba":
        dm = instantiate(cfg.dataset.init, _target_=OgbgMolPcbaDataModule)
    else:
        dm = instantiate(cfg.dataset.init, _target_=MoleculeNetDataModule)

    dm.setup("fit")

    # Get dataset properties
    task_type = dm.task_type
    num_node_features = dm.num_node_features
    num_edge_features = dm.num_edge_features
    num_tasks = dm.num_tasks

    print("Dataset properties:")
    print(f"Task type: {task_type}")
    print(f"Number of node features: {num_node_features}")
    print(f"Number of edge features: {num_edge_features}")
    print(f"Number of tasks: {num_tasks}")

    print("Model properties:")
    print(f"Model name: {cfg.model.name}")

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function for tuning the MeanTeacherRegressionModel.
        """

        params = {
            "learning_rate": trial.suggest_categorical("learning_rate", [1e-3, 5e-4, 1e-4, 5e-5]),
            "weight_decay": trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 1e-3, 5e-3]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            ""

        }

        # Set up Data and Model ---
        data_module = instantiate()

        data_module.setup()

        model = instantiate(
            cfg.model.init,
            num_tasks=num_tasks,
        )

        lightning_module = instantiate(
            cfg.lightning_module.init,
            _target_=BaselineModule,
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

        
        try:
            
            trainer.fit(model=lightning_module, datamodule=dm)

            best_model_path = trainer.checkpoint_callback.best_model_path
            print(f"Loading best model from: {best_model_path}")
            lightning_module = lightning_module.load_weights(best_model_path)
            
            trainer.test(model=lightning_module, datamodule=dm)
        
        except optuna.exceptions.TrialPruned:
            return optuna.trial.TrialPruned()
        except Exception:
            return optuna.trial.TrialPruned()

        best_val_loss = min(trainer.callback_metrics["val/loss"].cpu().detach().numpy())
        logging.info(f"Trial {trial.number} completed with val/loss: {best_val_loss}")
        return float(best_val_loss)

        # We use a MedianPruner to stop unpromising trials early
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Wait for 5 trials before pruning
        n_warmup_steps=cfg.trainer.max_epochs // 2,  # Wait for half the epochs
    )

    # We want to minimize the validation loss
    study = optuna.create_study(
        storage="sqlite:////work3/s250272/semi_supervised_gnn_drug_discovery/logs/optuna_regr_hparam_optimization.db",
        study_name="cls_hparam_optimization",
        direction="minimize",
        pruner=pruner,
        load_if_exists=True,
    )

    # Start the optimization
    study.optimize(
        objective,
        n_trials=1,  # Number of trials to run
        timeout=3600 * 2,  # Set a timeout (e.g., 1 hour)
    )

    # --- Print Results ---
    print("\n" + "=" * 50)
    print("Hyperparameter Search Finished!")
    print(f"Number of finished trials: {len(study.trials)}")

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (min val/loss): {trial.value:.4f}")

    print("\n  Best HParams:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()