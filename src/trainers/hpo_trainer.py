import logging
import gc
import torch
import hydra
import optuna
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from src.data.moleculenet import MoleculeNetDataModule
from src.data.pcba import OgbgMolPcbaDataModule
from src.lightning_modules.baseline import BaselineModule
from src.utils.path_utils import get_configs_dir, get_storage_dir
from optuna.integration import PyTorchLightningPruningCallback

from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)


logging.basicConfig(level=logging.INFO)


@hydra.main(
    config_path=get_configs_dir(),
    config_name="hpo_config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:
    """Main hyperparameter optimization pipeline."""

    print("--- Base Config ---")
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed, workers=True)

    if cfg.dataset.name == "ogbg-molpcba":
        dm_ref = instantiate(cfg.dataset.init, _target_=OgbgMolPcbaDataModule)
    else:
        dm_ref = instantiate(cfg.dataset.init, _target_=MoleculeNetDataModule)

    dm_ref.setup("fit")

    num_tasks = dm_ref.num_tasks
    pos_weights = dm_ref.get_pos_weights()

    print(f"Dataset: {cfg.dataset.name} | Num Tasks: {num_tasks}")

    del dm_ref
    gc.collect()

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        """
        cfg_trial = cfg.copy()

        # 1. Training Params
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        bs = trial.suggest_categorical("batch_size", [32, 64])

        cfg_trial.lightning_module.init.learning_rate = lr
        cfg_trial.lightning_module.init.weight_decay = wd
        cfg_trial.dataset.init.batch_size_train = bs

        # 2. Model Dimensions
        dim = trial.suggest_categorical("hidden_channels", [8, 16, 32, 64, 128, 256])
        cfg_trial.model.init.embedding_dim = dim
        cfg_trial.model.init.hidden_channels = dim

        # 3. GNN Structure
        cfg_trial.model.init.num_gnn_layers = trial.suggest_int("num_gnn_layers", 3, 5)
        cfg_trial.model.init.gnn_mlp_layers = trial.suggest_int("gnn_mlp_layers", 1, 2)

        # 4. Regularization & Stability
        cfg_trial.model.init.encoder_num_heads = trial.suggest_categorical(
            "encoder_num_heads", [4]
        )
        cfg_trial.model.init.encoder_dropout = trial.suggest_float(
            "encoder_dropout", 0.0, 0.3, step=0.1
        )
        cfg_trial.model.init.dropout = trial.suggest_float(
            "dropout", 0.2, 0.6, step=0.1
        )
        cfg_trial.model.init.pooling_type = trial.suggest_categorical(
            "pooling_type", ["mean", "add"]
        )
        cfg_trial.model.init.activation = trial.suggest_categorical(
            "activation", ["relu", "silu"]
        )

        if cfg.dataset.name == "ogbg-molpcba":
            dm = instantiate(cfg_trial.dataset.init, _target_=OgbgMolPcbaDataModule)
        else:
            dm = instantiate(cfg_trial.dataset.init, _target_=MoleculeNetDataModule)

        dm.setup("fit")

        model = instantiate(
            cfg_trial.model.init, num_tasks=num_tasks, _recursive_=False
        )

        lightning_module = instantiate(
            cfg_trial.lightning_module.init,
            _target_=BaselineModule,
            model=model,
            num_outputs=num_tasks,
            loss_weights=pos_weights,
        )

        logger = instantiate(cfg.logger.wandb, _target_=WandbLogger, group="optuna_hpo")

        callbacks = [
            ModelCheckpoint(
                monitor="val/roc_auc",
                mode="max",
                save_top_k=1,
                save_last=False,
                filename="best",
                auto_insert_metric_name=False,
            ),
            ModelCheckpoint(
                save_top_k=0,
                save_last=True,
                filename="last",
                verbose=False,
            ),
            LearningRateMonitor(
                logging_interval="epoch",
            ),
            # EarlyStopping(
            #     monitor="val/loss",
            #     patience=30,
            #     mode="min",
            # ),
            PyTorchLightningPruningCallback(trial, monitor="val/roc_auc"),
        ]

        # Create trainer directly without instantiate for callbacks
        trainer = Trainer(
            max_epochs=cfg_trial.trainer.init.max_epochs,
            accelerator=cfg_trial.trainer.init.accelerator,
            devices=cfg_trial.trainer.init.devices,
            precision=cfg_trial.trainer.init.precision,
            accumulate_grad_batches=cfg_trial.trainer.init.accumulate_grad_batches,
            log_every_n_steps=cfg_trial.trainer.init.log_every_n_steps,
            check_val_every_n_epoch=cfg_trial.trainer.init.check_val_every_n_epoch,
            enable_checkpointing=cfg_trial.trainer.init.enable_checkpointing,
            enable_progress_bar=cfg_trial.trainer.init.enable_progress_bar,
            enable_model_summary=cfg_trial.trainer.init.enable_model_summary,
            deterministic=cfg_trial.trainer.init.deterministic,
            benchmark=cfg_trial.trainer.init.benchmark,
            logger=logger,
            callbacks=callbacks,
        )

        try:
            trainer.fit(model=lightning_module, datamodule=dm)

            best_score = trainer.checkpoint_callback.best_model_score

            if best_score is None:
                best_score = trainer.callback_metrics.get("val/roc_auc")

            if best_score is None:
                logging.warning(f"Trial {trial.number}: No validation metric found.")
                raise optuna.exceptions.TrialPruned()

            # Load best checkpoint before testing
            best_model_path = trainer.checkpoint_callback.best_model_path
            print(f"Loading best model from: {best_model_path}")

            # Use your custom load_weights method
            lightning_module = lightning_module.load_weights(best_model_path)
            
            # Test with best model
            trainer.test(model=lightning_module, datamodule=dm)

            final_metric = float(best_score)

            logging.info(f"Trial {trial.number} finished. Val ROC_AUC: {final_metric}")

            return final_metric

        except optuna.exceptions.TrialPruned:
            logging.info(f"Trial {trial.number} was pruned.")
            raise

        except Exception as e:
            logging.error(f"Trial {trial.number} failed with error: {e}")
            raise optuna.exceptions.TrialPruned()

        finally:
            # Cleanup
            del model, lightning_module, trainer, dm, logger
            gc.collect()
            torch.cuda.empty_cache()

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
    )

    storage_path = f"sqlite:///{get_storage_dir()}/optuna_hparam_optimization.db"

    study = optuna.create_study(
        storage=storage_path,
        study_name=f"{cfg.dataset.name}_hparam_optimization",
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )

    print(f"Starting optimization. Storage: {storage_path}")

    study.optimize(
        objective,
        n_trials=50,
        timeout=5400,  # 1.5 hours
    )

    # --- Results ---
    print("\n" + "=" * 50)
    print("Hyperparameter Search Finished!")
    print(f"Number of finished trials: {len(study.trials)}")

    print("\nBest trial:")
    trial = study.best_trial
    print(f"  Value (Max val/roc_auc): {trial.value:.4f}")

    print("\n  Best HParams:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()