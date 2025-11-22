import copy
import gc
import logging

import hydra
import optuna
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from src.utils.path_utils import get_configs_dir, get_storage_dir

logging.basicConfig(level=logging.INFO)


@hydra.main(
    config_path=get_configs_dir(),
    config_name="hpo_config",
    version_base="1.3",
)
def main(cfg: DictConfig) -> None:

    print("--- Base Config ---")
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed, workers=True)

    dm_ref = instantiate(cfg.dataset.init)
    dm_ref.setup()

    num_tasks = dm_ref.num_tasks
    pos_weights = dm_ref.get_pos_weights()
    train_loader = dm_ref.train_dataloader()
    
    iter(train_loader) # Needed to compute length
    batches_per_epoch = len(train_loader)
    
    max_epochs = cfg.trainer.init.max_epochs

    print(f"Dataset: {cfg.dataset.name} | Num Tasks: {num_tasks}")

    del dm_ref
    gc.collect()

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_categorical("learning_rate", [1e-4, 3e-4, 5e-4, 8e-4, 1e-3])
        wd = trial.suggest_categorical("weight_decay", [1e-5, 1e-4, 5e-4, 1e-3, 5e-3])
        bs = trial.suggest_categorical("batch_size", [16, 32, 64])
        dim = trial.suggest_categorical("hidden_channels", [32, 64, 128, 256])
        num_gnn_layers = trial.suggest_int("num_gnn_layers", 1, 5)
        gnn_mlp_layers = trial.suggest_int("gnn_mlp_layers", 1, 2)
        encoder_dropout = trial.suggest_float("encoder_dropout", 0.0, 0.3, step=0.1)
        dropout = trial.suggest_float("dropout", 0.2, 0.6, step=0.1)
        pooling_type = trial.suggest_categorical("pooling_type", ["mean", "add", "max"])
        activation = trial.suggest_categorical(
            "activation", ["relu", "gelu", "silu", "leaky_relu", "elu"]
        )

        cfg_trial = copy.deepcopy(cfg)
        cfg_trial.lightning_module.init.learning_rate = lr
        cfg_trial.lightning_module.init.weight_decay = wd
        cfg_trial.dataset.init.batch_size_train = bs
        cfg_trial.model.init.embedding_dim = dim
        cfg_trial.model.init.hidden_channels = dim
        cfg_trial.model.init.num_gnn_layers = num_gnn_layers
        cfg_trial.model.init.gnn_mlp_layers = gnn_mlp_layers
        cfg_trial.model.init.encoder_dropout = encoder_dropout
        cfg_trial.model.init.dropout = dropout
        cfg_trial.model.init.pooling_type = pooling_type
        cfg_trial.model.init.activation = activation

        OmegaConf.resolve(cfg_trial)

        dm = instantiate(cfg_trial.dataset.init)
        dm.setup()

        model = instantiate(
            cfg_trial.model.init,
            num_tasks=num_tasks,
        )

        lightning_module = instantiate(
            cfg_trial.lightning_module.init,
            model=model,
            num_outputs=num_tasks,
            loss_weights=pos_weights,
        )

        logger = instantiate(cfg_trial.logger.wandb)

        trainer = instantiate(
            cfg_trial.trainer.init,
            _target_=Trainer,
            logger=logger,
        )

        callbacks = [
            TQDMProgressBar(refresh_rate=150),
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                monitor="val/roc_auc",
                mode="max",
                save_top_k=1,
                filename="best",
                enable_version_counter=False,
            ),
            PyTorchLightningPruningCallback(trial, monitor="val/roc_auc"),
        ]
        trainer.callbacks.extend(callbacks)

        try:
            trainer.fit(model=lightning_module, datamodule=dm)

            # Get the best validation ROC-AUC score
            val_roc_auc = trainer.callback_metrics.get("val/roc_auc", None)
            if val_roc_auc is None:
                val_roc_auc = trainer.checkpoint_callback.best_model_score

            # Test the best model on the test set
            best_model_path = trainer.checkpoint_callback.best_model_path
            lightning_module = lightning_module.load_weights(best_model_path)
            trainer.test(model=lightning_module, datamodule=dm)

            return float(
                val_roc_auc.item() if hasattr(val_roc_auc, "item") else val_roc_auc
            )

        except optuna.exceptions.TrialPruned:
            logging.info(f"Trial {trial.number} was pruned.")
            raise

        except Exception as e:
            logging.error(f"Trial {trial.number} failed with error: {e}")
            raise optuna.exceptions.TrialPruned()

    warmup_epochs = max(30, int(max_epochs * 0.50))
    n_warmup_steps = warmup_epochs * batches_per_epoch
    check_every_epochs = max(10, int(max_epochs * 0.03))
    interval_steps = check_every_epochs * batches_per_epoch
    n_min_trials = max(5, 15 // 3)

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=15,
        n_warmup_steps=n_warmup_steps,
        interval_steps=interval_steps,
        n_min_trials=n_min_trials,
    )

    study = optuna.create_study(
        storage=f"sqlite:///{get_storage_dir()}/optuna_hparam_optimization.db",
        study_name=f"{cfg.dataset.name}_tox21_GINE_optimization",
        direction="maximize",
        pruner=pruner,
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=1,
        timeout=5400,  # 1.5 hours
    )

    print("\n Trial:")
    trial = study.best_trial
    print(f"  Value (Max val/roc_auc): {trial.value:.4f}")

    print("\n  HParams:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
