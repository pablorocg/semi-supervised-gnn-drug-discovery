import optuna
import pytorch_lightning as L
from optuna_integration.pytorch_lightning import PyTorchLightningPruningCallback
from pytorch_lightning.callbacks import TQDMProgressBar

from src.data.qm9 import QM9DataModule
from src.lightning_modules.mean_teacher import MeanTeacherRegressionModel
from src.models.gcn import GCN
import logging

# Fixed parameters for the study
MAX_EPOCHS = 150  # A shorter epoch count for faster tuning
BATCH_SIZE = 256
NUM_WORKERS = 4
GCN_HIDDEN_CHANNELS = 128
QM9_TARGET = 0  # The specific QM9 target you are regressing

logging.basicConfig(level=logging.INFO)


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function for tuning the MeanTeacherRegressionModel.
    """

    # --- 1. Define the Search Space ---
    # We select the 4 most critical hyperparameters
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "ema_decay": trial.suggest_float("ema_decay", 0.99, 0.9999, log=True),
        "max_consistency_weight": trial.suggest_float(
            "max_consistency_weight", 1.0, 100.0, log=True
        ),
        "consistency_rampup_epochs": trial.suggest_int(
            "consistency_rampup_epochs", 1, 50
        ),
        "hidden_channels": trial.suggest_categorical(
            "hidden_channels", [64, 128, 256, 512]
        ),
    }

    # Set up Data and Model ---
    data_module = QM9DataModule(
        target=QM9_TARGET,
        batch_size_train=BATCH_SIZE,
        batch_size_inference=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    data_module.setup()
    # We must re-create the model for each trial

    # *params unpacks the dictionary into keyword arguments, **params would be incorrect here
    model = MeanTeacherRegressionModel(
        model=GCN(
            num_node_features=data_module.num_features,
            hidden_channels=params["hidden_channels"],
        ),
        num_outputs=1,
        learning_rate=params["learning_rate"],
        ema_decay=params["ema_decay"],
        max_consistency_weight=params["max_consistency_weight"],
        consistency_rampup_epochs=params["consistency_rampup_epochs"],

    )

    # Add a pruning callback
    pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val/loss")

    trainer = L.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices=1,
        logger=False,  # Disable default logger for speed
        enable_checkpointing=False,
        callbacks=[
            pruning_callback,
            TQDMProgressBar(refresh_rate=0),
        ],  # Disable progress bar
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Run Training ---
    try:
        trainer.fit(model, datamodule=data_module)
    except optuna.exceptions.TrialPruned:
        return optuna.trial.TrialPruned()
    except Exception:
        return optuna.trial.TrialPruned()

    # Return the Metric to Optimize ---
    # We are minimizing the validation loss
    # .item() is important to get the Python float
    # Select best val loss over all epochs
    best_val_loss = min(trainer.callback_metrics["val/loss"].cpu().detach().numpy())
    logging.info(f"Trial {trial.number} completed with val/loss: {best_val_loss}")
    return float(best_val_loss)


# -------------------------------------------------------------------
# --- OPTUNA STUDY CREATION (MAIN SCRIPT) ---
# -------------------------------------------------------------------

if __name__ == "__main__":
    # We use a MedianPruner to stop unpromising trials early
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,  # Wait for 5 trials before pruning
        n_warmup_steps=MAX_EPOCHS // 2,  # Wait for half the epochs
    )

    # We want to minimize the validation loss
    study = optuna.create_study(
        storage="sqlite:////work3/s250272/semi_supervised_gnn_drug_discovery/logs/optuna_regr_hparam_optimization.db",
        study_name="regr_hparam_optimization",
        direction="minimize",
        pruner=pruner,
        load_if_exists=True,
    )

    # Start the optimization
    study.optimize(
        objective,
        n_trials=2,  # Number of trials to run
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
