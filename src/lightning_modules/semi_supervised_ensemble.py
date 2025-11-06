import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemiSuperviseEnsembleTrainer(pl.LightningModule):
    """
    This is the PyTorch LightningModule that encapsulates the training,
    validation, and optimization logic, replacing the custom SemiSupervisedEnsemble trainer.
    """

    def __init__(
        self,
        model_cfg: dict,
        optim_cfg: dict,
        scheduler_cfg: dict,
        num_models: int = 1,
    ):
        """
        Args:
            model_cfg (dict): Hydra config for instantiating the core model (e.g., GCN).
            optim_cfg (dict): Hydra config for the optimizer (e.g., AdamW).
            scheduler_cfg (dict): Hydra config for the LR scheduler (e.g., StepLR).
            num_models (int): The number of models in the ensemble.
        """
        super().__init__()
        # Save all hyperparameters (model_cfg, optim_cfg, etc.) to self.hparams
        # This makes them accessible via self.hparams and logs them.
        self.save_hyperparameters()

        # Instantiate the model ensemble
        self.models = nn.ModuleList(
            [hydra.utils.instantiate(model_cfg) for _ in range(num_models)]
        )

        # Define the loss function
        self.supervised_criterion = torch.nn.MSELoss()

    def forward(self, data):
        """
        Defines the forward pass for inference/prediction.
        This averages the predictions of all models in the ensemble.
        """
        # data is the full Data object from PyG
        # (x, edge_index, batch) are extracted inside the GCN model
        preds = [model(data) for model in self.models]
        avg_preds = torch.stack(preds).mean(0)
        return avg_preds

    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.
        """
        # The custom Collater in qm9_utils.py yields (batch, y)
        data, targets = batch

        # Calculate supervised loss for each model
        supervised_losses = [
            self.supervised_criterion(model(data), targets) for model in self.models
        ]

        # Sum losses for backpropagation
        loss = sum(supervised_losses)

        # Log the average loss
        avg_loss = loss / len(self.models)
        self.log(
            "train/supervised_loss",
            avg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=data.num_graphs,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.
        """
        data, targets = batch

        # Get ensemble prediction
        avg_preds = self.forward(data)

        # Calculate validation loss
        val_loss = torch.nn.functional.mse_loss(avg_preds, targets)

        self.log(
            "val/MSE",
            val_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=data.num_graphs,
        )
        return val_loss

    def test_step(self, batch, batch_idx):
        """
        Performs a single test step.
        (This was not in the original trainer but is good practice)
        """
        data, targets = batch
        avg_preds = self.forward(data)
        test_loss = torch.nn.functional.mse_loss(avg_preds, targets)

        self.log(
            "test/MSE",
            test_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=data.num_graphs,
        )
        return test_loss

    def configure_optimizers(self):
        """
        Sets up the optimizer and learning rate scheduler.
        """
        # Get all parameters from all models in the ensemble
        all_params = [p for m in self.models for p in m.parameters()]

        # Instantiate optimizer
        optimizer = hydra.utils.instantiate(self.hparams.optim_cfg, params=all_params)

        # Instantiate scheduler
        scheduler = hydra.utils.instantiate(
            self.hparams.scheduler_cfg, optimizer=optimizer
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
