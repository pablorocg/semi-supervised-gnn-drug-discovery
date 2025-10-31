import logging
from abc import abstractmethod
from typing import Callable, Optional

import pytorch_lightning as L
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchmetrics import (
    AUROC,
    Accuracy,
    AveragePrecision,
    F1Score,
    MetricCollection,
)

# Set up logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-3,
        warmup_epochs: int = 5,
        cosine_period_ratio: float = 1,
        compile_mode: str = None,
        weights: str = None,
        optimizer: str = "SGD",
        weight_decay: float = 3e-5,
        nesterov: bool = True,
        momentum: float = 0.99,
        # RE-ADDED: Transforms to be applied on-the-fly
        train_transforms: Optional[Callable] = None,
        val_transforms: Optional[Callable] = None,
        test_transforms: Optional[Callable] = None,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.loss_fn = None  # Will be defined in the child class
        self.warmup_epochs = warmup_epochs
        self.cosine_period_ratio = cosine_period_ratio
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        self.momentum = momentum
        assert 0 < cosine_period_ratio <= 1

        # RE-ADDED: Save on-the-fly transforms
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.test_transforms = test_transforms

        self.num_classes = num_classes
        self.save_hyperparameters(ignore=["model"])

        self.train_metrics = self.configure_metrics("train")
        self.val_metrics = self.configure_metrics("val")
        self.test_metrics = self.configure_metrics("test")

        self.model = model

        if weights is not None:
            log.info(f"Loading weights from {weights}")
            self.load_weights(weights)

        self.model = (
            torch.compile(model, mode=compile_mode)
            if compile_mode is not None
            else model
        )

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        if self.optimizer == "SGD":
            optimizer = SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
                nesterov=self.nesterov,
            )
        elif self.optimizer == "AdamW":
            optimizer = AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                amsgrad=False,
                betas=(0.9, 0.98),
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

        try:
            if self.trainer.datamodule:
                steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            else:
                steps_per_epoch = len(self.trainer.train_dataloader)
        except Exception:
            log.warning("Could not determine steps_per_epoch. Using fallback of 1000.")
            steps_per_epoch = 1000

        if steps_per_epoch == 0:
            log.warning("steps_per_epoch is zero. Using fallback of 1000.")
            steps_per_epoch = 1000

        total_steps = self.trainer.max_epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch
        cosine_steps = max(
            1, int(self.cosine_period_ratio * (total_steps - warmup_steps))
        )

        log.info(f"Optimizer: {self.optimizer}, LR: {self.learning_rate}")
        log.info(
            f"Total steps: {total_steps}, Warmup steps: {warmup_steps}, Cosine steps: {cosine_steps}"
        )

        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_steps)

        if self.warmup_epochs > 0 and warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1.0 / 1000,
                total_iters=warmup_steps,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        else:
            scheduler = cosine_scheduler

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler_config]

    def configure_metrics(self, prefix: str):
        task = "multilabel"
        avg_strategy = "macro"

        return MetricCollection(
            {
                f"{prefix}/AUROC_macro": AUROC(
                    task=task,
                    num_labels=self.num_classes,
                    average=avg_strategy,
                    validate_args=True,
                ),
                f"{prefix}/AUPRC_macro": AveragePrecision(
                    task=task,
                    num_labels=self.num_classes,
                    average=avg_strategy,
                    validate_args=True,
                ),
                f"{prefix}/F1_macro": F1Score(
                    task=task,
                    num_labels=self.num_classes,
                    average=avg_strategy,
                    validate_args=True,
                ),
                f"{prefix}/Accuracy_macro": Accuracy(
                    task=task,
                    num_labels=self.num_classes,
                    average=avg_strategy,
                    validate_args=True,
                ),
            },
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Applies transforms to the batch *after* it has been moved to the
        target device (e.g., GPU).
        """
        if self.trainer.training and self.train_transforms is not None:
            batch = self.train_transforms(batch)
        elif (
            self.trainer.validating or self.trainer.sanity_checking
        ) and self.val_transforms is not None:
            batch = self.val_transforms(batch)
        elif (
            self.trainer.testing or self.trainer.predicting
        ) and self.test_transforms is not None:
            batch = self.test_transforms(batch)

        # CRITICAL FIX: Return the modified batch
        return batch

    def on_train_epoch_end(self):
        if self.train_metrics:
            metrics = self.train_metrics.compute()
            self.log_dict(
                metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
            )
            self.train_metrics.reset()

    def on_validation_epoch_end(self):
        if self.val_metrics:
            metrics = self.val_metrics.compute()
            self.log_dict(
                metrics, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True
            )
            self.val_metrics.reset()

    def load_weights(self, weights):
        # ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        # print(
        #     f"Loading weights trained for {ckpt['global_step']} steps / {ckpt['epoch']} epochs."
        # )
        # self.load_state_dict(ckpt["state_dict"], strict=False)
        pass

    def load_state_dict(self, state_dict, load_decoder=True, *args, **kwargs):
        # old_params = copy.deepcopy(self.state_dict())

        # target_compiled = "_orig" in next(iter(old_params.keys()))
        # source_compiled = "_orig" in next(iter(state_dict.keys()))

        # print(f"Target compiled: {target_compiled}, source compiled: {source_compiled}")

        # if not target_compiled and source_compiled:
        #     print(
        #         "Source state_dict is compiled, but target model is not. Removing _orig suffix from state_dict keys."
        #     )
        #     state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # # Repeat stem weights when state_dict num_channels is smaller than new_state_dict num_channels
        # if self.model.stem_weight_name is not None:
        #     stem_name = f"model.{self.model.stem_weight_name}"
        #     pt_input_channels = state_dict[stem_name].shape[
        #         1
        #     ]  # (N, C, H, W, Z) where N is num tokens.
        #     ft_input_channels = old_params[stem_name].shape[1]
        #     if pt_input_channels < ft_input_channels:
        #         assert pt_input_channels == 1, (
        #             "Stem weights can only be repeated if the input channels in the state_dict is 1."
        #         )
        #         print(
        #             f"Repeating stem weights from {pt_input_channels} to {ft_input_channels} channels for {stem_name}."
        #         )
        #         state_dict[stem_name] = (
        #             state_dict[stem_name].repeat(1, ft_input_channels, 1, 1, 1)
        #             / ft_input_channels
        #         )

        # # Filter out keys that are not in the old state dict or have different shapes
        # def should_load_key(key, state_dict, old_params, load_decoder):
        #     # reject all decoder keys regardless of their shape
        #     if not load_decoder and key.startswith("model.decoder"):
        #         return True
        #     # accept all keys that are in the old state dict and have the same shape
        #     return (key in old_params) and (
        #         old_params[key].shape == state_dict[key].shape
        #     )

        # # Filter state_dict to only include keys that should be loaded
        # state_dict = {
        #     k: v
        #     for k, v in state_dict.items()
        #     if should_load_key(k, state_dict, old_params, load_decoder)
        # }

        # # Lists used to inform master of our whereabouts
        # rejected_keys_new = [k for k in state_dict.keys() if k not in old_params]
        # state_dict = {k: v for k, v in state_dict.items() if k in old_params}
        # rejected_keys_shape = [
        #     k for k in state_dict.keys() if old_params[k].shape != state_dict[k].shape
        # ]
        # rejected_keys_decoder = [
        #     k
        #     for k in state_dict.keys()
        #     if not load_decoder and k.startswith("model.decoder")
        # ]

        # # Load the state dict
        # kwargs["strict"] = False
        # super().load_state_dict(state_dict, *args, **kwargs)

        # # Check if weights were actually loaded
        # new_params = self.state_dict()
        # rejected_keys_data = []

        # successful = 0
        # unsuccessful = 0
        # for param_name, p1, p2 in zip(
        #     old_params.keys(), old_params.values(), new_params.values()
        # ):
        #     if p1.data.ne(p2.data).sum() > 0:
        #         successful += 1
        #     else:
        #         unsuccessful += 1
        #         if (
        #             param_name not in rejected_keys_new
        #             and param_name not in rejected_keys_shape
        #         ):
        #             rejected_keys_data.append(param_name)

        # print(
        #     f"Succesfully transferred weights for {successful}/{successful + unsuccessful} layers"
        # )
        # print(
        #     f"Rejected the following keys:\n"
        #     f"Not in old dict: {rejected_keys_new}.\n"
        #     f"Wrong shape: {rejected_keys_shape}.\n"
        #     f"Post check not succesful: {rejected_keys_data}."
        # )
        # if not load_decoder:
        #     print(
        #         "Decoder weights were not loaded, as requested. If you want to load them, set `load_decoder=True`."
        #     )
        #     print(f"Rejected decoder keys: {rejected_keys_decoder}.")
        # else:
        #     print(
        #         "Warning! Also loaded the decoder. If you are finetuning, this might not be what you want."
        #     )

        # assert successful > 0, (
        #     "No weights were loaded. Check the state_dict and the model architecture."
        # )
        pass
