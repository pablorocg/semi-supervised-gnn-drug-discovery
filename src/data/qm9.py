import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch_geometric.datasets import QM9

from src.utils.dataset_utils import (
    DataLoader,
    GetTarget,
)
from src.utils.path_utils import get_data_dir


class QM9DataModule(pl.LightningDataModule):
    """
    DataModule for the QM9 dataset.

    Parameters
    ----------
    target : int, optional
        Index of the target property to predict (0-11), by default 0
    batch_size_train : int, optional
        Batch size for training dataloaders, by default 32
    batch_size_inference : int, optional
        Batch size for validation and test dataloaders, by default 32
    num_workers : int, optional
        Number of workers for data loading, by default 1
    splits : list of int or float, optional
        Sizes or proportions for dataset splits: [unlabeled, labeled, val, test], by default [0.72, 0.08, 0.1, 0.1]
    seed : int, optional
        Random seed for shuffling and splitting the dataset, by default 0
    subset_size : int or None, optional
        If specified, use only a subset of the dataset of this size, by default None
    data_augmentation : bool, optional
        Whether to apply data augmentation (not used here), by default False
    mode : str, optional
        Mode of operation: "supervised" or "semisupervised", by default "semisupervised"
    name : str, optional
        Name of the dataset, by default "qm9"
    """

    def __init__(
        self,
        target: int | None = 0,
        batch_size_train: int = 32,
        batch_size_inference: int = 32,
        num_workers: int = 1,
        splits: list[int] | list[float] = [
            0.72,
            0.08,
            0.1,
            0.1,
        ],  # Unlabeled, labeled, val, test
        seed: int = 0,
        subset_size: int | None = None,
        data_augmentation: bool = False,  # Unused but here for compatibility
        mode="semisupervised",
        name: str = "qm9",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = Path(get_data_dir()) / "QM9"

    def prepare_data(self) -> None:
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        # Download data
        QM9(root=self.data_dir)

    def setup(self, stage: str | None = None) -> None:
        dataset = QM9(root=self.data_dir, transform=GetTarget(self.hparams.target))

        # Shuffle dataset
        rng = np.random.default_rng(seed=self.hparams.seed)
        dataset = dataset[rng.permutation(len(dataset))]

        # Subset dataset
        if self.hparams.subset_size is not None:
            dataset = dataset[: self.hparams.subset_size]

        # Split dataset
        if all(isinstance(split, int) for split in self.hparams.splits):
            split_sizes = self.hparams.splits
        elif all(isinstance(split, float) for split in self.hparams.splits):
            split_sizes = [int(len(dataset) * prop) for prop in self.hparams.splits]

        split_idx = np.cumsum(split_sizes)

        self.data_train_labeled = dataset[split_idx[0] : split_idx[1]]

        # Semi-supervised: use part of training data as unlabeled data
        if self.hparams.mode == "semisupervised":
            self.data_train_unlabeled = dataset[: split_idx[0]]
        
        self.data_val = dataset[split_idx[1] : split_idx[2]]
        self.data_test = dataset[split_idx[2] :]

        self.batch_size_train_labeled = self.hparams.batch_size_train
        self.batch_size_train_unlabeled = self.hparams.batch_size_train

        self.print_dataset_info()


    def print_dataset_info(self) -> None:
        if self.hparams.mode == "supervised":
            print("Using supervised mode.")
            print(
                f"QM9 {self.dataset_name} dataset loaded with {len(self.data_train_labeled)} labeled,"
                f"{len(self.data_val)} validation, and {len(self.data_test)} test samples."
            )
            print(
                f"Batch sizes: labeled={self.batch_size_train_labeled}"
            )
        elif self.hparams.mode == "semisupervised":
            print("Using semi-supervised mode.")
            print(
                f"QM9 {self.dataset_name} dataset loaded with {len(self.data_train_labeled)} labeled, {len(self.data_train_unlabeled)} unlabeled, "
                f"{len(self.data_val)} validation, and {len(self.data_test)} test samples."
            )
            print(
                f"Batch sizes: labeled={self.batch_size_train_labeled}, unlabeled={self.batch_size_train_unlabeled}"
            )

    def train_dataloader(self) -> CombinedLoader:
        if self.hparams.mode == "supervised":
            return CombinedLoader(
                {
                    "labeled": self.supervised_train_dataloader(),
                },
                mode="max_size_cycle",
            )

        elif self.hparams.mode == "semisupervised":
            return CombinedLoader(
                {
                    "labeled": self.supervised_train_dataloader(),
                    "unlabeled": self.unsupervised_train_dataloader(),
                },
                mode="max_size_cycle",
            )

    def supervised_train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.data_train_labeled,
            batch_size=self.batch_size_train_labeled,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=True,
        )

    def unsupervised_train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.data_train_unlabeled,
            batch_size=self.batch_size_train_unlabeled,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size_inference,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size_inference,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )

    @property
    def num_features(self) -> int:
        return self.data_train_labeled.num_node_features

    @property
    def num_tasks(self) -> int:
        if isinstance(self.hparams.target, int):
            return 1
        elif isinstance(self.hparams.target, list):
            return len(self.hparams.target)
        else:
            return QM9(root=self.data_dir).num_classes

    @property
    def task_type(self) -> str:
        return "regression"


if __name__ == "__main__":
    dm = QM9DataModule(
        target=range(12),  # All 12 regression targets
        batch_size_train=256,
        batch_size_inference=256,
        num_workers=4,
        splits=[0.7, 0.1, 0.15, 0.05],  # Unlabeled, Labeled, Train, Val, Test
        seed=42,
        subset_size=10000,
        data_augmentation=False,
        mode="semisupervised",
        name="qm9",
    )

    dm.setup()
    dl = dm.train_dataloader()
    
    for batch, batch_idx, dataloader_idx in dl:
        print(
            f"""
            Batch idx: {batch_idx}, 
            Dataloader idx: {dataloader_idx}, 
            Labeled batch size: {batch["labeled"].num_graphs} 
            Unlabeled batch size: {batch["unlabeled"].num_graphs}

            Labels in labeled batch: {batch["labeled"].y.squeeze()}
            """
        )
        if batch_idx == 5:
            break

    
    _ = iter(dl)
    print(f"Len dataloader: {len(dl)} batches")
