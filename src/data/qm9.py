import os
import subprocess

import numpy as np
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import QM9
from torch_geometric.transforms import BaseTransform
from src.utils.dataset_utils import (
    DataLoader,
    ConvertTargetType,
    ConvertFeaturesToFloat,
    GetTarget,
)
from pytorch_lightning.utilities.combined_loader import CombinedLoader

# import Compose for transforming data in torch geometric
from torch_geometric.transforms import Compose
from src.utils.path_utils import get_data_dir
from pathlib import Path


class QM9DataModule(pl.LightningDataModule):
    def __init__(
        self,
        target: int = 0,
        batch_size_train: int = 32,
        batch_size_inference: int = 32,
        num_workers: int = 1,
        splits: list[int] | list[float] = [0.72, 0.08, 0.1, 0.1],
        seed: int = 0,
        subset_size: int | None = None,
        data_augmentation: bool = False,  # Unused but here for compatibility
        mode="semi_supervised", 
        name: str = "qm9",
        ood: bool = False,
    ) -> None:
        super().__init__()
        self.target = target

        self.data_dir = Path(get_data_dir()) / "QM9"

        self.batch_size_train = batch_size_train
        self.batch_size_inference = batch_size_inference
        self.num_workers = num_workers
        self.splits = splits
        self.seed = seed
        self.subset_size = subset_size
        self.data_augmentaion = data_augmentation
        self.name = name
        self.ood = ood
        self.mode = mode

        self.data_train_unlabeled = None
        self.data_train_labeled = None
        self.data_val = None
        self.data_test = None
        self.ood_datasets = None

        self.batch_size_train_labeled = None
        self.batch_size_train_unlabeled = None

        

    def prepare_data(self) -> None:
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
        # Download data
        QM9(root=self.data_dir)

    def setup(self, stage: str | None = None) -> None:
        dataset = QM9(
            root=self.data_dir,
            transform=GetTarget(self.target),
            # Compose(
            #     [
            #         ConvertTargetType(target=self.target, dtype=torch.float),
            #         ConvertFeaturesToFloat(),
            #     ]
            # ),
        )

        # Shuffle dataset
        rng = np.random.default_rng(seed=self.seed)
        dataset = dataset[rng.permutation(len(dataset))]

        # Subset dataset
        if self.subset_size is not None:
            dataset = dataset[: self.subset_size]

        # Split dataset
        if all([type(split) == int for split in self.splits]):
            split_sizes = self.splits
        elif all([type(split) == float for split in self.splits]):
            split_sizes = [int(len(dataset) * prop) for prop in self.splits]

        split_idx = np.cumsum(split_sizes)

        self.data_train_unlabeled = dataset[: split_idx[0]]
        self.data_train_labeled = dataset[split_idx[0] : split_idx[1]]
        self.data_val = dataset[split_idx[1] : split_idx[2]]
        self.data_test = dataset[split_idx[2] :]

        # Set batch sizes. We want the labeled batch size to be the one given by the user, 
        # and the unlabeled one to be so that we have the same number of batches
        self.batch_size_train_labeled = self.batch_size_train
        self.batch_size_train_unlabeled = self.batch_size_train


        print(
            f"QM9 dataset loaded with {len(self.data_train_labeled)} labeled, {len(self.data_train_unlabeled)} unlabeled, "
            f"{len(self.data_val)} validation, and {len(self.data_test)} test samples."
        )
        print(
            f"Batch sizes: labeled={self.batch_size_train_labeled}, unlabeled={self.batch_size_train_unlabeled}"
        )

    def train_dataloader(self) -> CombinedLoader:

        if self.mode == "supervised":
            return CombinedLoader(
                {
                    "labeled": self.supervised_train_dataloader(),
                },
                mode="max_size_cycle",
            )
        
        elif self.mode == "semi_supervised":
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
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=True,
        )

    def unsupervised_train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.data_train_unlabeled,
            batch_size=self.batch_size_train_unlabeled,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size_inference,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True,
        )

    @property
    def num_features(self) -> int:
        return self.data_train_labeled.num_node_features

    @property
    def num_classes(self) -> int:
        return 1  # QM9 is a regression task

    @property
    def task_type(self) -> str:
        return "regression"


if __name__ == "__main__":
    dm = QM9DataModule(subset_size=1000)

    
    dm.setup()

    print("Preparing data...")
    print(dm.num_features)
    print(dm.num_classes)

    train_loader = dm.train_dataloader()

    for batch, batch_idx, dataloader_idx in train_loader:
        print(f"{batch}, {batch_idx=}, {dataloader_idx=}")
        # {
        #     'labeled': DataBatch(x=[547, 11], edge_index=[2, 1142], edge_attr=[1142, 4], y=[32, 1], pos=[547, 3], z=[547], smiles=[32], name=[32], idx=[32], batch=[547], ptr=[33]), 
        #     'unlabeled': DataBatch(x=[580, 11], edge_index=[2, 1196], edge_attr=[1196, 4], y=[32, 1], pos=[580, 3], z=[580], smiles=[32], name=[32], idx=[32], batch=[580], ptr=[33])
        # }, 
        # batch_idx=0, 
        # dataloader_idx=0

        # Show features of first node in labeled batch and its type
        print(batch['labeled'].x[0], batch['labeled'].x[0].dtype)
        # Show target of first graph in labeled batch
        print(batch['labeled'].y[0], batch['labeled'].y[0].dtype)

        break
