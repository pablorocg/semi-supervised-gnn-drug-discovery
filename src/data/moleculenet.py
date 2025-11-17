import numpy as np
import pytorch_lightning as pl
from torch_geometric.datasets import MoleculeNet
from src.utils.dataset_utils import (
    DataLoader,
    ConvertTargetType,
    ConvertFeaturesToFloat,
)
from torch_geometric.transforms import Compose
from src.utils.path_utils import get_data_dir
import torch
from pytorch_lightning.utilities.combined_loader import CombinedLoader




class MoleculeNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        target: int = 0,
        batch_size_train: int = 32,
        batch_size_inference: int = 32,
        num_workers: int = 0,
        splits: list[int] | list[float] = [0.72, 0.08, 0.1, 0.1],
        seed: int = 0,
        subset_size: int | None = None,
        data_augmentation: bool = False,  # Unused but here for compatibility
        name: str = "PCBA",
        ood: bool = False,
    ) -> None:
        super().__init__()
        self.target = target
        self.data_dir = get_data_dir()
        self.batch_size_train = batch_size_train
        self.batch_size_inference = batch_size_inference
        self.num_workers = num_workers
        self.splits = splits
        self.seed = seed
        self.subset_size = subset_size
        self.data_augmentaion = data_augmentation
        self.name = name
        self.ood = ood

        self.data_train_unlabeled = None
        self.data_train_labeled = None
        self.data_val = None
        self.data_test = None
        self.ood_datasets = None

        self.batch_size_train_labeled = None
        self.batch_size_train_unlabeled = None

        self.setup()  # Call setup to initialize the datasets

    def prepare_data(self) -> None:
        # Download data
        MoleculeNet(root=self.data_dir, name=self.name)

    def setup(self, stage: str | None = None) -> None:
        # 1. Load the raw dataset *without* any transforms
        dataset = MoleculeNet(
            root=self.data_dir,
            name=self.name,
            transform=None,  # Load raw data first
        )

        # Shuffle dataset
        rng = np.random.default_rng(seed=self.hparams.seed)
        dataset = dataset[rng.permutation(len(dataset))]
        
        # Subset dataset
        if self.hparams.subset_size is not None:
            dataset = dataset[: self.hparams.subset_size]

        # Split dataset
        if all([isinstance(split, int) for split in self.hparams.splits]):
            split_sizes = self.hparams.splits
        elif all([isinstance(split, float) for split in self.hparams.splits]):
            split_sizes = [int(len(dataset) * prop) for prop in self.hparams.splits]

        split_idx = np.cumsum(split_sizes)

        self.data_train_labeled = dataset[split_idx[0] : split_idx[1]]

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
                f"Moleculenet {self.hparams.name} dataset loaded with {len(self.data_train_labeled)} labeled, "
                f"{len(self.data_val)} validation, and {len(self.data_test)} test samples."
            )
            print(
                f"Batch sizes: labeled={self.batch_size_train_labeled}"
            )
        elif self.hparams.mode == "semisupervised":
            print("Using semi-supervised mode.")
            print(
                f"Moleculenet {self.hparams.name} dataset loaded with {len(self.data_train_labeled)} labeled, "
                f"{len(self.data_train_unlabeled)} unlabeled, {len(self.data_val)} validation, and {len(self.data_test)} test samples."
            )
            print(
                f"Batch sizes: labeled={self.batch_size_train_labeled}, unlabeled={self.batch_size_train_unlabeled}"
            )

    def compute_class_stats(self) -> dict[str, float]:
        labels = torch.tensor([
            data.y.item() for data in self.data_train_labeled
        ])
        num_positive = (labels == 1).sum().item()
        num_negative = (labels == 0).sum().item()
        total = len(labels)
        pos_ratio = num_positive / total
        neg_ratio = num_negative / total
        return {
            "num_positive": num_positive,
            "num_negative": num_negative,
            "pos_ratio": pos_ratio,
            "neg_ratio": neg_ratio,
        }
    

    def train_dataloader(self) -> CombinedLoader | DataLoader:
        
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

    def ood_dataloaders(self) -> dict[str, DataLoader]:
        return {
            dataset_name: DataLoader(
                dataset,
                batch_size=self.batch_size_inference,
                num_workers=self.num_workers,
                shuffle=False,
                pin_memory=True,
                persistent_workers=True,
            )
            for dataset_name, dataset in self.ood_datasets.items()
        }

    def ood_dataloader(self) -> list[DataLoader]:
        """Returns a list of DataLoader for each OOD dataset."""
        if self.ood_datasets is None:
            return [], []
        else:
            ood_dataloaders = []
            ood_names = []
            # for dm in self.ood_datasets:
            for dataset_name, dataset in self.ood_datasets.items():
                val_dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size_inference,
                    num_workers=self.num_workers,
                    shuffle=False,
                    pin_memory=True,
                    persistent_workers=True,
                )
                ood_dataloaders.append(val_dataloader)
                ood_names.append(dataset_name)
            return ood_names, ood_dataloaders

    @property
    def num_features(self) -> int:
        return self.data_train_labeled.num_node_features

    @property
    def num_classes(self) -> int:
        return 1  # MoleculeNet is a binary classification task

    @property
    def task_type(self) -> str:
        return "classification"


if __name__ == "__main__":
    dm = MoleculeNetDataModule(num_workers=1)
    dm.prepare_data()
    dm.setup()
    # train_loader = dm.train_dataloader()
    # for batch in train_loader:
    #     print(batch)
    #     break

    for target in range(128):
        dm = MoleculeNetDataModule(target=target, num_workers=8)
        dm.prepare_data()
        dm.setup()
        class_stats = dm.compute_class_stats()
        print(f"Target {target}: {class_stats}")
