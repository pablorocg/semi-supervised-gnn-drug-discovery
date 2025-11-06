import numpy as np
import pytorch_lightning as pl
from torch_geometric.datasets import MoleculeNet
from src.utils.dataset_utils import DataLoader, GetTarget, ToLongTensor
from torch_geometric.transforms import Compose
from src.utils.path_utils import get_data_dir
import torch


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
        transforms = Compose([
            GetTarget(self.target),
            ToLongTensor(),
        ])
        # Add a transform to get the target as Long
        dataset = MoleculeNet(
            root=self.data_dir, name=self.name, transform=
        )

        rng = np.random.default_rng(seed=self.seed)

        all_indices = rng.permutation(len(dataset))
        all_labels = torch.tensor([data.y for data in dataset])
        nan_mask = torch.isnan(all_labels).squeeze().tolist()

        valid_indices = [i for i in all_indices if not nan_mask[i]]
        nan_indices = [i for i in all_indices if nan_mask[i]]

        self.data_train_unlabeled = dataset[nan_indices]

        valid_dataset = dataset[valid_indices]

        if self.subset_size is not None:
            valid_dataset = valid_dataset[: self.subset_size]

        valid_splits = np.array(self.splits[1:])
        valid_props = valid_splits / valid_splits.sum()

        split_sizes = [int(len(valid_dataset) * prop) for prop in valid_props]

        split_idx = np.cumsum(split_sizes)

        self.data_train_labeled = valid_dataset[: split_idx[0]]
        self.data_val = valid_dataset[split_idx[0] : split_idx[1]]
        self.data_test = valid_dataset[split_idx[1] :]

        # Set batch sizes. We want the labeled batch size to be the one given by the user, and the unlabeled one to be so that we have the same number of batches
        self.batch_size_train_labeled = self.batch_size_train
        self.batch_size_train_unlabeled = self.batch_size_train
        # self.batch_size_train_unlabeled = int(
        #    self.batch_size_train * len(self.data_train_unlabeled) / len(self.data_train_labeled)
        # )

        print(
            f"{self.name} dataset loaded with {len(self.data_train_labeled)} labeled, {len(self.data_train_unlabeled)} unlabeled, "
            f"{len(self.data_val)} validation, and {len(self.data_test)} test samples."
        )
        print(
            f"Batch sizes: labeled={self.batch_size_train_labeled}, unlabeled={self.batch_size_train_unlabeled}"
        )

    def train_dataloader(self, shuffle=True) -> DataLoader:
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
        return 1  # QM9 is a regression task
    
    @property
    def task_type(self) -> str:
        return "classification"

if __name__ == "__main__":
    dm = MoleculeNetDataModule(num_workers=1)
    dm.prepare_data()
    dm.setup()
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        print(batch)
        break
