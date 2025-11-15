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

        rng = np.random.default_rng(seed=self.seed)
        all_indices = rng.permutation(len(dataset))

        # 2. Build the nan_mask from the raw data.y, checking the specific target
        # This is slow, but necessary for this splitting logic
        all_labels_raw = torch.tensor([dataset[i].y[0, self.target] for i in all_indices])
        nan_mask = torch.isnan(all_labels_raw).squeeze().tolist()

        # 3. Split indices based on the *raw* NaN values
        valid_indices = [i for i, is_nan in zip(all_indices, nan_mask) if not is_nan]
        nan_indices = [i for i, is_nan in zip(all_indices, nan_mask) if is_nan]

        # 4. Create subsets from the raw dataset
        self.data_train_unlabeled = dataset[nan_indices]
        valid_dataset = dataset[valid_indices]

        if self.subset_size is not None:
            # Note: This subsets the valid data, not the whole dataset
            valid_dataset = valid_dataset[: self.subset_size]

        # --- Your original splitting logic for valid_dataset ---
        # (This splits the `valid_dataset` into train/val/test)
        valid_splits = np.array(self.splits[1:])  # Assumes splits[0] is not used here
        valid_props = valid_splits / valid_splits.sum()

        split_sizes = [int(len(valid_dataset) * prop) for prop in valid_props]
        # Ensure splits cover the whole dataset
        split_sizes[-1] = len(valid_dataset) - sum(split_sizes[:-1])
        
        split_idx = np.cumsum(split_sizes)

        self.data_train_labeled = valid_dataset[: split_idx[0]]
        self.data_val = valid_dataset[split_idx[0] : split_idx[1]]
        self.data_test = valid_dataset[split_idx[1] :]
        # --- End of splitting logic ---

        # 5. Now, assign the correct transforms to the *final* datasets
        
        # Unlabeled data only needs features converted
        self.data_train_unlabeled.transform = ConvertFeaturesToFloat()

        # Labeled, validation, and test data need both transforms
        labeled_transform = Compose(
            [
                ConvertTargetType(target=self.target, dtype=torch.long),
                ConvertFeaturesToFloat(),
            ]
        )
        
        self.data_train_labeled.transform = labeled_transform
        self.data_val.transform = labeled_transform
        self.data_test.transform = labeled_transform

        # --- Set batch sizes and print info ---
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
        return 1  # MoleculeNet is a binary classification task

    @property
    def task_type(self) -> str:
        return "classification"


if __name__ == "__main__":
    dm = MoleculeNetDataModule(num_workers=8)
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
