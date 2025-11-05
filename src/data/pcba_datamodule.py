import logging
import math
from typing import Callable, Optional

import torch
import torch_geometric
from ogb.graphproppred import PygGraphPropPredDataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import IterableDataset
from torch_geometric.loader import DataLoader

from src.utils.utils import get_data_dir

torch.serialization.add_safe_globals(
    [
        torch_geometric.data.data.DataEdgeAttr,
        torch_geometric.data.data.DataTensorAttr,
        torch_geometric.data.storage.GlobalStorage,
    ]
)

SUPPORTED_DATASETS = {
    "ogbg-molpcba",
}

logging.basicConfig(level=logging.INFO)

class OGBGDataModule(LightningDataModule):
    def __init__(
        self,
        name: str = "ogbg-molpcba",
        labeled_batch_size: int = 16,  # SSL labeled batch size
        unlabeled_batch_size: int = 48,  # SSL unlabeled batch size
        num_workers: int = 8,
        labeled_ratio: float = 0.1,  # 0.1, 0.3, 0.5
        ssl_mode: bool = False,
        seed: int = 42,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
    ):
        super().__init__()

        if name not in SUPPORTED_DATASETS:
            raise ValueError(
                f"Dataset '{name}' not supported. "
                f"Available: {', '.join(sorted(SUPPORTED_DATASETS))}"
            )

        self.root = get_data_dir()
        self.name = name

        if 0.0 < labeled_ratio <= 1.0:
            self.labeled_ratio = labeled_ratio
        else:
            raise ValueError("labeled_ratio must be in the range (0.0, 1.0]")

        self.ssl_mode = ssl_mode

        self.labeled_batch_size = labeled_batch_size
        self.unlabeled_batch_size = unlabeled_batch_size
        self.val_test_batch_size = labeled_batch_size + unlabeled_batch_size

        if num_workers < 0:
            raise ValueError("num_workers must be non-negative.")
        else:
            self.num_workers = num_workers

        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        self.seed = seed

        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        self.dataset = PygGraphPropPredDataset(name=self.name, root=self.root)

        split_idx = self.dataset.get_idx_split()

        if stage == "fit" or stage is None:
            full_train_ds = self.dataset[split_idx["train"]]
            labeled_idx, unlabeled_idx = self._split_training_set(full_train_ds)

            self.train_labeled_ds = full_train_ds[labeled_idx]
            self.train_unlabeled_ds = full_train_ds[unlabeled_idx]

            self.valid_dataset = self.dataset[split_idx["valid"]]

        elif stage == "test":
            self.test_dataset = self.dataset[split_idx["test"]]

    def _split_training_set(self, full_train_ds):
        n_train = len(full_train_ds)
        n_labeled = int(n_train * self.labeled_ratio)

        g = torch.Generator().manual_seed(self.seed)
        perm = torch.randperm(n_train, generator=g)

        labeled_indices = perm[:n_labeled]
        unlabeled_indices = perm[n_labeled:]

        return labeled_indices, unlabeled_indices

    def train_dataloader(self) -> dict:
        """
        Returns the training dataloader(s).
        - Baseline: Returns a single DataLoader.
        - SSL Mode: Returns a DICTIONARY of DataLoaders.
        """
        if self.ssl_mode:
            labeled_loader = DataLoader(
                self.train_labeled_ds,
                batch_size=self.labeled_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
            )

            unlabeled_loader = DataLoader(
                self.train_unlabeled_ds,
                batch_size=self.unlabeled_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
            )

            # This is the key: return a dictionary
            return {"labeled": labeled_loader, "unlabeled": unlabeled_loader}

        else:
            # --- Baseline (Supervised) Mode ---
            return DataLoader(
                self.train_labeled_ds,
                batch_size=self.val_test_batch_size, 
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=True if self.num_workers > 0 else False,
            )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.val_test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )


if __name__ == "__main__":
    dm_baseline = OGBGDataModule(
        name="ogbg-molpcba",
        labeled_batch_size=16,  # SSL labeled batch size
        unlabeled_batch_size=48,  # SSL unlabeled batch size
        num_workers=1,
        labeled_ratio=0.1,  # 0.1, 0.3, 0.5
        ssl_mode=True,
    )

    dm_baseline.setup("fit")

    # train_loaders is now a DICTIONARY: {'labeled': <DataLoader>, 'unlabeled': <DataLoader>}
    train_loaders = dm_baseline.train_dataloader()

    # Check the type and keys
    print(f"Type of train_dataloader() output: {type(train_loaders)}")
    print(f"Keys: {train_loaders.keys()}")

    # Check the length of the loaders (this part was correct!)
    # This shows how many batches each loader will produce
    print(f"Labeled Loader Length (batches): {len(train_loaders['labeled'])}")
    print(f"Unlabeled Loader Length (batches): {len(train_loaders['unlabeled'])}")

    # --- THIS IS THE FIX ---
    # You must create an iterator for EACH loader in the dictionary
    labeled_iter = iter(train_loaders["labeled"])
    unlabeled_iter = iter(train_loaders["unlabeled"])

    # Get one batch from each iterator
    batch_l = next(labeled_iter)
    batch_u = next(unlabeled_iter)

    # This 'batch' dictionary is what your training_step will receive
    batch = {"labeled": batch_l, "unlabeled": batch_u}

    print("\n--- Example Batch ---")
    print(f"Labeled Batch: {batch['labeled']}")
    print(f"Labeled Batch size (num_graphs): {batch['labeled'].num_graphs}")
    print(f"Unlabeled Batch: {batch['unlabeled']}")
    print(f"Unlabeled Batch size (num_graphs): {batch['unlabeled'].num_graphs}")
