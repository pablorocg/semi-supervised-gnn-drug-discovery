from typing import Callable, Optional

import torch
import torch_geometric
from pytorch_lightning import LightningDataModule

from src.utils.utils import get_data_dir

torch.serialization.add_safe_globals(
    [
        torch_geometric.data.data.DataEdgeAttr,
        torch_geometric.data.data.DataTensorAttr,
        torch_geometric.data.storage.GlobalStorage,
    ]
)


from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.loader import DataLoader

SUPPORTED_DATASETS = {
    "ogbg-molpcba",
}


class OgbgMolpcbaDataModule(LightningDataModule):
    """ """

    def __init__(
        self,
        name: str = "ogbg-molpcba",
        batch_size: int = 32,
        num_workers: int = 8,
        unlabeled_ratio: float = 0.1,
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
        self.unlabeled_ratio = unlabeled_ratio
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter
        self.num_tasks = None
        self.node_attrs_dim = None
        self.edge_attrs_dim = None

    def setup(self, stage: Optional[str] = None):
        """
        Initial setup - loads dataset and metadata.
        To configure a fold, use setup_fold(fold_idx).
        """
        self.dataset = PygGraphPropPredDataset(name=self.name, root=self.root)

        # 128 binary classification tasks
        self.num_tasks = self.dataset.num_tasks
        self.node_attrs_dim = self.dataset.num_node_features
        self.edge_attrs_dim = self.dataset.num_edge_features

        split_idx = self.dataset.get_idx_split()

        if stage == "fit" or stage is None:
            train_dataset = self.dataset[split_idx["train"]]

            # Split train dataset into labeled and unlabeled datasets (Get unlabeled ratio )
            self.train_dataset = train_dataset[
                : int(len(train_dataset) * (1 - self.unlabeled_ratio))
            ]

            self.valid_dataset = self.dataset[split_idx["valid"]]
        elif stage == "test":
            self.test_dataset = self.dataset[split_idx["test"]]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    @property
    def num_node_features(self) -> int:
        if self.dataset.num_node_features is None:
            raise RuntimeError("Run setup() first")
        return self.dataset.num_node_features

    @property
    def num_edge_features(self) -> int:
        if self.dataset.num_edge_features is None:
            raise RuntimeError("Run setup() first")
        return self.dataset.num_edge_features

    @property
    def num_classes(self) -> int:
        if self.dataset.num_tasks is None:
            raise RuntimeError("Run setup() first")
        return self.dataset.num_tasks

if __name__ == "__main__":
    dm = OgbgMolpcbaDataModule()
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        print(batch)
        break