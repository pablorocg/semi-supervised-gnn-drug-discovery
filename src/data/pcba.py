import numpy as np
import pytorch_lightning as pl
import torch
import torch.serialization
from ogb.graphproppred import PygGraphPropPredDataset
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.loader import DataLoader

from src.utils.dataset_utils import (
    GetTarget,
)
from src.utils.path_utils import get_data_dir

PYG_SAFE_GLOBALS = [Data, DataEdgeAttr, DataTensorAttr, GlobalStorage]


class OgbgMolPcbaDataModule(pl.LightningDataModule):
    def __init__(
        self,
        target: int | list[int] | None = None,
        batch_size_train: int = 32,
        batch_size_inference: int = 32,
        num_workers: int = 0,
        splits: list[float] = [0.72, 0.08, 0.1, 0.1],  # Train set (Unlabeled, Labeled), Val, Test
        seed: int = 0,
        subset_size: int | None = None, 
        data_augmentation: bool = False,  
        mode: str = "semisupervised",
        name: str = "ogbg-molpcba",
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.data_dir = get_data_dir()
        self.dataset_name = "ogbg-molpcba"

    def prepare_data(self) -> None:
        with torch.serialization.safe_globals(PYG_SAFE_GLOBALS):
            PygGraphPropPredDataset(name=self.dataset_name, root=self.data_dir)

    def setup(self, stage: str | None = None) -> None:
        with torch.serialization.safe_globals(PYG_SAFE_GLOBALS):
            
            # Only apply transform if target is specified
            transform = GetTarget(self.hparams.target) if self.hparams.target is not None else None

            self.dataset = PygGraphPropPredDataset(
                name=self.dataset_name,
                root=self.data_dir,
                transform=transform,
            )

        split_idx = self.dataset.get_idx_split()
        train_idx = split_idx["train"]
        val_idx = split_idx["valid"]
        test_idx = split_idx["test"]

        rng = np.random.default_rng(seed=self.hparams.seed)

        if self.hparams.subset_size is not None:
            train_idx = train_idx[: self.hparams.subset_size]

        train_idx = train_idx[rng.permutation(len(train_idx))]

        num_labeled = int(len(train_idx) * self.hparams.splits[1])
        labeled_indices = train_idx[:num_labeled]
        unlabeled_indices = train_idx[num_labeled:]

        self.data_train_labeled = self.dataset[labeled_indices]
        self.data_train_unlabeled = self.dataset[unlabeled_indices]
        self.data_val = self.dataset[val_idx]
        self.data_test = self.dataset[test_idx]

        self.batch_size_train_labeled = self.hparams.batch_size_train
        self.batch_size_train_unlabeled = self.hparams.batch_size_train

        print(
            f"OGB {self.dataset_name} dataset loaded with {len(self.data_train_labeled)} labeled, {len(self.data_train_unlabeled)} unlabeled, "
            f"{len(self.data_val)} validation, and {len(self.data_test)} test samples."
        )
        print(
            f"Batch sizes: labeled={self.batch_size_train_labeled}, unlabeled={self.batch_size_train_unlabeled}"
        )

    def train_dataloader(self) -> CombinedLoader | DataLoader:
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
            persistent_workers=self.hparams.num_workers > 0,
        )

    def unsupervised_train_dataloader(self, shuffle=True) -> DataLoader:
        return DataLoader(
            self.data_train_unlabeled,
            batch_size=self.batch_size_train_unlabeled,
            num_workers=self.hparams.num_workers,
            shuffle=shuffle,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size_inference,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size_inference,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    @property
    def num_features(self) -> int:
        return self.dataset.num_node_features

    @property
    def num_tasks(self) -> int:
        if self.hparams.target is None:
            return self.dataset.num_tasks  # 128 for PCBA
        elif isinstance(self.hparams.target, int):
            return 1
        elif isinstance(self.hparams.target, list):
            return len(self.hparams.target)


    @property
    def task_type(self) -> str:
        return "classification"
    
    @property
    def train_idx(self):
        # Return indices of labeled training data
        return range(len(self.data_train_labeled))


if __name__ == "__main__":
    dm = OgbgMolPcbaDataModule(
        batch_size_train=2,
        batch_size_inference=256,
        num_workers=4,
        splits=[0.72, 0.08, 0.1, 0.1],
        seed=42,
        subset_size=10000,  
        mode="semisupervised",
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

            Node features shape: {batch["labeled"].x.shape}
            Edge index shape: {batch["labeled"].edge_index.shape}
            Edge attributes shape: {batch["labeled"].edge_attr.shape}
            """
        )

        # Mostrar el contenido de cada uno de los tensores de características de nodos, índices de bordes y atributos de bordes
        print(f"Node features:\n{batch['labeled'].x}")
        print(f"Edge indices:\n{batch['labeled'].edge_index}")
        print(f"Edge attributes:\n{batch['labeled'].edge_attr}")
        print(f"Labels:\n{batch['labeled'].y}")
        if batch_idx == 5:
            break

    _ = iter(dl)
    print(f"Len dataloader: {len(dl)} batches")
