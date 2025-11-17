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
        target: int = 0,
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
<<<<<<< HEAD
        
=======
>>>>>>> main
        with torch.serialization.safe_globals(PYG_SAFE_GLOBALS):
            self.dataset = PygGraphPropPredDataset(
                name=self.dataset_name,
                root=self.data_dir,
                transform=GetTarget(self.hparams.target),
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
<<<<<<< HEAD
        
        labeled_indices = train_idx[:num_labeled]
        self.data_train_labeled = self.dataset[labeled_indices]

        if self.hparams.mode == "semisupervised":
            unlabeled_indices = train_idx[num_labeled:]
            self.data_train_unlabeled = self.dataset[unlabeled_indices]


=======
        labeled_indices = train_idx[:num_labeled]
        unlabeled_indices = train_idx[num_labeled:]

        self.data_train_labeled = self.dataset[labeled_indices]
        self.data_train_unlabeled = self.dataset[unlabeled_indices]
>>>>>>> main
        self.data_val = self.dataset[val_idx]
        self.data_test = self.dataset[test_idx]

        self.batch_size_train_labeled = self.hparams.batch_size_train
        self.batch_size_train_unlabeled = self.hparams.batch_size_train

<<<<<<< HEAD
        self.print_dataset_info()


    def print_dataset_info(self) -> None:
        if self.hparams.mode == "supervised":
            print("Using supervised mode.")
            print(
                f"OGB {self.dataset_name} dataset loaded with {len(self.data_train_labeled)} labeled,"
                f"{len(self.data_val)} validation, and {len(self.data_test)} test samples."
            )
            print(
                f"Batch sizes: labeled={self.batch_size_train_labeled}"
            )
        elif self.hparams.mode == "semisupervised":
            print("Using semi-supervised mode.")
            print(
                f"OGB {self.dataset_name} dataset loaded with {len(self.data_train_labeled)} labeled, {len(self.data_train_unlabeled)} unlabeled, "
                f"{len(self.data_val)} validation, and {len(self.data_test)} test samples."
            )
            print(
                f"Batch sizes: labeled={self.batch_size_train_labeled}, unlabeled={self.batch_size_train_unlabeled}"
            )
=======
        print(
            f"OGB {self.dataset_name} dataset loaded with {len(self.data_train_labeled)} labeled, {len(self.data_train_unlabeled)} unlabeled, "
            f"{len(self.data_val)} validation, and {len(self.data_test)} test samples."
        )
        print(
            f"Batch sizes: labeled={self.batch_size_train_labeled}, unlabeled={self.batch_size_train_unlabeled}"
        )
>>>>>>> main

    def train_dataloader(self) -> CombinedLoader | DataLoader:
        if self.hparams.mode == "supervised":
            return CombinedLoader(
                {
<<<<<<< HEAD
                    "labeled": self.supervised_train_dataloader(), # 8% train data
=======
                    "labeled": self.supervised_train_dataloader(),
>>>>>>> main
                },
                mode="max_size_cycle",
            )
        elif self.hparams.mode == "semisupervised":
            return CombinedLoader(
                {
<<<<<<< HEAD
                    "labeled": self.supervised_train_dataloader(), # 8% train data
                    "unlabeled": self.unsupervised_train_dataloader(), # 92% train data
=======
                    "labeled": self.supervised_train_dataloader(),
                    "unlabeled": self.unsupervised_train_dataloader(),
>>>>>>> main
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

<<<<<<< HEAD
    def compute_class_weights(self) -> torch.Tensor | None:
        # 128 tasks with binary classification (0/1/nan)
        y = self.data_train_labeled.y  # shape: [num_samples, num_tasks]
        num_tasks = y.size(1)
        class_weights = []
        for task_idx in range(num_tasks):
            task_labels = y[:, task_idx]
            mask = ~torch.isnan(task_labels)
            labels = task_labels[mask]
            num_positive = (labels == 1).sum().item()
            num_negative = (labels == 0).sum().item()
            if num_positive == 0 or num_negative == 0:
                weight = 1.0
            else:
                weight = num_negative / num_positive
            class_weights.append(weight)
        return torch.tensor(class_weights, dtype=torch.float) # shape: [num_tasks]
    
    @property
    def class_weights(self) -> torch.Tensor | None:
        return self.compute_class_weights()
    
=======
>>>>>>> main
    @property
    def num_features(self) -> int:
        return self.dataset.num_node_features

    @property
    def num_tasks(self) -> int:
        if isinstance(self.hparams.target, int):
            return 1
        elif isinstance(self.hparams.target, list):
            return len(self.hparams.target)
        else:
            return self.dataset.num_tasks

    @property
    def task_type(self) -> str:
        return "classification"


if __name__ == "__main__":
    dm = OgbgMolPcbaDataModule(
        batch_size_train=2,
        batch_size_inference=256,
        num_workers=4,
        splits=[0.72, 0.08, 0.1, 0.1],
        seed=42,
<<<<<<< HEAD
         
        mode="supervised",
=======
        subset_size=10000,  
        mode="semisupervised",
>>>>>>> main
    )

    dm.setup()

    dl = dm.train_dataloader()

<<<<<<< HEAD
    class_weights = dm.class_weights
    print(f"Class weights: {class_weights}")

=======
>>>>>>> main
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

<<<<<<< HEAD
        
=======
        # Mostrar el contenido de cada uno de los tensores de características de nodos, índices de bordes y atributos de bordes
>>>>>>> main
        print(f"Node features:\n{batch['labeled'].x}")
        print(f"Edge indices:\n{batch['labeled'].edge_index}")
        print(f"Edge attributes:\n{batch['labeled'].edge_attr}")
        print(f"Labels:\n{batch['labeled'].y}")
        if batch_idx == 5:
            break

    _ = iter(dl)
    print(f"Len dataloader: {len(dl)} batches")
