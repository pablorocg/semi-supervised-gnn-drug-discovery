import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from torch_geometric.datasets import MoleculeNet
import torch 
from src.utils.dataset_utils import (
    DataLoader,
    GetTarget,
)
from src.utils.path_utils import get_data_dir


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
        mode: str = "semisupervised",
    ) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.data_dir = get_data_dir()

    def prepare_data(self) -> None:
        # Download data
        MoleculeNet(root=self.data_dir, name=self.hparams.name)

    def setup(self, stage: str | None = None) -> None:
        dataset = MoleculeNet(
            root=self.data_dir,
            name=self.hparams.name,
            transform=GetTarget(self.hparams.target),
        )

        # Shuffle dataset
        rng = np.random.default_rng(seed=self.hparams.seed)
        self.dataset = dataset[rng.permutation(len(dataset))]
        
        # Subset dataset
        if self.hparams.subset_size is not None:
            self.dataset = self.dataset[: self.hparams.subset_size]

        # Split dataset
        if all([isinstance(split, int) for split in self.hparams.splits]):
            split_sizes = self.hparams.splits
        elif all([isinstance(split, float) for split in self.hparams.splits]):
            split_sizes = [int(len(dataset) * prop) for prop in self.hparams.splits]

        split_idx = np.cumsum(split_sizes)

        self.data_train_labeled = self.dataset[split_idx[0] : split_idx[1]]

        if self.hparams.mode == "semisupervised":
            self.data_train_unlabeled = self.dataset[: split_idx[0]]
        
        self.data_val = self.dataset[split_idx[1] : split_idx[2]]
        self.data_test = self.dataset[split_idx[2] :]

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


    def get_pos_weights(self):
    # dataset.y is usually size [num_graphs, n_tasks]
    # Convert to float and handle NaNs if necessary for counting
        labels = self.dataset.y.float()
        
        weights = []
        for i in range(labels.shape[1]):
            # Get valid labels for task i (not NaN)
            valid = labels[:, i][~torch.isnan(labels[:, i])]
            
            n_pos = valid.sum().item()
            n_neg = len(valid) - n_pos
            
            # Avoid division by zero if a task has no positives in a split (rare but possible)
            if n_pos == 0:
                weight = 1.0
            else:
                weight = n_neg / n_pos
                
            weights.append(weight)
        
        return torch.tensor(weights)

    @property
    def num_node_features(self) -> int:
        print("Num node features:", self.data_train_labeled.num_node_features)
        return self.data_train_labeled.num_node_features
    
    @property
    def num_edge_features(self) -> int:
        print("Num edge features:", self.data_train_labeled.num_edge_features)
        return self.data_train_labeled.num_edge_features

    @property
    def num_tasks(self) -> int:
        if isinstance(self.hparams.target, list):
            return len(self.hparams.target)
        elif isinstance(self.hparams.target, int):
            return 1
        else:
            return MoleculeNet(root=self.data_dir, name=self.hparams.name).num_classes

    @property
    def task_type(self) -> str:
        return "classification"


if __name__ == "__main__":
    dm = MoleculeNetDataModule(
        target=None,  # All 12 regression targets
        batch_size_train=2,
        batch_size_inference=256,
        num_workers=4,
        splits=[0.7, 0.08, 0.15, 0.05],  # Unlabeled, Labeled, Train, Val, Test
        seed=42,
        data_augmentation=False,
        mode="supervised",
        name="tox21",
    )

    dm.setup()
    
    
    # Compute weights 
    print("Pos weights:", dm.get_pos_weights())

    print(f"Number of node features: {dm.num_node_features}")
    print(f"Number of edge features: {dm.num_edge_features}")
    print(f"Number of tasks: {dm.num_tasks}")

    dl = dm.train_dataloader()

    for batch, batch_idx, dataloader_idx in dl:
        print(
            f"""
            Batch idx: {batch_idx}, 
            Dataloader idx: {dataloader_idx}, 
            Labeled batch size: {batch["labeled"].num_graphs} 
            

            Labels in labeled batch: {batch["labeled"].y.squeeze()}

            Node features shape: {batch["labeled"].x.shape}
            Edge index shape: {batch["labeled"].edge_index.shape}
            Edge attributes shape: {batch["labeled"].edge_attr.shape}
            """
        )

        # Mostrar el contenido de cada uno de los tensores de características de nodos, índices de bordes y atributos de bordes
        # Unlabeled batch size: {batch["unlabeled"].num_graphs}
        print(f"Node features:\n{batch['labeled'].x}")
        print(f"Edge indices:\n{batch['labeled'].edge_index}")
        print(f"Edge attributes:\n{batch['labeled'].edge_attr}")
        print(f"Labels:\n{batch['labeled'].y}")
        if batch_idx == 5:
            break

    _ = iter(dl)
    print(f"Len dataloader: {len(dl)} batches")