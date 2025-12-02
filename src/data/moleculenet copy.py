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
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

class MoleculeNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        target: int = 0,
        batch_size_train: int = 32,
        batch_size_inference: int = 32,
        num_workers: int = 0,
        splits: list[int] | list[float] = [0.72, 0.08, 0.1, 0.1],
        seed: int = 0,
        mu = 5,
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
        print(f"Loading {self.hparams.name} dataset...")
        dataset = MoleculeNet(
            root=self.data_dir,
            name=self.hparams.name,
            transform=GetTarget(self.hparams.target),
        )

        # 1. Parse Split Ratios
        # Expected order from your list: [Unlabeled, Labeled, Val, Test]
        if all([isinstance(split, float) for split in self.hparams.splits]):
            ratios = self.hparams.splits
        else:
            # Convert ints to ratios
            total = len(dataset)
            ratios = [x / total for x in self.hparams.splits]

        # 2. Get Stratified Indices
        # We pass the full label matrix Y. 
        # CAUTION: MoleculeNet has NaNs. We treat NaNs as negative (0) for stratification 
        # purposes, ensuring we stratify based on KNOWN positives.
        y_all = dataset.y.cpu().numpy()
        y_all = np.nan_to_num(y_all, nan=0.0)

        # Perform the chained split
        print(f"Calculating stratified splits for {len(dataset)} samples...")
        indices = self._get_stratified_indices(y_all, ratios, seed=self.hparams.seed)

        # 3. Assign Subsets
        self.data_train_unlabeled = dataset[indices["unlabeled"]]
        self.data_train_labeled   = dataset[indices["labeled"]]
        self.data_val             = dataset[indices["val"]]
        self.data_test            = dataset[indices["test"]]

        # 4. Critical Safety Check for Exp 4
        self._check_label_integrity()
        
        # 5. Apply Mu Ratio & Print Info
        self.batch_size_train_labeled = self.hparams.batch_size_train
        self.batch_size_train_unlabeled = self.hparams.batch_size_train * self.hparams.mu
        
        self.print_dataset_info()

    def _get_stratified_indices(self, y, ratios, seed):
        """
        Performs chained iterative stratification.
        Ratios order assumed: [Unlabeled, Labeled, Val, Test]
        """
        # Map ratios to names for clarity
        r_unlabeled, r_labeled, r_val, r_test = ratios
        
        # We work with an array of indices [0, 1, ... N]
        all_indices = np.arange(len(y))
        
        # --- SPLIT 1: Separate TEST from REST ---
        # Test size relative to total
        test_split = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=r_test, random_state=seed
        )
        # split returns indices relative to the input array
        rest_idx_local, test_idx_local = next(test_split.split(all_indices, y))
        
        # Map back to global indices
        idx_test = all_indices[test_idx_local]
        idx_rest = all_indices[rest_idx_local]
        y_rest = y[idx_rest]

        # --- SPLIT 2: Separate VAL from TRAIN_TOTAL ---
        # Val size relative to the REST (Total - Test)
        # Math: 0.1 / (1.0 - 0.2) = 0.1 / 0.8 = 0.125
        val_rel_size = r_val / (1.0 - r_test)
        
        val_split = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=val_rel_size, random_state=seed
        )
        train_total_idx_local, val_idx_local = next(val_split.split(idx_rest, y_rest))
        
        idx_val = idx_rest[val_idx_local]
        idx_train_total = idx_rest[train_total_idx_local]
        y_train_total = y[idx_train_total]

        # --- SPLIT 3: Separate LABELED from UNLABELED ---
        # Labeled size relative to TRAIN_TOTAL
        # Math for Exp 4: 0.03 / (0.03 + 0.67) = 0.03 / 0.70 = ~0.0428
        labeled_rel_size = r_labeled / (r_labeled + r_unlabeled)
        
        # Safety clamp for float precision issues
        if labeled_rel_size <= 0: labeled_rel_size = 0.0
        if labeled_rel_size >= 1: labeled_rel_size = 1.0

        label_split = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=labeled_rel_size, random_state=seed
        )
        unlabeled_idx_local, labeled_idx_local = next(label_split.split(idx_train_total, y_train_total))
        
        idx_labeled = idx_train_total[labeled_idx_local]
        idx_unlabeled = idx_train_total[unlabeled_idx_local]

        return {
            "unlabeled": idx_unlabeled,
            "labeled": idx_labeled,
            "val": idx_val,
            "test": idx_test
        }

    def _check_label_integrity(self):
        """Verifies that the labeled set is not degenerate."""
        y_labeled = self.data_train_labeled.y.float()
        # Treat NaNs as 0 for counting
        y_labeled = torch.nan_to_num(y_labeled, nan=0.0)
        
        pos_counts = y_labeled.sum(dim=0)
        dead_tasks = (pos_counts == 0).sum().item()
        
        if dead_tasks > 0:
            print(f"\n[WARNING] Stratification Warning: {dead_tasks}/{y_labeled.shape[1]} tasks have ZERO positive samples in the labeled set.")
            print(f"Labeled Set Size: {len(self.data_train_labeled)}")
            print("This is expected for very rare tasks in Exp 4, but limits supervised signal.\n")


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
    
        labels = self.data_train_labeled.y.float()
        
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