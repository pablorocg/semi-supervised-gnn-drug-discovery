import logging
from dataclasses import dataclass
from typing import Callable, Optional

from pytorch_lightning import LightningDataModule
from torch.utils.data import Subset
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

from src.data.split_manager import SplitManager
from src.utils.utils import get_data_dir, get_splits_dir

log = logging.getLogger(__name__)

SUPPORTED_DATASETS = {"Tox21", "ToxCast", "SIDER", "ClinTox", "BBBP"}


@dataclass
class DatasetInfo:
    """
    Basic dataset information loaded once
    """

    num_node_features: int
    num_edge_features: int
    num_classes: int
    length: int

    def __str__(self) -> str:
        return (
            f"Samples: {self.length} | "
            f"Node features: {self.num_node_features} | "
            f"Classes: {self.num_classes}"
        )


class MoleculeNetDataModule(LightningDataModule):
    """
    DataModule for MoleculeNet with rigorous and efficient protocol.

    Protocol: Fixed test (10%) + 5-fold CV (90%)
    - Only requires 5 training runs
    - Allows hyperparameter optimization without bias
    - Final evaluation on independent test set

    Typical usage:

    # 1. Prepare data
    dm = MoleculeNetDataModule(name="BBBP", k_folds=5, seed=42)
    dm.prepare_data()

    # 2. Optimize hyperparameters with CV (5 training runs)
    cv_results = []
    for fold in range(dm.k_folds):
        dm.setup_fold(fold)
        model = YourModel(...)
        trainer.fit(model, dm)
        cv_results.append(trainer.validate(model, dm))

    print(f"CV: {np.mean(cv_results):.4f} ± {np.std(cv_results):.4f}")

    # 3. Final evaluation on test (1 optional training)
    dm.setup_fold(0)  # Any fold, test is the same
    final_model = YourModel(best_hyperparameters)
    trainer.fit(final_model, dm)
    test_result = trainer.test(final_model, dm)

    print(f"Test: {test_result:.4f}")
    """

    def __init__(
        self,
        name: str = "BBBP",
        batch_size: int = 32,
        num_workers: int = 8,
        k_folds: int = 5,
        test_ratio: float = 0.1,
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

        self.name = name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.k_folds = k_folds
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        self.split_manager = SplitManager(
            dataset_name=name,
            splits_dir=get_splits_dir(),
            k_folds=k_folds,
            test_ratio=test_ratio,
            seed=seed,
        )

        self.root = get_data_dir()
        self.dataset: Optional[MoleculeNet] = None
        self.dataset_info: Optional[DatasetInfo] = None

        # Current state
        self.current_fold: Optional[int] = None
        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None
        self.test_dataset: Optional[Subset] = None

    def prepare_data(self):
        """Download dataset and create splits (only on main process)"""
        log.info(f"Preparing dataset {self.name}...")

        dataset = self._load_base_dataset()

        if self.split_manager.splits_exist():
            log.info("✓ Splits found")
        else:
            log.warning("Splits not found, creating new ones...")
            self.split_manager.create_splits(len(dataset))

    def setup(self, stage: Optional[str] = None):
        """
        Initial setup - loads dataset and metadata.
        To configure a fold, use setup_fold(fold_idx).
        """
        if self.dataset is None:
            self.dataset = self._load_base_dataset()

            self.dataset_info = DatasetInfo(
                num_node_features=self.dataset.num_node_features,
                num_edge_features=self.dataset.num_edge_features,
                num_classes=self.dataset.num_classes,
                length=len(self.dataset),
            )

            log.info(f"Dataset: {self.name} | {self.dataset_info}")

        # Configure fold 0 by default if not specified
        if self.current_fold is None:
            log.warning("No fold specified, using fold 0 by default")
            self.setup_fold(0, stage)

    def setup_fold(self, fold_idx: int, stage: Optional[str] = None):
        """
        Configure a specific CV fold.

        Args:
            fold_idx: Fold index (0 to k_folds-1)
            stage: 'fit', 'validate', 'test', or None
        """
        if not (0 <= fold_idx < self.k_folds):
            raise ValueError(
                f"fold_idx must be in [0, {self.k_folds - 1}], received: {fold_idx}"
            )

        self.current_fold = fold_idx

        if self.dataset is None:
            self.setup()

        # Load indices
        train_idx, val_idx, test_idx = self.split_manager.load_splits(fold_idx)

        log.info(
            f"Fold {fold_idx}/{self.k_folds - 1} | "
            f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}"
        )

        # Create subsets
        if stage in ("fit", "validate", None):
            self.train_dataset = Subset(self.dataset, train_idx)
            self.val_dataset = Subset(self.dataset, val_idx)

        if stage in ("test", None):
            self.test_dataset = Subset(self.dataset, test_idx)

    def _load_base_dataset(self) -> MoleculeNet:
        return MoleculeNet(
            self.root,
            self.name,
            pre_transform=self.pre_transform,
            pre_filter=self.pre_filter,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Run setup_fold(fold_idx) first")
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Run setup_fold(fold_idx) first")
        return self._create_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Run setup_fold(fold_idx) first")
        return self._create_dataloader(self.test_dataset, shuffle=False)

    def _create_dataloader(self, dataset: Subset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    @property
    def num_node_features(self) -> int:
        if self.dataset_info is None:
            raise RuntimeError("Run setup() first")
        return self.dataset_info.num_node_features

    @property
    def num_edge_features(self) -> int:
        if self.dataset_info is None:
            raise RuntimeError("Run setup() first")
        return self.dataset_info.num_edge_features

    @property
    def num_classes(self) -> int:
        if self.dataset_info is None:
            raise RuntimeError("Run setup() first")
        return self.dataset_info.num_classes

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name}, k_folds={self.k_folds}, "
            f"current_fold={self.current_fold})"
        )


if __name__ == "__main__":
    # # Configure basic logging to see the output
    # logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    dm = MoleculeNetDataModule(
        name="ToxCast",  # BBBP, ClinTox, SIDER, Tox21, ToxCast
        batch_size=128,
        k_folds=5,  # Only 5 folds = 5 training runs
        test_ratio=0.2,  # 10% fixed test
        seed=42,  # Reproducibility
    )

    dm.prepare_data()

    print("\n" + "=" * 60)
    print("5-FOLD CROSS-VALIDATION")
    print("=" * 60)

    cv_results = []

    for fold_idx in range(dm.k_folds):
        print(f"\n--- Fold {fold_idx + 1}/{dm.k_folds} ---")

        dm.setup_fold(fold_idx)

        for batch in dm.train_dataloader():
            print("\nBatch sample:")
            print(f"Nodes: {batch.x.shape}, Edges: {batch.edge_index.shape}")
            print(
                f"Node features: {dm.num_node_features}, Edge features: {dm.num_edge_features}"
            )
            print(f"Labels: {batch.y.shape}")
            print(f"Batch vector: {batch.batch.shape}")

    # # This will load the splits into datasets
    # dm.setup()

    # print(f"\nNumber of training samples: {len(dm.train_dataset)}")
    # print(f"Number of validation samples: {len(dm.val_dataset)}")
    # print(f"Number of test samples: {len(dm.test_dataset)}")

    # total_in_splits = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
    # print(f"Total samples in splits: {total_in_splits}")

    # # Check for overlap
    # train_set = set(dm.train_dataset.indices)
    # val_set = set(dm.val_dataset.indices)
    # test_set = set(dm.test_dataset.indices)

    # print(f"Train/Val Overlap: {len(train_set.intersection(val_set))}")
    # print(f"Train/Test Overlap: {len(train_set.intersection(test_set))}")
    # print(f"Val/Test Overlap: {len(val_set.intersection(test_set))}")

    # for batch in dm.train_dataloader():
    #     print("\nBatch sample:")
    #     print(f"Nodes: {batch.x.shape}, Edges: {batch.edge_index.shape}")
    #     print(f"Labels: {batch.y.shape}")
    #     print(f"Batch vector: {batch.batch.shape}")
    #     break

    # # Example usage of the MoleculeNetDataModule with 5-fold CV

    # from pytorch_lightning import Trainer
    # from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    # import numpy as np

    # # ============================================
    # # CONFIGURATION
    # # ============================================
    # dm = MoleculeNetDataModule(
    #     name="BBBP",
    #     batch_size=32,
    #     k_folds=5,          # Only 5 folds = 5 training runs
    #     test_ratio=0.1,      # 10% fixed test
    #     seed=42,             # Reproducibility
    # )

    # dm.prepare_data()

    # # ============================================
    # # 5-FOLD CROSS-VALIDATION
    # # (For hyperparameter selection and variance estimation)
    # # ============================================
    # print("\n" + "="*60)
    # print("5-FOLD CROSS-VALIDATION")
    # print("="*60)

    # cv_results = []

    # for fold_idx in range(dm.k_folds):
    #     print(f"\n--- Fold {fold_idx + 1}/{dm.k_folds} ---")

    #     dm.setup_fold(fold_idx)

    #     # Create model (reinitialize weights for each fold)
    #     model = YourGNNModel(
    #         num_node_features=dm.num_node_features,
    #         num_classes=dm.num_classes,
    #         hidden_dim=64,      # Hyperparameters to optimize
    #         num_layers=3,
    #         dropout=0.2,
    #     )

    #     # Callbacks
    #     checkpoint = ModelCheckpoint(
    #         dirpath=f"checkpoints/{dm.name}/fold_{fold_idx}",
    #         filename="best",
    #         monitor="val_loss",
    #         mode="min",
    #     )

    #     early_stop = EarlyStopping(
    #         monitor="val_loss",
    #         patience=20,
    #         mode="min",
    #     )

    #     # Train
    #     trainer = Trainer(
    #         max_epochs=200,
    #         callbacks=[checkpoint, early_stop],
    #         enable_progress_bar=True,
    #         accelerator="auto",
    #         devices=1,
    #     )

    #     trainer.fit(model, dm)

    #     # Validate (DO NOT use test here)
    #     val_results = trainer.validate(model, dm, ckpt_path="best")
    #     cv_results.append(val_results[0])

    #     print(f"Val Loss: {val_results[0]['val_loss']:.4f}")
    #     print(f"Val Acc:  {val_results[0]['val_acc']:.4f}")

    # # ============================================
    # # CV RESULTS (report in paper)
    # # ============================================
    # print("\n" + "="*60)
    # print("CROSS-VALIDATION RESULTS")
    # print("="*60)

    # val_accs = [r['val_acc'] for r in cv_results]
    # val_losses = [r['val_loss'] for r in cv_results]

    # print(f"Validation Accuracy: {np.mean(val_accs):.4f} ± {np.std(val_accs):.4f}")
    # print(f"Validation Loss:     {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    # print(f"Per-fold accuracies: {[f'{acc:.4f}' for acc in val_accs]}")

    # # ============================================
    # # FINAL TEST SET EVALUATION
    # # (Report this as the paper's final result)
    # # ============================================
    # print("\n" + "="*60)
    # print("FINAL TEST EVALUATION")
    # print("="*60)

    # # Option 1: Use best model from best fold
    # best_fold = np.argmax(val_accs)
    # dm.setup_fold(best_fold)

    # best_model = YourGNNModel.load_from_checkpoint(
    #     f"checkpoints/{dm.name}/fold_{best_fold}/best.ckpt"
    # )

    # trainer = Trainer()
    # test_results = trainer.test(best_model, dm)

    # print(f"Test Accuracy: {test_results[0]['test_acc']:.4f}")
    # print(f"Test Loss:     {test_results[0]['test_loss']:.4f}")

    # # Option 2 (more rigorous): Train on full train+val with best hyperparameters
    # # (commented, use if you want maximum rigor)
    # """
    # # Combine train+val from all folds
    # all_train_val_indices = []
    # for fold_idx in range(dm.k_folds):
    #     train_idx, val_idx, _ = dm.split_manager.load_splits(fold_idx)
    #     all_train_val_indices.extend(train_idx)
    #     all_train_val_indices.extend(val_idx)

    # # Train final model
    # final_model = YourGNNModel(best_hyperparameters)
    # # ... train on all_train_val_indices
    # # ... evaluate on test
    # """
