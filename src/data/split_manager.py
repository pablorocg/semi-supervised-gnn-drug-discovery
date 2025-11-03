from pathlib import Path
import json
import logging
import numpy as np
from sklearn.model_selection import KFold, train_test_split


log = logging.getLogger(__name__)


class SplitManager:
    """
    Split manager: Fixed test set (10%) + 5-fold CV (90%)

    This is the standard protocol in ML papers:
    - Completely independent test set (never touched during development)
    - 5-fold CV on the rest for hyperparameter selection and validation
    - Final evaluation on test set

    Advantages:
    - Only requires 5 training runs (vs 25 for nested CV)
    - Unbiased estimation of final performance
    - Reproducible and documented
    """

    def __init__(
        self,
        dataset_name: str,
        splits_dir: Path,
        k_folds: int = 5,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.k_folds = k_folds
        self.test_ratio = test_ratio
        self.seed = seed

        self.splits_dir = Path(splits_dir) / dataset_name
        self.splits_dir.mkdir(parents=True, exist_ok=True)

    @property
    def splits_file(self) -> Path:
        """
        Unique filename based on configuration
        """
        return (
            self.splits_dir
            / f"holdout_{self.k_folds}fold_test{self.test_ratio}_seed{self.seed}.json"
        )

    def splits_exist(self) -> bool:
        return self.splits_file.exists()

    def create_splits(self, n_samples: int) -> None:
        """
        Create and save splits following this protocol:
        1. Separate fixed test set (10%)
        2. Create 5-fold CV on remaining 90%
        """
        log.info(
            f"Creating splits: {self.k_folds}-fold CV + fixed test set | "
            f"n={n_samples}, test_ratio={self.test_ratio}, seed={self.seed}"
        )

        indices = np.arange(n_samples)

        # 1. Fixed test set (NEVER used in training)
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_ratio,
            random_state=self.seed,
            shuffle=True,
        )

        n_test = len(test_idx)
        log.info(
            f"  Test set: {n_test} ({n_test / n_samples * 100:.1f}%) - FINAL HOLD-OUT"
        )

        # 2. K-fold CV on train+val
        kfold = KFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)

        cv_splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(train_val_idx)):
            n_train = len(train_idx)
            n_val = len(val_idx)

            cv_splits.append(
                {
                    "fold": fold_idx,
                    "train": train_val_idx[train_idx].tolist(),
                    "val": train_val_idx[val_idx].tolist(),
                }
            )

            log.info(
                f"  Fold {fold_idx}: "
                f"train={n_train} ({n_train / n_samples * 100:.1f}%), "
                f"val={n_val} ({n_val / n_samples * 100:.1f}%)"
            )

        # Save with metadata for reproducibility
        output = {
            "metadata": {
                "n_samples": n_samples,
                "k_folds": self.k_folds,
                "test_ratio": self.test_ratio,
                "seed": self.seed,
                "created_at": str(np.datetime64("now")),
            },
            "test": test_idx.tolist(),
            "cv_splits": cv_splits,
        }

        self._save_json(output, self.splits_file)
        log.info(f"✓ Splits saved: {self.splits_file.name}")

    def load_splits(self, fold_idx: int) -> tuple[list[int], list[int], list[int]]:
        """
        Load splits for a specific fold.

        Args:
            fold_idx: Fold index (0 to k_folds-1)

        Returns:
            tuple: (train_indices, val_indices, test_indices)
        """
        if not self.splits_exist():
            raise FileNotFoundError(
                f"Splits file not found: {self.splits_file}\nRun prepare_data() first."
            )

        data = self._load_json(self.splits_file)

        # Validate configuration
        metadata = data["metadata"]
        if metadata["seed"] != self.seed:
            log.warning(
                f"⚠️  Seed in file ({metadata['seed']}) "
                f"differs from requested ({self.seed})"
            )

        # Validate fold_idx
        if not (0 <= fold_idx < self.k_folds):
            raise ValueError(
                f"fold_idx must be in [0, {self.k_folds - 1}], received: {fold_idx}"
            )

        cv_splits = data["cv_splits"]
        test_indices = data["test"]

        fold = cv_splits[fold_idx]
        return fold["train"], fold["val"], test_indices

    @staticmethod
    def _save_json(data, file_path: Path) -> None:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _load_json(file_path: Path):
        with open(file_path, "r") as f:
            return json.load(f)