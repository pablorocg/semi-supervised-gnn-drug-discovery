"""
Script for downloading datasets for semi-supervised GNN drug discovery project.
Uses OGB and MoleculeNet.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch_geometric
import torch_geometric.data
from dotenv import load_dotenv
from ogb.graphproppred import PygGraphPropPredDataset
from torch.serialization import safe_globals
from torch_geometric.datasets import MoleculeNet

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Dataset configurations
OGB_DATASETS = ["ogbg-molhiv", "ogbg-molpcba"]
MOLECULENET_DATASETS = [
    "Tox21",
    "ToxCast",
    "SIDER",
    "ClinTox",
    "PCBA",
    "MUV",
    "HIV",
    "BACE",
    "BBBP",
]


def get_data_dir() -> Path:
    """Get data directory from environment or use default."""
    data_dir = os.getenv("SOURCE_DATA_DIR", None)

    assert data_dir is not None, "SOURCE_DATA_DIR environment variable not set."

    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_exists(data_dir: Path, dataset_name: str) -> bool:
    """Check if dataset is already downloaded."""
    dataset_path = data_dir / dataset_name
    return dataset_path.exists()


def download_ogb_dataset(data_dir: Path, dataset_name: str, override: bool) -> None:
    """Download a single OGB dataset."""
    if dataset_exists(data_dir, dataset_name) and not override:
        logger.info(f"✓ {dataset_name} already exists (skipping)")
        return

    try:
        logger.info(f"Downloading {dataset_name}...")
        dataset = PygGraphPropPredDataset(name=dataset_name, root=str(data_dir))
        logger.info(f"✓ {dataset_name}: {len(dataset)} graphs")
    except Exception as e:
        logger.error(f"✗ Failed to download {dataset_name}: {e}")
        raise


def download_moleculenet_dataset(
    data_dir: Path, dataset_name: str, override: bool
) -> None:
    """Download a MoleculeNet dataset."""
    if dataset_exists(data_dir, dataset_name) and not override:
        logger.info(f"✓ {dataset_name} already exists (skipping)")
        return

    try:
        logger.info(f"Downloading {dataset_name} from MoleculeNet...")
        dataset = MoleculeNet(root=str(data_dir), name=dataset_name)
        logger.info(f"✓ {dataset_name}: {len(dataset)} graphs")
    except Exception as e:
        logger.error(f"✗ Failed to download {dataset_name}: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for semi-supervised GNN drug discovery"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific datasets to download (space-separated). If not specified, downloads all available datasets.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Re-download datasets that already exist",
    )
    parser.add_argument("--list", action="store_true", help="List available datasets")

    args = parser.parse_args()

    if args.list:
        logger.info("Available OGB datasets: " + ", ".join(OGB_DATASETS))
        logger.info(
            "Available MoleculeNet datasets: " + ", ".join(MOLECULENET_DATASETS)
        )
        return

    data_dir = get_data_dir()
    logger.info(f"Using data directory: {data_dir.absolute()}\n")

    # Determine datasets to download based on user input or default to all datasets
    datasets_to_download = (
        args.datasets if args.datasets else OGB_DATASETS + MOLECULENET_DATASETS
    )

    # Download each dataset
    for dataset_name in datasets_to_download:
        if dataset_name in OGB_DATASETS:
            download_ogb_dataset(data_dir, dataset_name, args.override)
        elif dataset_name in MOLECULENET_DATASETS:
            download_moleculenet_dataset(data_dir, dataset_name, args.override)
        else:
            logger.warning(f"Unknown dataset: {dataset_name}")

    logger.info("\n✓ Done")


if __name__ == "__main__":
    main()
