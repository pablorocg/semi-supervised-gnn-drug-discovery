# Semi-Supervised GNN for Drug Discovery

A PyTorch Lightning framework for molecular property prediction. This project implements a **Mean Teacher** semi-supervised learning approach using an **Attentive GINE** backbone, designed to improve performance on drug discovery datasets with sparse labels.

## Quick Start

### 1\. Installation

Clone the repo and set up the environment.

```bash
# Clone repository
git clone https://github.com/pablorocg/semi-supervised-gnn-drug-discovery
cd semi-supervised-gnn-drug-discovery

# Create environment (recommended)
conda create -n gnn_env python=3.11 -y
conda activate gnn_env

# Install dependencies
pip install -r requirements.txt
```

### 2\. Configuration

The project needs to know where to save data and logs.

1.  Create a `.env` file from the template:
    ```bash
    cp .env.template .env
    ```
2.  Open `.env` and set your absolute paths:
    ```bash
    SOURCE_DATA_DIR=/abs/path/to/data    # Datasets will be downloaded here
    CONFIGS_DIR=/abs/path/to/config      # Path to the 'config' folder in this repo
    LOGS_DIR=/abs/path/to/logs           # Where to save training logs
    ```

-----

## Usage

Run experiments using the scripts in `src/trainers/`. You can override parameters (like dataset or model) directly from the command line.

### Train Supervised Baseline

Standard training using only labeled data.

```bash
python -m src.trainers.baseline_trainer \
    dataset.init.name=SIDER \
    "dataset.init.splits=[0.67, 0.03, 0.1, 0.2]"\
    dataset.init.batch_size_train=16 \
    dataset.init.mu=5
```

### Train Semi-Supervised (Mean Teacher)

Training using both labeled and unlabeled data. Ideal for low-data regimes.

```bash
python -m src.trainers.mean_teacher_trainer \
    dataset.init.name=SIDER \
    "dataset.init.splits=[0.35, 0.35, 0.1, 0.2]" \
    dataset.init.batch_size_train=32 \
    dataset.init.mu=1
```

-----

## Project Structure

  * **`config/`**: Hydra configuration files (datasets, models, training params).
  * **`src/data/`**: DataModules for MoleculeNet and OGB datasets.
  * **`src/models/`**: GNN implementations (GINE, Attentive Encoders).
  * **`src/lightning_modules/`**: PyTorch Lightning modules for Baseline and Mean Teacher logic.
  * **`src/trainers/`**: Entry points for training scripts.


