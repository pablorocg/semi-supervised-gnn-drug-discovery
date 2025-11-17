# Semi-Supervised GNN for Molecular Property Prediction

This repository contains a framework for training Graph Neural Networks (GNNs) for molecular property prediction, with a focus on semi-supervised learning techniques. The project is built using PyTorch, PyTorch Lightning, PyTorch Geometric, and Hydra for configuration management. 

It includes implementations for:
- Supervised baseline models (GCN, GIN).
- Semi-supervised Mean Teacher models.
- Data modules for QM9 and MoleculeNet (PCBA) datasets, supporting semi-supervised splits.
- Hydra-configurable training pipelines.

## 1. Installation

Clone the repository and install the required dependencies.
```bash
# Clone the repository
git clone <your-repo-url>
cd semi-supervised-gnn-drug-discovery

# Create a conda virtual environment (recommended)
conda create -n gnn_env python=3.11 -y
conda activate gnn_env

# Install dependencies
pip install -r requirements.txt
```
*Note on PyTorch Geometric:* requirements.txt includes torch-geometric, torch-scatter, and torch-sparse. Depending on your system (OS, CUDA version), you might need to install these manually by following the official PyTorch Geometric installation guide.

## 2. Configuration

This project relies on environment variables to manage paths for data, logs, and configs.

**Environment Variables:** 

You must create a .env file in the root of the project. You can copy the template:

```bash
cp .env.template .env
```
Now, edit the .env file to set the correct paths for your system. These variables are loaded by src/utils/path_utils.py.

```bash
# .env

# Path to directory where datasets (QM9, MoleculeNet) will be downloaded/stored
SOURCE_DATA_DIR=/path/to/your/data/directory

# Path to the 'config' directory in this project
# Example: /home/user/semi-supervised-gnn-drug-discovery/config
CONFIGS_DIR=/path/to/this/projects/config

# Path to directory where logs (Wandb, checkpoints, hydra logs) will be saved
LOGS_DIR=/path/to/your/logs/directory

# Path to directory where trained models will be saved (if not saved with logs)
MODELS_DIR=/path/to/your/models/directory
```

**Hydra Configuration**

To be documented.

## Tasks

|  **Task**  |  **Status**  |  **Comments**  |
|--------|----------|------------|
|Setup Dataset Module for QM9|Done|Regression task|
|Setup Dataset Module for MoleculeNet (PCBA)|Done|It can use any of the Moleculenet Datasets for classification/regression|
|Configure Data Modules for SSL splits (labeled/unlabeled)|Done||
|Implement GNN Models (GCN, GIN)|Done|We can try/implement more architectures|
|Implement Supervised Baseline LightningModule|Done|Default params not optimized yet|
|Create Configurable Trainer for Baseline|Done||
|Implement Hyperparameter Search Framework|In progress||
|Create Configurable Trainer for Hyperparameter Search Framework|In progress|40%|
|Implement Mean-Teacher LightningModule|In progress||
|Implement Configurable Trainer for Mean-Teacher|Not Started||
|Run Baseline Experiments|Not Started||
|Run Mean-Teacher Experiments|Not Started||
|Implement Noisy NCP LightningModule|Not Started||
|Implement Configurable Trainer for Noisy NCP|Not Started||
|Run Noisy NCP Experiments|Not Started||
|Write-up & Analysis|Not Started||



## Experimental Setup
Each model will be trained and evaluated on:

- **Train set:** 70% of the labeled data divided into labeled and unlabeled subsets
- **Validation set:** 10% of the labeled data
- **Test set:** 20% of the labeled data

### Regression Task (QM9 dataset)
**Metric:** Mean Absolute Error (MAE)

|Experiment|Model|% Labeled Data|Result on Test Set|
|---|---|---|---|
|Supervised Baseline|GCN|10%|-|
|Supervised Baseline|GCN|20%|-|
|Supervised Baseline|GCN|50%|-|
|Mean Teacher|GCN|10%|-|
|Mean Teacher|GCN|20%|-|
|Mean Teacher|GCN|50%|-|
|Noisy NCP|GCN|10%|-|
|Noisy NCP|GCN|20%|-|
|Noisy NCP|GCN|50%|-|



### Classification Task (MoleculeNet - PCBA dataset)
**Metric:** ROC-AUC

|Experiment|Model|% Labeled Data|Result on Test Set|
|---|---|---|---|
|Supervised Baseline|GCN|10%|-|
|Supervised Baseline|GCN|20%|-|
|Supervised Baseline|GCN|50%|-|
|Mean Teacher|GCN|10%|-|
|Mean Teacher|GCN|20%|-|
|Mean Teacher|GCN|50%|-|
|Noisy NCP|GCN|10%|-|
|Noisy NCP|GCN|20%|-|
|Noisy NCP|GCN|50%|-|

