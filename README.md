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
|Setup Dataset Module for MoleculeNet (PCBA)|Done|It can use any of the Moleculenet Datasets for classification|
|Configure Data Modules for SSL splits (labeled/unlabeled)|Done||
|Implement GNN Models (GCN, GIN)|Done|We can try/implement more architectures|
|Implement Supervised Baseline LightningModule|Done|Default params not optimized yet|
|Create Configurable Trainer for Baseline|Done||
|Implement Hyperparameter Search Framework|Done||
|Create Configurable Trainer for Hyperparameter Search Framework|Done|40%|
|Implement Mean-Teacher LightningModule|In progress||
|Implement Configurable Trainer for Mean-Teacher|Not Started||
|Run Baseline Experiments|Done||
|Run Mean-Teacher Experiments|Not Started||
|Write-up & Analysis|Not Started||



## Experimental Setup
Each model will be trained and evaluated on:

- **Train set:** 70% of the labeled data divided into labeled and unlabeled subsets
- **Validation set:** 10% of the labeled data
- **Test set:** 20% of the labeled data


### Classification Task (MoleculeNet - Tox21)

**Metric:** ROC-AUC

|Job ID|Experiment|Model|% Labeled Data|Result on Test Set (783 test samples)|
|---|---|---|---|---|
|2435|Supervised Baseline|GINE|10%  (782 labeled)|0.7406|
|2436|Supervised Baseline|GINE|20% (1564 labeled)|0.7559|
|2437|Supervised Baseline|GINE|50% (3911 labeled)|0.8189|
||Mean Teacher|GINE|10%|-|
||Mean Teacher|GINE|20%|-|
||Mean Teacher|GINE|50%|-|


### Classification Task (MoleculeNet - PCBA dataset)

**Metric:** ROC-AUC
Supervised training details: 
- batch size 256
- scheduler cosine scheduler with warmup, 
- learning rate 0.005, 
- Max 1000 epochs (early stopping patience 10 (monitoring val loss).
- Model: GINE with 5 layers, hidden dim 128, dropout 0.5 (654 K parameters)

  embedding_dim: 16 # 256      
  hidden_channels: 256   
  encoder_num_heads: 4     # 256 / 4 = 64 dim per head
  encoder_dropout: 0.1    
  num_gnn_layers: 4        
  gnn_mlp_layers: 2        
  readout_mlp_layers: 2    
  dropout: 0.5             
  activation: "relu"       
  pooling_type: "mean"    
  use_residual: true      
  learn_eps: true  








|Experiment|Model|% Labeled Data|Result on Test Set|
|---|---|---|---|
|Supervised Baseline|GINE|10%  43792 labeled|0.7901|
|Supervised Baseline|GINE|20%  87585 labeled|-|
|Supervised Baseline|GINE|50% 218963 labeled|-|
|Mean Teacher|GINE|10%|-|
|Mean Teacher|GINE|20%|-|
|Mean Teacher|GINE|50%|-|




