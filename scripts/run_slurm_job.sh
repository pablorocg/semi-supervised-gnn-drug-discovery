#!/bin/bash
#BSUB -J Dataloader_Test
#BSUB -q hpc
#BSUB -W 02:30
#BSUB -n 12
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -o Output_%J.out
#BSUB -e Output_%J.err


cd /your/path/to/semi-supervised-gnn-drug-discovery

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate drug_discovery


# Try running the data module script to test dataloaders
python -m src.data.datamodules

