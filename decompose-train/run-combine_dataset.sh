#!/bin/bash

#SBATCH --job-name=combine_dataset
#SBATCH --output=combine_dataset.out
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00


source ~/venv/bin/activate

python combine_dataset.py