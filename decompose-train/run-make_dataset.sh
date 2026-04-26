#!/bin/bash

#SBATCH --job-name=make-d2c
#SBATCH --output=d2c.out
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00


source ~/venv/bin/activate

python make_dataset.py