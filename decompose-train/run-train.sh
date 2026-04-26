#!/bin/bash

#SBATCH --job-name=train
#SBATCH --output=train.out
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00


source ~/venv/bin/activate

python train.py \
    --dataset_path /home/b5ag/yuxuan.b5ag/project/0.6B/decompose-train/generated_facts/dataset \
    --output_path $SCRATCH/checkpoints/new_decompose