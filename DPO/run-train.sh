#!/bin/bash

#SBATCH --job-name=dpo
#SBATCH --output=dpo.out
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00


source ~/venv/bin/activate

python train.py \
    --model_name $SCRATCH/checkpoints/last_judge/checkpoint-2168 \
    --dataset_path /home/b5ag/yuxuan.b5ag/project/0.6B/reason-train/dataset/train-dpo.csv \
    --output_path $SCRATCH/checkpoints/DPO