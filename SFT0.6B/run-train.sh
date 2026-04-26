#!/bin/bash

#SBATCH --job-name=sft-last
#SBATCH --output=sft-last.out
#SBATCH --gpus=1
#SBATCH --time=10:00:00


source ~/venv/bin/activate

python train.py \
    --model_name Qwen/Qwen3-0.6B \
    --dataset_path /home/b5ag/yuxuan.b5ag/project/0.6B/reason-train/dataset \
    --output_path $SCRATCH/checkpoints/last_judge