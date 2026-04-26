#!/bin/bash

#SBATCH --job-name=de-eval-raw
#SBATCH --output=de-eval-raw.out
#SBATCH --gpus=1
#SBATCH --time=10:00:00

source ~/venv/bin/activate

python decompose-eval.py \
    --reference_path /home/b5ag/yuxuan.b5ag/project/0.6B/decompose-results/A3B \
    --prediction_path /home/b5ag/yuxuan.b5ag/project/0.6B/decompose-results/Raw0.6B