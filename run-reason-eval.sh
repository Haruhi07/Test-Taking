#!/bin/bash

#SBATCH --job-name=reason-eval
#SBATCH --output=reason-eval.out
#SBATCH --gpus=1
#SBATCH --time=10:00:00


source ~/venv/bin/activate

python reason-eval.py --folder_path  /home/b5ag/yuxuan.b5ag/project/rebuttal/-Decomp-LA-30B-1
