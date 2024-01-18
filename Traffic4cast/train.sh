#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=32G

nvidia-smi

python rec_cc.py --city london --device 0 --batch_size 2 --hidden_channels 32 --epochs 20 --fill -1