#!/bin/bash
#SBATCH --job-name=ft_roberta
#SBATCH --output=logs/ft/output/roberta.log
#SBATCH --error=logs/ft/error/roberta.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=owner1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:00:00
#SBATCH --mem=24G

mkdir -p logs/ft/output logs/ft/error

echo "Job started: $(date)"

python finetune_roberta.py --dataset sst2

echo "Job ended: $(date)"
