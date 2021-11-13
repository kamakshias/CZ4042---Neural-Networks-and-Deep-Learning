#!/bin/bash
#SBATCH --job-name=gender_lr
#SBATCH --output=Output_learning_rate_gender.out
#SBATCH --error=Error_learning_rate_gender.err
#SBATCH --nodes=1
#SBATCH --partition=SCSEGPU_UG
#SBATCH --mem=8000M
#SBATCH --gres=gpu:1

module load anaconda
source activate /apps/conda_env/CZ4042_v3
python optimal_gender_learning_rate_final.py
