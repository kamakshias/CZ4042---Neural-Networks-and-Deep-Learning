#!/bin/bash
#SBATCH --job-name=aug_age
#SBATCH --output=Output_aug_age.out
#SBATCH --error=Error_aug_age.err
#SBATCH --nodes=1
#SBATCH --partition=SCSEGPU_UG
#SBATCH --mem=8000M
#SBATCH --gres=gpu:1

module load anaconda
source activate /apps/conda_env/CZ4042_v3
python optimal_age_augmentation_final.py