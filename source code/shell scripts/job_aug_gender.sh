#!/bin/bash
#SBATCH --job-name=aug_gender
#SBATCH --output=Output_aug_gender.out
#SBATCH --error=Error_aug_gender.err
#SBATCH --nodes=1
#SBATCH --partition=SCSEGPU_UG
#SBATCH --mem=8000M
#SBATCH --gres=gpu:1

module load anaconda
source activate /apps/conda_env/CZ4042_v3
python optimal_gender_augmentation_final.py