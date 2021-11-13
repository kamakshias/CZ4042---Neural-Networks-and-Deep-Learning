#!/bin/bash
#SBATCH --job-name=CZ4042
#SBATCH --output=Output_batch_age.out
#SBATCH --error=Error_batch_age.err
#SBATCH --nodes=1
#SBATCH --partition=SCSEGPU_UG
#SBATCH --mem=8000M
#SBATCH --gres=gpu:1

module load anaconda
source activate /apps/conda_env/CZ4042_v3
python optimal_gender_batch_size_final.py

