#!/bin/bash
#SBATCH --job-name=age_layers
#SBATCH --output=Output_no_of_layers_age.out
#SBATCH --error=Error_no_of_layers_age.err
#SBATCH --nodes=1
#SBATCH --partition=SCSEGPU_UG
#SBATCH --mem=8000M
#SBATCH --gres=gpu:1

module load anaconda
source activate /apps/conda_env/CZ4042_v3
python optimal_age_no_of_layers_final.py
