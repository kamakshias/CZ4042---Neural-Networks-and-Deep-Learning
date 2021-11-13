#!/bin/bash
#SBATCH --job-name=CZ4042
#SBATCH --output=Output_neurons_gender.out
#SBATCH --error=Error_neurons_gender.err
#SBATCH --nodes=1
#SBATCH --partition=SCSEGPU_UG
#SBATCH --mem=8000M
#SBATCH --gres=gpu:1

module load anaconda
source activate /apps/conda_env/CZ4042_v3
python optimal_gender_no_of_neurons_final.py

