#!/bin/bash
#SBATCH --job-name=pre_age
#SBATCH --output=Output_pre_age.out
#SBATCH --error=Error_pre_age.err
#SBATCH --nodes=1
#SBATCH --partition=SCSEGPU_UG
#SBATCH --mem=8000M
#SBATCH --gres=gpu:1

module load anaconda
source activate /apps/conda_env/CZ4042_v3
python pretraining_on_imdb_wiki_dataset.py