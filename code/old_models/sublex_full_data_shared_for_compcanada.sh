#!/bin/bash
# ---------------------------------------------------------------------
# SLURM script for job resubmission on a Compute Canada cluster. 
# ---------------------------------------------------------------------
#SBATCH --job-name=sublexicalization-shared
#SBATCH --account=def-timod
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00
#SBATCH --mem=7G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tmorita@mit.edu
# ---------------------------------------------------------------------
source /home/tmorita/VI/bin/activate
python sublex_shared_for_compcanada.py