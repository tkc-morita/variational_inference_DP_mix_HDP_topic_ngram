#!/bin/bash

#SBATCH --output=sublex_distinct_final_indep_array_%A_%a.out
#SBATCH --error=sublex_distinct_final_indep_array_%A_%a.err
#SBATCH --job-name=distinct_final-indep
#SBATCH --array=1-60
#SBATCH --time=7-00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=9G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tmorita@mit.edu


######################
# Begin work section #
######################

source /home/tmorita/.bashrc
python sublex_indep_poisson_distinct_final.py