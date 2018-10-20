#!/bin/bash

#SBATCH --output=sublex_forward_backward_indep_array_%A_%a.out
#SBATCH --error=sublex_forward_backward_indep_array_%A_%a.err
#SBATCH --job-name=forward-backward
#SBATCH --account=def-timod
#SBATCH --array=1-30
#SBATCH --time=7-00:00
#SBATCH --cpus-per-task=5
#SBATCH --mem=9G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tmorita@mit.edu


######################
# Begin work section #
######################

source /home/tmorita/VI/bin/activate
python sublex_indep_poisson_forward_backward.py -j $SLURM_JOB_ID