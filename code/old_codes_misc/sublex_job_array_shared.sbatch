#!/bin/bash

#SBATCH --output=sublex_shared_array_%A_%a.out
#SBATCH --error=sublex_shared_array_%A_%a.err
#SBATCH --job-name=sublexicalization-shared
#SBATCH --array=1-77
#SBATCH --time=7-00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tmorita@mit.edu


######################
# Begin work section #
######################

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

source /home/tmorita/.bashrc
python sublex_shared_for_bcs.py -c 0.0001