#!/bin/bash

#SBATCH --output=sublex_mora_indep_array_%A_%a.out
#SBATCH --error=sublex_mora_indep_array_%A_%a.err
#SBATCH --job-name=mora-indep
#SBATCH --account=def-timod
#SBATCH --array=1-30
#SBATCH --time=7-00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=7G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tmorita@mit.edu


######################
# Begin work section #
######################

# Print this sub-job's task ID
echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

source /home/tmorita/VI/bin/activate
python sublex_indep_mora.py