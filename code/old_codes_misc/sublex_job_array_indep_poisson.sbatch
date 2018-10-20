#!/bin/bash

#SBATCH --output=sublex_indep_poisson_array_%A_%a.out
#SBATCH --error=sublex_indep_poisson_array_%A_%a.err
#SBATCH --job-name=Poisson-indep
#SBATCH --account=def-timod
#SBATCH --array=1-100
#SBATCH --time=7-00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=7G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tmorita@mit.edu


######################
# Begin work section #
######################


source /home/tmorita/VI/bin/activate
python sublex_indep_poisson.py -s 6 -j $SLURM_JOB_ID -d ../data/BCCWJ_frequencylist_suw_ver1_0_core-nouns_revised.tsv