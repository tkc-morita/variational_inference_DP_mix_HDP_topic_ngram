#!/bin/bash

#=======SLURM Options==========
#SBATCH -c 30
#SBATCH -t 7-0
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user==tmorita@mit.edu

#=======shell scipt============
python variational_inference_DP_ngram_mix_LSE_rev.py ../data/BCCWJ_frequencylist_suw_ver1_0_core-nouns.tsv 10 1 3 1 0.0001 10000 0.01