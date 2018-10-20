#!/bin/bash

#=======SLURM Options==========
#SBATCH -c 12
#SBATCH -t 7-0
#SBATCH --mem=6G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tmorita@mit.edu
#=======shell scipt============
source ~/.bashrc
python train_and_test_indep.py