#!/bin/bash
 
#SBATCH --job-name=CTM_normal_mnist
#SBATCH --comment="CTM training"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justus.fischer@campus.lmu.de
#SBATCH --ntasks=1
#SBATCH --chdir=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/examples
#SBATCH --output=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/examples/slurm_mnist_normal.%j.%N.out
 
python3 -u mnist_normal.py
#sbatch --partition=NvidiaAll mnist_normal.sh