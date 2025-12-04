#!/bin/bash
#
#SBATCH --job-name=CTM_kan_mnist
#SBATCH --comment="CTM training"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justus.fischer@campus.lmu.de
#SBATCH --ntasks=1
#SBATCH --chdir=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/examples
#SBATCH --output=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/examples/slurm_mnist_kan.%j.%N.out


python3 -u mnist_kan.py

#sbatch --partition=NvidiaAll mnist_kan.sh