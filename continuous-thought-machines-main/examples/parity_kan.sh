#!/bin/bash
#
#SBATCH --job-name=CTM_kan_parity
#SBATCH --comment="CTM training"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justus.fischer@campus.lmu.de
#SBATCH --ntasks=1
#SBATCH --chdir=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/examples
#SBATCH --output=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/examples/slurm_kan.%j.%N.out


python3 -u 04_parity_kan.py
#to submit the job, use:
#sbatch --partition=NvidiaAll parity_kan.sh