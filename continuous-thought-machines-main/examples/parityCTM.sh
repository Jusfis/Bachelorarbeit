#!/bin/bash
#
#SBATCH --job-name=CTM
#SBATCH --comment="CTM training"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justus.fischer@campus.lmu.de
#SBATCH --chdir=/home/fischerjus/Bachelorarbeit/continuous-thought-machines-main/Examples
#SBATCH --output=slurm.%j.%N.out
#SBATCH --ntasks=1

python3 -u 04_parity.py
