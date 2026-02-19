#!/bin/bash
#
#SBATCH --job-name=Base_Cifar10_Parity
#SBATCH --comment="CTM tuning with wandb"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justus.fischer@campus.lmu.de
#SBATCH --ntasks=1
#SBATCH --chdir=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/
#SBATCH --output=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/tasks/image_classification/baseline_slurm_.%j.%N.out

RUN=1
LOG_DIR="logs/image_classification/run${RUN}/baseline${model}"
SEED=$((RUN - 1))

export PYTHONPATH=$PYTHONPATH:.

python -u tasks/image_classification/image_baseline_mlp.py \
--log_dir $LOG_DIR \
--dataset cifar10 \
--d_model 256 \
--d_input 64 \
--memory_hidden_dims 64 \
--deep_memory \
--dropout 0.0 \
--backbone_type resnet18-1 \
--training_iterations 600001 \
--warmup_steps 1000 \
--use_scheduler \
--scheduler_type cosine \
--weight_decay 0.0001 \
--save_every 1000 \
--track_every 2000 \
--n_test_batches 50 \
--num_workers_train 8 \
--batch_size 512 \
--batch_size_test 512 \
--lr 1e-4 \
--seed $SEED \
--useWandb 1 \
--model "mlp" \
 --device 0

# set --device 0 to allow slurm to assign GPU automatically
# use Wandb set to 0 for local testing, set to 1 for slurm runs
# to submit the job on slurm, use from ctm main folder:
# sbatch --partition=NvidiaAll baseline_cifar10.sh