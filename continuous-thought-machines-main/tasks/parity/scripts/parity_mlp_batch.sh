#!/bin/bash
#
#SBATCH --job-name=MLP_Parity
#SBATCH --comment="CTM tuning with wandb"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justus.fischer@campus.lmu.de
#SBATCH --ntasks=1
#SBATCH --chdir=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/
#SBATCH --output=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/tasks/parity/mlp_final_100_50.%j.%N.out

#RUN=1
ITERATIONS=100
MEMORY_LENGTH=50
#LOG_DIR="logs/parity/run${RUN}/ctm_${ITERATIONS}_${MEMORY_LENGTH}"
#SEED=$((RUN - 1))
MODEL="ctm"
POSTACTIVATION="mlp"
#    --model_type "ctm"\
# IMPORTANT D_model % 5 == 0 for MLP postactivation production

export PYTHONPATH=$PYTHONPATH:.

python -u tasks/parity/train_sweeps_efficient.py \
    --model_type $MODEL \
    --log_dir "logs/parity/mlp"\
    --seed 1 \
    --iterations $ITERATIONS \
    --memory_length $MEMORY_LENGTH \
    --parity_sequence_length 64  \
    --n_test_batches 20 \
    --d_model 1024 \
    --d_input 512 \
    --n_synch_out 32 \
    --n_synch_action 32 \
    --synapse_depth 1 \
    --heads 8 \
    --memory_hidden_dims 16 \
    --dropout 0.0 \
    --deep_memory \
    --no-do_normalisation \
    --positional_embedding_type="custom-rotational-1d" \
    --backbone_type="parity_backbone" \
    --no-full_eval \
    --weight_decay 0.0 \
    --gradient_clipping 0.9 \
    --use_scheduler \
    --scheduler_type "cosine" \
    --milestones 0 0 0 \
    --gamma 0 \
    --dataset "parity" \
    --batch_size 64 \
    --batch_size_test 256 \
    --lr=0.0001 \
    --training_iterations 200001 \
    --warmup_steps 500 \
    --track_every 1000 \
    --save_every 20000 \
    --no-reload \
    --no-reload_model_only \
    --no-use_amp \
    --neuron_select_type "random" \
    --device 0 \
    --postactivation_production $POSTACTIVATION \
    --useWandb 1


# set --device 0 to allow slurm to assign GPU automatically
# use Wandb set to 0 for local testing, set to 1 for slurm runs
# to submit the job on slurm, use from ctm main folder:
# sbatch --partition=NvidiaAll parity_kan_batch.sh script for slurm
