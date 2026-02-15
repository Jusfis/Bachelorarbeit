#!/bin/bash
#
#SBATCH --job-name=CTM_parity_wandb_sweeps
#SBATCH --comment="CTM tuning with wandb"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justus.fischer@campus.lmu.de
#SBATCH --ntasks=1
#SBATCH --chdir=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/tasks/parity
#SBATCH --output=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/tasks/parity/slurm_kan.%j.%N.out

#RUN=1
#ITERATIONS=10
#MEMORY_LENGTH=5
#LOG_DIR="logs/parity/run${RUN}/ctm_${ITERATIONS}_${MEMORY_LENGTH}"
#SEED=$((RUN - 1))
#    --model_type "ctm"\

# IMPORTANT D_model % 5 == 0 for MLP postactivation production

python -m tasks.parity.train_sweeps_efficient \
    --log_dir "logs/parity/run10/MLP"\
    --seed 1 \
    --iterations 10 \
    --memory_length 5 \
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
    --training_iterations 50001 \
    --warmup_steps 500 \
    --track_every 1000 \
    --save_every 20000 \
    --no-reload \
    --no-reload_model_only \
    --no-use_amp \
    --neuron_select_type "random" \
    --postactivation_production 'mlp'\
    --useWandb 0


# set --device 0 to allow slurm to assign GPU automatically
# use Wandb set to 0 for local testing, set to 1 for slurm runs
# to submit the job on slurm, use from ctm main folder:
#sbatch --partition=NvidiaAll parityBatchKAN.sh script for slurm
