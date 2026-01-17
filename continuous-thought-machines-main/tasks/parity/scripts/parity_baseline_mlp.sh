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
python -m tasks.parity.parity_baseline_mlp \
    --log_dir "logs/parity/run3/baseline_sweeps"\
    --seed 1 \
    --iterations 75 \
    --memory_length 25 \
    --parity_sequence_length 64  \
    --n_test_batches 20 \
    --d_model 60 \
    --d_input 64 \
    --n_synch_out 16 \
    --n_synch_action 16 \
    --synapse_depth 1 \
    --heads 4 \
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
    --batch_size 32 \
    --batch_size_test 120 \
    --lr=0.0001 \
    --training_iterations 200001 \
    --warmup_steps 500 \
    --track_every 1000 \
    --save_every 20000 \
    --no-reload \
    --no-reload_model_only \
    --no-use_amp \
    --neuron_select_type "random" \
    --postactivation_production 'kan'

#to submit the job on slurm, use from ctm main folder:
#sbatch --partition=NvidiaAll parityBatch.sh
# remove --device 0 to allow slurm to assign GPU automatically