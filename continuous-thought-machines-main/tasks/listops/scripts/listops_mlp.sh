#!/bin/bash
#
#SBATCH --job-name=CTM_listops_wandb_sweeps
#SBATCH --comment="CTM tuning with wandb"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justus.fischer@campus.lmu.de
#SBATCH --ntasks=1
#SBATCH --chdir=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/tasks/listops
#SBATCH --output=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/tasks/listops/slurm_mlp.%j.%N.out

LOG_DIR="logs/qamnist/mlp/test"


python -m tasks.listops.train_listops \
    --log_dir $LOG_DIR \
    --seed 1 \
    --iterations 75 \
    --memory_length 25 \
    --n_test_batches 20 \
    --d_model 1024 \
    --d_input 100 \
    --n_synch_out 32 \
    --n_synch_action 32 \
    --synapse_depth 1 \
    --heads 4 \
    --memory_hidden_dims 16 \
    --dropout 0.0 \
    --deep_memory \
    --no-do_normalisation \
    --positional_embedding_type="learned-1d" \
    --backbone_type="listops"\
    --no-full_eval \
    --weight_decay 0.0 \
    --gradient_clipping 0.9 \
    --use_scheduler \
    --scheduler_type "cosine" \
    --milestones 0 0 0 \
    --gamma 0 \
    --dataset "listops" \
    --batch_size 64 \
    --batch_size_test 256 \
    --lr 0.0002 \
    --training_iterations 50001 \
    --warmup_steps 500 \
    --track_every 1000 \
    --save_every 2000 \
    --no-reload \
    --no-reload_model_only \
    --no-use_amp \
    --neuron_select_type "random" \
    --postactivation_production 'mlp' \
    --useWandb 1 \
    --device 0

# use Wandb set to 0 for local testing, set to 1 for slurm runs
# to submit the job on slurm, use from ctm main folder:
# sbatch --partition=NvidiaAll listops_mlp.sh
# remove --device 0 to allow slurm to assign GPU automatically