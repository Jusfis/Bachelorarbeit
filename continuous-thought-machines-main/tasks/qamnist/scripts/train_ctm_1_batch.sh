#!/bin/bash
#
#SBATCH --job-name=CTM_qamnist_kan
#SBATCH --comment="CTM tuning with wandb"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justus.fischer@campus.lmu.de
#SBATCH --ntasks=1
#SBATCH --chdir=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/tasks/qamnist
#SBATCH --output=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/tasks/qamnist/slurm_kan.%j.%N.out

#RUN=1
#ITERATIONS=10
#MEMORY_LENGTH=5
#LOG_DIR="logs/parity/run${RUN}/ctm_${ITERATIONS}_${MEMORY_LENGTH}"
#SEED=$((RUN - 1))

#!/bin/bash
RUN=2
MEMORY_LENGTH=3
MODEL_TYPE="ctm"
Q_NUM_REPEATS_PER_INPUT=1
LOG_DIR="logs/qamnist/run${RUN}/${MODEL_TYPE}_${Q_NUM_REPEATS_PER_INPUT}"
SEED=$((RUN - 1))

python -u train_sweep.py \
    --log_dir $LOG_DIR \
    --seed $SEED \
    --memory_length $MEMORY_LENGTH \
    --model_type $MODEL_TYPE \
    --q_num_images 3 \
    --q_num_images_delta 2 \
    --q_num_repeats_per_input $Q_NUM_REPEATS_PER_INPUT \
    --q_num_operations 3 \
    --q_num_operations_delta 2 \
    --q_num_answer_steps 1 \
    --n_test_batches 20 \
    --d_model 1024 \
    --d_input 64 \
    --n_synch_out 32 \
    --n_synch_action 32 \
    --synapse_depth 1 \
    --heads 4 \
    --memory_hidden_dims 16 \
    --dropout 0.0 \
    --deep_memory \
    --no-do_normalisation \
    --weight_decay 0.0 \
    --use_scheduler \
    --scheduler_type "cosine" \
    --milestones 0 0 0 \
    --gamma 0 \
    --batch_size 64 \
    --batch_size_test 256 \
    --lr=0.0001 \
    --training_iterations 300001 \
    --warmup_steps 500 \
    --track_every 1000 \
    --save_every 10000 \
    --no-reload \
    --no-reload_model_only \
    --no-use_amp \
    --neuron_select_type "random" \
    --postactivation_production 'kan'
#    --device 0

# use Wandb set to 0 for local testing, set to 1 for slurm runs
# to submit the job on slurm, use from ctm main folder:
# sbatch --partition=NvidiaAll parityBatchKAN.sh
# remove --device 0 to allow slurm to assign GPU automatically