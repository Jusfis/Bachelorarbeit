#!/bin/bash
#
#SBATCH --job-name=Base_Cifar10_Parity
#SBATCH --comment="CTM tuning with wandb"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=justus.fischer@campus.lmu.de
#SBATCH --ntasks=1
#SBATCH --chdir=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/
#SBATCH --output=/home/f/fischerjus/Bachelorarbeit/continuous-thought-machines-main/tasks/image_classification/KAN_slurm_.%j.%N.out

RUN=1
LOG_DIR="logs/image_classification/run${RUN}/kan${MODEL}"
SEED=$((RUN - 1))
MODEL="ctm"
POSTACTIVATION="mlp"


export PYTHONPATH=$PYTHONPATH:.

python -u tasks/image_classification/train.py \
--log_dir $LOG_DIR \
--model $MODEL \
--dataset cifar10 \
--d_model 256 \
--d_input 64 \
--synapse_depth 5 \
--heads 16 \
--n_synch_out 256 \
--n_synch_action 512 \
--n_random_pairing_self 0 \
--neuron_select_type random-pairing \
--iterations 50 \
--memory_length 15 \
--deep_memory \
--memory_hidden_dims 64 \
--dropout 0.0 \
--dropout_nlm 0 \
--no-do_normalisation \
--positional_embedding_type none \
--backbone_type resnet18-1 \
--training_iterations 600001 \
--warmup_steps 1000 \
--use_scheduler \
--scheduler_type cosine \
--weight_decay 0.0001 \
--save_every 10000 \
--track_every 2000 \
--n_test_batches 50 \
--num_workers_train 8 \
--batch_size 512 \
--batch_size_test 512 \
--lr 1e-4 \
--seed 1 \
--useWandb 1 \
--postactivation_production $POSTACTIVATION \
--device 1 \



#python -m tasks.image_classification.train \
#--log_dir logs/cifar10-versus-humans/ctm/d=256--i=64--heads=16--sd=5--synch=256-512-0-h=64-random-pairing--iters=50x15--backbone=18-1--seed=2 \
#--model ctm
#--dataset cifar10 \
#--d_model 256 \
#--d_input 64 \
#--synapse_depth 5 \
#--heads 16 \
#--n_synch_out 256 \
#--n_synch_action 512 \
#--n_random_pairing_self 0 \
#--neuron_select_type random-pairing \
#--iterations 50 \
#--memory_length 15 \
#--deep_memory \
#--memory_hidden_dims 64 \
#--dropout 0.0 \
#--dropout_nlm 0 \
#--no-do_normalisation \
#--positional_embedding_type none \
#--backbone_type resnet18-1 \
#--training_iterations 600001 \
#--warmup_steps 1000 \
#--use_scheduler \
#--scheduler_type cosine \
#--weight_decay 0.0001 \
#--save_every 1000 \
#--track_every 2000 \
#--n_test_batches 50 \
#--num_workers_train 8 \
#--batch_size 512 \
#--batch_size_test 512 \
#--lr 1e-4 \
#--device 0 \
#--seed 2\
#--useWandb 0 \
#--postactivation_production "mlp" \
#
#python -m tasks.image_classification.train \
#--log_dir logs/cifar10-versus-humans/ctm/d=256--i=64--heads=16--sd=5--synch=256-512-0-h=64-random-pairing--iters=50x15--backbone=18-1--seed=42 \
#--model ctm
#--dataset cifar10 \
#--d_model 256 \
#--d_input 64 \
#--synapse_depth 5 \
#--heads 16 \
#--n_synch_out 256 \
#--n_synch_action 512 \
#--n_random_pairing_self 0 \
#--neuron_select_type random-pairing \
#--iterations 50 \
#--memory_length 15 \
#--deep_memory \
#--memory_hidden_dims 64 \
#--dropout 0.0 \
#--dropout_nlm 0 \
#--no-do_normalisation \
#--positional_embedding_type none \
#--backbone_type resnet18-1 \
#--training_iterations 600001 \
#--warmup_steps 1000 \
#--use_scheduler \
#--scheduler_type cosine \
#--weight_decay 0.0001 \
#--save_every 1000 \
#--track_every 2000 \
#--n_test_batches 50 \
#--num_workers_train 8 \
#--batch_size 512 \
#--batch_size_test 512 \
#--lr 1e-4 \
#--device 0 \
#--seed 42 \
#--useWandb 0 \
#--postactivation_production "mlp" \
#



#
#python -m tasks.image_classification.train \
#--log_dir logs/cifar10-versus-humans/lstm/nlayers=2--d=256--i=64--heads=16--synch=256-512-0-h=64-random-pairing--iters=50x15--backbone=18-1--seed=1 \
#--dataset cifar10 \
#--model lstm \
#--num_layers 2 \
#--d_model 256 \
#--d_input 64 \
#--heads 16 \
#--iterations 50 \
#--dropout 0.0  \
#--positional_embedding_type none \
#--backbone_type resnet18-1 \
#--training_iterations 600001 \
#--warmup_steps 2000 \
#--use_scheduler \
#--scheduler_type cosine \
#--weight_decay 0.0001 \
#--save_every 1000 \
#--track_every 2000 \
#--n_test_batches 50 \
#--reload  \
#--num_workers_train 8 \
#--batch_size 512 \
#--batch_size_test 512 \
#--lr 1e-4 \
#--device 0 \
#--seed 1 \
#--no-reload
#
#
#python -m tasks.image_classification.train \
#--log_dir logs/cifar10-versus-humans/lstm/nlayers=2--d=256--i=64--heads=16--synch=256-512-0-h=64-random-pairing--iters=50x15--backbone=18-1--seed=2 \
#--dataset cifar10 \
#--model lstm \
#--num_layers 2 \
#--d_model 256 \
#--d_input 64 \
#--heads 16 \
#--iterations 50 \
#--dropout 0.0  \
#--positional_embedding_type none \
#--backbone_type resnet18-1 \
#--training_iterations 600001 \
#--warmup_steps 2000 \
#--use_scheduler \
#--scheduler_type cosine \
#--weight_decay 0.0001 \
#--save_every 1000 \
#--track_every 2000 \
#--n_test_batches 50 \
#--reload  \
#--num_workers_train 8 \
#--batch_size 512 \
#--batch_size_test 512 \
#--lr 1e-4 \
#--device 0 \
#--seed 2 \
#--no-reload
#
#
#python -m tasks.image_classification.train \
#--log_dir logs/cifar10-versus-humans/lstm/nlayers=2--d=256--i=64--heads=16--synch=256-512-0-h=64-random-pairing--iters=50x15--backbone=18-1--seed=42 \
#--dataset cifar10 \
#--model lstm \
#--num_layers 2 \
#--d_model 256 \
#--d_input 64 \
#--heads 16 \
#--iterations 50 \
#--dropout 0.0  \
#--positional_embedding_type none \
#--backbone_type resnet18-1 \
#--training_iterations 600001 \
#--warmup_steps 2000 \
#--use_scheduler \
#--scheduler_type cosine \
#--weight_decay 0.0001 \
#--save_every 1000 \
#--track_every 2000 \
#--n_test_batches 50 \
#--reload  \
#--num_workers_train 8 \
#--batch_size 512 \
#--batch_size_test 512 \
#--lr 1e-4 \
#--device 0 \
#--seed 42 \
#--no-reload
#
#
#
#
#
#python -m tasks.image_classification.train \
#--log_dir logs/cifar10-versus-humans/ff/d=256--backbone=18-1--seed=1 \
#--dataset cifar10 \
#--model ff \
#--d_model 256 \
#--memory_hidden_dims 64 \
#--dropout 0.0 \
#--dropout_nlm 0 \
#--backbone_type resnet18-1 \
#--training_iterations 600001 \
#--warmup_steps 1000 \
#--use_scheduler \
#--scheduler_type cosine \
#--weight_decay 0.0001 \
#--save_every 1000 \
#--track_every 2000 \
#--n_test_batches 50 \
#--num_workers_train 8 \
#--batch_size 512 \
#--batch_size_test 512 \
#--lr 1e-4 \
#--device 0 \
#--seed 1
#
#
#python -m tasks.image_classification.train \
#--log_dir logs/cifar10-versus-humans/ff/d=256--backbone=18-1--seed=2 \
#--dataset cifar10 \
#--model ff \
#--d_model 256 \
#--memory_hidden_dims 64 \
#--dropout 0.0 \
#--dropout_nlm 0 \
#--backbone_type resnet18-1 \
#--training_iterations 600001 \
#--warmup_steps 1000 \
#--use_scheduler \
#--scheduler_type cosine \
#--weight_decay 0.0001 \
#--save_every 1000 \
#--track_every 2000 \
#--n_test_batches 50 \
#--num_workers_train 8 \
#--batch_size 512 \
#--batch_size_test 512 \
#--lr 1e-4 \
#--device 0 \
#--seed 2
#
#python -m tasks.image_classification.train \
#--log_dir logs/cifar10-versus-humans/ff/d=256--backbone=18-1--seed=42 \
#--dataset cifar10 \
#--model ff \
#--d_model 256 \
#--memory_hidden_dims 64 \
#--dropout 0.0 \
#--dropout_nlm 0 \
#--backbone_type resnet18-1 \
#--training_iterations 600001 \
#--warmup_steps 1000 \
#--use_scheduler \
#--scheduler_type cosine \
#--weight_decay 0.0001 \
#--save_every 1000 \
#--track_every 2000 \
#--n_test_batches 50 \
#--num_workers_train 8 \
#--batch_size 512 \
#--batch_size_test 512 \
#--lr 1e-4 \
#--device 0 \
#--seed 42
#






