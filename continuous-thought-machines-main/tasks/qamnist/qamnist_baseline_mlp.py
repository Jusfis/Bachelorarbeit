import argparse
import multiprocessing # Used for GIF generation
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')
import torch
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
from tqdm.auto import tqdm

from utils.samplers import QAMNISTSampler
from tasks.image_classification.plotting import plot_neural_dynamics
from tasks.qamnist.plotting import make_qamnist_gif
from utils.housekeeping import set_seed, zip_python_code
from utils.losses import qamnist_loss
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup
from tasks.parity.utils import reshape_attention_weights
from tasks.qamnist.utils import get_dataset, prepare_model
from models.utils import reshape_predictions, get_latest_checkpoint
import wandb
def parse_args():
    parser = argparse.ArgumentParser()

    # Task Configuration
    parser.add_argument('--q_num_images', type=int, default=3, help='Number of inputs per min mnist view')
    parser.add_argument('--q_num_images_delta', type=int, default=2, help='Range of numbers for QMNIST dataset')
    parser.add_argument('--q_num_repeats_per_input', type=int, default=10, help='Number of MNIST repeats to show model')
    parser.add_argument('--q_num_operations', type=int, default=3, help='The number of operations to apply.')
    parser.add_argument('--q_num_operations_delta', type=int, default=2, help='The range of operations to apply.')
    parser.add_argument('--q_num_answer_steps', type=int, default=10, help='The number of steps to answer a question, after observing digits and operator embeddings.')

    # Model Architecture
    parser.add_argument('--model_type', type=str, default='ctm', choices=['ctm', 'lstm','baseline'], help='Type of model to use.')
    parser.add_argument('--d_model', type=int, default=1024, help='Dimension of the model.')
    parser.add_argument('--d_input', type=int, default=64, help='Dimension of the input.')
    parser.add_argument('--synapse_depth', type=int, default=1, help='Depth of U-NET model for synapse. 1=linear, no unet.')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--n_synch_out', type=int, default=32, help='Number of neurons to use for output synch.')
    parser.add_argument('--n_synch_action', type=int, default=32, help='Number of neurons to use for observation/action synch.')
    parser.add_argument('--neuron_select_type', type=str, default='random', help='Protocol for selecting neuron subset.')
    parser.add_argument('--n_random_pairing_self', type=int, default=256, help='Number of neurons paired self-to-self for synch.')
    parser.add_argument('--iterations', type=int, default=50, help='Number of internal ticks.')
    parser.add_argument('--memory_length', type=int, default=30, help='Length of the pre-activation history for NLMS.')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True, help='Use deep memory.')
    parser.add_argument('--memory_hidden_dims', type=int, default=4, help='Hidden dimensions of the memory if using deep memory.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False, help='Apply normalization in NLMs.')
    parser.add_argument('--postactivation_production', type=str, default='mlp', choices=['mlp', 'kan'],
                        help='Type neural network for post-activiation production.')

    # Training
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--batch_size_test', type=int, default=256, help='Batch size for testing.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the model.')
    parser.add_argument('--training_iterations', type=int, default=100001, help='Number of training iterations.')
    parser.add_argument('--warmup_steps', type=int, default=5000, help='Number of warmup steps.')
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Use a learning rate scheduler.')
    parser.add_argument('--scheduler_type', type=str, default='cosine', choices=['multistep', 'cosine'], help='Type of learning rate scheduler.')
    parser.add_argument('--milestones', type=int, default=[8000, 15000, 20000], nargs='+', help='Learning rate scheduler milestones.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate scheduler gamma for multistep.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay factor.')

    # Housekeeping
    parser.add_argument('--log_dir', type=str, default='logs/qamnist', help='Directory for logging.')
    parser.add_argument('--data_root', type=str, default='data/', help='Where to save dataset.')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoints every this many iterations.')
    parser.add_argument('--seed', type=int, default=412, help='Random seed.')
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=False, help='Reload from disk?')
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=False, help='Reload only the model from disk?')
    parser.add_argument('--track_every', type=int, default=1000, help='Track metrics every this many iterations.')
    parser.add_argument('--n_test_batches', type=int, default=20, help='How many minibatches to approx metrics. Set to -1 for full eval')
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help='List of GPU(s) to use. Set to -1 to use CPU.')
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False, help='AMP autocast.')
    parser.add_argument('--useWandb', type=int, help='Use wandb logging.', default=0)

    args = parser.parse_args()
    return args

def qamnist_baseline_model(args,config, run):
    if config is not None:
        print(f"Using config: {config}\n Parsed args: {args}\n")
    else:
        print(f"Using args: {args}")

    set_seed(args.seed)

    # ----------------------------- LOAD DATA ------------------------------- #
    train_data, test_data, class_labels, dataset_mean, dataset_std = get_dataset(args.q_num_images,
                                                                                 args.q_num_images_delta,
                                                                                 args.q_num_repeats_per_input,
                                                                                 args.q_num_operations,
                                                                                 args.q_num_operations_delta)
    train_sampler = QAMNISTSampler(train_data, batch_size=args.batch_size)
    trainloader = torch.utils.data.DataLoader(train_data, num_workers=0, batch_sampler=train_sampler)

    test_sampler = QAMNISTSampler(test_data, batch_size=args.batch_size_test)
    testloader = torch.utils.data.DataLoader(test_data, num_workers=0, batch_sampler=test_sampler)
    # For total reproducibility
    # Python 3.x
    zip_python_code(f'{args.log_dir}/repo_state.zip')
    with open(f'{args.log_dir}/args.txt', 'w') as f:
        print(args, file=f)

    # ----------------------------- # Configure device string (support MPS on macOS) ------------------------------- #
    if args.device[0] != -1:
        device = f'cuda:{args.device[0]}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Running model {args.model_type} on {device}')

    # Build model
    model = prepare_model(args, device)

    # For lazy modules so that we can get param count
    pseudo_data = train_data.__getitem__(0)
    pseudo_inputs = pseudo_data[0].unsqueeze(0).to(device)
    pseudo_z = torch.tensor(pseudo_data[1]).unsqueeze(0).unsqueeze(2).to(device)
    model(pseudo_inputs, pseudo_z)

    model.train()

    print(f'Total params: {sum(p.numel() for p in model.parameters())}')

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr,
                                  eps=1e-8,
                                  weight_decay=args.weight_decay)

    warmup_schedule = warmup(args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_schedule.step)
    if args.use_scheduler:
        if args.scheduler_type == 'multistep':
            scheduler = WarmupMultiStepLR(optimizer, warmup_steps=args.warmup_steps, milestones=args.milestones,
                                          gamma=args.gamma)
        elif args.scheduler_type == 'cosine':
            scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_steps, args.training_iterations,
                                                warmup_start_lr=1e-20, eta_min=1e-7)
        else:
            raise NotImplementedError



    start_iter = 0  # For reloading, keep track of this (pretty tqdm stuff needs it)
    train_losses = []
    test_losses = []
    train_accuracies = []  # This will be per internal tick, not so simple
    test_accuracies = []
    train_accuracies_most_certain = []  # This will be selected according to what is returned by loss function
    test_accuracies_most_certain = []
    iters = []
    scaler = torch.amp.GradScaler("cuda" if "cuda" in device else "cpu", enabled=args.use_amp)

    # Now that everything is initliased, reload if desired
    if args.reload and (latest_checkpoint_path := get_latest_checkpoint(args.log_dir)):
        print(f'Reloading from: {latest_checkpoint_path}')
        checkpoint = torch.load(f'{latest_checkpoint_path}', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        if not args.reload_model_only:
            print('Reloading optimizer etc.')
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_iter = checkpoint['iteration']
            train_losses = checkpoint['train_losses']
            train_accuracies_most_certain = checkpoint['train_accuracies_most_certain']
            train_accuracies = checkpoint['train_accuracies']
            test_losses = checkpoint['test_losses']
            test_accuracies_most_certain = checkpoint['test_accuracies_most_certain']
            test_accuracies = checkpoint['test_accuracies']
            iters = checkpoint['iters']
        else:
            print('Only relading model!')
        if 'torch_rng_state' in checkpoint:
            # Reset seeds, otherwise mid-way training can be obscure (particularly for imagenet)
            torch.set_rng_state(checkpoint['torch_rng_state'].cpu().byte())
            np.random.set_state(checkpoint['numpy_rng_state'])
            random.setstate(checkpoint['random_rng_state'])

        del checkpoint
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------------- TRAINING BEGINS ------------------------------- #
    iterator = iter(trainloader)  # Not training in epochs, but rather iterations. Need to reset this from time to time

    with tqdm(total=args.training_iterations, initial=start_iter, leave=False, position=0, dynamic_ncols=True) as pbar:
        for bi in range(start_iter, args.training_iterations):
            current_lr = optimizer.param_groups[-1]['lr']

            try:
                inputs, z, _, targets = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                inputs, z, _, targets = next(iterator)





def run_sweep():
    """ Function to be called by wandb agent for hyperparameter sweeps """
    with wandb.init(entity="justus-fischer-ludwig-maximilian-university-of-munich", project="ctm-qamnist-baseline") as run:
        config = wandb.config
        # --------------------- Input from wandb sweep -------------------------- #
        args.batch_size = config.batch_size
        args.lr = config.learning_rate
        args.training_iterations = config.training_iterations
        args.use_amp = config.use_amp
        args.use_scheduler = config.use_scheduler
        args.model_type = config.model_type

        # ------------------ Modell laufen lassen ------------------------------- #
        qamnist_baseline_model(args, config, run)

if __name__=='__main__':
    args = parse_args()

    if args.useWandb == 1:
        # Sweep configuration for wandb
        sweep_configuration = {
            "program": "train_sweeps.py",
            "name": "ctm-qamnist-baseline",
            "method": "random",
            "metric": {
                "name": "Train/Losses",
                "goal": "minimize"
            },
            "parameters": {
                "batch_size": {"values": [64]},
                "learning_rate": {"min": 1e-4, "max": 3e-4},
                "use_amp": {"values": [True]},
                "use_scheduler": {"values": [True]},
                "training_iterations": {"values": [200000]},
                "postactivation_production": {"values": ["kan"]},
                "model_type": {"values": ["baseline"]},
                # ------------------ Hyperparameters from paper  ------------------------- #
                # "memory_length": {"values": [3]},
                # "q_num_repeats_per_input": {'values': [1]},
                # "q_num_answer_steps": {'values': [1]},

            }
        }

        # ------------------ RUN WITH WANDB SWEEPS ------------------------- #
        sweep_id = wandb.sweep(sweep_configuration, project="ctm-qamnist-baseline")
        wandb.agent(sweep_id, function=run_sweep, count=50)

    else:
        # ------------------- RUN WITHOUT WANDB ------------------------ #
        qamnist_baseline_model(args, None, None)

