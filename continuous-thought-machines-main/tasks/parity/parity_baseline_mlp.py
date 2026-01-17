import argparse
import math
import multiprocessing # Used for GIF generation
import random # Used for saving/loading RNG state
import os
import sys
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib as mpl
mpl.use('Agg') # Use Agg backend before pyplot import
import matplotlib.pyplot as plt # Used for loss/accuracy plots
import numpy as np
np.seterr(divide='ignore', invalid='warn') # Keep basic numpy settings
import seaborn as sns
sns.set_style('darkgrid')
import torch
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
import torchvision # For disabling warning
from tqdm.auto import tqdm # Used for progress bars

from autoclip.torch import QuantileClip # Used for gradient clipping
from data.custom_datasets import ParityDataset
from tasks.image_classification.plotting import plot_neural_dynamics
from models.utils import reshape_predictions, get_latest_checkpoint
from tasks.parity.plotting import make_parity_gif
from tasks.parity.utils_efficient import prepare_baseline, reshape_attention_weights, reshape_inputs
from utils.housekeeping import set_seed, zip_python_code
from utils.losses import parity_loss_baseline
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup
import wandb

torchvision.disable_beta_transforms_warning()
torch.serialization.add_safe_globals([argparse.Namespace])


def parse_args():
    parser = argparse.ArgumentParser(description="Train CTM on Parity Task")

    # Model Architecture
    parser.add_argument('--model_type', type=str, default="ctm", choices=['ctm', 'lstm'], help='The type of model to train.')
    parser.add_argument('--parity_sequence_length', type=int, default=64, help='Sequence length for parity task.')
    parser.add_argument('--d_model', type=int, default=1024, help='Dimension of the model.')
    parser.add_argument('--d_input', type=int, default=512, help='Dimension of the input projection.')
    parser.add_argument('--synapse_depth', type=int, default=1, help='Depth of U-NET model for synapse. 1=linear.')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--n_synch_out', type=int, default=32, help='Number of neurons for output sync.')
    parser.add_argument('--n_synch_action', type=int, default=32, help='Number of neurons for action sync.')
    parser.add_argument('--neuron_select_type', type=str, default='random', choices=['first-last', 'random', 'random-pairing'], help='Protocol for selecting neuron subset.')
    parser.add_argument('--n_random_pairing_self', type=int, default=256, help='Number of neurons paired self-to-self for synch.')
    parser.add_argument('--iterations', type=int, default=75, help='Number of internal ticks.')
    parser.add_argument('--memory_length', type=int, default=25, help='Length of pre-activation history for NLMs.')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True, help='Use deep NLMs.')
    parser.add_argument('--memory_hidden_dims', type=int, default=16, help='Hidden dimensions for deep NLMs.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False, help='Apply normalization in NLMs.')
    parser.add_argument('--positional_embedding_type', type=str, default='custom-rotational-1d', help='Type of positional embedding.') # Choices removed for simplicity if not strictly needed by argparse functionality here
    parser.add_argument('--backbone_type', type=str, default='parity_backbone', help='Type of backbone feature extractor.')
    parser.add_argument('--postactivation_production', type=str, default='mlp', choices=['mlp', 'kan'], help='Type neural network for post-activiation production.')

    # Training Configuration
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--batch_size_test', type=int, default=256, help='Batch size for testing.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--training_iterations', type=int, default=50001, help='Number of training iterations.')
    parser.add_argument('--warmup_steps', type=int, default=0, help='Number of warmup steps.')
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True, help='Use a learning rate scheduler.')
    parser.add_argument('--scheduler_type', type=str, default='multistep', choices=['multistep', 'cosine'], help='Type of learning rate scheduler.')
    parser.add_argument('--milestones', type=int, default=[8000, 15000, 20000], nargs='+', help='Scheduler milestones for multistep.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Scheduler gamma for multistep.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay factor.')
    parser.add_argument('--gradient_clipping', type=float, default=-1, help='Quantile clipping value (-1 to disable).')
    parser.add_argument('--use_most_certain_with_lstm', action=argparse.BooleanOptionalAction, default=False, help='Use most certain loss with LSTM baseline.')

    # Housekeeping
    parser.add_argument('--log_dir', type=str, default='logs/parity', help='Directory for logging.')
    parser.add_argument('--dataset', type=str, default='parity', help='Dataset name (used for assertion).')
    parser.add_argument('--save_every', type=int, default=1000, help='Save checkpoint frequency.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=True, help='Reload checkpoint from log_dir?')
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=True, help='Reload only model weights?')
    parser.add_argument('--track_every', type=int, default=1000, help='Track metrics frequency.')
    parser.add_argument('--n_test_batches', type=int, default=20, help='Num batches for metrics approx. (-1 for full).')
    parser.add_argument('--full_eval',  action=argparse.BooleanOptionalAction, default=False, help='Perform full evaluation instead of approx.')
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help='GPU(s) or -1 for CPU.')
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=False, help='AMP autocast.')

    args = parser.parse_args()
    return args


def create_long_df(data_array, iters, metric_name='Accuracy'):
    """
    Converts numpy array (iters x runs) into a Long-Form DataFrame for Seaborn.
    """
    # Ensure data is (iters, runs)
    data = np.array(data_array)

    # Create DataFrame with Iterations as index
    df = pd.DataFrame(data, index=iters)

    # Reset index to make 'Iteration' a column
    df = df.reset_index().rename(columns={'index': 'Iteration'})

    # Melt into long format: [Iteration, Run_ID, Value]
    df_long = df.melt(id_vars='Iteration', var_name='Run', value_name=metric_name)
    return df_long

def main():
    # todo anpassen an neue wandb api mlp vs kan
    # with wandb.init(entity="justus-fischer-ludwig-maximilian-university-of-munich",project="ctm-parity") as run:
    #     config = wandb.config
        args = parse_args()
        # input from wandb sweep
        # args.batch_size = config.batch_size
        # args.lr = config.learning_rate
        # # args.postactivation_production = config.postactivation_production
        # args.training_iterations = config.training_iterations
        # # args.model_type = config.model_type
        # args.use_amp = config.use_amp
        # args.use_scheduler = config.use_scheduler
        # args.memory_length = config.memory_length
        # args.iterations = config.internal_ticks

        # print(f"Using config: {config}\n Parsed args: {args}\n")


        set_seed(args.seed)

        if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)

        assert int(math.sqrt(args.parity_sequence_length)) ** 2 == args.parity_sequence_length, "parity_sequence_length must be a perfect square."

        train_data = ParityDataset(sequence_length=args.parity_sequence_length, length=100000)
        test_data = ParityDataset(sequence_length=args.parity_sequence_length, length=10000)


        trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True, num_workers=0, drop_last=False)

        prediction_reshaper = [args.parity_sequence_length, 2]

        # ueberlegen wie
        args.out_dims = args.parity_sequence_length * 2

        args.use_most_certain = args.model_type == "ctm" or (args.use_most_certain_with_lstm and args.model_type == "lstm")

        # For total reproducibility
        # Python 3.x
        zip_python_code(f'{args.log_dir}/repo_state.zip')
        with open(f'{args.log_dir}/args.txt', 'w') as f:
            print(args, file=f)

        # Configure device string (support MPS on macOS)

        if args.device[0] != -1:
            device = f'cuda:{args.device[0]}'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        print(f'Running model {args.model_type} on {device}')

        # Build model
        # Prediction Reshaper
        model = prepare_baseline(args=args, device=device)

        model.train()

        # For lazy modules so that we can get param count
        pseudo_inputs = train_data.__getitem__(0)[0].unsqueeze(0).to(device)
        model(pseudo_inputs)

        print(f'Total params: {sum(p.numel() for p in model.parameters())}')

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.lr,
                                      eps=1e-8,
                                      weight_decay=args.weight_decay)
        if args.gradient_clipping!=-1:  # Not using, but handy to have
            optimizer = QuantileClip.as_optimizer(optimizer=optimizer, quantile=args.gradient_clipping, history_length=1000)

        warmup_schedule = warmup(args.warmup_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_schedule.step)
        if args.use_scheduler:
            if args.scheduler_type == 'multistep':
                scheduler = WarmupMultiStepLR(optimizer, warmup_steps=args.warmup_steps, milestones=args.milestones, gamma=args.gamma)
            elif args.scheduler_type == 'cosine':
                scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_steps, args.training_iterations, warmup_start_lr=1e-20, eta_min=1e-7)
            else:
                raise NotImplementedError


        # Metrics tracking (I like custom)
        # Using batched estimates
        start_iter = 0  # For reloading, keep track of this (pretty tqdm stuff needs it)
        train_losses = []
        test_losses = []
        train_accuracies = []  # This will be per internal tick, not so simple
        test_accuracies = []
        train_accuracies_most_certain = []  # This will be selected according to what is returned by loss function
        test_accuracies_most_certain = []
        train_accuracies_most_certain_per_input = []
        test_accuracies_most_certain_per_input = []
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
                train_accuracies_most_certain_per_input = checkpoint['train_accuracies_most_certain_per_input'] if 'train_accuracies_most_certain_per_input' in checkpoint else train_accuracies_most_certain_per_input
                train_accuracies = checkpoint['train_accuracies']
                test_losses = checkpoint['test_losses']
                test_accuracies_most_certain = checkpoint['test_accuracies_most_certain']
                test_accuracies_most_certain_per_input = checkpoint['test_accuracies_most_certain_per_input'] if 'test_accuracies_most_certain_per_input' in checkpoint else test_accuracies_most_certain_per_input
                test_accuracies = checkpoint['test_accuracies']
                iters = checkpoint['iters']
            else:
                print('Only relading model!')
            if 'torch_rng_state' in checkpoint:
                print("Reloading rng state")
                # Reset seeds, otherwise mid-way training can be obscure (particularly for imagenet)
                torch.set_rng_state(checkpoint['torch_rng_state'].cpu().byte())
                np.random.set_state(checkpoint['numpy_rng_state'])
                random.setstate(checkpoint['random_rng_state'])

            del checkpoint
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        # Training
        iterator = iter(trainloader)  # Not training in epochs, but rather iterations. Need to reset this from time to time
        with tqdm(total=args.training_iterations, initial=start_iter, leave=False, position=0, dynamic_ncols=True) as pbar:
            for bi in range(start_iter, args.training_iterations):
                current_lr = optimizer.param_groups[-1]['lr']

                try:
                    inputs, targets = next(iterator)
                except StopIteration:
                    iterator = iter(trainloader)
                    inputs, targets = next(iterator)

                inputs = inputs.to(device)
                targets = targets.to(device)

                with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.float16, enabled=args.use_amp):
                    predictions = model(inputs)
                    # predictions = predictions.reshape(predictions.size(0), -1, 2, predictions.size(-1))
                    # loss, where_most_certain = parity_loss(predictions, certainties, targets, use_most_certain=args.use_most_certain)
                    # print(f"Size of predictions:{predictions.size()}, Size of targets:{targets.size()}")
                    loss = parity_loss_baseline(predictions,targets)
                    # print(loss)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                predicted_classes = predictions.argmax(dim=1)

                # targets: [32, 64] -> true_classes: [32]
                true_classes = targets[:, -1]

                accuracy = (predicted_classes == true_classes).float().mean().item()
                # accuracy_finegrained = (predictions.argmax(2)[torch.arange(predictions.size(0), device=predictions.device),:,where_most_certain] == targets).float().mean().item()
                pbar.set_description(f'Dataset=Parity. Loss={loss.item():0.3f}. Accuracy={accuracy:0.3f}. LR={current_lr:0.6f}. Iter={bi}')

                # run.log({
                #     "Train/Losses": loss.item(),
                #     "Train/Accuracies": accuracy_finegrained,
                # }, step=bi)


                # Metrics tracking and plotting ####################### TRACK ##############
                if bi%args.track_every==0 and bi != 0:
                    model.eval()
                    with torch.inference_mode():

                        inputs, targets = next(iter(testloader))
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                        predictions, certainties, synchronisation, pre_activations, post_activations, attention = model(inputs, track=True)

                        predictions = reshape_predictions(predictions, prediction_reshaper)
                        attention = reshape_attention_weights(attention)
                        inputs = reshape_inputs(inputs, args.iterations, grid_size=int(math.sqrt(args.parity_sequence_length)))

                        pbar.set_description('Tracking: Neural dynamics')
                        plot_neural_dynamics(post_activations, args.d_model, args.log_dir, axis_snap=True)

                        pbar.set_description('Tracking: Producing attention gif')

                        process = multiprocessing.Process(
                            target=make_parity_gif,
                            args=(
                            predictions.detach().cpu().numpy(),
                            certainties.detach().cpu().numpy(),
                            targets.detach().cpu().numpy(),
                            pre_activations,
                            post_activations,
                            attention,
                            inputs,
                            f"{args.log_dir}/eval_output_val_{0}_iter_{0}.gif",
                        ))
                        process.start()

                        ##################################### TRAIN METRICS ##########################
                        all_predictions = []
                        all_targets = []
                        all_predictions_most_certain = []
                        all_losses = []

                        iters.append(bi)
                        with torch.inference_mode():
                            loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_test, shuffle=True, num_workers=0)
                            with tqdm(total=len(loader), initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:

                                for inferi, (inputs, targets) in enumerate(loader):

                                    inputs = inputs.to(device)
                                    targets = targets.to(device)
                                    these_predictions, certainties, synchronisation = model(inputs)

                                    these_predictions = reshape_predictions(these_predictions, prediction_reshaper)
                                    loss, where_most_certain = parity_loss(these_predictions, certainties, targets, use_most_certain=args.use_most_certain)
                                    all_losses.append(loss.item())

                                    all_targets.append(targets.detach().cpu().numpy())

                                    all_predictions_most_certain.append(these_predictions.argmax(2)[torch.arange(these_predictions.size(0), device=these_predictions.device), :, where_most_certain].detach().cpu().numpy())
                                    all_predictions.append(these_predictions.argmax(2).detach().cpu().numpy())

                                    if inferi%args.n_test_batches==0 and inferi!=0 and not args.full_eval: break
                                    pbar_inner.set_description('Computing metrics for train')
                                    pbar_inner.update(1)

                            all_predictions = np.concatenate(all_predictions)
                            all_targets = np.concatenate(all_targets)
                            all_predictions_most_certain = np.concatenate(all_predictions_most_certain)


                            train_accuracies.append(np.mean(all_predictions == all_targets[...,np.newaxis], axis=tuple(range(all_predictions.ndim-1))))
                            train_accuracies_most_certain.append((all_targets == all_predictions_most_certain).mean())
                            train_accuracies_most_certain_per_input.append((all_targets == all_predictions_most_certain).reshape(all_targets.shape[0], -1).all(-1).mean())
                            train_losses.append(np.mean(all_losses))

                            # # log to wandb
                            # run.log({
                            #     "Train/Accuracies_Most_Certain": train_accuracies_most_certain[-1] if len(train_accuracies_most_certain) > 0 else 0,
                            #     "Test/Accuracies_Most_Certain": test_accuracies_most_certain[-1] if len(test_accuracies_most_certain) > 0 else 0,
                            #     "Train/Losses": train_losses[-1] if len(train_losses) > 0 else 0,
                            #     "Train/Accuracies": train_accuracies[-1] if len(train_accuracies) > 0 else 0,
                            # }, step=bi)


                            ##################################### TEST METRICS ##################################
                            all_predictions = []
                            all_predictions_most_certain = []
                            all_targets = []
                            all_losses = []
                            loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=True, num_workers=0)
                            with tqdm(total=len(loader), initial=0, leave=False, position=1, dynamic_ncols=True) as pbar_inner:
                                for inferi, (inputs, targets) in enumerate(loader):

                                    inputs = inputs.to(device)
                                    targets = targets.to(device)
                                    these_predictions, certainties, synchronisation = model(inputs)

                                    these_predictions = these_predictions.reshape(these_predictions.size(0), -1, 2, these_predictions.size(-1))
                                    loss, where_most_certain = parity_loss(these_predictions, certainties, targets, use_most_certain=args.use_most_certain)
                                    all_losses.append(loss.item())

                                    all_targets.append(targets.detach().cpu().numpy())

                                    all_predictions_most_certain.append(these_predictions.argmax(2)[torch.arange(these_predictions.size(0), device=these_predictions.device), :, where_most_certain].detach().cpu().numpy())
                                    all_predictions.append(these_predictions.argmax(2).detach().cpu().numpy())

                                    if inferi%args.n_test_batches==0 and inferi!=0 and not args.full_eval: break
                                    pbar_inner.set_description('Computing metrics for test')
                                    pbar_inner.update(1)

                            all_predictions = np.concatenate(all_predictions)
                            all_targets = np.concatenate(all_targets)
                            all_predictions_most_certain = np.concatenate(all_predictions_most_certain)

                            test_accuracies.append(np.mean(all_predictions == all_targets[...,np.newaxis], axis=tuple(range(all_predictions.ndim-1))))
                            test_accuracies_most_certain.append((all_targets == all_predictions_most_certain).mean())
                            test_accuracies_most_certain_per_input.append((all_targets == all_predictions_most_certain).reshape(all_targets.shape[0], -1).all(-1).mean())
                            test_losses.append(np.mean(all_losses))

                            sns.set_theme(style="whitegrid")

                            # Define Color Palette for the "All Runs" plots
                            cm = sns.color_palette("viridis", as_cmap=True)

                            # Ensure data is numpy array for easier handling
                            train_acc_arr = np.array(train_accuracies)
                            test_acc_arr = np.array(test_accuracies)
                            train_loss_arr = np.array(train_losses)
                            test_loss_arr = np.array(test_losses)

                            if args.dataset != 'sort':
                                # =========================================================
                                # PART A: ACCURACIES
                                # =========================================================

                                # --- A1. Accuracies: Old Style (All Runs) ---
                                figacc_raw = plt.figure(figsize=(10, 10))
                                axacc_train = figacc_raw.add_subplot(211)
                                axacc_test = figacc_raw.add_subplot(212)

                                # Loop through runs (Transpose .T to iterate over columns/runs)
                                num_runs = train_acc_arr.shape[1]
                                for ti in range(num_runs):
                                    color_val = ti / num_runs
                                    axacc_train.plot(iters, train_acc_arr[:, ti], color=cm(color_val), alpha=0.3)
                                    axacc_test.plot(iters, test_acc_arr[:, ti], color=cm(color_val), alpha=0.3)

                                # Baselines
                                for ax in [axacc_train, axacc_test]:
                                    ax.plot(iters, train_accuracies_most_certain, 'k--', alpha=0.7,
                                            label='Most certain')
                                    ax.plot(iters, train_accuracies_most_certain_per_input, 'r', alpha=0.6,
                                            label='Full Input')

                                axacc_train.set_title('Train Accuracy (All Runs)')
                                axacc_test.set_title('Test Accuracy (All Runs)')
                                axacc_train.legend(loc='lower right')
                                axacc_train.set_xlim([0, args.training_iterations])
                                axacc_test.set_xlim([0, args.training_iterations])

                                figacc_raw.tight_layout()
                                figacc_raw.savefig(f'{args.log_dir}/accuracies_all_runs.png', dpi=150)
                                plt.close(figacc_raw)

                                # --- A2. Accuracies: New Style (Seaborn CI) ---
                                figacc_ci, (ax_ci_train, ax_ci_test) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

                                df_acc_train = create_long_df(train_acc_arr, iters, 'Accuracy')
                                df_acc_test = create_long_df(test_acc_arr, iters, 'Accuracy')

                                sns.lineplot(data=df_acc_train, x='Iteration', y='Accuracy', ax=ax_ci_train,
                                             color='tab:blue', errorbar=('ci', 95), label='Mean Train Acc')
                                sns.lineplot(data=df_acc_test, x='Iteration', y='Accuracy', ax=ax_ci_test,
                                             color='tab:green', errorbar=('ci', 95), label='Mean Test Acc')

                                # Baselines
                                ax_ci_train.plot(iters, train_accuracies_most_certain, 'k--', alpha=0.7,
                                                 label='Most certain')
                                ax_ci_train.plot(iters, train_accuracies_most_certain_per_input, 'r', alpha=0.6,
                                                 label='Full Input')
                                ax_ci_test.plot(iters, test_accuracies_most_certain, 'k--', alpha=0.7,
                                                label='Most certain')
                                ax_ci_test.plot(iters, test_accuracies_most_certain_per_input, 'r', alpha=0.6,
                                                label='Full Input')

                                ax_ci_train.set_title('Train Accuracy (Mean + 95% CI)')
                                ax_ci_test.set_title('Test Accuracy (Mean + 95% CI)')
                                ax_ci_train.legend(loc='lower right')
                                ax_ci_test.legend(loc='lower right')
                                ax_ci_train.set_xlim([0, args.training_iterations])

                                figacc_ci.tight_layout()
                                figacc_ci.savefig(f'{args.log_dir}/accuracies.png', dpi=150)
                                plt.close(figacc_ci)

                                # =========================================================
                                # PART B: LOSSES
                                # =========================================================

                                # --- B1. Losses: Old Style (All Runs) ---
                                figloss_raw = plt.figure(figsize=(10, 5))
                                axloss_raw = figloss_raw.add_subplot(111)

                                # Loop through runs
                                num_runs_loss = train_loss_arr.shape[1]
                                for ti in range(num_runs_loss):
                                    # We use a very low alpha (0.2) because losses can be noisy
                                    axloss_raw.plot(iters, train_loss_arr[:, ti], 'b-', alpha=0.2)
                                    axloss_raw.plot(iters, test_loss_arr[:, ti], 'r-', alpha=0.2)

                                # Create custom legend handles since we plotted many lines
                                from matplotlib.lines import Line2D
                                custom_lines = [Line2D([0], [0], color='blue', lw=2),
                                                Line2D([0], [0], color='red', lw=2)]
                                axloss_raw.legend(custom_lines, ['Train Loss', 'Test Loss'], loc='upper right')

                                axloss_raw.set_title('Losses (All Runs)')
                                axloss_raw.set_xlim([0, args.training_iterations])

                                figloss_raw.tight_layout()
                                figloss_raw.savefig(f'{args.log_dir}/losses_all_runs.png', dpi=150)
                                plt.close(figloss_raw)

                                # --- B2. Losses: New Style (Seaborn CI) ---
                                figloss_ci = plt.figure(figsize=(10, 5))
                                axloss_ci = figloss_ci.add_subplot(111)

                                df_loss_train = create_long_df(train_loss_arr, iters, 'Loss')
                                df_loss_test = create_long_df(test_loss_arr, iters, 'Loss')

                                # Seaborn automatically handles the mean line and shadow
                                sns.lineplot(data=df_loss_train, x='Iteration', y='Loss', ax=axloss_ci,
                                             color='blue', errorbar=('ci', 95), label='Train Loss')
                                sns.lineplot(data=df_loss_test, x='Iteration', y='Loss', ax=axloss_ci,
                                             color='red', errorbar=('ci', 95), label='Test Loss')

                                axloss_ci.set_title('Losses (Mean + 95% Confidence Interval)')
                                axloss_ci.legend(loc='upper right')
                                axloss_ci.set_xlim([0, args.training_iterations])

                                figloss_ci.tight_layout()
                                figloss_ci.savefig(f'{args.log_dir}/losses.png', dpi=150)
                                plt.close(figloss_ci)


                # todo why model.train() twice?
                    model.train()



                # Save model ########################### AND make fig##############################
                if (bi%args.save_every==0 or bi==args.training_iterations-1):
                    torch.save(
                        {
                        'model_state_dict':model.state_dict(),
                        'optimizer_state_dict':optimizer.state_dict(),
                        'scheduler_state_dict':scheduler.state_dict(),
                        'scaler_state_dict':scaler.state_dict(),
                        'iteration':bi,
                        'train_accuracies_most_certain':train_accuracies_most_certain,
                        'train_accuracies_most_certain_per_input':train_accuracies_most_certain_per_input,
                        'train_accuracies':train_accuracies,
                        'test_accuracies_most_certain':test_accuracies_most_certain,
                        'test_accuracies_most_certain_per_input':test_accuracies_most_certain_per_input,
                        'test_accuracies':test_accuracies,
                        'train_losses':train_losses,
                        'test_losses':test_losses,
                        'iters':iters,
                        'args':args,
                        'torch_rng_state': torch.get_rng_state(),
                        'numpy_rng_state': np.random.get_state(),
                        'random_rng_state': random.getstate(),
                        } , f'{args.log_dir}/checkpoint_{bi}.pt')

                pbar.update(1)



if __name__=='__main__':
        main()
        # Sweep configuration for wandb
        # sweep_configuration = {
        #     "program": "parity_baseline_mlp.py",
        #     "name": "ctm-parity",
        #     "method": "random",
        #     "metric": {
        #         "name": "Train/Losses",
        #         "goal": "minimize"# todo decide if maximize accuracies or minimize loss and add iterations and memory length
        #     },
        #     "parameters": {
        #         "batch_size": {"values": [64]},
        #         "learning_rate": {"min": 1e-4, "max": 3e-4},
        #         "use_amp": {"values": [True]},
        #         "use_scheduler": {"values": [True]},
        #         "memory_length": {"values": [75]},
        #         "internal_ticks": {"values": [100]},
        #         "training_iterations": {"values": [200000]},
        #         # "postactivation_production": {"values": ["kan"]},
        #         # "model_type": {"values": ["ctm"]},
        #         # Todo"parity_sequence_length": {"values": [16, 64]},
        #     }
        # }
        #
        # sweep_id = wandb.sweep(sweep_configuration, project="ctm-parity")
        # wandb.agent(sweep_id, function=main, count=50)