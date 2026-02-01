import argparse
import math
import multiprocessing
import random
import os
import sys
import matplotlib as mpl
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
from tqdm.auto import tqdm

# Adjust paths as necessary for your project structure
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from autoclip.torch import QuantileClip
from data.custom_datasets import ListOpsDataset
from utils.housekeeping import set_seed, zip_python_code
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup
from models.utils import get_latest_checkpoint

# Import your Baseline Model Preparer
# Ensure this imports the BaselineMLP or LSTM we defined previously
from tasks.parity.utils_efficient import prepare_baseline

mpl.use('Agg')
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='warn')
sns.set_style('darkgrid')

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')

torchvision.disable_beta_transforms_warning()
torch.serialization.add_safe_globals([argparse.Namespace])


def parse_args():
    parser = argparse.ArgumentParser(description="Train Baseline on ListOps Task")

    # --- Model Architecture ---
    parser.add_argument('--model_type', type=str, default="mlp", choices=['mlp', 'lstm'],
                        help='Type of baseline model.')
    parser.add_argument('--parity_sequence_length', type=int, default=2048, help='Max sequence length for ListOps.')
    parser.add_argument('--d_model', type=int, default=512, help='Hidden dimension size.')
    parser.add_argument('--output_dim', type=int, default=10, help='10 digits (0-9) for ListOps.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')

    # MLP Specifics (if using MLP baseline)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 512, 512],
                        help='Hidden layer sizes for MLP.')

    # --- Training Configuration ---
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--batch_size_test', type=int, default=256, help='Batch size for testing.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--training_iterations', type=int, default=100001, help='Number of training iterations.')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Number of warmup steps.')
    parser.add_argument('--use_scheduler', action=argparse.BooleanOptionalAction, default=True,
                        help='Use a learning rate scheduler.')
    parser.add_argument('--scheduler_type', type=str, default='multistep', choices=['multistep', 'cosine'],
                        help='Type of scheduler.')
    parser.add_argument('--milestones', type=int, default=[30000, 60000], nargs='+', help='Milestones for multistep.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for multistep.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay factor.')
    parser.add_argument('--gradient_clipping', type=float, default=1.0, help='Gradient clipping value.')

    # --- Housekeeping ---
    parser.add_argument('--log_dir', type=str, default='logs/listops_baseline', help='Directory for logging.')
    parser.add_argument('--dataset', type=str, default='listops', help='Dataset name.')
    parser.add_argument('--save_every', type=int, default=5000, help='Save checkpoint frequency.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--reload', action=argparse.BooleanOptionalAction, default=True, help='Reload checkpoint?')
    parser.add_argument('--reload_model_only', action=argparse.BooleanOptionalAction, default=False,
                        help='Reload only weights?')
    parser.add_argument('--track_every', type=int, default=1000, help='Track metrics frequency.')
    parser.add_argument('--n_test_batches', type=int, default=50, help='Num batches for metrics approx.')
    parser.add_argument('--full_eval', action=argparse.BooleanOptionalAction, default=False,
                        help='Perform full evaluation?')
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help='GPU(s) or -1 for CPU.')
    parser.add_argument('--use_amp', action=argparse.BooleanOptionalAction, default=True, help='Use AMP autocast.')

    args = parser.parse_args()
    return args


def create_long_df(data_array, iters, metric_name='Accuracy'):
    """Converts numpy array (iters x runs) into a Long-Form DataFrame for Seaborn."""
    data = np.array(data_array)
    # If 1D, reshape to (iters, 1)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    df = pd.DataFrame(data, index=iters)
    df = df.reset_index().rename(columns={'index': 'Iteration'})
    df_long = df.melt(id_vars='Iteration', var_name='Run', value_name=metric_name)
    return df_long


def compute_metrics(dataloader, model, device, loss_fn, limit_batches=-1):
    """
    Helper to compute aggregated metrics over a dataset to avoid 'last-batch' bias.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward
            outputs = model(inputs)  # Baseline returns only predictions

            # Loss
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            # Accuracy
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

            if limit_batches != -1 and i >= limit_batches:
                break

    avg_loss = total_loss / (i + 1)
    avg_acc = correct / total if total > 0 else 0
    return avg_loss, avg_acc


def main():
    args = parse_args()
    print(f"Parsed args: {args}\n")
    set_seed(args.seed)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --- 1. Dataset Setup ---
    # Adjust filenames if needed
    csv_train_file = 'tasks/listops/dataset/train_d20s.tsv'
    csv_test_file = 'tasks/listops/dataset/test_d20s.tsv'

    if not os.path.exists('tasks/listops/dataset'):
        print("Warning: Dataset folder missing.")
        # Add logic here to download/generate if missing

    train_data = ListOpsDataset(csv_train_file)
    test_data = ListOpsDataset(csv_test_file)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=False, num_workers=0)

    # --- 2. Device Setup ---
    if args.device[0] != -1:
        device = f'cuda:{args.device[0]}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Running model {args.model_type} on {device}')

    # --- 3. Model Setup ---
    # prepare_baseline should return your MLP/LSTM instance
    model = prepare_baseline(args=args, device=device)
    model.train()

    # Calculate Params
    print(f'Total params: {sum(p.numel() for p in model.parameters())}')

    # --- 4. Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Optional Gradient Clipping Wrapper
    if args.gradient_clipping != -1:
        optimizer = QuantileClip.as_optimizer(optimizer=optimizer, quantile=args.gradient_clipping, history_length=1000)

    warmup_schedule = warmup(args.warmup_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_schedule.step)

    if args.use_scheduler:
        if args.scheduler_type == 'multistep':
            scheduler = WarmupMultiStepLR(optimizer, warmup_steps=args.warmup_steps, milestones=args.milestones,
                                          gamma=args.gamma)
        elif args.scheduler_type == 'cosine':
            scheduler = WarmupCosineAnnealingLR(optimizer, args.warmup_steps, args.training_iterations,
                                                warmup_start_lr=1e-20, eta_min=1e-7)

    scaler = torch.amp.GradScaler("cuda" if "cuda" in device else "cpu", enabled=args.use_amp)
    criterion = nn.CrossEntropyLoss()  # Standard Loss for Baselines

    # --- 5. State Initialization ---
    start_iter = 0
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    iters = []

    # Reload Logic
    if args.reload and (latest_checkpoint_path := get_latest_checkpoint(args.log_dir)):
        print(f'Reloading from: {latest_checkpoint_path}')
        checkpoint = torch.load(f'{latest_checkpoint_path}', weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)

        if not args.reload_model_only:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_iter = checkpoint['iteration']
            train_losses = checkpoint['train_losses']
            test_losses = checkpoint['test_losses']
            train_accuracies = checkpoint['train_accuracies']
            test_accuracies = checkpoint['test_accuracies']
            iters = checkpoint['iters']

            if 'torch_rng_state' in checkpoint:
                torch.set_rng_state(checkpoint['torch_rng_state'].cpu().byte())
                np.random.set_state(checkpoint['numpy_rng_state'])
                random.setstate(checkpoint['random_rng_state'])

    # --- 6. Training Loop ---
    iterator = iter(trainloader)

    with tqdm(total=args.training_iterations, initial=start_iter, dynamic_ncols=True) as pbar:
        for bi in range(start_iter, args.training_iterations):
            current_lr = optimizer.param_groups[-1]['lr']

            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                inputs, targets = next(iterator)

            inputs, targets = inputs.to(device), targets.to(device)

            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu", enabled=args.use_amp):
                # Baseline Forward: returns only predictions
                predictions = model(inputs)

                # Baseline Loss: Standard CrossEntropy
                loss = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            # Live progress bar updates (Approximation)
            acc_batch = (predictions.argmax(1) == targets).float().mean().item()
            pbar.set_description(f'Loss={loss.item():.3f} | Acc={acc_batch:.3f} | LR={current_lr:.2e}')

            # --- 7. Metrics & Plotting ---
            if bi % args.track_every == 0 and bi != 0:
                iters.append(bi)

                # Compute Train Metrics (Average over subset)
                t_loss, t_acc = compute_metrics(trainloader, model, device, criterion, args.n_test_batches)
                train_losses.append(t_loss)
                train_accuracies.append(t_acc)

                # Compute Test Metrics (Average over subset)
                v_loss, v_acc = compute_metrics(testloader, model, device, criterion, args.n_test_batches)
                test_losses.append(v_loss)
                test_accuracies.append(v_acc)

                model.train()  # Important: Switch back to train mode

                # --- Plotting ---
                sns.set_theme(style="whitegrid")

                # Data Prep
                df_acc_train = create_long_df(train_accuracies, iters, 'Accuracy')
                df_acc_test = create_long_df(test_accuracies, iters, 'Accuracy')
                df_loss_train = create_long_df(train_losses, iters, 'Loss')
                df_loss_test = create_long_df(test_losses, iters, 'Loss')

                # Plot 1: Accuracies
                fig_acc, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

                sns.lineplot(data=df_acc_train, x='Iteration', y='Accuracy', ax=ax1, color='tab:blue',
                             label='Train Acc')
                sns.lineplot(data=df_acc_test, x='Iteration', y='Accuracy', ax=ax2, color='tab:green', label='Test Acc')

                # ListOps Baseline: 10 classes -> 0.1 Accuracy
                ax1.axhline(0.1, color='gray', linestyle=':', label='Random (0.1)')
                ax2.axhline(0.1, color='gray', linestyle=':', label='Random (0.1)')

                ax1.set_ylim([-0.05, 1.05])
                ax2.set_ylim([-0.05, 1.05])
                ax1.set_title('Train Accuracy');
                ax2.set_title('Test Accuracy')
                fig_acc.savefig(f'{args.log_dir}/accuracies.png')
                plt.close(fig_acc)

                # Plot 2: Losses
                fig_loss, ax_loss = plt.subplots(figsize=(10, 5))

                sns.lineplot(data=df_loss_train, x='Iteration', y='Loss', ax=ax_loss, color='blue', label='Train Loss')
                sns.lineplot(data=df_loss_test, x='Iteration', y='Loss', ax=ax_loss, color='red', label='Test Loss')

                # ListOps Baseline: ln(10) ~ 2.302
                ax_loss.axhline(np.log(10), color='gray', linestyle=':', label='Random Loss (ln 10)')

                ax_loss.set_title('Losses')
                fig_loss.savefig(f'{args.log_dir}/losses.png')
                plt.close(fig_loss)

            # --- 8. Checkpointing ---
            if (bi % args.save_every == 0 or bi == args.training_iterations - 1):
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'iteration': bi,
                    'train_accuracies': train_accuracies,
                    'test_accuracies': test_accuracies,
                    'train_losses': train_losses,
                    'test_losses': test_losses,
                    'iters': iters,
                    'args': args,
                    'torch_rng_state': torch.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate(),
                }, f'{args.log_dir}/checkpoint_{bi}.pt')

                pbar.update(1)


if __name__ == "__main__":
    main()