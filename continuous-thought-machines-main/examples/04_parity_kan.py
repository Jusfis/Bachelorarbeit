import os
import sys

import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time
import wandb
# import imageio



# Add parent directory to path to access the CTM package
# Assumes this script is run from the 'examples' folder or similar depth
sys.path.append("..")

# Try importing CTM modules. 
# If these fail, ensure you are running this script from the correct directory 
# relative to the 'continuous-thought-machines' repo.
try:
    from models.ctm_kan import ContinuousThoughtMachine as CTM
    from tasks.parity.plotting import make_parity_gif
    from tasks.parity.utils import reshape_attention_weights, reshape_inputs
except ImportError as e:
    print("Error importing CTM modules. Make sure you are running this script inside the CTM repository structure.")
    print(f"Details: {e}")
    sys.exit(1)


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False


def make_pbar_desc(train_loss, train_accuracy, test_loss, test_accuracy, lr, where_most_certain):
    """A helper function to create a description for the tqdm progress bar"""
    pbar_desc = f'Train Loss={train_loss:0.3f}. Train Acc={train_accuracy:0.3f}. Test Loss={test_loss:0.3f}. Test Acc={test_accuracy:0.3f}. LR={lr:0.6f}.'
    pbar_desc += f' Where_certain={where_most_certain.float().mean().item():0.2f}+-{where_most_certain.float().std().item():0.2f} ({where_most_certain.min().item():d}<->{where_most_certain.max().item():d}).'
    return pbar_desc


def update_training_curve_plot(fig, ax1, ax2, train_losses, test_losses, train_accuracies, test_accuracies, steps,
                               log_dir):
    """Saves the training curve plot to disk instead of displaying inline."""

    # Plot loss
    ax1.clear()
    ax1.plot(range(len(train_losses)), train_losses, 'b-', alpha=0.7, label=f'Train Loss: {train_losses[-1]:.3f}')
    ax1.plot(steps, test_losses, 'r-', marker='o', label=f'Test Loss: {test_losses[-1]:.3f}')
    ax1.set_title('Loss')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot accuracy
    ax2.clear()
    ax2.plot(range(len(train_accuracies)), train_accuracies, 'b-', alpha=0.7,
             label=f'Train Accuracy: {train_accuracies[-1]:.3f}')
    ax2.plot(steps, test_accuracies, 'r-', marker='o', label=f'Test Accuracy: {test_accuracies[-1]:.3f}')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(log_dir, 'training_curves.png'))


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------

class ParityDataset(Dataset):
    def __init__(self, sequence_length=64, length=100000):
        self.sequence_length = sequence_length
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        vector = 2 * torch.randint(0, 2, (self.sequence_length,)) - 1
        vector = vector.float()
        negatives = (vector == -1).to(torch.long)
        cumsum = torch.cumsum(negatives, dim=0)
        target = (cumsum % 2 != 0).to(torch.long)
        return vector, target


# -----------------------------------------------------------------------------
# Loss Function
# -----------------------------------------------------------------------------

def parity_loss(predictions, certainties, targets, use_most_certain=True):
    """
    Computes the parity loss.

    Predictions are of shape: (B, parity_sequence_length, class, internal_ticks),
        where classes are in [0,1,2,3,4] for [Up, Down, Left, Right, Wait]
    Certainties are of shape: (B, 2, internal_ticks), 
        where the inside dimension (2) is [normalised_entropy, 1-normalised_entropy]
    Targets are of shape: [B, parity_sequence_length]

    use_most_certain will select either the most certain point or the final point. For baselines,
        the final point proved the only usable option. 
    """

    # Losses are of shape [B, parity_sequence_length, internal_ticks]
    losses = nn.CrossEntropyLoss(reduction='none')(predictions.flatten(0, 1),
                                                   torch.repeat_interleave(targets.unsqueeze(-1), predictions.size(-1),
                                                                           -1).flatten(0, 1).long()).reshape(
        predictions[:, :, 0].shape)

    # Average the loss over the parity sequence dimension
    losses = losses.mean(1)

    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:, 1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1

    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()

    loss = (loss_minimum_ce + loss_selected) / 2
    return loss, loss_index_2


# -----------------------------------------------------------------------------
# Visualization Helper
# -----------------------------------------------------------------------------

def create_viz(model, testloader, device, log_dir, grid_size):
    """Generates and saves the prediction GIF."""
    model.eval()
    with torch.no_grad():
        inputs_viz, targets_viz = next(iter(testloader))
        inputs_viz = inputs_viz.to(device)
        targets_viz = targets_viz.to(device)

        predictions_raw, certainties, _, pre_activations, post_activations, attention = model(inputs_viz, track=True)

        # Reshape predictions
        predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 2, predictions_raw.size(-1))

        attention = reshape_attention_weights(attention)
        inputs = reshape_inputs(inputs_viz, 50, grid_size=grid_size)

        # Generate the parity GIF
        save_path = f'{log_dir}/prediction.gif'
        make_parity_gif(
            predictions.detach().cpu().numpy(),
            certainties.detach().cpu().numpy(),
            targets_viz.detach().cpu().numpy(),
            pre_activations,
            post_activations,
            attention,
            inputs,
            save_path,
        )
        print(f"Saved visualization to {save_path}")


# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

def train(model, trainloader, testloader, grid_size, device='cpu', training_iterations=100000, test_every=1000, lr=1e-4,
          log_dir='./logs'):
    os.makedirs(log_dir, exist_ok=True)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    iterator = iter(trainloader)

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    steps = []

    # Setup figure for static saving
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    with tqdm(total=training_iterations) as pbar:
        for stepi in range(training_iterations):

            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                inputs, targets = next(iterator)

            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            predictions_raw, certainties, _ = model(inputs)

            # Reshape: (B, SeqLength, C, T)
            predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 2, predictions_raw.size(-1))

            # Compute loss
            train_loss, where_most_certain = parity_loss(predictions, certainties, targets, use_most_certain=True)
            train_accuracy = (predictions.argmax(2)[torch.arange(predictions.size(0), device=predictions.device), :,
                              where_most_certain] == targets).float().mean().item()

            train_losses.append(train_loss.item())
            train_accuracies.append(train_accuracy)

            # log to wandb
            wandb.log({"loss/train": train_loss.item(), "accuracy/train": train_accuracy, "step": stepi})


            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Evaluation Step
            if stepi % test_every == 0 or stepi == 0 or stepi == training_iterations - 1:
                model.eval()
                with torch.no_grad():
                    all_test_predictions = []
                    all_test_targets = []
                    all_test_where_most_certain = []
                    all_test_losses = []

                    # Run over a subset or full test loader
                    # Using a limited number of batches for speed if dataset is large, 
                    # but here testloader is small enough.
                    for inputs, targets in testloader:
                        inputs = inputs.to(device)
                        targets = targets.to(device)

                        predictions_raw, certainties, where_most_certain = model(inputs)
                        predictions = predictions_raw.reshape(predictions_raw.size(0), -1, 2, predictions_raw.size(-1))

                        test_loss, where_most_certain = parity_loss(predictions, certainties, targets,
                                                                    use_most_certain=True)
                        all_test_losses.append(test_loss.item())
                        all_test_predictions.append(predictions)
                        all_test_targets.append(targets)
                        all_test_where_most_certain.append(where_most_certain)

                    all_test_predictions = torch.cat(all_test_predictions, dim=0)
                    all_test_targets = torch.cat(all_test_targets, dim=0)
                    all_test_where_most_certain = torch.cat(all_test_where_most_certain, dim=0)

                    test_accuracy = (all_test_predictions.argmax(2)[
                                     torch.arange(all_test_predictions.size(0), device=predictions.device), :,
                                     all_test_where_most_certain] == all_test_targets).float().mean().item()
                    test_loss = sum(all_test_losses) / len(all_test_losses)

                    test_losses.append(test_loss)
                    test_accuracies.append(test_accuracy)
                    steps.append(stepi)

                    # Save visualization
                    create_viz(model, testloader, device, log_dir, grid_size)

                model.train()

                # Save plot to disk
                update_training_curve_plot(fig, ax1, ax2, train_losses, test_losses, train_accuracies, test_accuracies,
                                           steps, log_dir)

            pbar_desc = make_pbar_desc(train_loss=train_loss.item(), train_accuracy=train_accuracy, test_loss=test_loss,
                                       test_accuracy=test_accuracy, lr=optimizer.param_groups[-1]["lr"],
                                       where_most_certain=where_most_certain)
            pbar.set_description(pbar_desc)
            pbar.update(1)

    plt.close(fig)
    return model


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    # Configuration
    GRID_SIZE = 4
    PARITY_SEQUENCE_LENGTH = GRID_SIZE ** 2
    BATCH_SIZE = 20
    ITERATIONS = 10000
    LOG_DIR = './parity_logs_kan'
    LEARNINGRATE = 1e-2
    set_seed(42)

# Initialize WandB
    run = wandb.init(
        project="ctm-parity-kan",
        entity="justus-fischer-ludwig-maximilian-university-of-munich",

        config={
            "learning_rate": LEARNINGRATE,
            "architecture": "CTM-KAN",
            "dataset": "Parity",
            "batch_size": BATCH_SIZE,
            "iterations": ITERATIONS,
            "parity_sequence_length": PARITY_SEQUENCE_LENGTH,
        },
    )




    print("Initializing Data...")
    train_data = ParityDataset(sequence_length=PARITY_SEQUENCE_LENGTH, length=100000)
    test_data = ParityDataset(sequence_length=PARITY_SEQUENCE_LENGTH, length=10000)

    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Initializing Model...")
    model = CTM(
        iterations=20,
        d_model=32,
        d_input=32,
        heads=8,
        n_synch_out= 32,
        n_synch_action=32,
        synapse_depth=8,
        memory_length=16,
        deep_nlms=True,
        memory_hidden_dims=16,
        backbone_type='parity_backbone',
        out_dims=PARITY_SEQUENCE_LENGTH * 2,
        prediction_reshaper=[PARITY_SEQUENCE_LENGTH, 2],
        dropout=0.0,
        do_layernorm_nlm=False,
        positional_embedding_type='custom-rotational-1d'
    ).to(device)

    # Initialize model parameters with dummy forward pass
    sample_batch = next(iter(trainloader))
    dummy_input = sample_batch[0][:1].to(device)
    with torch.no_grad():
        _ = model(dummy_input)

    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    print("Starting Training...")
    start = time.time()
    model = train(
        model=model,
        trainloader=trainloader,
        testloader=testloader,
        grid_size=GRID_SIZE,
        device=device,
        training_iterations=ITERATIONS,
        lr=LEARNINGRATE,
        log_dir=LOG_DIR
    )

    end = time.time()
    duration = end - start
    print("Training Complete. Time: duration {:.2f} seconds ({:.2f} minutes)".format(duration, duration / 60))
    print(f"Check {LOG_DIR} for logs and visualizations.")
    run.finish()


if __name__ == "__main__":
    main()
