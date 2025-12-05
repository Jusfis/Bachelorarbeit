import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import random
from scipy.special import softmax
import math
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import seaborn as sns
import imageio
import mediapy

import torch._dynamo as dynamo

dynamo.config.suppress_errors = False  # Raise errors on fallback


def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False

class Identity(nn.Module):
    """Identity Module."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Squeeze(nn.Module):
    """Squeeze Module."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

class SuperLinear(nn.Module):
    """SuperLinear Layer: Implements Neuron-Level Models (NLMs) for the CTM."""
    def __init__(self, in_dims, out_dims, N, dropout=0.0):
        super().__init__()
        self.in_dims = in_dims
        self.dropout = nn.Dropout(dropout) if dropout > 0 else Identity()
        self.register_parameter('w1', nn.Parameter(
            torch.empty((in_dims, out_dims, N)).uniform_(
                -1/math.sqrt(in_dims + out_dims),
                 1/math.sqrt(in_dims + out_dims)
            ), requires_grad=True)
        )
        self.register_parameter('b1', nn.Parameter(torch.zeros((1, N, out_dims)), requires_grad=True))

    def forward(self, x):
            out = self.dropout(x)
            out = torch.einsum('BDM,MHD->BDH', out, self.w1) + self.b1
            out = out.squeeze(-1)
            return out


def compute_normalized_entropy(logits, reduction='mean'):
    """Computes the normalized entropy for certainty-loss."""
    preds = F.softmax(logits, dim=-1)
    log_preds = torch.log_softmax(logits, dim=-1)
    entropy = -torch.sum(preds * log_preds, dim=-1)
    num_classes = preds.shape[-1]
    max_entropy = torch.log(torch.tensor(num_classes, dtype=torch.float32))
    normalized_entropy = entropy / max_entropy
    if len(logits.shape)>2 and reduction == 'mean':
        normalized_entropy = normalized_entropy.flatten(1).mean(-1)
    return normalized_entropy

class ContinuousThoughtMachine(nn.Module):
    def __init__(self,
                 iterations,
                 d_model,
                 d_input,
                 memory_length,
                 heads,
                 n_synch_out,
                 n_synch_action,
                 out_dims,
                 memory_hidden_dims,
                 dropout=0.0
                 ):
        super(ContinuousThoughtMachine, self).__init__()

        # --- Core Parameters ---
        self.iterations = iterations
        self.d_model = d_model
        self.d_input = d_input
        self.memory_length = memory_length
        self.n_synch_out = n_synch_out
        self.n_synch_action = n_synch_action
        self.out_dims = out_dims
        self.memory_length = memory_length
        self.memory_hidden_dims = memory_hidden_dims

        # --- Input Processing  ---
        self.backbone = nn.Sequential(
            nn.LazyConv2d(d_input, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_input),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.LazyConv2d(d_input, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(d_input),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.attention = nn.MultiheadAttention(self.d_input, heads, dropout, batch_first=True)
        self.kv_proj = nn.Sequential(nn.LazyLinear(self.d_input), nn.LayerNorm(self.d_input))
        self.q_proj = nn.LazyLinear(self.d_input)

        # --- Core CTM Modules ---
        self.synapses = nn.Sequential(
                nn.Dropout(dropout),
                nn.LazyLinear(d_model * 2),
                nn.GLU(),
                nn.LayerNorm(d_model)
            )
        self.trace_processor = nn.Sequential(
            SuperLinear(in_dims=memory_length, out_dims=2 * memory_hidden_dims, N=d_model, dropout=dropout),
            nn.GLU(),
            SuperLinear(in_dims=memory_hidden_dims, out_dims=2, N=d_model, dropout=dropout),
            nn.GLU(),
            Squeeze(-1)
        )

        #  --- Start States ---
        self.register_parameter('start_activated_state', nn.Parameter(
                torch.zeros((d_model)).uniform_(-math.sqrt(1/(d_model)), math.sqrt(1/(d_model))),
                requires_grad=True
            ))

        self.register_parameter('start_trace', nn.Parameter(
            torch.zeros((d_model, memory_length)).uniform_(-math.sqrt(1/(d_model+memory_length)), math.sqrt(1/(d_model+memory_length))),
            requires_grad=True
        ))

        # --- Synchronisation ---
        self.synch_representation_size_action = (self.n_synch_action * (self.n_synch_action+1))//2
        self.synch_representation_size_out = (self.n_synch_out * (self.n_synch_out+1))//2

        for synch_type, size in [('action', self.synch_representation_size_action), ('out', self.synch_representation_size_out)]:
            print(f"Synch representation size {synch_type}: {size}")

        self.set_synchronisation_parameters('out')
        self.set_synchronisation_parameters('action')

        # --- Output Procesing ---
        self.output_projector = nn.Sequential(nn.LazyLinear(self.out_dims))

    def set_synchronisation_parameters(self, synch_type: str):
        synch_representation_size = self.synch_representation_size_action if synch_type == 'action' else self.synch_representation_size_out
        self.register_parameter(f'decay_params_{synch_type}', nn.Parameter(torch.zeros(synch_representation_size), requires_grad=True))


    def compute_synchronisation(self, activated_state, decay_alpha, decay_beta, r, synch_type):
        if synch_type == 'action':
            n_synch = self.n_synch_action
            selected_left = selected_right = activated_state[:, -n_synch:]
        elif synch_type == 'out':
            n_synch = self.n_synch_out
            selected_left = selected_right = activated_state[:, :n_synch]

        outer = selected_left.unsqueeze(2) * selected_right.unsqueeze(1)
        i, j = torch.triu_indices(n_synch, n_synch)
        pairwise_product = outer[:, i, j]

        if decay_alpha is None or decay_beta is None:
            decay_alpha = pairwise_product
            decay_beta = torch.ones_like(pairwise_product)
        else:
            decay_alpha = r * decay_alpha + pairwise_product
            decay_beta = r * decay_beta + 1

        synchronisation = decay_alpha / (torch.sqrt(decay_beta))
        return synchronisation, decay_alpha, decay_beta

    def compute_features(self, x):
        input_features = self.backbone(x)
        kv = self.kv_proj(input_features.flatten(2).transpose(1, 2))
        return kv

    def compute_certainty(self, current_prediction):
        ne = compute_normalized_entropy(current_prediction)
        current_certainty = torch.stack((ne, 1-ne), -1)
        return current_certainty

    def forward(self, x, track=False):
        B = x.size(0)
        device = x.device

        # --- Tracking Initialization ---
        pre_activations_tracking = []
        post_activations_tracking = []
        synch_out_tracking = []
        synch_action_tracking = []
        attention_tracking = []

        # --- Featurise Input Data ---
        kv = self.compute_features(x)

        # --- Initialise Recurrent State ---
        state_trace = self.start_trace.unsqueeze(0).expand(B, -1, -1) # Shape: (B, H, T)
        activated_state = self.start_activated_state.unsqueeze(0).expand(B, -1) # Shape: (B, H)

        # --- Storage for outputs per iteration
        predictions = torch.empty(B, self.out_dims, self.iterations, device=device, dtype=x.dtype)
        certainties = torch.empty(B, 2, self.iterations, device=device, dtype=x.dtype)

        decay_alpha_action, decay_beta_action = None, None
        r_action, r_out = torch.exp(-self.decay_params_action).unsqueeze(0).repeat(B, 1), torch.exp(-self.decay_params_out).unsqueeze(0).repeat(B, 1)

        _, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, None, None, r_out, synch_type='out')

        # --- Recurrent Loop  ---
        for stepi in range(self.iterations):

            # --- Calculate Synchronisation for Input Data Interaction ---
            synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action')

            # --- Interact with Data via Attention ---
            q = self.q_proj(synchronisation_action).unsqueeze(1)
            attn_out, attn_weights = self.attention(q, kv, kv, average_attn_weights=False, need_weights=True)
            attn_out = attn_out.squeeze(1)
            pre_synapse_input = torch.concatenate((attn_out, activated_state), dim=-1)

            # --- Apply Synapses ---
            state = self.synapses(pre_synapse_input)
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)

            # --- Activate ---
            activated_state = self.trace_processor(state_trace)

            # --- Calculate Synchronisation for Output Predictions ---
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out')

            # --- Get Predictions and Certainties ---
            current_prediction = self.output_projector(synchronisation_out)
            current_certainty = self.compute_certainty(current_prediction)

            predictions[..., stepi] = current_prediction
            certainties[..., stepi] = current_certainty

            # --- Tracking ---
            if track:
                pre_activations_tracking.append(state_trace[:,:,-1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                attention_tracking.append(attn_weights.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())

        # --- Return Values ---
        if track:
            return predictions, certainties, (np.array(synch_out_tracking), np.array(synch_action_tracking)), np.array(pre_activations_tracking), np.array(post_activations_tracking), np.array(attention_tracking)
        return predictions, certainties, synchronisation_out


def get_loss(predictions, certainties, targets, use_most_certain=True):
    """use_most_certain will select either the most certain point or the final point."""

    losses = nn.CrossEntropyLoss(reduction='none')(predictions,
                                                   torch.repeat_interleave(targets.unsqueeze(-1), predictions.size(-1), -1))

    loss_index_1 = losses.argmin(dim=1)
    loss_index_2 = certainties[:,1].argmax(-1)
    if not use_most_certain:
        loss_index_2[:] = -1

    batch_indexer = torch.arange(predictions.size(0), device=predictions.device)
    loss_minimum_ce = losses[batch_indexer, loss_index_1].mean()
    loss_selected = losses[batch_indexer, loss_index_2].mean()

    loss = (loss_minimum_ce + loss_selected)/2
    return loss, loss_index_2

def calculate_accuracy(predictions, targets, where_most_certain):
    """Calculate the accuracy based on the prediction at the most certain internal tick."""
    B = predictions.size(0)
    device = predictions.device

    predictions_at_most_certain_internal_tick = predictions.argmax(1)[torch.arange(B, device=device), where_most_certain].detach().cpu().numpy()
    accuracy = (targets.detach().cpu().numpy() == predictions_at_most_certain_internal_tick).mean()

    return accuracy

def prepare_data():
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, num_workers=1)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=True, num_workers=1, drop_last=False)
    return trainloader, testloader


def update_training_curve_plot(fig, ax1, ax2, train_losses, test_losses, train_accuracies, test_accuracies, steps):
    clear_output(wait=True)

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

    out_dir = './training_mnist_normal_figs'
    os.makedirs(out_dir, exist_ok=True)
    step_label = steps[-1] if len(steps) > 0 else len(train_losses) - 1
    filename = os.path.join(out_dir, f'training_curve_step_{step_label}.jpg')
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    display(fig)


def train(model, trainloader, testloader, iterations, test_every, device):
    optimizer = torch.optim.AdamW(params=list(model.parameters()), lr=0.0001, eps=1e-8)
    iterator = iter(trainloader)
    model.train()

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    steps = []

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    with tqdm(total=iterations, initial=0, dynamic_ncols=True) as pbar:
        test_loss = None
        test_accuracy = None
        for stepi in range(iterations):

            try:
                inputs, targets = next(iterator)
            except StopIteration:
                iterator = iter(trainloader)
                inputs, targets = next(iterator)
            inputs, targets = inputs.to(device), targets.to(device)
            predictions, certainties, _ = model(inputs, track=False)
            train_loss, where_most_certain = get_loss(predictions, certainties, targets)
            train_accuracy = calculate_accuracy(predictions, targets, where_most_certain)

            train_losses.append(train_loss.item())
            train_accuracies.append(train_accuracy)

            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if stepi % test_every == 0 or stepi == iterations - 1:
                model.eval()
                with torch.inference_mode():
                    all_test_predictions = []
                    all_test_targets = []
                    all_test_where_most_certain = []
                    all_test_losses = []

                    for inputs, targets in testloader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        predictions, certainties, _ = model(inputs, track=False)
                        test_loss, where_most_certain = get_loss(predictions, certainties, targets)
                        all_test_losses.append(test_loss.item())

                        all_test_predictions.append(predictions)
                        all_test_targets.append(targets)
                        all_test_where_most_certain.append(where_most_certain)

                    all_test_predictions = torch.cat(all_test_predictions, dim=0)
                    all_test_targets = torch.cat(all_test_targets, dim=0)
                    all_test_where_most_certain = torch.cat(all_test_where_most_certain, dim=0)

                    test_accuracy = calculate_accuracy(all_test_predictions, all_test_targets,
                                                       all_test_where_most_certain)
                    test_loss = sum(all_test_losses) / len(all_test_losses)

                    test_losses.append(test_loss)
                    test_accuracies.append(test_accuracy)
                    steps.append(stepi)
                model.train()

                update_training_curve_plot(fig, ax1, ax2, train_losses, test_losses, train_accuracies, test_accuracies,
                                           steps)

            pbar.set_description(
                f'Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f} Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}')
            pbar.update(1)

    plt.ioff()
    plt.close(fig)
    return model






def main():
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    trainloader, testloader = prepare_data()

    model = ContinuousThoughtMachine(
        iterations=15,
        d_model=128,
        d_input=128,
        memory_length=10,
        heads=2,
        n_synch_out=16,
        n_synch_action=16,
        memory_hidden_dims=8,
        out_dims=10,
        dropout=0.0
    ).to(device)

    sample_batch = next(iter(trainloader))
    dummy_input = sample_batch[0][:1].to(device)
    with torch.no_grad():
        _ = model(dummy_input)

    # Now compile the model
    model = torch.compile(model)

    def clamp_decay_params(module, _input):
        with torch.no_grad():
            module.decay_params_action.data.clamp_(0, 15)
            module.decay_params_out.data.clamp_(0, 15)

    model.register_forward_pre_hook(clamp_decay_params)

    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    model = train(model=model, trainloader=trainloader, testloader=testloader, iterations=100000, test_every=10000,
                  device=device)





if __name__ == "__main__":
    main()
