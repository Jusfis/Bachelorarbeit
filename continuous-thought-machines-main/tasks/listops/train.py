import argparse
import math
import random # Used for saving/loading RNG state
import os
import sys
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

from tasks.listops.utils  import prepare_model
from autoclip.torch import QuantileClip # Used for gradient clipping
from data.custom_datasets import ListOpsDataset
from tasks.image_classification.plotting import plot_neural_dynamics
from models.utils import reshape_predictions, get_latest_checkpoint

# from tasks.listops import prepare_model, reshape_attention_weights, reshape_inputs
from utils.housekeeping import set_seed, zip_python_code
# from utils.losses import listops_loss
from utils.schedulers import WarmupCosineAnnealingLR, WarmupMultiStepLR, warmup
import wandb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

torchvision.disable_beta_transforms_warning()
torch.serialization.add_safe_globals([argparse.Namespace])

def parse_args():
    parser = argparse.ArgumentParser(description="Train CTM on Listops Task")

    # Model Architecture
    parser.add_argument('--model_type', type=str, default="ctm", choices=['ctm', 'lstm'], help='The type of model to train.')
    parser.add_argument('--listops_size', type=int, default=64, help='Size parameter for listops task.')
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
    parser.add_argument('--log_dir', type=str, default='logs/listops', help='Directory for logging.')
    parser.add_argument('--dataset', type=str, default='listops', help='Dataset name (used for assertion).')
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


def main():
    args = parse_args()
    #print(f"Using config: {config}\n Parsed args: {args}\n")
    print(f"Parsed args: {args}\n")
    set_seed(args.seed)

    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)


    csv_train_file='tasks/listops/dataset/train_d20s.tsv'
    csv_test_file='tasks/listops/dataset/test_d20s.tsv'
    if not os.path.exists('tasks/listops/dataset'):
        os.makedirs('tasks/listops/dataset')
        print("Fehler, dataset folder is missing, created dataset")
    train_data = ListOpsDataset(csv_train_file)
    test_data = ListOpsDataset(csv_test_file)
    # loads batches from train_data and test_data which are insctances of torch.utils.data.Dataset
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size_test, shuffle=False, num_workers=0, drop_last=False)

    prediction_reshaper = [1, 10]
    # da ZMOD 10Z Restklasse
    args.out_dims = 10
    pass
    # todo implement baseline or keep?

    args.use_most_certain = args.model_type == "ctm" or (args.use_most_certain_with_lstm and args.model_type == "lstm")

    # Set device mps for mac cuda for nvidia else cpu
    if args.device[0] != -1:
        device = f'cuda:{args.device[0]}'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Running model {args.model_type} on {device}')

    # Build model
    #prediction_reshaper was genau?
    model = prepare_model(prediction_reshaper,args, device)
    model.train()

    # as per usual: lazy modules so that we can get param count
    pseudo_inputs = train_data.__getitem__(0)[0].unsqueeze(0).to(device)
    model(pseudo_inputs)

    print(f'Total params: {sum(p.numel() for p in model.parameters())}')





if __name__ == "__main__":

        main()