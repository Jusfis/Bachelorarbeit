import torch
import torch.nn as nn
import torch.optim as optim


class BaselineMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU()):
        """
        Args:
            input_dim (int): Number of input features (e.g., sequence length for parity).
            hidden_dims (list): List of integers defining hidden layer sizes.
            output_dim (int): Output size (usually 1 for binary parity or 2 for logits).
            activation (nn.Module): Activation function to use between layers.
        """
        super(BaselineMLP, self).__init__()

        layers = []
        current_dim = input_dim

        # Hidden Layers
        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(activation)
            current_dim = h_dim


        layers.append(nn.Linear(current_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten input if necessary (e.g., if input is [Batch, Sequence])
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)