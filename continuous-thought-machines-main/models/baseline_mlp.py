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
            linear_layer = nn.Linear(current_dim, h_dim)

            # Initialization: Specifically for ReLU/LeakyReLU to prevent dead neurons
            nn.init.kaiming_normal_(linear_layer.weight, nonlinearity='relu')
            nn.init.constant_(linear_layer.bias, 0)

            layers.append(linear_layer)
            layers.append(activation)  # Instantiate unique instance
            current_dim = h_dim

            # Output Layer
        final_layer = nn.Linear(current_dim, output_dim)
        # Xavier initialization is often preferred for the final layer before a Softmax/Logit
        nn.init.xavier_normal_(final_layer.weight)
        layers.append(final_layer)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Flatten input if necessary (e.g., if input is [Batch, Sequence])
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)


class SequentialBaselineMLP(nn.Module):


    def __init__(self, hidden_dim=512):
        super().__init__()
        #[act_pix. logits from before...]
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)  # outpu: Logits for Parität 0 und 1
        self.relu = nn.ReLU()


    def forward(self, x):
        # x: [Batch, 64]
        batch_size, seq_len = x.size()
        x = x.unsqueeze(-1).float()  # [Batch, 64, 1]

        # use on every time step window
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        logits = self.fc3(out)  # [Batch, 64, 2]

        return logits