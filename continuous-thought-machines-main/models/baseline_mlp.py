import torch
import torch.nn as nn
import torch.optim as optim


class BaselineMLPParity(nn.Module):
    def __init__(self, in_dim=64, h_dims=[1024, 512], out_dim=2):
        super().__init__()
        self.seq_len = in_dim

        layers = []
        curr_dim = in_dim
        # Build hidden layers in for loop
        for h in h_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            curr_dim = h

        # Layer gives 128 logits for 64 positions (2 logits each)
        # splat operator * to unpack list of layers into sequential
        self.network = nn.Sequential(*layers)
        # 128 or 32 logits for 64 or 16 positions (2 logits each)
        self.classifier = nn.Linear(curr_dim, self.seq_len * 2)

    def forward(self, x):
        # x: [32, 64], [32, 16] batch size 32, sequence length 64 or 16
        features = self.network(x)
        logits = self.classifier(features)  # [32, 128] or [32, 32]

        # Transform from [Batch, Seq_Len, Classes] -> [32, 64, 2]
        return logits.view(-1, self.seq_len, 2)



class BaselineMLPListops(nn.Module):
    def __init__(self, in_dim=64, h_dims=[1024, 512], out_dim=2):
        pass
    def forward(self, x):
        pass

class SimpleMLP(nn.Module):


    def __init__(self, input_dim, hidden_dim, out_dims, dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dims)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.net(x)