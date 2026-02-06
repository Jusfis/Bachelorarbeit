import torch
import torch.nn as nn
import torch.optim as optim


class BaselineMLP(nn.Module):
    def __init__(self, in_dim=64, h_dims=[1024, 512], out_dim=2):
        super().__init__()
        # 2 logits for each sequence position (0 or 1)
        self.seq_len = in_dim

        layers = []
        curr_dim = in_dim
        for h in h_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.ReLU())
            curr_dim = h

        # Layer gives 128 logits for 64 positions (2 logits each)
        self.network = nn.Sequential(*layers)
        self.classifier = nn.Linear(curr_dim, self.seq_len * 2)

    def forward(self, x):
        # x: [32, 64]
        features = self.network(x)
        logits = self.classifier(features)  # [32, 128]

        # transform from [Batch, Seq_Len, Classes] -> [32, 64, 2]
        return logits.view(-1, self.seq_len, 2)

#
# class SequentialBaselineMLP(nn.Module):
#
#
#     def __init__(self, hidden_dim=512):
#         super().__init__()
#         #[act_pix. logits from before...]
#         self.fc1 = nn.Linear(1, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, 2)  # outpu: Logits for Parität 0 und 1
#         self.relu = nn.ReLU()
#
#
#     def forward(self, x):
#         # x: [Batch, 64]
#         batch_size, seq_len = x.size()
#         x = x.unsqueeze(-1).float()  # [Batch, 64, 1]
#
#         # use on every time step window
#         out = self.relu(self.fc1(x))
#         out = self.relu(self.fc2(out))
#         logits = self.fc3(out)  # [Batch, 64, 2]
#
#         return logits