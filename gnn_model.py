import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class MoleculeGNN(torch.nn.Module):

    def __init__(self):

        super(MoleculeGNN, self).__init__()

        self.conv1 = GCNConv(10, 64)
        self.conv2 = GCNConv(64, 128)

        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x