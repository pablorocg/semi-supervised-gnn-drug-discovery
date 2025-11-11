import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class GCN(torch.nn.Module):
    """Simple GCN model for graph classification."""

    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        
        x, edge_index, batch, batch_size = (
            data.x,
            data.edge_index,
            data.batch,
            data.num_graphs,
        )

        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch, size=batch_size)  # [batch_size, hidden_channels]

        x = self.linear(x)  # [batch_size, 1] 
        return x
