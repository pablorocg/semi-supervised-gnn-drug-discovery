import torch
from torch_geometric.nn import GCNConv, global_mean_pool, MLP
import torch.nn as nn

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = GCNConv(in_channels, hidden_channels)
            self.convs.append(conv)
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels], norm=None, dropout=0.5)
        self.gelu = nn.GELU()

    def forward(self, data):
        # Ensure x is float, edge_index is long as required by GCNConv
        x, edge_index, batch, batch_size = (
            data.x.float(),              # <-- ensure float type
            data.edge_index.long(),      # <-- ensure long type
            data.batch,
            data.num_graphs,
        )

        for conv in self.convs:
            x = conv(x, edge_index)
            x = self.gelu(x)

        x = global_mean_pool(x, batch, size=batch_size)
        x = self.mlp(x)
        return x