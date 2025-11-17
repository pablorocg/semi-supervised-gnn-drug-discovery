import torch
import torch.nn.functional as F
from torch_geometric.nn import MLP, ResGatedGraphConv, global_mean_pool
import torch.nn as nn

class RGG(torch.nn.Module):
    """Simple ResGatedGraphConv model for graph classification."""

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()

        # First layer
        self.convs.append(ResGatedGraphConv(in_channels, hidden_channels))

        #   Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(ResGatedGraphConv(hidden_channels, hidden_channels))
        self.mlp = MLP(
            [hidden_channels, hidden_channels, out_channels], norm=None, dropout=0.5
        )
        self.gelu = nn.GELU()

    def forward(self, data):
        x, edge_index, batch, batch_size = (
            data.x,
            data.edge_index,
            data.batch,
            data.num_graphs,
        )

        for conv in self.convs:
            x = conv(x, edge_index) 
            x = self.gelu(x)
        x = global_mean_pool(x, batch, size=batch_size)  # [batch_size, hidden_channels]
        x = self.mlp(x)  # [batch_size, out_channels]
        return x

