import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.data import Data


class GINE(torch.nn.Module):
    """
    Modelo GNN basado en GINE para predicción multi-tarea, usando Edge Attributes (data.edge_attr)
    y la activación ELU en el Encoder GNN.
    """

    def __init__(
        self,
        num_node_features: int = 1,
        num_edge_features: int = 1,
        hidden_channels: int = 128,
        num_tasks: int = 1,
        dropout: float = 0.5,
    ):
        super(GINE, self).__init__()

        self.dropout = dropout

        self.conv1 = GINEConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(num_node_features, hidden_channels),
                torch.nn.ELU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            ),
            edge_dim=num_edge_features,
        )

        self.conv2 = GINEConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ELU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            ),
            edge_dim=num_edge_features,
        )

        self.pool = global_add_pool

        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, num_tasks)

    def forward(self, data: Data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        edge_attr = edge_attr.float()

        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)

        h_graph = self.pool(x, batch)
        h_graph = self.lin1(h_graph)
        h_graph = F.relu(h_graph)

        out = self.lin2(h_graph)

        return out
