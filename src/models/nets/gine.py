import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.data import Data
from torch import nn
from src.models.encoders.moleculenet_encoders import (
    MoleculeNetAtomEncoder,
    MoleculeNetBondEncoder,
)
from src.models.encoders.ogbg_encoders import OGBGAtomEncoder, OGBGBondEncoder


class GINE(nn.Module):
    """
    Graph Isomorphism Network with Edge attributes (GINE) for molecular property prediction.
    
    Supports both MoleculeNet and OGB datasets with proper feature encoding.
    
    Args:
        dataset: Dataset name ('moleculenet' or 'ogbg-molpcba').
        embedding_dim: Dimension of atom/bond embeddings.
        hidden_channels: Hidden dimension for GNN layers.
        num_tasks: Number of prediction tasks.
        num_layers: Number of GNN layers (default: 2).
        dropout: Dropout probability.
        pool: Global pooling function (default: global_add_pool).
    """
    
    def __init__(
        self,
        dataset: str = "moleculenet",
        embedding_dim: int = 16,
        hidden_channels: int = 128,
        num_tasks: int = 1,
        num_layers: int = 2,
        dropout: float = 0.5,
       
    ):
        super().__init__()
        
        # Validate dataset
        if dataset not in ["moleculenet", "ogbg-molpcba"]:
            raise ValueError(
                f"Dataset '{dataset}' not supported. "
                f"Choose 'moleculenet' or 'ogbg-molpcba'."
            )
        
        self.dataset = dataset
        self.embedding_dim = embedding_dim
        self.hidden_channels = hidden_channels
        self.num_tasks = num_tasks
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initialize encoders based on dataset
        if dataset == "moleculenet":
            self.atom_encoder = MoleculeNetAtomEncoder(embedding_dim)
            self.bond_encoder = MoleculeNetBondEncoder(embedding_dim)
        else:  # ogbg-molpcba
            self.atom_encoder = OGBGAtomEncoder(embedding_dim)
            self.bond_encoder = OGBGBondEncoder(embedding_dim)
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_channels
            self.convs.append(
                GINEConv(
                    nn=nn.Sequential(
                        nn.Linear(in_dim, hidden_channels),
                        nn.ELU(),
                        nn.Linear(hidden_channels, hidden_channels),
                    ),
                    edge_dim=embedding_dim,
                )
            )
        
        # Pooling
        self.pool = global_add_pool
        
        # Prediction head
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_tasks)
    
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: PyG Data object with x, edge_index, edge_attr, batch.
            
        Returns:
            Predictions [batch_size, num_tasks]
        """
        # Encode features
        x = self.atom_encoder(data.x)
        edge_attr = self.bond_encoder(data.edge_attr)
        edge_index = data.edge_index
        batch = data.batch
        
        # GNN layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        h_graph = self.pool(x, batch)
        
        # Prediction head
        h_graph = self.lin1(h_graph)
        h_graph = F.relu(h_graph)
        h_graph = F.dropout(h_graph, p=self.dropout, training=self.training)
        out = self.lin2(h_graph)
        
        return out
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  dataset={self.dataset},\n"
            f"  embedding_dim={self.embedding_dim},\n"
            f"  hidden_channels={self.hidden_channels},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_tasks={self.num_tasks},\n"
            f"  dropout={self.dropout}\n"
            f")"
        )