from torch_geometric.nn import GINEConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GINEConv, global_mean_pool, global_add_pool, 
                                 global_max_pool, GlobalAttention)


class GINE(nn.Module):
    """Improved GINE with virtual nodes, layer normalization, and better pooling."""
    
    def __init__(self,
                 num_tasks,
                 num_node_features=9,
                 num_edge_features=3,
                 hidden_dim=256,
                 num_layers=5,
                 dropout=0.3,
                 use_residual=True,
                 use_virtual_node=True,
                 use_attention_pool=True,
                 num_mlp_layers=2,
                 use_layer_norm=True):  # NEW: option to use LayerNorm
        super().__init__()
        
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_residual = use_residual
        self.use_virtual_node = use_virtual_node
        self.use_attention_pool = use_attention_pool
        self.use_layer_norm = use_layer_norm
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()  # Can be BatchNorm or LayerNorm
        
        # Virtual node (learnable global node)
        if self.use_virtual_node:
            self.virtualnode_embedding = nn.Embedding(1, hidden_dim)
            self.virtualnode_mlps = nn.ModuleList()
            for _ in range(num_layers - 1):
                self.virtualnode_mlps.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim) if use_layer_norm else nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim) if use_layer_norm else nn.BatchNorm1d(hidden_dim),
                        nn.ReLU(),
                    )
                )
        
        # First layer
        mlp = self._make_mlp(num_node_features, hidden_dim, num_mlp_layers)
        self.convs.append(GINEConv(mlp, edge_dim=num_edge_features))
        self.norms.append(
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.BatchNorm1d(hidden_dim)
        )
        
        # Remaining layers
        for _ in range(num_layers - 1):
            mlp = self._make_mlp(hidden_dim, hidden_dim, num_mlp_layers)
            self.convs.append(GINEConv(mlp, edge_dim=num_edge_features))
            self.norms.append(
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.BatchNorm1d(hidden_dim)
            )
        
        # Attention-based pooling (optional but often better than simple pooling)
        if self.use_attention_pool:
            self.attention_pool = GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
            )
            pool_dim = hidden_dim * 4  # mean + add + max + attention
        else:
            pool_dim = hidden_dim * 3  # mean + add + max
        
        # Classifier with deeper MLP
        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2) if use_layer_norm else nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks)
        )
        
    def _make_mlp(self, in_dim, out_dim, num_layers=2):
        """Create MLP for GINEConv."""
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(in_dim, out_dim),
                    nn.LayerNorm(out_dim) if self.use_layer_norm else nn.BatchNorm1d(out_dim),
                    nn.ReLU()
                ])
            else:
                layers.extend([
                    nn.Linear(out_dim, out_dim),
                    nn.LayerNorm(out_dim) if self.use_layer_norm else nn.BatchNorm1d(out_dim),
                    nn.ReLU()
                ])
        return nn.Sequential(*layers)
    
    def forward(self, data):
        x = data.x.float()
        edge_index = data.edge_index
        edge_attr = data.edge_attr.float()
        batch = data.batch
        
        # Initialize virtual node
        if self.use_virtual_node:
            virtualnode_embedding = self.virtualnode_embedding(
                torch.zeros(batch[-1].item() + 1, dtype=torch.long, device=batch.device)
            )
        
        # GINE conv stack with optional residual connections and virtual node
        for layer_idx, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Add virtual node to node features ONLY after first layer
            if self.use_virtual_node and layer_idx > 0:
                x = x + virtualnode_embedding[batch]
    
            # Store for residual
            x_residual = x
            
            # GINEConv
            x = conv(x, edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            
            # Residual connection
            if self.use_residual and layer_idx > 0:
                x = x + x_residual
            
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # Update virtual node (aggregate node features)
            if self.use_virtual_node and layer_idx < len(self.convs) - 1:
                virtualnode_temp = global_add_pool(x, batch)
                virtualnode_embedding = self.virtualnode_mlps[layer_idx](
                    virtualnode_embedding + virtualnode_temp
                )
        
        # Multi-scale pooling
        x_mean = global_mean_pool(x, batch)
        x_add = global_add_pool(x, batch)
        x_max = global_max_pool(x, batch)
        
        if self.use_attention_pool:
            x_att = self.attention_pool(x, batch)
            x_pooled = torch.cat([x_mean, x_add, x_max, x_att], dim=1)
        else:
            x_pooled = torch.cat([x_mean, x_add, x_max], dim=1)
        
        # Classification
        return self.classifier(x_pooled)