from torch_geometric.nn import GINConv, GINEConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_max_pool

class GIN_GINE(nn.Module):
  def __init__(self,
               num_tasks,
               num_node_features=9,
               num_edge_features=3,
               hidden_dim=64,
               num_layers=5,
               dropout=0.5,
               use_edge_features=True):
    super(GIN_GINE, self).__init__()

    self.num_layers = num_layers
    self.dropout = dropout
    self.use_edge_features = use_edge_features

    self.convs = nn.ModuleList()
    self.batch_norms = nn.ModuleList()

    # First layer
    if use_edge_features:
      # FIne for first layer (uses edge features)
      nn1 = nn.Sequential(
          nn.Linear(num_node_features, hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, hidden_dim)
      )
      self.convs.append(GINEConv(nn1, edge_dim=num_edge_features))
    else:
      # Gin for first layer (no edge features)
      nn1 = nn.Sequential(
          nn.Linear(num_node_features, hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim, hidden_dim)
      )
      self.convs.append(GINConv(nn1))
    
    self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    # Hidden layers - alternate between GIN and GINE
    for i in range(num_layers - 1):
      if use_edge_features and i % 2 == 0:
        # GINE layers (even indices)
        nn_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINEConv(nn_layer, edge_dim=num_edge_features))
      else:
        # Gin layers (odd indices or when not using edge features)
        nn_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.convs.append(GINConv(nn_layer))
      self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
    
    # Classifier
    self.classifier = nn.Linear(hidden_dim * 2, num_tasks)

  def forward(self, data):
    x, edge_index, batch = data.x, data.edge_index, data.batch
    edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None

    x = x.float()
    if edge_attr is not None:
      edge_attr = edge_attr.float()

    # Graph convolutions
    for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
      # Use edge attributes for GINE layers
      if self.use_edge_features and isinstance(conv, GINEConv) and edge_attr is not None:
        x = conv(x, edge_index, edge_attr)
      else:
        x = conv(x, edge_index)
      
      x = bn(x)
      x = F.relu(x)
      x = F.dropout(x, p=self.dropout, training=self.training)
    
    # Global pooling
    x_add = global_add_pool(x, batch)
    x_max = global_max_pool(x, batch)
    x = torch.cat([x_add, x_max], dim=1)

    # Classification
    x = self.classifier(x)
    return x
