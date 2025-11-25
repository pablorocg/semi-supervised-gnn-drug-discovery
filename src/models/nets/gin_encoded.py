from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm1d, ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import (
    MLP,
    GINEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


class EncodedGINE(torch.nn.Module):
    def __init__(
        self,
        num_tasks: int,
        embedding_dim: int = 256,
        hidden_channels: int = 256,
        encoder_num_heads: int = 4,
        encoder_dropout: float = 0.1,
        num_gnn_layers: int = 4,
        gnn_mlp_layers: int = 2,
        readout_mlp_layers: int = 2,
        dropout: float = 0.5,
        activation: str = "relu",
        pooling_type: str = "add",
        use_residual: bool = True,
        learn_eps: bool = True,
    ):
        super().__init__()

        assert embedding_dim % encoder_num_heads == 0, (
            f"embedding_dim ({embedding_dim}) has to be divisible by encoder_num_heads ({encoder_num_heads})"
        )

        self.dropout_p = dropout
        self.use_residual = use_residual
        self.pooling_type = pooling_type

        self.node_encoder = MoleculeNetAtomicEncoder(
            out_dim=embedding_dim, num_heads=encoder_num_heads, dropout=encoder_dropout
        )
        self.edge_encoder = MoleculeNetBondEncoder(
            out_dim=embedding_dim, num_heads=encoder_num_heads, dropout=encoder_dropout
        )

        

        self.act_fn = self._get_activation(activation)

        self.convs = ModuleList()
        self.bns = ModuleList()

        self.input_proj = None
        if embedding_dim != hidden_channels:
            self.input_proj = nn.Linear(embedding_dim, hidden_channels)
            current_dim = hidden_channels
        else:
            current_dim = embedding_dim

        for _ in range(num_gnn_layers):
            mlp_channels = [current_dim] * (gnn_mlp_layers + 1)

            internal_mlp = MLP(
                channel_list=mlp_channels, act=activation, norm=None, plain_last=False
            )

            self.convs.append(
                GINEConv(nn=internal_mlp, edge_dim=embedding_dim, train_eps=learn_eps)
            )
            self.bns.append(BatchNorm1d(hidden_channels))

        readout_channels = [hidden_channels] * readout_mlp_layers + [num_tasks]

        self.mlp_out = MLP(
            channel_list=readout_channels,
            norm="batch_norm",
            dropout=dropout,
            act=activation,
            plain_last=True,
        )

    def _get_activation(self, name: str):
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(),
            "elu": nn.ELU(),
        }
        return activations.get(name.lower(), nn.ReLU())

    def _pool(self, x, batch):
        if self.pooling_type == "mean":
            return global_mean_pool(x, batch)
        elif self.pooling_type == "max":
            return global_max_pool(x, batch)
        else:
            return global_add_pool(x, batch)

    def forward(self, data):
        
        x = self.node_encoder(data)
        edge_index = data.edge_index
        edge_attr = self.edge_encoder(data)
        batch = data.batch

        

        if self.input_proj is not None:
            x = self.input_proj(x)

        for conv, bn in zip(self.convs, self.bns):
            x_in = x

            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = self.act_fn(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

            if self.use_residual and x_in.shape == x.shape:
                x = x + x_in

        x = self._pool(x, batch)

        x = self.mlp_out(x)

        return x


if __name__ == "__main__":
    from torch_geometric.data import Data

    node_features = torch.tensor(
        [
            [6, 0, 2, 0, 0, 0, 2, 1, 0],
            [6, 0, 3, 0, 0, 0, 3, 1, 0],
            [8, 0, 1, 0, 0, 0, 2, 1, 0],
        ],
        dtype=torch.float,
    )

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)

    edge_attr = torch.tensor(
        [
            [1, 0, 1],
            [1, 0, 1],
            [2, 0, 0],
            [2, 0, 0],
        ],
        dtype=torch.float,
    )

    data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    model = EncodedGINE(num_tasks=128)

    out = model(data)
    print(out)  # Should print a tensor of shape [1, 1]
