import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm1d, ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import MLP, GINConv, GINEConv, global_add_pool
from typing import Any, Dict, List

X_MAP: Dict[str, List[Any]] = {
    "atomic_num": list(range(0, 119)),
    "chirality": [
        "CHI_UNSPECIFIED",
        "CHI_TETRAHEDRAL_CW",
        "CHI_TETRAHEDRAL_CCW",
        "CHI_OTHER",
        "CHI_TETRAHEDRAL",
        "CHI_ALLENE",
        "CHI_SQUAREPLANAR",
        "CHI_TRIGONALBIPYRAMIDAL",
        "CHI_OCTAHEDRAL",
    ],
    "degree": list(range(0, 11)),
    "formal_charge": list(range(-5, 7)),
    "num_hs": list(range(0, 9)),
    "num_radical_electrons": list(range(0, 5)),
    "hybridization": [
        "UNSPECIFIED",
        "S",
        "SP",
        "SP2",
        "SP3",
        "SP3D",
        "SP3D2",
        "OTHER",
    ],
    "is_aromatic": [False, True],
    "is_in_ring": [False, True],
}

E_MAP: Dict[str, List[Any]] = {
    "bond_type": [
        "UNSPECIFIED",
        "SINGLE",
        "DOUBLE",
        "TRIPLE",
        "QUADRUPLE",
        "QUINTUPLE",
        "HEXTUPLE",
        "ONEANDAHALF",
        "TWOANDAHALF",
        "THREEANDAHALF",
        "FOURANDAHALF",
        "FIVEANDAHALF",
        "AROMATIC",
        "IONIC",
        "HYDROGEN",
        "THREECENTER",
        "DATIVEONE",
        "DATIVE",
        "DATIVEL",
        "DATIVER",
        "OTHER",
        "ZERO",
    ],
    "stereo": [
        "STEREONONE",
        "STEREOANY",
        "STEREOZ",
        "STEREOE",
        "STEREOCIS",
        "STEREOTRANS",
    ],
    "is_conjugated": [False, True],
}


# class MoleculeNetAtomicEncoder(nn.Module):
#     VOCAB_SIZE_PER_DIM = [119, 9, 11, 12, 9, 5, 8, 2, 2]

#     def __init__(self, out_dim: int = 64):
#         super(MoleculeNetAtomicEncoder, self).__init__()

#         self.out_dim = out_dim
#         self.num_features = len(self.VOCAB_SIZE_PER_DIM)

#         self.encoders = nn.ModuleList()

#         for vocab_size in self.VOCAB_SIZE_PER_DIM:
#             self.encoders.append(nn.Embedding(vocab_size, out_dim))

#         self.proj = nn.Linear(len(self.VOCAB_SIZE_PER_DIM) * out_dim, out_dim)

#     def forward(self, data: Data):
#         """Encode node attributes."""

#         x_encoded = [
#             self.encoders[i](data.x[:, i].long())
#             for i in range(len(self.VOCAB_SIZE_PER_DIM))
#         ]

#         x_encoded = torch.cat(x_encoded, dim=-1)
#         data.x = self.proj(x_encoded)

#         return data


# class MoleculeNetBondEncoder(nn.Module):
#     VOCAB_SIZE_PER_DIM = [22, 6, 2]

#     def __init__(self, out_dim: int = 64):
#         super(MoleculeNetBondEncoder, self).__init__()

#         self.encoders = nn.ModuleList()

#         for vocab_size in self.VOCAB_SIZE_PER_DIM:
#             self.encoders.append(nn.Embedding(vocab_size, out_dim))

#         self.proj = nn.Linear(len(self.VOCAB_SIZE_PER_DIM) * out_dim, out_dim)

#     def forward(self, data: Data) -> Data:
#         edge_attr_encoded = [
#             self.encoders[i](data.edge_attr[:, i].long())
#             for i in range(len(self.VOCAB_SIZE_PER_DIM))
#         ]
#         edge_attr_encoded = torch.cat(edge_attr_encoded, dim=-1)

#         data.edge_attr = self.proj(edge_attr_encoded)

#         return data


class MoleculeNetAtomicEncoder(nn.Module):
    VOCAB_SIZE_PER_DIM = [119, 9, 11, 12, 9, 5, 8, 2, 2]

    def __init__(self, out_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super(MoleculeNetAtomicEncoder, self).__init__()

        self.out_dim = out_dim
        self.num_features = len(self.VOCAB_SIZE_PER_DIM)

        self.encoders = nn.ModuleList()
        for vocab_size in self.VOCAB_SIZE_PER_DIM:
            self.encoders.append(nn.Embedding(vocab_size, out_dim))

        self.attention = nn.MultiheadAttention(
            embed_dim=out_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, data: Data):
        x_encoded_list = [
            self.encoders[i](data.x[:, i].long()) for i in range(self.num_features)
        ]

        x_sequence = torch.stack(x_encoded_list, dim=0)

        attn_output, _ = self.attention(
            query=x_sequence, key=x_sequence, value=x_sequence
        )
        x_sequence = self.norm(x_sequence + attn_output)

        node_embeddings = torch.mean(x_sequence, dim=0)  # [Num_Nodes, Emb_Dim]

        data.x = node_embeddings

        return data

class MoleculeNetBondEncoder(nn.Module):
    VOCAB_SIZE_PER_DIM = [22, 6, 2]

    def __init__(self, out_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super(MoleculeNetBondEncoder, self).__init__()
        self.out_dim = out_dim
        self.num_features = len(self.VOCAB_SIZE_PER_DIM)

        self.encoders = nn.ModuleList()
        for vocab_size in self.VOCAB_SIZE_PER_DIM:
            self.encoders.append(nn.Embedding(vocab_size, out_dim))

        self.attention = nn.MultiheadAttention(
            embed_dim=out_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, data: Data) -> Data:
        edge_attr_encoded_list = [
            self.encoders[i](data.edge_attr[:, i].long())
            for i in range(self.num_features)
        ]

        edge_attr_sequence = torch.stack(edge_attr_encoded_list, dim=0)

        attn_output, _ = self.attention(
            query=edge_attr_sequence,
            key=edge_attr_sequence,
            value=edge_attr_sequence,
        )
        edge_attr_sequence = self.norm(edge_attr_sequence + attn_output)

        bond_embeddings = torch.mean(edge_attr_sequence, dim=0)  # [Num_Edges, Emb_Dim]

        data.edge_attr = bond_embeddings

        return data


class EncodedGINE(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int = 256,
        hidden_channels: int = 256,
        num_tasks: int = 12,
        num_layers: int = 4,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.dropout_p = dropout

        self.node_encoder = MoleculeNetAtomicEncoder(out_dim=embedding_dim)
        self.edge_encoder = MoleculeNetBondEncoder(out_dim=embedding_dim)

        self.convs = ModuleList()
        self.bns = ModuleList()

        current_in_dim = embedding_dim

        for i in range(num_layers):
            mlp = MLP([current_in_dim, hidden_channels, hidden_channels])

            self.convs.append(GINEConv(nn=mlp, edge_dim=embedding_dim, train_eps=True))
            self.bns.append(BatchNorm1d(hidden_channels))

            current_in_dim = hidden_channels

        self.mlp = MLP(
            [hidden_channels, hidden_channels, num_tasks],
            norm="batch_norm",
            dropout=dropout,
        )

        self.gelu = torch.nn.GELU()

    def forward(self, data):
        data = self.node_encoder(data)
        data = self.edge_encoder(data)

        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        for conv, bn in zip(self.convs, self.bns):
            x_in = x

            x = conv(x, edge_index, edge_attr=edge_attr)
            x = bn(x)
            x = self.gelu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)

            if x_in.shape == x.shape:
                x = x + x_in

        x = global_add_pool(x, batch, size=data.num_graphs)

        x = self.mlp(x)

        return x
