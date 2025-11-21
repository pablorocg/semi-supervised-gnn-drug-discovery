import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data  # Importamos Data para type hinting
from torch_geometric.nn import MLP, GINConv, GINEConv, global_add_pool

# x_map: Dict[str, List[Any]] = {
#     'atomic_num':
#     list(range(0, 119)),
#     'chirality': [
#         'CHI_UNSPECIFIED',
#         'CHI_TETRAHEDRAL_CW',
#         'CHI_TETRAHEDRAL_CCW',
#         'CHI_OTHER',
#         'CHI_TETRAHEDRAL',
#         'CHI_ALLENE',
#         'CHI_SQUAREPLANAR',
#         'CHI_TRIGONALBIPYRAMIDAL',
#         'CHI_OCTAHEDRAL',
#     ],
#     'degree':
#     list(range(0, 11)),
#     'formal_charge':
#     list(range(-5, 7)),
#     'num_hs':
#     list(range(0, 9)),
#     'num_radical_electrons':
#     list(range(0, 5)),
#     'hybridization': [
#         'UNSPECIFIED',
#         'S',
#         'SP',
#         'SP2',
#         'SP3',
#         'SP3D',
#         'SP3D2',
#         'OTHER',
#     ],
#     'is_aromatic': [False, True],
#     'is_in_ring': [False, True],
# }

# e_map: Dict[str, List[Any]] = {
#     'bond_type': [
#         'UNSPECIFIED',
#         'SINGLE',
#         'DOUBLE',
#         'TRIPLE',
#         'QUADRUPLE',
#         'QUINTUPLE',
#         'HEXTUPLE',
#         'ONEANDAHALF',
#         'TWOANDAHALF',
#         'THREEANDAHALF',
#         'FOURANDAHALF',
#         'FIVEANDAHALF',
#         'AROMATIC',
#         'IONIC',
#         'HYDROGEN',
#         'THREECENTER',
#         'DATIVEONE',
#         'DATIVE',
#         'DATIVEL',
#         'DATIVER',
#         'OTHER',
#         'ZERO',
#     ],
#     'stereo': [
#         'STEREONONE',
#         'STEREOANY',
#         'STEREOZ',
#         'STEREOE',
#         'STEREOCIS',
#         'STEREOTRANS',
#     ],
#     'is_conjugated': [False, True],
# }


class MoleculeNetAtomicEncoder(nn.Module):
    VOCAB_SIZE_PER_DIM = [119, 9, 11, 12, 9, 5, 8, 2, 2]

    def __init__(self, out_dim: int = 64):
        super(MoleculeNetAtomicEncoder, self).__init__()

        self.encoders = nn.ModuleList()

        for vocab_size in self.VOCAB_SIZE_PER_DIM:
            self.encoders.append(nn.Embedding(vocab_size, out_dim))

        self.proj = nn.Linear(len(self.VOCAB_SIZE_PER_DIM) * out_dim, out_dim)

    def forward(self, data: Data):
        """Encode node attributes."""

        x_encoded = [
            self.encoders[i](data.x[:, i].long())
            for i in range(len(self.VOCAB_SIZE_PER_DIM))
        ]

        x_encoded = torch.cat(x_encoded, dim=-1)
        data.x = self.proj(x_encoded)

        return data


class MoleculeNetBondEncoder(nn.Module):
    VOCAB_SIZE_PER_DIM = [22, 6, 2]

    def __init__(self, out_dim: int = 64):
        super(MoleculeNetBondEncoder, self).__init__()

        self.encoders = nn.ModuleList()

        for vocab_size in self.VOCAB_SIZE_PER_DIM:
            self.encoders.append(nn.Embedding(vocab_size, out_dim))

        self.proj = nn.Linear(len(self.VOCAB_SIZE_PER_DIM) * out_dim, out_dim)

    def forward(self, data: Data) -> Data:
        edge_attr_encoded = [
            self.encoders[i](data.edge_attr[:, i].long())
            for i in range(len(self.VOCAB_SIZE_PER_DIM))
        ]
        edge_attr_encoded = torch.cat(edge_attr_encoded, dim=-1)

        data.edge_attr = self.proj(edge_attr_encoded)

        return data


# class EncodedGINE(torch.nn.Module):
#     """
#     Modelo GNN basado en GINE para predicción multi-tarea, usando Edge Attributes (data.edge_attr)
#     y la activación ELU en el Encoder GNN.
#     """

#     def __init__(
#         self,
#         num_node_features: int = 32,
#         num_edge_features: int = 32,
#         embedding_dim: int = 64,
#         hidden_channels: int = 128,
#         num_tasks: int = 1,
#         dropout: float = 0.5,
#     ):
#         super(EncodedGINE, self).__init__()

#         self.node_encoder = MoleculeNetAtomicEncoder(out_dim=embedding_dim)
#         self.edge_encoder = MoleculeNetBondEncoder(out_dim=embedding_dim)

#         self.dropout = dropout

#         self.conv1 = GINEConv(
#             nn=torch.nn.Sequential(
#                 torch.nn.Linear(embedding_dim, hidden_channels),
#                 torch.nn.ELU(),
#                 torch.nn.Linear(hidden_channels, hidden_channels),
#             ),
#             edge_dim=embedding_dim,
#         )

#         self.conv2 = GINEConv(
#             nn=torch.nn.Sequential(
#                 torch.nn.Linear(hidden_channels, hidden_channels),
#                 torch.nn.ELU(),
#                 torch.nn.Linear(hidden_channels, hidden_channels),
#             ),
#             edge_dim=embedding_dim,
#         )

#         self.pool = global_add_pool

#         self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
#         self.lin2 = torch.nn.Linear(hidden_channels, num_tasks)

#     def forward(self, data: Data):
#         data = self.node_encoder(data)
#         data = self.edge_encoder(data)

#         x, edge_index, edge_attr, batch = (
#             data.x,
#             data.edge_index,
#             data.edge_attr,
#             data.batch,
#         )

#         x = self.conv1(x, edge_index, edge_attr=edge_attr)
#         x = F.elu(x)
#         x = F.dropout(x, p=self.dropout, training=self.training)

#         x = self.conv2(x, edge_index, edge_attr=edge_attr)
#         x = F.elu(x)

#         h_graph = self.pool(x, batch)
#         h_graph = self.lin1(h_graph)
#         h_graph = F.relu(h_graph)

#         out = self.lin2(h_graph)

#         return out


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

        self.node_encoder = MoleculeNetAtomicEncoder(out_dim=embedding_dim)
        self.edge_encoder = MoleculeNetBondEncoder(out_dim=embedding_dim)

        self.convs = torch.nn.ModuleList()

        current_in_dim = embedding_dim

        for i in range(num_layers):
            mlp = MLP([current_in_dim, hidden_channels, hidden_channels])

            self.convs.append(GINEConv(nn=mlp, edge_dim=embedding_dim, train_eps=False))
            current_in_dim = hidden_channels

        
        self.mlp = MLP(
            [hidden_channels, hidden_channels, num_tasks],
            norm="batch_norm",
            dropout=dropout,
        )

        self.gelu = nn.GELU()

    def forward(self, data: Data):
        
        data = self.node_encoder(data)
        data = self.edge_encoder(data)

        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,  
            data.batch,
        )

      
        for conv in self.convs:
           
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.gelu(x)

        
        x = global_add_pool(x, batch, size=data.num_graphs)

        
        x = self.mlp(x)
        return x
