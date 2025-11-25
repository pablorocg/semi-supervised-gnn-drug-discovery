from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

X_MAP: Dict[str, List[Any]] = {
    "atomic_num": list(range(0, 119)),  # N of elements in the list is 119 from 0 to 118
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


class MoleculeNetAtomEncoder(nn.Module):
    """
    Atom encoder for MoleculeNet datasets.

    Encodes 9 atomic features: atomic_num, chirality, degree, formal_charge,
    num_hs, num_radical_electrons, hybridization, is_aromatic, is_in_ring.

    Args:
        emb_dim: Embedding dimension for each feature.
        vocab_sizes: Custom vocabulary sizes per dimension (optional).
    """

    DEFAULT_VOCAB_SIZES = [119, 9, 11, 12, 9, 5, 8, 2, 2]

    def __init__(self, emb_dim: int = 16, vocab_sizes: Optional[List[int]] = None):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_sizes = vocab_sizes or self.DEFAULT_VOCAB_SIZES
        self.n_features = len(self.vocab_sizes)

        # Create embedding layers
        self.embeddings = nn.ModuleList(
            [
                self._create_embedding(vocab_size, emb_dim)
                for vocab_size in self.vocab_sizes
            ]
        )

    @staticmethod
    def _create_embedding(vocab_size: int, emb_dim: int) -> nn.Embedding:
        """Create and initialize an embedding layer."""
        emb = nn.Embedding(vocab_size, emb_dim)
        nn.init.xavier_uniform_(emb.weight.data)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [num_nodes, num_features]

        Returns:
            Node embeddings [num_nodes, emb_dim]
        """
        if x.shape[1] != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {x.shape[1]}")

        x = x.long()
        # Sum embeddings across all features
        x_emb = sum(self.embeddings[i](x[:, i]) for i in range(self.n_features))
        return x_emb


class MoleculeNetBondEncoder(nn.Module):
    """
    Bond encoder for MoleculeNet datasets.

    Encodes 3 bond features: bond_type, stereo, is_conjugated.

    Args:
        emb_dim: Embedding dimension for each feature.
        vocab_sizes: Custom vocabulary sizes per dimension (optional).
    """

    DEFAULT_VOCAB_SIZES = [22, 6, 2]

    def __init__(self, emb_dim: int = 64, vocab_sizes: Optional[List[int]] = None):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_sizes = vocab_sizes or self.DEFAULT_VOCAB_SIZES
        self.n_features = len(self.vocab_sizes)

        self.embeddings = nn.ModuleList(
            [
                self._create_embedding(vocab_size, emb_dim)
                for vocab_size in self.vocab_sizes
            ]
        )

    @staticmethod
    def _create_embedding(vocab_size: int, emb_dim: int) -> nn.Embedding:
        """Create and initialize an embedding layer."""
        emb = nn.Embedding(vocab_size, emb_dim)
        nn.init.xavier_uniform_(emb.weight.data)
        return emb

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_attr: Edge attributes [num_edges, num_features]

        Returns:
            Edge embeddings [num_edges, emb_dim]
        """
        if edge_attr.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {edge_attr.shape[1]}"
            )

        edge_attr = edge_attr.long()
        edge_emb = sum(
            self.embeddings[i](edge_attr[:, i]) for i in range(self.n_features)
        )
        return edge_emb
    


class MoleculeNetAttnAtomEncoder(nn.Module):
    VOCAB_SIZE_PER_DIM = [119, 9, 11, 12, 9, 5, 8, 2, 2]

    def __init__(self, out_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super(MoleculeNetAttnAtomEncoder, self).__init__()

        self.out_dim = out_dim
        self.num_features = len(self.VOCAB_SIZE_PER_DIM)

        self.encoders = nn.ModuleList()
        for vocab_size in self.VOCAB_SIZE_PER_DIM:
            self.encoders.append(nn.Embedding(vocab_size, out_dim))

        self.attention = nn.MultiheadAttention(
            embed_dim=out_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()


        x_encoded_list = [
            self.encoders[i](x[:, i]) for i in range(self.num_features)
        ]

        x_sequence = torch.stack(x_encoded_list, dim=0)

        attn_output, _ = self.attention(
            query=x_sequence, key=x_sequence, value=x_sequence
        )
        x_sequence = self.norm(x_sequence + attn_output)

        node_embeddings = torch.mean(x_sequence, dim=0)  # [Num_Nodes, Emb_Dim]

        return node_embeddings


class MoleculeNetAttnBondEncoder(nn.Module):
    VOCAB_SIZE_PER_DIM = [22, 6, 2]

    def __init__(self, out_dim: int = 64, num_heads: int = 4, dropout: float = 0.1):
        super(MoleculeNetAttnBondEncoder, self).__init__()
        self.out_dim = out_dim
        self.num_features = len(self.VOCAB_SIZE_PER_DIM)

        self.encoders = nn.ModuleList()
        for vocab_size in self.VOCAB_SIZE_PER_DIM:
            self.encoders.append(nn.Embedding(vocab_size, out_dim))

        self.attention = nn.MultiheadAttention(
            embed_dim=out_dim, num_heads=num_heads, dropout=dropout
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        
        edge_attr = edge_attr.long()
        
        edge_attr_encoded_list = [
            self.encoders[i](edge_attr[:, i]) for i in range(self.num_features)
        ]

        edge_attr_sequence = torch.stack(edge_attr_encoded_list, dim=0)

        attn_output, _ = self.attention(
            query=edge_attr_sequence,
            key=edge_attr_sequence,
            value=edge_attr_sequence,
        )
        edge_attr_sequence = self.norm(edge_attr_sequence + attn_output)

        bond_embeddings = torch.mean(edge_attr_sequence, dim=0)  # [Num_Edges, Emb_Dim]

        return bond_embeddings
