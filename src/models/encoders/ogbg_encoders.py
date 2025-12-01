import torch
import torch.nn as nn


class OGBGAtomEncoder(nn.Module):
    """
    Atom encoder for OGB molecular datasets.

    Dynamically retrieves feature dimensions from OGB utilities.

    Args:
        emb_dim: Embedding dimension for each feature.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        try:
            from ogb.utils.features import get_atom_feature_dims
        except ImportError:
            raise ImportError("OGB not installed. Install with: pip install ogb")

        self.emb_dim = emb_dim
        feature_dims = get_atom_feature_dims()

        self.embeddings = nn.ModuleList(
            [self._create_embedding(dim, emb_dim) for dim in feature_dims]
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
        x = x.long()
        x_emb = sum(self.embeddings[i](x[:, i]) for i in range(x.shape[1]))
        return x_emb


class OGBGBondEncoder(nn.Module):
    """
    Bond encoder for OGB molecular datasets.

    Dynamically retrieves feature dimensions from OGB utilities.

    Args:
        emb_dim: Embedding dimension for each feature.
    """

    def __init__(self, emb_dim: int):
        super().__init__()
        try:
            from ogb.utils.features import get_bond_feature_dims
        except ImportError:
            raise ImportError("OGB not installed. Install with: pip install ogb")

        self.emb_dim = emb_dim
        feature_dims = get_bond_feature_dims()

        self.embeddings = nn.ModuleList(
            [self._create_embedding(dim, emb_dim) for dim in feature_dims]
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
        edge_attr = edge_attr.long()
        edge_emb = sum(
            self.embeddings[i](edge_attr[:, i]) for i in range(edge_attr.shape[1])
        )
        return edge_emb
