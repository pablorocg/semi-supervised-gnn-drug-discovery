"""Graph augmentation utilities for Mean Teacher."""

import torch
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge, dropout_node


class GraphAugmentor:
    """Graph augmentation for creating different views of molecular graphs."""

    def __init__(
        self,
        node_drop_rate: float = 0.1,
        edge_drop_rate: float = 0.1,
        feature_mask_rate: float = 0.1,
        feature_noise_std: float = 0.01,
        edge_attr_noise_std: float = 0.0,
    ):
        self.node_drop_rate = node_drop_rate
        self.edge_drop_rate = edge_drop_rate
        self.feature_mask_rate = feature_mask_rate
        self.feature_noise_std = feature_noise_std
        self.edge_attr_noise_std = edge_attr_noise_std

    def __call__(self, data: Data) -> Data:
        """Apply augmentation to a graph."""
        # Clone the data by creating a new Data object with copied tensors
        aug_data = Data(
            x=data.x.clone(),
            edge_index=data.edge_index.clone(),
            edge_attr=data.edge_attr.clone() if data.edge_attr is not None else None,
            y=data.y.clone() if data.y is not None else None,
            batch=data.batch.clone() if hasattr(data, 'batch') and data.batch is not None else None,
        )

        # Apply augmentations
        if self.edge_drop_rate > 0:
            aug_data = self._drop_edges(aug_data)

        if self.node_drop_rate > 0:
            aug_data = self._mask_nodes(aug_data)

        if self.feature_mask_rate > 0:
            aug_data = self._mask_features(aug_data)

        if self.feature_noise_std > 0:
            aug_data = self._add_node_noise(aug_data)

        if self.edge_attr_noise_std > 0 and aug_data.edge_attr is not None:
            aug_data = self._add_edge_noise(aug_data)

        return aug_data

    def _drop_edges(self, data: Data) -> Data:
        """Randomly drop edges from the graph."""
        edge_index, edge_attr = dropout_edge(
            data.edge_index,
            p=self.edge_drop_rate,
            force_undirected=False,
            training=True,
        )
        data.edge_index = edge_index
        
        # If edge attributes exist, we need to filter them too
        if data.edge_attr is not None and edge_attr is not None:
            # dropout_edge returns a mask, we need to apply it
            data.edge_attr = data.edge_attr
            
        return data

    def _mask_nodes(self, data: Data) -> Data:
        """Randomly mask node features."""
        num_nodes = data.x.size(0)
        mask = torch.rand(num_nodes, device=data.x.device) > self.node_drop_rate
        mask = mask.unsqueeze(1).expand_as(data.x)
        data.x = data.x * mask.float()
        return data

    def _mask_features(self, data: Data) -> Data:
        """Randomly mask individual features."""
        mask = torch.rand_like(data.x) > self.feature_mask_rate
        data.x = data.x * mask.float()
        return data

    def _add_node_noise(self, data: Data) -> Data:
        """Add Gaussian noise to node features."""
        noise = torch.randn_like(data.x) * self.feature_noise_std
        data.x = data.x + noise
        return data

    def _add_edge_noise(self, data: Data) -> Data:
        """Add Gaussian noise to edge attributes."""
        if data.edge_attr is not None:
            noise = torch.randn_like(data.edge_attr) * self.edge_attr_noise_std
            data.edge_attr = data.edge_attr + noise
        return data

    def __repr__(self) -> str:
        return (
            f"GraphAugmentor(\n"
            f"  node_drop_rate={self.node_drop_rate},\n"
            f"  edge_drop_rate={self.edge_drop_rate},\n"
            f"  feature_mask_rate={self.feature_mask_rate},\n"
            f"  feature_noise_std={self.feature_noise_std},\n"
            f"  edge_attr_noise_std={self.edge_attr_noise_std}\n"
            f")"
        )


class NoAugmentation:
    """No augmentation - for supervised baseline."""

    def __call__(self, data: Data) -> Data:
        """Return data unchanged."""
        return data

    def __repr__(self) -> str:
        return "NoAugmentation()"