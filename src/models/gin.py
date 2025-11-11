import torch
from torch_geometric.nn import MLP, GINConv, global_add_pool


class GIN(torch.nn.Module):
    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int
    ):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP(
            [hidden_channels, hidden_channels, out_channels], norm=None, dropout=0.5
        )

    def forward(self, data):
        x, edge_index, batch, batch_size = (
            data.x,
            data.edge_index,
            data.batch,
            data.num_graphs,
        )
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        # Pass the batch size to avoid CPU communication/graph breaks:
        x = global_add_pool(x, batch, size=batch_size)
        return self.mlp(x)

if __name__ == "__main__":
    # Simple test
    from torch_geometric.data import Data, Batch

    # Create a batch of 2 simple graphs
    data1 = Data(
        x=torch.randn(3, 5),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
    )
    data2 = Data(
        x=torch.randn(4, 5),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]]),
    )
    batch = Batch.from_data_list([data1, data2])

    model = GIN(in_channels=5, hidden_channels=16, out_channels=1, num_layers=3)
    out = model(batch)
    print(out.shape)  # Should be [2, 1] for the batch of 2 graphs
