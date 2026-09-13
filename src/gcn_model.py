import torch
from torch import nn
from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    global_max_pool
)


class ApChagiGCN(nn.Module):
    """GCN model for Ap Chagi correctness classification."""

    def __init__(
        self,
        input_features=4,
        hidden_channels=64,
        num_classes=2,
        dropout=0.3
    ):
        super().__init__()

        self.conv1 = GCNConv(
            input_features,
            hidden_channels
        )

        self.conv2 = GCNConv(
            hidden_channels,
            hidden_channels
        )

        self.dropout = nn.Dropout(
            p=dropout
        )

        self.classifier = nn.Linear(
            hidden_channels * 2,
            num_classes
        )

    def forward(
        self,
        x,
        edge_index,
        batch
    ):
        x = self.conv1(
            x,
            edge_index
        )

        x = torch.relu(x)

        x = self.dropout(x)

        x = self.conv2(
            x,
            edge_index
        )

        x = torch.relu(x)

        mean_features = global_mean_pool(
            x,
            batch
        )

        max_features = global_max_pool(
            x,
            batch
        )

        x = torch.cat(
            [
                mean_features,
                max_features
            ],
            dim=1
        )

        x = self.dropout(
            x
        )

        logits = self.classifier(
            x
        )

        return logits


def test_model():
    """Run one batch through the GCN model."""
    from torch_geometric.loader import DataLoader

    from graph_dataset import ApChagiGraphDataset

    dataset = ApChagiGraphDataset()

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False
    )

    model = ApChagiGCN()

    batch = next(
        iter(loader)
    )

    output = model(
        batch.x,
        batch.edge_index,
        batch.batch
    )

    print("\n--- GCN Test ---")
    print(f"Graphs in batch: {batch.num_graphs}")
    print(f"Input node features: {batch.x.shape}")
    print(f"Edge index: {batch.edge_index.shape}")
    print(f"Output shape: {output.shape}")
    print("Output:")
    print(output)
    print("----------------")


if __name__ == "__main__":
    test_model()