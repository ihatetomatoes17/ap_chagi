from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

from file_utils import find_class_files


GRAPHS_DIR = Path("../processed/graphs")


class ApChagiGraphDataset(Dataset):
    """PyTorch Geometric dataset for Ap Chagi graphs."""

    def __init__(self, root_dir=GRAPHS_DIR):
        super().__init__()

        self.root_dir = Path(root_dir)

        self.graph_files = find_class_files(
            self.root_dir,
            "*_graph.npz"
        )

        if not self.graph_files:
            raise ValueError(
                f"No graph files found in {self.root_dir}."
            )

    def len(self):
        """Return the number of graph samples."""
        return len(self.graph_files)

    def get(self, index):
        """Load one graph and convert it to a Data object."""
        input_path = self.graph_files[index]

        with np.load(input_path) as data:
            node_features = data[
                "node_features"
            ].astype(np.float32)

            edge_index = data[
                "edge_index"
            ].astype(np.int64)

            label = int(
                data["label"]
            )

            sample_name = str(
                data["sample_name"]
            )

            participant_id = str(
                data["participant_id"]
            )

        self.validate_graph(
            node_features,
            edge_index,
            label,
            input_path
        )

        graph = Data(
            x=torch.from_numpy(
                node_features
            ),
            edge_index=torch.from_numpy(
                edge_index
            ).long(),
            y=torch.tensor(
                [label],
                dtype=torch.long
            )
        )

        graph.sample_name = sample_name
        graph.participant_id = participant_id

        return graph

    @staticmethod
    def validate_graph(
        node_features,
        edge_index,
        label,
        input_path
    ):
        """Validate one graph before conversion."""
        if node_features.ndim != 2:
            raise ValueError(
                f"Invalid node feature shape in {input_path}: "
                f"{node_features.shape}"
            )

        if node_features.shape[1] != 4:
            raise ValueError(
                f"Expected 4 node features in {input_path}, "
                f"but got {node_features.shape[1]}."
            )

        if not np.all(
            np.isfinite(node_features)
        ):
            raise ValueError(
                f"Invalid node feature values in {input_path}."
            )

        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
        ):
            raise ValueError(
                f"Invalid edge_index shape in {input_path}: "
                f"{edge_index.shape}"
            )

        num_nodes = node_features.shape[0]

        if np.min(edge_index) < 0:
            raise ValueError(
                f"Negative node index in {input_path}."
            )

        if np.max(edge_index) >= num_nodes:
            raise ValueError(
                f"Edge references a non-existing node "
                f"in {input_path}."
            )

        if label not in (0, 1):
            raise ValueError(
                f"Invalid label in {input_path}: {label}"
            )

def inspect_dataset():
    """Load the dataset and print one graph sample."""
    dataset = ApChagiGraphDataset()

    print(f"Number of graphs: {len(dataset)}")

    graph = dataset[0]

    print("\n--- First Graph ---")
    print(f"Sample: {graph.sample_name}")
    print(f"Participant: {graph.participant_id}")
    print(f"Label: {graph.y.item()}")
    print(f"Node features shape: {graph.x.shape}")
    print(f"Edge index shape: {graph.edge_index.shape}")
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Number of edges: {graph.num_edges}")
    print("-------------------")


if __name__ == "__main__":
    inspect_dataset()