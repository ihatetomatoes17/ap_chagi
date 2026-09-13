from pathlib import Path
import re

import numpy as np

from file_utils import (
    find_class_files,
    load_cleaned_keypoints_file
)
from landmark_config import AP_CHAGI_LANDMARKS


CLEANED_DIR = Path("../processed/cleaned")
GRAPHS_DIR = Path("../processed/graphs")

TARGET_FRAMES = 80


SELECTED_LANDMARK_NAMES = (
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index"
)


LABEL_MAP = {
    "incorrect": 0,
    "correct": 1
}


SPATIAL_CONNECTIONS = (
    ("left_shoulder", "right_shoulder"),

    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),

    ("left_hip", "right_hip"),

    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),

    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),

    ("left_ankle", "left_heel"),
    ("right_ankle", "right_heel"),

    ("left_heel", "left_foot_index"),
    ("right_heel", "right_foot_index"),

    ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_foot_index")
)

def select_relevant_landmarks(keypoints):
    """Select landmarks relevant for Ap Chagi analysis."""
    landmark_ids = [
        AP_CHAGI_LANDMARKS[name]
        for name in SELECTED_LANDMARK_NAMES
    ]

    selected_keypoints = keypoints[
        :,
        landmark_ids,
        :
    ]

    return selected_keypoints.astype(
        np.float32
    )

def resample_keypoints(
    keypoints,
    target_frames
):
    """Resample a keypoint sequence to a fixed number of frames."""
    num_frames = keypoints.shape[0]

    if num_frames < 2:
        raise ValueError(
            "At least two frames are required for resampling."
        )

    original_positions = np.linspace(
        0.0,
        1.0,
        num_frames
    )

    target_positions = np.linspace(
        0.0,
        1.0,
        target_frames
    )

    num_landmarks = keypoints.shape[1]
    num_features = keypoints.shape[2]

    resampled_keypoints = np.empty(
        (
            target_frames,
            num_landmarks,
            num_features
        ),
        dtype=np.float32
    )

    for landmark_index in range(num_landmarks):
        for feature_index in range(num_features):
            resampled_keypoints[
                :,
                landmark_index,
                feature_index
            ] = np.interp(
                target_positions,
                original_positions,
                keypoints[
                    :,
                    landmark_index,
                    feature_index
                ]
            )

    return resampled_keypoints

def normalize_keypoints(keypoints):
    """
    Normalize coordinates relative to the hip center
    and torso length for every frame.
    """
    normalized_keypoints = keypoints.copy()

    landmark_index_map = {
        name: index
        for index, name in enumerate(
            SELECTED_LANDMARK_NAMES
        )
    }

    left_hip_index = landmark_index_map[
        "left_hip"
    ]

    right_hip_index = landmark_index_map[
        "right_hip"
    ]

    left_shoulder_index = landmark_index_map[
        "left_shoulder"
    ]

    right_shoulder_index = landmark_index_map[
        "right_shoulder"
    ]

    for frame_index in range(
        normalized_keypoints.shape[0]
    ):
        frame = normalized_keypoints[
            frame_index
        ]

        left_hip = frame[
            left_hip_index,
            :3
        ]

        right_hip = frame[
            right_hip_index,
            :3
        ]

        left_shoulder = frame[
            left_shoulder_index,
            :3
        ]

        right_shoulder = frame[
            right_shoulder_index,
            :3
        ]

        hip_center = (
            left_hip + right_hip
        ) / 2.0

        shoulder_center = (
            left_shoulder + right_shoulder
        ) / 2.0

        torso_length = np.linalg.norm(
            shoulder_center - hip_center
        )

        if torso_length <= 1e-6:
            raise ValueError(
                "Torso length is too small for normalization."
            )

        normalized_keypoints[
            frame_index,
            :,
            :3
        ] = (
            frame[:, :3] - hip_center
        ) / torso_length

    return normalized_keypoints


def build_spatial_edges(
    num_frames,
    num_landmarks
):
    """Create anatomical edges inside every frame."""
    landmark_index_map = {
        name: index
        for index, name in enumerate(
            SELECTED_LANDMARK_NAMES
        )
    }

    edges = []

    for frame_index in range(num_frames):
        frame_offset = (
            frame_index * num_landmarks
        )

        for (
            source_name,
            target_name
        ) in SPATIAL_CONNECTIONS:
            source_index = (
                frame_offset
                + landmark_index_map[source_name]
            )

            target_index = (
                frame_offset
                + landmark_index_map[target_name]
            )

            edges.append(
                (source_index, target_index)
            )

            edges.append(
                (target_index, source_index)
            )

    return edges

def build_temporal_edges(
    num_frames,
    num_landmarks
):
    """
    Connect each landmark with the same landmark
    in the next video frame.
    """
    edges = []

    for frame_index in range(
        num_frames - 1
    ):
        current_offset = (
            frame_index * num_landmarks
        )

        next_offset = (
            (frame_index + 1) * num_landmarks
        )

        for landmark_index in range(
            num_landmarks
        ):
            current_node = (
                current_offset
                + landmark_index
            )

            next_node = (
                next_offset
                + landmark_index
            )

            edges.append(
                (current_node, next_node)
            )

            edges.append(
                (next_node, current_node)
            )

    return edges

def build_edge_index(
    num_frames,
    num_landmarks
):
    """Create spatial and temporal graph edges."""
    spatial_edges = build_spatial_edges(
        num_frames,
        num_landmarks
    )

    temporal_edges = build_temporal_edges(
        num_frames,
        num_landmarks
    )

    all_edges = (
        spatial_edges
        + temporal_edges
    )

    edge_index = np.asarray(
        all_edges,
        dtype=np.int64
    ).T

    return edge_index

def extract_participant_id(sample_name):
    """Extract participant ID from the sample name."""
    match = re.search(
        r"(p\d+)",
        sample_name
    )

    if match is None:
        raise ValueError(
            f"Participant ID not found in: {sample_name}"
        )

    return match.group(1)

def build_graph(input_path):
    """Build one spatio-temporal graph from a cleaned video."""
    (
        keypoints,
        pose_detected,
        interpolated_frames,
        fps
    ) = load_cleaned_keypoints_file(
        input_path
    )

    selected_keypoints = (
        select_relevant_landmarks(
            keypoints
        )
    )

    resampled_keypoints = resample_keypoints(
        selected_keypoints,
        TARGET_FRAMES
    )

    normalized_keypoints = normalize_keypoints(
        resampled_keypoints
    )

    num_frames = normalized_keypoints.shape[0]
    num_landmarks = normalized_keypoints.shape[1]

    node_features = normalized_keypoints.reshape(
        num_frames * num_landmarks,
        4
    )

    edge_index = build_edge_index(
        num_frames,
        num_landmarks
    )

    class_name = input_path.parent.name

    if class_name not in LABEL_MAP:
        raise ValueError(
            f"Unknown class: {class_name}"
        )

    label = LABEL_MAP[class_name]

    sample_name = input_path.stem.replace(
        "_cleaned",
        ""
    )

    participant_id = extract_participant_id(
        sample_name
    )

    return (
        node_features,
        edge_index,
        label,
        sample_name,
        participant_id
    )

def create_output_path(input_path):
    """Create the output path for one graph."""
    class_name = input_path.parent.name

    sample_name = input_path.stem.replace(
        "_cleaned",
        ""
    )

    return (
        GRAPHS_DIR
        / class_name
        / f"{sample_name}_graph.npz"
    )

def save_graph(
    output_path,
    node_features,
    edge_index,
    label,
    sample_name,
    participant_id
):
    """Save one graph and its metadata."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        output_path,
        node_features=node_features,
        edge_index=edge_index,
        label=np.int64(label),
        sample_name=np.asarray(sample_name),
        participant_id=np.asarray(participant_id),
        landmark_names=np.asarray(
            SELECTED_LANDMARK_NAMES
        ),
        target_frames=np.int32(
            TARGET_FRAMES
        )
    )

def process_all_graphs():
    """Build graphs for all cleaned videos."""
    cleaned_files = find_class_files(
        CLEANED_DIR,
        "*_cleaned.npz"
    )

    if not cleaned_files:
        raise ValueError(
            f"No cleaned files found in {CLEANED_DIR}."
        )

    print(
        f"Found {len(cleaned_files)} cleaned files."
    )

    for file_index, input_path in enumerate(
        cleaned_files,
        start=1
    ):
        output_path = create_output_path(
            input_path
        )

        print(
            f"\nProcessing graph "
            f"{file_index}/{len(cleaned_files)}:"
        )

        print(
            f"  Input: {input_path}"
        )

        (
            node_features,
            edge_index,
            label,
            sample_name,
            participant_id
        ) = build_graph(
            input_path
        )

        save_graph(
            output_path,
            node_features,
            edge_index,
            label,
            sample_name,
            participant_id
        )

        print(
            f"  Nodes: {node_features.shape[0]}"
        )

        print(
            f"  Node features: {node_features.shape[1]}"
        )

        print(
            f"  Edges: {edge_index.shape[1]}"
        )

        print(
            f"  Label: {label}"
        )

        print(
            f"  Participant: {participant_id}"
        )

        print(
            f"  Saved: {output_path}"
        )

if __name__ == "__main__":
    process_all_graphs()