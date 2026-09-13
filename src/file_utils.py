from pathlib import Path

import numpy as np


CLASS_NAMES = (
    "correct",
    "incorrect"
)


def find_class_files(
    root_dir,
    patterns
):
    """Find files inside correct and incorrect class directories."""
    if isinstance(patterns, str):
        patterns = (patterns,)

    found_files = []

    for class_name in CLASS_NAMES:
        class_dir = root_dir / class_name

        if not class_dir.exists():
            continue

        for pattern in patterns:
            found_files.extend(
                class_dir.glob(pattern)
            )

    return sorted(found_files)


def load_keypoints_file(input_path):
    """Load keypoints and basic metadata from an NPZ file."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    with np.load(input_path) as data:
        keypoints = data[
            "keypoints"
        ].astype(np.float32)

        pose_detected = data[
            "pose_detected"
        ].astype(np.uint8)

        fps = float(
            data["fps"]
        )

    return keypoints, pose_detected, fps

def load_cleaned_keypoints_file(input_path):
    """Load cleaned keypoints and metadata from an NPZ file."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {input_path}"
        )

    with np.load(input_path) as data:
        keypoints = data["keypoints"].astype(np.float32)
        pose_detected = data["pose_detected"].astype(np.uint8)
        interpolated_frames = data[
            "interpolated_frames"
        ].astype(np.uint8)
        fps = float(data["fps"])

    return (
        keypoints,
        pose_detected,
        interpolated_frames,
        fps
    )