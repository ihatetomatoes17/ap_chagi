from pathlib import Path
from landmark_config import NUM_LANDMARKS, NUM_FEATURES, SMOOTHING_WINDOW_SIZE
from file_utils import (
    find_class_files,
    load_keypoints_file
)
import numpy as np

KEYPOINTS_DIR = Path("../processed/keypoints")
CLEANED_DIR = Path("../processed/cleaned")

def create_output_path(input_path):
    """Create cleaned output path for a keypoint file."""
    class_name = input_path.parent.name

    output_path = (
        CLEANED_DIR
        / class_name
        / f"{input_path.stem}_cleaned.npz"
    )

    return output_path

def validate_keypoints(keypoints):
    """Check whether the keypoint array has the expected shape."""
    expected_shape = (
        NUM_LANDMARKS,
        NUM_FEATURES
    )

    if keypoints.ndim != 3 or keypoints.shape[1:] != expected_shape:
        raise ValueError(
            "Expected shape: "
            f"(num_frames, {NUM_LANDMARKS}, {NUM_FEATURES}), "
            f"but got: {keypoints.shape}"
        )

    if keypoints.shape[0] == 0:
        raise ValueError(
            "The keypoint array does not contain any frames."
        )

def interpolate_series(values):
    """Fill missing values in one time series using linear interpolation."""
    frame_ids = np.arange(len(values))
    valid_mask = ~np.isnan(values)

    if not np.any(valid_mask):
        raise ValueError(
            "Cannot interpolate a series without valid values."
        )

    if np.all(valid_mask):
        return values.astype(
            np.float32,
            copy=True
        )

    interpolated_values = np.interp(
        frame_ids,
        frame_ids[valid_mask],
        values[valid_mask]
    )

    return interpolated_values.astype(np.float32)

def interpolate_keypoints(keypoints):
    """Fill missing x, y, and z coordinates across video frames."""
    interpolated_keypoints = keypoints.copy()

    num_coordinates = 3

    for landmark_id in range(NUM_LANDMARKS):
        for coordinate_id in range(num_coordinates):
            values = interpolated_keypoints[
                :,
                landmark_id,
                coordinate_id
            ]

            interpolated_keypoints[
                :,
                landmark_id,
                coordinate_id
            ] = interpolate_series(values)

    return interpolated_keypoints

def save_cleaned_keypoints(
    keypoints,
    pose_detected,
    fps,
    output_path
):
    """Save cleaned keypoints and metadata."""
    if keypoints.shape[0] != len(pose_detected):
        raise ValueError(
            "The number of keypoint frames must match "
            "the number of pose detection values."
        )

    if fps <= 0:
        raise ValueError(
            "FPS must be greater than zero."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    interpolated_frames = (
        pose_detected == 0
    ).astype(np.uint8)

    np.savez_compressed(
        output_path,
        keypoints=keypoints,
        pose_detected=pose_detected,
        interpolated_frames=interpolated_frames,
        fps=fps
    )

    print(
        f"Cleaned keypoints saved to: {output_path}"
    )
    print(f"Keypoints shape: {keypoints.shape}")
    print(
        "Interpolated frames: "
        f"{int(np.sum(interpolated_frames))}"
    )

def trim_undetected_edge_frames(
    keypoints,
    pose_detected
):
    """Remove undetected frames from the start and end."""
    detected_frame_ids = np.where(
        pose_detected == 1
    )[0]

    if detected_frame_ids.size == 0:
        raise ValueError(
            "Pose was not detected in any video frame."
        )

    first_detected_frame_id = detected_frame_ids[0]
    last_detected_frame_id = detected_frame_ids[-1]

    trimmed_keypoints = keypoints[
        first_detected_frame_id:last_detected_frame_id + 1
    ]

    trimmed_pose_detected = pose_detected[
        first_detected_frame_id:last_detected_frame_id + 1
    ]

    removed_start_frames = first_detected_frame_id

    removed_end_frames = (
        len(pose_detected)
        - last_detected_frame_id
        - 1
    )

    return (
        trimmed_keypoints,
        trimmed_pose_detected,
        removed_start_frames,
        removed_end_frames
    )

def smooth_series(values, window_size):
    """Smooth one time series using a moving average."""
    if window_size < 1:
        raise ValueError(
            "Smoothing window size must be greater than zero."
        )

    if window_size % 2 == 0:
        raise ValueError(
            "Smoothing window size must be an odd number."
        )

    if len(values) < window_size:
        return values.astype(
            np.float32,
            copy=True
        )

    padding_size = window_size // 2

    padded_values = np.pad(
        values,
        (padding_size, padding_size),
        mode="edge"
    )

    smoothing_kernel = np.ones(
        window_size,
        dtype=np.float32
    ) / window_size

    smoothed_values = np.convolve(
        padded_values,
        smoothing_kernel,
        mode="valid"
    )

    return smoothed_values.astype(np.float32)

def smooth_keypoints(keypoints, window_size):
    """Smooth x, y, and z coordinates across video frames."""
    smoothed_keypoints = keypoints.copy()

    num_coordinates = 3

    for landmark_id in range(NUM_LANDMARKS):
        for coordinate_id in range(num_coordinates):
            coordinate_values = keypoints[
                :,
                landmark_id,
                coordinate_id
            ]

            smoothed_keypoints[
                :,
                landmark_id,
                coordinate_id
            ] = smooth_series(
                coordinate_values,
                window_size
            )

    return smoothed_keypoints

def preprocess_keypoints(input_path, output_path):
    """Load, preprocess, and save keypoints from one video."""
    keypoints, pose_detected, fps = load_keypoints_file(
        input_path
    )

    validate_keypoints(keypoints)

    original_missing_frame_count = int(
        np.sum(pose_detected == 0)
    )

    print(f"Original keypoints shape: {keypoints.shape}")
    print(
        "Frames without detected pose: "
        f"{original_missing_frame_count}"
    )

    (
        trimmed_keypoints,
        trimmed_pose_detected,
        removed_start_frames,
        removed_end_frames
    ) = trim_undetected_edge_frames(
        keypoints,
        pose_detected
    )

    print(
        "Removed undetected frames from start: "
        f"{removed_start_frames}"
    )

    print(
        "Removed undetected frames from end: "
        f"{removed_end_frames}"
    )

    remaining_missing_frame_count = int(
        np.sum(trimmed_pose_detected == 0)
    )

    print(
        "Internal frames to interpolate: "
        f"{remaining_missing_frame_count}"
    )

    interpolated_keypoints = interpolate_keypoints(trimmed_keypoints)
    interpolated_keypoints[trimmed_pose_detected == 0,:,3] = 0.0

    if np.isnan(interpolated_keypoints).any():
        raise RuntimeError(
            "Some missing values remain after interpolation."
        )

    cleaned_keypoints = smooth_keypoints(interpolated_keypoints, SMOOTHING_WINDOW_SIZE)

    save_cleaned_keypoints(
        cleaned_keypoints,
        trimmed_pose_detected,
        fps,
        output_path
    )

    print(
        f"Cleaned keypoints shape: "
        f"{cleaned_keypoints.shape}"
    )

    print("Preprocessing completed successfully.")

def process_all_keypoints():
    """Preprocess all keypoint files."""
    keypoint_files = find_class_files(
        KEYPOINTS_DIR,
        "*.npz"
    )

    if not keypoint_files:
        raise ValueError(
            f"No keypoint files found in {KEYPOINTS_DIR}."
        )

    print(
        f"Found {len(keypoint_files)} keypoint files."
    )

    processed_count = 0
    skipped_count = 0

    for file_index, input_path in enumerate(
        keypoint_files,
        start=1
    ):
        output_path = create_output_path(
            input_path
        )

        if output_path.exists():
            print(
                f"\nSkipping file "
                f"{file_index}/{len(keypoint_files)} "
                f"(already processed):"
            )
            print(f"  Input: {input_path}")

            skipped_count += 1
            continue

        print(
            f"\nProcessing file "
            f"{file_index}/{len(keypoint_files)}:"
        )
        print(f"  Input: {input_path}")
        print(f"  Output: {output_path}")

        preprocess_keypoints(
            input_path,
            output_path
        )

        processed_count += 1

    print("\n" + "=" * 50)
    print("Preprocessing summary")
    print("=" * 50)
    print(f"Total files: {len(keypoint_files)}")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")

if __name__ == "__main__":
    process_all_keypoints()