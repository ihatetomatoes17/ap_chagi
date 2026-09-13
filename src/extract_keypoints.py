from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np

from file_utils import find_class_files
from extract_pose import open_video, create_pose_landmarker
from landmark_config import NUM_FEATURES, NUM_LANDMARKS

MODEL_PATH = Path("../models/pose_landmarker_full.task")
DATASET_DIR = Path("../dataset/ap_chagi")
KEYPOINTS_DIR = Path("../processed/keypoints")


def create_output_path(video_path):
    """Create the keypoint output path for a video."""
    class_name = video_path.parent.name

    output_path = (
        KEYPOINTS_DIR
        / class_name
        / f"{video_path.stem}.npz"
    )

    return output_path

def landmarks_to_array(landmarks):
    """Convert MediaPipe landmarks to a NumPy array."""
    keypoints = np.zeros(
        (NUM_LANDMARKS, NUM_FEATURES),
        dtype=np.float32
    )

    for landmark_id, landmark in enumerate(landmarks):
        keypoints[landmark_id] = [
            landmark.x,
            landmark.y,
            landmark.z,
            landmark.visibility
        ]

    return keypoints

def save_keypoints(
    keypoints,
    pose_detected,
    fps,
    output_path
):
    """Validate and save keypoints and metadata to an NPZ file."""
    if keypoints.ndim != 3:
        raise ValueError(
            "Keypoints must have three dimensions."
        )

    expected_shape = (
        NUM_LANDMARKS,
        NUM_FEATURES
    )

    if keypoints.shape[1:] != expected_shape:
        raise ValueError(
            "Expected keypoints shape "
            f"(num_frames, {NUM_LANDMARKS}, {NUM_FEATURES}), "
            f"but received {keypoints.shape}."
        )

    if len(pose_detected) != keypoints.shape[0]:
        raise ValueError(
            "The number of pose detection values must "
            "match the number of video frames."
        )

    if fps <= 0:
        raise ValueError(
            "FPS must be greater than zero."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        output_path,
        keypoints=keypoints,
        pose_detected=pose_detected,
        fps=fps
    )

    print(f"Saved file: {output_path}")
    print(f"Keypoints shape: {keypoints.shape}")

def extract_keypoints(video_path, output_path):
    """Extract pose keypoints from a video and save them."""
    video_capture, fps = open_video(video_path)

    all_keypoints = []
    pose_detected_flags = []

    try:
        with create_pose_landmarker(MODEL_PATH) as landmarker:
            frame_id = 0

            while True:
                success, frame = video_capture.read()

                if not success:
                    break

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = int(
                    frame_id * 1000 / fps
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                if result.pose_landmarks:
                    frame_keypoints = landmarks_to_array(
                        result.pose_landmarks[0]
                    )

                    all_keypoints.append(
                        frame_keypoints
                    )

                    pose_detected_flags.append(1)

                else:
                    empty_keypoints = np.full(
                        (NUM_LANDMARKS, NUM_FEATURES),
                        np.nan,
                        dtype=np.float32
                    )

                    all_keypoints.append(
                        empty_keypoints
                    )

                    pose_detected_flags.append(0)

                frame_id += 1

        if not all_keypoints:
            raise ValueError(
                "No video frames were read."
            )

        keypoints_array = np.stack(
            all_keypoints
        ).astype(np.float32)

        pose_detected_array = np.asarray(
            pose_detected_flags,
            dtype=np.uint8
        )

        save_keypoints(
            keypoints_array,
            pose_detected_array,
            fps,
            output_path
        )

        detected_frames = int(
            np.sum(pose_detected_array)
        )

        total_frames = len(
            pose_detected_array
        )

        detection_percentage = (
            detected_frames / total_frames * 100
        )

        print(
            "Pose detected in: "
            f"{detected_frames}/{total_frames} frames "
            f"({detection_percentage:.2f}%)"
        )

    finally:
        video_capture.release()

def process_dataset():
    """Extract keypoints from all videos in the dataset."""
    video_files = find_class_files(
        DATASET_DIR,
        (
            "*.mp4",
            "*.mov",
            "*.avi"
        )
    )

    if not video_files:
        raise ValueError(
            f"No video files found in {DATASET_DIR}."
        )

    print(
        f"Found {len(video_files)} video files."
    )

    for video_index, video_path in enumerate(
        video_files,
        start=1
    ):
        output_path = create_output_path(
            video_path
        )

        if output_path.exists():
            print(
                f"\nSkipping video "
                f"{video_index}/{len(video_files)} "
                f"(already processed):"
            )
            print(f"  Input: {video_path}")
            continue

        print(
            f"\nProcessing video "
            f"{video_index}/{len(video_files)}:"
        )

        extract_keypoints(
            video_path,
            output_path
        )

if __name__ == "__main__":
    process_dataset()