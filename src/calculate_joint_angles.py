from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from file_utils import (
    find_class_files,
    load_cleaned_keypoints_file
)
from landmark_config import AP_CHAGI_LANDMARKS


CLEANED_DIR = Path("../processed/cleaned")
FEATURES_DIR = Path("../processed/features")

SUPPORT_FOOT_BASELINE_FRAMES = 5
ANGLE_BASELINE_FRAMES = 5


def calculate_angle(
    point_a,
    vertex,
    point_c
):
    """
    Calculate the angle formed by three 2D points.

    The vertex represents the joint at which
    the angle is calculated.
    """
    vector_a = point_a - vertex
    vector_c = point_c - vertex

    vector_a_length = np.linalg.norm(
        vector_a
    )

    vector_c_length = np.linalg.norm(
        vector_c
    )

    if (
        vector_a_length == 0
        or vector_c_length == 0
    ):
        return np.nan

    cosine_angle = np.dot(
        vector_a,
        vector_c
    ) / (
        vector_a_length
        * vector_c_length
    )

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0
    )

    angle_radians = np.arccos(
        cosine_angle
    )

    angle_degrees = np.degrees(
        angle_radians
    )

    return float(
        angle_degrees
    )


def create_output_paths(
    input_path
):
    """Create feature output paths based on the input file."""
    class_name = input_path.parent.name

    sample_name = input_path.stem.replace(
        "_cleaned",
        ""
    )

    output_dir = (
        FEATURES_DIR
        / class_name
    )

    output_path = (
        output_dir
        / f"{sample_name}_features.npz"
    )

    plot_output_path = (
        output_dir
        / f"{sample_name}_joint_angles.png"
    )

    trajectory_plot_output_path = (
        output_dir
        / f"{sample_name}_right_foot_trajectory.png"
    )

    kick_extension_plot_output_path = (
        output_dir
        / f"{sample_name}_kick_extension_trajectory.png"
    )

    return (
        output_path,
        plot_output_path,
        trajectory_plot_output_path,
        kick_extension_plot_output_path
    )


def calculate_knee_angles(
    keypoints
):
    """Calculate left and right knee angles for every frame."""
    num_frames = keypoints.shape[0]

    left_knee_angles = np.full(
        num_frames,
        np.nan,
        dtype=np.float32
    )

    right_knee_angles = np.full(
        num_frames,
        np.nan,
        dtype=np.float32
    )

    left_hip_id = AP_CHAGI_LANDMARKS[
        "left_hip"
    ]

    left_knee_id = AP_CHAGI_LANDMARKS[
        "left_knee"
    ]

    left_ankle_id = AP_CHAGI_LANDMARKS[
        "left_ankle"
    ]

    right_hip_id = AP_CHAGI_LANDMARKS[
        "right_hip"
    ]

    right_knee_id = AP_CHAGI_LANDMARKS[
        "right_knee"
    ]

    right_ankle_id = AP_CHAGI_LANDMARKS[
        "right_ankle"
    ]

    for frame_index in range(
        num_frames
    ):
        frame_keypoints = keypoints[
            frame_index
        ]

        left_hip = frame_keypoints[
            left_hip_id,
            :2
        ]

        left_knee = frame_keypoints[
            left_knee_id,
            :2
        ]

        left_ankle = frame_keypoints[
            left_ankle_id,
            :2
        ]

        right_hip = frame_keypoints[
            right_hip_id,
            :2
        ]

        right_knee = frame_keypoints[
            right_knee_id,
            :2
        ]

        right_ankle = frame_keypoints[
            right_ankle_id,
            :2
        ]

        left_knee_angles[
            frame_index
        ] = calculate_angle(
            left_hip,
            left_knee,
            left_ankle
        )

        right_knee_angles[
            frame_index
        ] = calculate_angle(
            right_hip,
            right_knee,
            right_ankle
        )

    return (
        left_knee_angles,
        right_knee_angles
    )


def calculate_right_hip_angles(
    keypoints
):
    """
    Calculate the right hip angle for every frame.

    The angle is formed by the right shoulder,
    right hip and right knee.
    """
    num_frames = keypoints.shape[0]

    right_hip_angles = np.full(
        num_frames,
        np.nan,
        dtype=np.float32
    )

    right_shoulder_id = AP_CHAGI_LANDMARKS[
        "right_shoulder"
    ]

    right_hip_id = AP_CHAGI_LANDMARKS[
        "right_hip"
    ]

    right_knee_id = AP_CHAGI_LANDMARKS[
        "right_knee"
    ]

    for frame_index in range(
        num_frames
    ):
        frame_keypoints = keypoints[
            frame_index
        ]

        right_shoulder = frame_keypoints[
            right_shoulder_id,
            :2
        ]

        right_hip = frame_keypoints[
            right_hip_id,
            :2
        ]

        right_knee = frame_keypoints[
            right_knee_id,
            :2
        ]

        right_hip_angles[
            frame_index
        ] = calculate_angle(
            right_shoulder,
            right_hip,
            right_knee
        )

    return right_hip_angles


def calculate_ankle_angles(
    keypoints
):
    """Calculate left and right ankle angles for every frame."""
    num_frames = keypoints.shape[0]

    left_ankle_angles = np.full(
        num_frames,
        np.nan,
        dtype=np.float32
    )

    right_ankle_angles = np.full(
        num_frames,
        np.nan,
        dtype=np.float32
    )

    left_knee_id = AP_CHAGI_LANDMARKS[
        "left_knee"
    ]

    left_ankle_id = AP_CHAGI_LANDMARKS[
        "left_ankle"
    ]

    left_foot_index_id = AP_CHAGI_LANDMARKS[
        "left_foot_index"
    ]

    right_knee_id = AP_CHAGI_LANDMARKS[
        "right_knee"
    ]

    right_ankle_id = AP_CHAGI_LANDMARKS[
        "right_ankle"
    ]

    right_foot_index_id = AP_CHAGI_LANDMARKS[
        "right_foot_index"
    ]

    for frame_index in range(
        num_frames
    ):
        frame_keypoints = keypoints[
            frame_index
        ]

        left_knee = frame_keypoints[
            left_knee_id,
            :2
        ]

        left_ankle = frame_keypoints[
            left_ankle_id,
            :2
        ]

        left_foot_index = frame_keypoints[
            left_foot_index_id,
            :2
        ]

        right_knee = frame_keypoints[
            right_knee_id,
            :2
        ]

        right_ankle = frame_keypoints[
            right_ankle_id,
            :2
        ]

        right_foot_index = frame_keypoints[
            right_foot_index_id,
            :2
        ]

        left_ankle_angles[
            frame_index
        ] = calculate_angle(
            left_knee,
            left_ankle,
            left_foot_index
        )

        right_ankle_angles[
            frame_index
        ] = calculate_angle(
            right_knee,
            right_ankle,
            right_foot_index
        )

    return (
        left_ankle_angles,
        right_ankle_angles
    )

def calculate_right_foot_angle(
    keypoints,
    frame_index
):
    """
    Calculate right foot posture angle at a selected frame.

    The angle is formed by:
    right heel -> right ankle -> right foot index.

    This is used as a proxy for foot/toe posture because
    MediaPipe does not provide individual toe landmarks.
    """
    right_heel_id = AP_CHAGI_LANDMARKS[
        "right_heel"
    ]

    right_ankle_id = AP_CHAGI_LANDMARKS[
        "right_ankle"
    ]

    right_foot_index_id = AP_CHAGI_LANDMARKS[
        "right_foot_index"
    ]

    frame_keypoints = keypoints[
        frame_index
    ]

    right_heel = frame_keypoints[
        right_heel_id,
        :2
    ]

    right_ankle = frame_keypoints[
        right_ankle_id,
        :2
    ]

    right_foot_index = frame_keypoints[
        right_foot_index_id,
        :2
    ]

    return calculate_angle(
        right_heel,
        right_ankle,
        right_foot_index
    )


def calculate_right_knee_height(
    keypoints
):
    """
    Calculate right knee height.

    This feature is kept for compatibility and
    event detection, but is not used in the
    final Random Forest feature set.
    """
    num_frames = keypoints.shape[0]

    right_knee_height = np.full(
        num_frames,
        np.nan,
        dtype=np.float32
    )

    left_shoulder_id = AP_CHAGI_LANDMARKS[
        "left_shoulder"
    ]

    right_shoulder_id = AP_CHAGI_LANDMARKS[
        "right_shoulder"
    ]

    left_hip_id = AP_CHAGI_LANDMARKS[
        "left_hip"
    ]

    right_hip_id = AP_CHAGI_LANDMARKS[
        "right_hip"
    ]

    right_knee_id = AP_CHAGI_LANDMARKS[
        "right_knee"
    ]

    for frame_index in range(
        num_frames
    ):
        frame_keypoints = keypoints[
            frame_index
        ]

        left_shoulder_y = frame_keypoints[
            left_shoulder_id,
            1
        ]

        right_shoulder_y = frame_keypoints[
            right_shoulder_id,
            1
        ]

        left_hip_y = frame_keypoints[
            left_hip_id,
            1
        ]

        right_hip_y = frame_keypoints[
            right_hip_id,
            1
        ]

        right_knee_y = frame_keypoints[
            right_knee_id,
            1
        ]

        shoulder_center_y = (
            left_shoulder_y
            + right_shoulder_y
        ) / 2.0

        hip_center_y = (
            left_hip_y
            + right_hip_y
        ) / 2.0

        chest_y = (
            shoulder_center_y
            + hip_center_y
        ) / 2.0

        right_knee_height[
            frame_index
        ] = (
            chest_y
            - right_knee_y
        )

    return right_knee_height


def calculate_right_foot_trajectory(
    keypoints
):
    """Extract right foot trajectory for every frame."""
    num_frames = keypoints.shape[0]

    right_foot_trajectory = np.full(
        (
            num_frames,
            2
        ),
        np.nan,
        dtype=np.float32
    )

    right_foot_index_id = AP_CHAGI_LANDMARKS[
        "right_foot_index"
    ]

    for frame_index in range(
        num_frames
    ):
        frame_keypoints = keypoints[
            frame_index
        ]

        right_foot_trajectory[
            frame_index
        ] = frame_keypoints[
            right_foot_index_id,
            :2
        ]

    return right_foot_trajectory


def find_max_right_knee_height(
    right_knee_height
):
    """Find maximum right knee height and its frame."""
    max_knee_height_frame = np.nanargmax(
        right_knee_height
    )

    max_right_knee_height = (
        right_knee_height[
            max_knee_height_frame
        ]
    )

    return (
        max_knee_height_frame,
        max_right_knee_height
    )


def find_knee_flexion_before_extension(
    knee_angles,
    reference_frame,
    fps
):
    """
    Find maximum knee flexion shortly before
    the kick reaches its highest point.
    """
    search_duration = 0.6

    start_frame = max(
        0,
        reference_frame
        - int(
            search_duration
            * fps
        )
    )

    flexion_angles = knee_angles[
        start_frame:
        reference_frame + 1
    ]

    relative_flexion_frame = np.nanargmin(
        flexion_angles
    )

    flexion_frame = (
        start_frame
        + relative_flexion_frame
    )

    min_knee_angle = knee_angles[
        flexion_frame
    ]

    return (
        flexion_frame,
        min_knee_angle
    )


def find_max_knee_extension(
    knee_angles,
    ankle_angles,
    start_frame,
    fps
):
    """
    Find maximum knee extension shortly after
    the start of the extension phase.
    """
    max_extension_duration = 0.5

    end_frame = min(
        start_frame
        + int(
            max_extension_duration
            * fps
        ),
        len(knee_angles)
    )

    extension_angles = knee_angles[
        start_frame:
        end_frame
    ]

    relative_max_frame = np.nanargmax(
        extension_angles
    )

    max_extension_frame = (
        start_frame
        + relative_max_frame
    )

    max_knee_angle = knee_angles[
        max_extension_frame
    ]

    ankle_angle_at_max_extension = (
        ankle_angles[
            max_extension_frame
        ]
    )

    return (
        max_extension_frame,
        max_knee_angle,
        ankle_angle_at_max_extension
    )


def calculate_trajectory_straightness(
    trajectory
):
    """Calculate straightness of a 2D trajectory."""
    if len(trajectory) < 2:
        return np.nan

    start_point = trajectory[0]
    end_point = trajectory[-1]

    direct_distance = np.linalg.norm(
        end_point
        - start_point
    )

    point_differences = np.diff(
        trajectory,
        axis=0
    )

    segment_lengths = np.linalg.norm(
        point_differences,
        axis=1
    )

    total_path_length = np.sum(
        segment_lengths
    )

    if total_path_length == 0:
        return np.nan

    straightness = (
        direct_distance
        / total_path_length
    )

    return float(
        straightness
    )



def calculate_baseline_angle(
    angle_values,
    baseline_frames=ANGLE_BASELINE_FRAMES
):
    """
    Calculate the average joint angle
    at the beginning of the video.
    """
    baseline_values = angle_values[
        :min(
            baseline_frames,
            len(angle_values)
        )
    ]

    baseline_values = baseline_values[
        np.isfinite(
            baseline_values
        )
    ]

    if len(
        baseline_values
    ) == 0:
        return np.nan

    return float(
        np.mean(
            baseline_values
        )
    )


def calculate_foot_orientation(
    heel,
    foot_index
):
    """
    Calculate foot orientation using
    the heel-to-foot-index vector.
    """
    foot_vector = (
        foot_index
        - heel
    )

    vector_length = np.linalg.norm(
        foot_vector
    )

    if vector_length == 0:
        return np.nan

    angle_radians = np.arctan2(
        foot_vector[1],
        foot_vector[0]
    )

    return float(
        np.degrees(
            angle_radians
        )
    )


def calculate_circular_mean(
    angles
):
    """Calculate mean angle while handling angle wrap-around."""
    angles = np.asarray(
        angles,
        dtype=np.float32
    )

    angles = angles[
        np.isfinite(
            angles
        )
    ]

    if len(
        angles
    ) == 0:
        return np.nan

    radians = np.radians(
        angles
    )

    mean_sin = np.mean(
        np.sin(
            radians
        )
    )

    mean_cos = np.mean(
        np.cos(
            radians
        )
    )

    return float(
        np.degrees(
            np.arctan2(
                mean_sin,
                mean_cos
            )
        )
    )


def calculate_angle_difference(
    angle,
    reference_angle
):
    """
    Calculate the smallest signed difference
    between two angles.
    """
    difference = (
        angle
        - reference_angle
        + 180.0
    ) % 360.0 - 180.0

    return float(
        difference
    )


def calculate_support_foot_rotation(
    keypoints,
    max_extension_frame
):
    """
    Calculate support-foot rotation at maximum kick extension.

    The left foot is the support foot.

    Rotation is calculated in the X-Z plane and measured
    relative to the average orientation of the foot
    at the beginning of the video.
    """
    left_heel_id = AP_CHAGI_LANDMARKS[
        "left_heel"
    ]

    left_foot_index_id = AP_CHAGI_LANDMARKS[
        "left_foot_index"
    ]

    baseline_count = min(
        SUPPORT_FOOT_BASELINE_FRAMES,
        len(keypoints)
    )

    baseline_orientations = []

    for frame_index in range(
        baseline_count
    ):
        frame_keypoints = keypoints[
            frame_index
        ]

        heel = frame_keypoints[
            left_heel_id,
            [0, 2]
        ]

        foot_index = frame_keypoints[
            left_foot_index_id,
            [0, 2]
        ]

        orientation = calculate_foot_orientation(
            heel,
            foot_index
        )

        if np.isfinite(
            orientation
        ):
            baseline_orientations.append(
                orientation
            )

    baseline_orientation = (
        calculate_circular_mean(
            baseline_orientations
        )
    )

    if not np.isfinite(
        baseline_orientation
    ):
        return (
            np.nan,
            np.nan
        )

    frame_keypoints = keypoints[
        max_extension_frame
    ]

    heel = frame_keypoints[
        left_heel_id,
        [0, 2]
    ]

    foot_index = frame_keypoints[
        left_foot_index_id,
        [0, 2]
    ]

    extension_orientation = (
        calculate_foot_orientation(
            heel,
            foot_index
        )
    )

    if not np.isfinite(
        extension_orientation
    ):
        return (
            np.nan,
            baseline_orientation
        )

    support_foot_rotation = abs(
        calculate_angle_difference(
            extension_orientation,
            baseline_orientation
        )
    )

    return (
        float(
            support_foot_rotation
        ),
        float(
            baseline_orientation
        )
    )


def save_features(
    output_path,
    left_knee_angles,
    right_knee_angles,
    right_hip_angles,
    left_ankle_angles,
    right_ankle_angles,
    right_knee_height,
    flexion_frame,
    min_right_knee_angle,
    right_hip_angle_at_flexion,
    max_extension_frame,
    max_right_knee_angle,
    right_ankle_angle_at_max_extension,
    left_knee_angle_at_max_extension,
    right_foot_angle_at_max_extension,
    support_foot_rotation,
    support_foot_baseline_orientation,
    baseline_right_knee_angle,
    baseline_left_knee_angle,
    baseline_right_hip_angle,
    baseline_right_ankle_angle,
    right_knee_flexion_change,
    right_hip_flexion_change,
    right_knee_extension_change,
    left_knee_change_at_max_extension,
    right_ankle_change_at_max_extension,
    max_knee_height_frame,
    max_right_knee_height,
    trajectory_straightness,
    pose_detected,
    fps
):
    """Save calculated movement features and metadata."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez_compressed(
        output_path,

        # Full angle sequences.
        left_knee_angles=(
            left_knee_angles
        ),

        right_knee_angles=(
            right_knee_angles
        ),

        right_hip_angles=(
            right_hip_angles
        ),

        left_ankle_angles=(
            left_ankle_angles
        ),

        right_ankle_angles=(
            right_ankle_angles
        ),

        right_knee_height=(
            right_knee_height
        ),

        # Preparation / chamber.
        flexion_frame=np.int32(
            flexion_frame
        ),

        min_right_knee_angle=np.float32(
            min_right_knee_angle
        ),

        right_hip_angle_at_flexion=np.float32(
            right_hip_angle_at_flexion
        ),

        # Maximum extension.
        max_extension_frame=np.int32(
            max_extension_frame
        ),

        max_right_knee_angle=np.float32(
            max_right_knee_angle
        ),

        right_ankle_angle_at_max_extension=np.float32(
            right_ankle_angle_at_max_extension
        ),

        left_knee_angle_at_max_extension=np.float32(
            left_knee_angle_at_max_extension
        ),

        right_foot_angle_at_max_extension=np.float32(
            right_foot_angle_at_max_extension
        ),

        # Support foot.
        support_foot_rotation=np.float32(
            support_foot_rotation
        ),

        support_foot_baseline_orientation=np.float32(
            support_foot_baseline_orientation
        ),

        # Baseline joint angles.
        baseline_right_knee_angle=np.float32(
            baseline_right_knee_angle
        ),

        baseline_left_knee_angle=np.float32(
            baseline_left_knee_angle
        ),

        baseline_right_hip_angle=np.float32(
            baseline_right_hip_angle
        ),

        baseline_right_ankle_angle=np.float32(
            baseline_right_ankle_angle
        ),

        # Relative features.
        right_knee_flexion_change=np.float32(
            right_knee_flexion_change
        ),

        right_hip_flexion_change=np.float32(
            right_hip_flexion_change
        ),

        right_knee_extension_change=np.float32(
            right_knee_extension_change
        ),

        left_knee_change_at_max_extension=np.float32(
            left_knee_change_at_max_extension
        ),

        right_ankle_change_at_max_extension=np.float32(
            right_ankle_change_at_max_extension
        ),

        # Additional diagnostic feature.
        max_knee_height_frame=np.int32(
            max_knee_height_frame
        ),

        max_right_knee_height=np.float32(
            max_right_knee_height
        ),

        # Trajectory.
        trajectory_straightness=np.float32(
            trajectory_straightness
        ),

        # Metadata.
        pose_detected=(
            pose_detected
        ),

        fps=np.float32(
            fps
        )
    )


def plot_joint_angles(
    left_knee_angles,
    right_knee_angles,
    left_ankle_angles,
    right_ankle_angles,
    fps,
    output_path
):
    """Plot left and right joint angles through time."""
    num_frames = len(
        left_knee_angles
    )

    time_seconds = np.arange(
        num_frames,
        dtype=np.float32
    ) / fps

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        time_seconds,
        left_knee_angles,
        label="Left knee"
    )

    plt.plot(
        time_seconds,
        right_knee_angles,
        label="Right knee"
    )

    plt.plot(
        time_seconds,
        left_ankle_angles,
        label="Left ankle"
    )

    plt.plot(
        time_seconds,
        right_ankle_angles,
        label="Right ankle"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Joint angle (degrees)"
    )

    plt.title(
        "Joint angles during Ap Chagi"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()


def plot_right_foot_trajectory(
    right_foot_trajectory,
    output_path
):
    """Plot the right foot trajectory in image coordinates."""
    x_values = (
        right_foot_trajectory[
            :,
            0
        ]
    )

    y_values = (
        right_foot_trajectory[
            :,
            1
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.plot(
        x_values,
        y_values,
        marker="o",
        markersize=2,
        linewidth=1
    )

    plt.scatter(
        x_values[0],
        y_values[0],
        label="Start"
    )

    plt.scatter(
        x_values[-1],
        y_values[-1],
        label="End"
    )

    plt.xlabel(
        "X"
    )

    plt.ylabel(
        "Y"
    )

    plt.title(
        "Right foot trajectory during Ap Chagi"
    )

    plt.gca().invert_yaxis()

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()


def plot_kick_extension_trajectory(
    kick_extension_trajectory,
    flexion_frame,
    max_extension_frame,
    output_path
):
    """Plot the right foot trajectory during kick extension."""
    x_values = (
        kick_extension_trajectory[
            :,
            0
        ]
    )

    y_values = (
        kick_extension_trajectory[
            :,
            1
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(
        figsize=(6, 6)
    )

    plt.plot(
        x_values,
        y_values,
        marker="o"
    )

    plt.scatter(
        x_values[0],
        y_values[0],
        label=(
            f"Start of extension "
            f"(frame {flexion_frame})"
        )
    )

    plt.scatter(
        x_values[-1],
        y_values[-1],
        label=(
            f"Maximum extension "
            f"(frame {max_extension_frame})"
        )
    )

    plt.annotate(
        f"Frame {flexion_frame}",
        (
            x_values[0],
            y_values[0]
        ),
        textcoords="offset points",
        xytext=(10, -15)
    )

    plt.annotate(
        f"Frame {max_extension_frame}",
        (
            x_values[-1],
            y_values[-1]
        ),
        textcoords="offset points",
        xytext=(10, 10)
    )

    plt.xlabel(
        "X"
    )

    plt.ylabel(
        "Y"
    )

    plt.title(
        "Right foot trajectory during kick extension"
    )

    plt.gca().invert_yaxis()

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()


def process_video_features(
    input_path,
    generate_plots=True
):
    """
    Calculate and save all movement features
    for one cleaned video sample.
    """
    (
        output_path,
        plot_output_path,
        trajectory_plot_output_path,
        kick_extension_plot_output_path
    ) = create_output_paths(
        input_path
    )

    (
        keypoints,
        pose_detected,
        _,
        fps
    ) = load_cleaned_keypoints_file(
        input_path
    )

    # -------------------------------------------------
    # Joint angle sequences
    # -------------------------------------------------

    (
        left_knee_angles,
        right_knee_angles
    ) = calculate_knee_angles(
        keypoints
    )

    right_hip_angles = (
        calculate_right_hip_angles(
            keypoints
        )
    )

    (
        left_ankle_angles,
        right_ankle_angles
    ) = calculate_ankle_angles(
        keypoints
    )

    # -------------------------------------------------
    # Baseline angles
    # -------------------------------------------------

    baseline_right_knee_angle = (
        calculate_baseline_angle(
            right_knee_angles
        )
    )

    baseline_left_knee_angle = (
        calculate_baseline_angle(
            left_knee_angles
        )
    )

    baseline_right_hip_angle = (
        calculate_baseline_angle(
            right_hip_angles
        )
    )

    baseline_right_ankle_angle = (
        calculate_baseline_angle(
            right_ankle_angles
        )
    )

    # -------------------------------------------------
    # Knee height and foot trajectory
    # -------------------------------------------------

    right_knee_height = (
        calculate_right_knee_height(
            keypoints
        )
    )

    right_foot_trajectory = (
        calculate_right_foot_trajectory(
            keypoints
        )
    )

    # -------------------------------------------------
    # Detect preparation / chamber phase
    # -------------------------------------------------

    (
        max_knee_height_frame,
        max_right_knee_height
    ) = find_max_right_knee_height(
        right_knee_height
    )

    (
        flexion_frame,
        min_right_knee_angle
    ) = find_knee_flexion_before_extension(
        right_knee_angles,
        max_knee_height_frame,
        fps
    )

    right_hip_angle_at_flexion = (
        right_hip_angles[
            flexion_frame
        ]
    )

    # -------------------------------------------------
    # Detect maximum kick extension
    # -------------------------------------------------

    (
        max_extension_frame,
        max_right_knee_angle,
        right_ankle_angle_at_max_extension
    ) = find_max_knee_extension(
        right_knee_angles,
        right_ankle_angles,
        flexion_frame,
        fps
    )

    left_knee_angle_at_max_extension = (
        left_knee_angles[
            max_extension_frame
        ]
    )

    right_foot_angle_at_max_extension = (
        calculate_right_foot_angle(
            keypoints,
            max_extension_frame
        )
    )

    # -------------------------------------------------
    # Support foot rotation
    # -------------------------------------------------

    (
        support_foot_rotation,
        support_foot_baseline_orientation
    ) = calculate_support_foot_rotation(
        keypoints,
        max_extension_frame
    )

    # -------------------------------------------------
    # Relative features
    # -------------------------------------------------

    right_knee_flexion_change = (
        baseline_right_knee_angle
        - min_right_knee_angle
    )

    right_hip_flexion_change = (
        baseline_right_hip_angle
        - right_hip_angle_at_flexion
    )

    right_knee_extension_change = (
        max_right_knee_angle
        - min_right_knee_angle
    )

    left_knee_change_at_max_extension = (
        baseline_left_knee_angle
        - left_knee_angle_at_max_extension
    )

    right_ankle_change_at_max_extension = (
        right_ankle_angle_at_max_extension
        - baseline_right_ankle_angle
    )

    # -------------------------------------------------
    # Kick trajectory
    # -------------------------------------------------

    kick_extension_trajectory = (
        right_foot_trajectory[
            flexion_frame:
            max_extension_frame + 1
        ]
    )

    trajectory_straightness = (
        calculate_trajectory_straightness(
            kick_extension_trajectory
        )
    )

    # -------------------------------------------------
    # Save features
    # -------------------------------------------------

    save_features(
        output_path,
        left_knee_angles,
        right_knee_angles,
        right_hip_angles,
        left_ankle_angles,
        right_ankle_angles,
        right_knee_height,
        flexion_frame,
        min_right_knee_angle,
        right_hip_angle_at_flexion,
        max_extension_frame,
        max_right_knee_angle,
        right_ankle_angle_at_max_extension,
        left_knee_angle_at_max_extension,
        right_foot_angle_at_max_extension,
        support_foot_rotation,
        support_foot_baseline_orientation,
        baseline_right_knee_angle,
        baseline_left_knee_angle,
        baseline_right_hip_angle,
        baseline_right_ankle_angle,
        right_knee_flexion_change,
        right_hip_flexion_change,
        right_knee_extension_change,
        left_knee_change_at_max_extension,
        right_ankle_change_at_max_extension,
        max_knee_height_frame,
        max_right_knee_height,
        trajectory_straightness,
        pose_detected,
        fps
    )

    # -------------------------------------------------
    # Plots
    # -------------------------------------------------

    if generate_plots:
        plot_kick_extension_trajectory(
            kick_extension_trajectory,
            flexion_frame,
            max_extension_frame,
            kick_extension_plot_output_path
        )

        plot_joint_angles(
            left_knee_angles,
            right_knee_angles,
            left_ankle_angles,
            right_ankle_angles,
            fps,
            plot_output_path
        )

        plot_right_foot_trajectory(
            right_foot_trajectory,
            trajectory_plot_output_path
        )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    print(
        "\n--- Ap Chagi Feature Summary ---"
    )

    print(
        f"Processed frames: "
        f"{len(right_knee_angles)}"
    )

    print(
        "\nKnee preparation:"
    )

    print(
        f"  Knee flexion frame: "
        f"{flexion_frame}"
    )

    print(
        f"  Right knee angle at flexion: "
        f"{min_right_knee_angle:.2f}°"
    )

    print(
        f"  Right hip angle at flexion: "
        f"{right_hip_angle_at_flexion:.2f}°"
    )

    print(
        "\nKnee extension:"
    )

    print(
        f"  Maximum extension frame: "
        f"{max_extension_frame}"
    )

    print(
        f"  Maximum right knee angle: "
        f"{max_right_knee_angle:.2f}°"
    )

    print(
        f"  Left knee angle at maximum extension: "
        f"{left_knee_angle_at_max_extension:.2f}°"
    )

    print(
        f"  Right ankle angle at maximum extension: "
        f"{right_ankle_angle_at_max_extension:.2f}°"
    )

    print(
        f"  Right foot angle at maximum extension: "
        f"{right_foot_angle_at_max_extension:.2f}°"
    )

    print(
        "\nSupport foot:"
    )

    print(
        f"  Support foot rotation at maximum extension: "
        f"{support_foot_rotation:.2f}°"
    )

    print(
        "\nRelative features:"
    )

    print(
        f"  Right knee flexion change: "
        f"{right_knee_flexion_change:.2f}°"
    )

    print(
        f"  Right hip flexion change: "
        f"{right_hip_flexion_change:.2f}°"
    )

    print(
        f"  Right knee extension change: "
        f"{right_knee_extension_change:.2f}°"
    )

    print(
        f"  Left knee change at maximum extension: "
        f"{left_knee_change_at_max_extension:.2f}°"
    )

    print(
        f"  Right ankle change at maximum extension: "
        f"{right_ankle_change_at_max_extension:.2f}°"
    )

    print(
        "\nKick trajectory:"
    )

    print(
        f"  Extension frames: "
        f"{flexion_frame} -> {max_extension_frame}"
    )

    print(
        f"  Number of trajectory points: "
        f"{len(kick_extension_trajectory)}"
    )

    print(
        f"  Trajectory straightness: "
        f"{trajectory_straightness:.3f}"
    )

    print(
        "\nLegacy/debug feature:"
    )

    print(
        f"  Maximum right knee height: "
        f"{max_right_knee_height:.4f}"
    )

    print(
        "\nSaved files:"
    )

    print(
        f"  Features: "
        f"{output_path}"
    )

    print(
        f"  Joint angles plot: "
        f"{plot_output_path}"
    )

    print(
        f"  Foot trajectory plot: "
        f"{trajectory_plot_output_path}"
    )

    print(
        f"  Kick extension trajectory plot: "
        f"{kick_extension_plot_output_path}"
    )

    print(
        "---------------------------------\n"
    )


def process_all_cleaned_files():
    """Calculate features for all cleaned files."""
    cleaned_files = find_class_files(
        CLEANED_DIR,
        "*_cleaned.npz"
    )

    if not cleaned_files:
        raise ValueError(
            f"No cleaned files found in "
            f"{CLEANED_DIR}."
        )

    print(
        f"Found "
        f"{len(cleaned_files)} "
        f"cleaned files."
    )

    for (
        file_index,
        input_path
    ) in enumerate(
        cleaned_files,
        start=1
    ):
        print(
            f"\nProcessing file "
            f"{file_index}/"
            f"{len(cleaned_files)}:"
        )

        print(
            f"  Input: "
            f"{input_path}"
        )

        process_video_features(
            input_path,
            generate_plots=True
        )


if __name__ == "__main__":
    process_all_cleaned_files()