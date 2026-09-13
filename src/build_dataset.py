from pathlib import Path
import csv

import numpy as np

from file_utils import find_class_files


FEATURES_DIR = Path("../processed/features")
DATASET_DIR = Path("../processed/dataset")

CSV_OUTPUT_PATH = DATASET_DIR / "ap_chagi_features.csv"
NPZ_OUTPUT_PATH = DATASET_DIR / "ap_chagi_features.npz"


FEATURE_NAMES = [
    "min_right_knee_angle",
    "right_hip_angle_at_flexion",
    "max_right_knee_angle",
    "left_knee_angle_at_max_extension",
    "right_ankle_angle_at_max_extension",
    "support_foot_rotation",
    "trajectory_straightness",
]


LABEL_MAP = {
    "incorrect": 0,
    "correct": 1
}


def load_feature_sample(input_path):
    """Load scalar features and label for one video sample."""
    class_name = input_path.parent.name

    if class_name not in LABEL_MAP:
        raise ValueError(
            f"Unknown class: {class_name}"
        )

    with np.load(input_path) as data:
        missing_features = [
            feature_name
            for feature_name in FEATURE_NAMES
            if feature_name not in data
        ]

        if missing_features:
            raise ValueError(
                f"Missing features in {input_path}: "
                f"{missing_features}"
            )

        feature_values = np.array(
            [
                float(data[feature_name])
                for feature_name in FEATURE_NAMES
            ],
            dtype=np.float32
        )

    invalid_features = []

    for feature_name, feature_value in zip(
            FEATURE_NAMES,
            feature_values
    ):
        if not np.isfinite(feature_value):
            invalid_features.append(
                (
                    feature_name,
                    feature_value
                )
            )

    if invalid_features:
        print(
            f"\nInvalid features in: {input_path}"
        )

        for feature_name, feature_value in invalid_features:
            print(
                f"  {feature_name}: {feature_value}"
            )

        raise ValueError(
            f"Invalid feature values in {input_path}."
        )

    sample_name = input_path.stem.replace(
        "_features",
        ""
    )

    label = LABEL_MAP[class_name]

    return (
        sample_name,
        class_name,
        feature_values,
        label
    )


def save_dataset_csv(
    samples,
    output_path
):
    """Save the feature dataset as a readable CSV file."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "sample_name",
                "class_name",
                *FEATURE_NAMES,
                "label"
            ]
        )

        for (
            sample_name,
            class_name,
            feature_values,
            label
        ) in samples:
            writer.writerow(
                [
                    sample_name,
                    class_name,
                    *feature_values.tolist(),
                    label
                ]
            )


def save_dataset_npz(
    samples,
    output_path
):
    """Save the feature dataset for machine learning."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_matrix = np.stack(
        [
            sample[2]
            for sample in samples
        ]
    ).astype(np.float32)

    labels = np.asarray(
        [
            sample[3]
            for sample in samples
        ],
        dtype=np.uint8
    )

    sample_names = np.asarray(
        [
            sample[0]
            for sample in samples
        ]
    )

    class_names = np.asarray(
        [
            sample[1]
            for sample in samples
        ]
    )

    np.savez_compressed(
        output_path,
        features=feature_matrix,
        labels=labels,
        feature_names=np.asarray(FEATURE_NAMES),
        sample_names=sample_names,
        class_names=class_names
    )


def build_dataset():
    """Build one dataset from all extracted feature files."""
    feature_files = find_class_files(
        FEATURES_DIR,
        "*_features.npz"
    )

    if not feature_files:
        raise ValueError(
            f"No feature files found in {FEATURES_DIR}."
        )

    samples = []

    for input_path in feature_files:
        sample = load_feature_sample(
            input_path
        )

        samples.append(sample)

    save_dataset_csv(
        samples,
        CSV_OUTPUT_PATH
    )

    save_dataset_npz(
        samples,
        NPZ_OUTPUT_PATH
    )

    labels = np.asarray(
        [
            sample[3]
            for sample in samples
        ]
    )

    correct_count = int(
        np.sum(labels == LABEL_MAP["correct"])
    )

    incorrect_count = int(
        np.sum(labels == LABEL_MAP["incorrect"])
    )

    print("\n--- Dataset Summary ---")
    print(f"Total samples: {len(samples)}")
    print(f"Correct samples: {correct_count}")
    print(f"Incorrect samples: {incorrect_count}")
    print(f"Number of features: {len(FEATURE_NAMES)}")

    print("\nFeatures:")
    for feature_name in FEATURE_NAMES:
        print(f"  {feature_name}")

    print("\nSaved files:")
    print(f"  CSV: {CSV_OUTPUT_PATH}")
    print(f"  NPZ: {NPZ_OUTPUT_PATH}")
    print("-----------------------\n")


if __name__ == "__main__":
    build_dataset()