from pathlib import Path
import subprocess
import sys
import time


SRC_DIR = Path(__file__).resolve().parent


PIPELINE_STEPS = [
    (
        "Extracting keypoints",
        "extract_keypoints.py"
    ),
    (
        "Preprocessing keypoints",
        "preprocess_keypoints.py"
    ),
    (
        "Extracting movement features for Random Forest",
        "calculate_joint_angles.py"
    ),
    (
        "Building Random Forest dataset",
        "build_dataset.py"
    ),
    (
        "Building spatio-temporal graphs for GCN",
        "graph_builder.py"
    ),
    (
        "Training and evaluating models GCN and Random Forest with 5-fold cross-validation",
        "train_cross_validation.py"
    ),
]


def run_script(description, script_name):
    """Run one pipeline step."""
    script_path = SRC_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Script not found: {script_path}"
        )

    print()
    print("=" * 70)
    print(f"START: {description}")
    print(f"Script: {script_name}")
    print("=" * 70)

    start_time = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=SRC_DIR
    )

    elapsed_time = time.time() - start_time

    if result.returncode != 0:
        print()
        print("=" * 70)
        print(f"FAILED: {description}")
        print(f"Pipeline stopped at: {script_name}")
        print("=" * 70)

        sys.exit(result.returncode)

    print(
        f"DONE: {description} "
        f"({elapsed_time:.2f} s)"
    )


def main():
    """Run the complete Ap Chagi pipeline."""
    print()
    print("=" * 70)
    print("AP CHAGI - FULL PIPELINE")
    print("=" * 70)

    total_start_time = time.time()

    for description, script_name in PIPELINE_STEPS:
        run_script(
            description,
            script_name
        )

    total_time = time.time() - total_start_time

    print()
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print(f"Total execution time: {total_time:.2f} s")
    print("=" * 70)


if __name__ == "__main__":
    main()