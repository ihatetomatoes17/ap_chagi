from pathlib import Path
import csv
import random
import re

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold

from torch import nn
from torch.optim import Adam
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from graph_dataset import ApChagiGraphDataset
from gcn_model import ApChagiGCN


RF_DATASET_PATH = Path(
    "../processed/dataset/ap_chagi_features.npz"
)

MODEL_OUTPUT_DIR = Path(
    "../processed/models"
)

RESULTS_DIR = Path(
    "../processed/results"
)

RESULTS_OUTPUT_PATH = (
    RESULTS_DIR
    / "cross_validation_results.csv"
)

COMPARISON_PLOT_PATH = (
    RESULTS_DIR
    / "model_comparison_cross_validation.png"
)


NUM_FOLDS = 5

BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

NUM_TREES = 200

RANDOM_SEED = 42


def set_random_seed(seed):
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_random_forest_dataset():
    """Load handcrafted features for Random Forest."""
    if not RF_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RF_DATASET_PATH}"
        )

    with np.load(RF_DATASET_PATH) as data:
        features = data[
            "features"
        ].astype(np.float32)

        labels = data[
            "labels"
        ].astype(np.int64)

        feature_names = data[
            "feature_names"
        ].astype(str)

        sample_names = data[
            "sample_names"
        ].astype(str)

    if features.ndim != 2:
        raise ValueError(
            f"Invalid feature shape: {features.shape}"
        )

    if len(features) != len(labels):
        raise ValueError(
            "Number of features and labels does not match."
        )

    if len(features) != len(sample_names):
        raise ValueError(
            "Number of samples and sample names does not match."
        )

    if not np.all(
        np.isfinite(features)
    ):
        raise ValueError(
            "Dataset contains invalid feature values."
        )

    return (
        features,
        labels,
        feature_names,
        sample_names
    )


def extract_participant_id(sample_name):
    """Extract participant ID from sample name."""
    match = re.search(
        r"(p\d+)",
        sample_name
    )

    if match is None:
        raise ValueError(
            f"Could not extract participant "
            f"from sample: {sample_name}"
        )

    return match.group(1)


def get_participant_ids(sample_names):
    """Return participant ID for every sample."""
    return np.array(
        [
            extract_participant_id(
                sample_name
            )
            for sample_name in sample_names
        ]
    )


def calculate_metrics(
    labels,
    predictions
):
    """Calculate classification metrics."""
    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0
    )

    matrix = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix
    }


def create_random_forest():
    """Create Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=NUM_TREES,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )


def validate_datasets(
    graph_dataset,
    sample_names,
    labels,
    participant_ids
):
    """
    Check that GCN and Random Forest datasets
    contain the same samples and labels.
    """
    graph_lookup = {}

    for index in range(
        len(graph_dataset)
    ):
        graph = graph_dataset[index]

        graph_lookup[
            graph.sample_name
        ] = {
            "index": index,
            "label": int(
                graph.y.item()
            ),
            "participant_id":
                graph.participant_id
        }

    rf_sample_names = set(
        sample_names.tolist()
    )

    graph_sample_names = set(
        graph_lookup.keys()
    )

    if (
        rf_sample_names
        != graph_sample_names
    ):
        raise ValueError(
            "GCN and Random Forest datasets "
            "do not contain the same samples."
        )

    for (
        sample_name,
        label,
        participant_id
    ) in zip(
        sample_names,
        labels,
        participant_ids
    ):
        graph_info = graph_lookup[
            sample_name
        ]

        if (
            graph_info["label"]
            != int(label)
        ):
            raise ValueError(
                f"Label mismatch for "
                f"{sample_name}."
            )

        if (
            graph_info["participant_id"]
            != participant_id
        ):
            raise ValueError(
                f"Participant mismatch for "
                f"{sample_name}."
            )

    print(
        "Dataset validation passed: "
        "samples and labels match."
    )

    return graph_lookup


def create_stratification_labels(
    participant_ids,
    labels
):
    """
    Create stratification labels using
    participant ID and class label.
    """
    return np.array(
        [
            f"{participant_id}_{label}"
            for participant_id, label
            in zip(
                participant_ids,
                labels
            )
        ]
    )


def create_folds(
    participant_ids,
    labels
):
    """Create five stratified folds."""
    stratification_labels = (
        create_stratification_labels(
            participant_ids,
            labels
        )
    )

    splitter = StratifiedKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED
    )

    return list(
        splitter.split(
            np.zeros(
                len(labels)
            ),
            stratification_labels
        )
    )


def count_labels(labels):
    """Count incorrect and correct labels."""
    incorrect_count = int(
        np.sum(
            labels == 0
        )
    )

    correct_count = int(
        np.sum(
            labels == 1
        )
    )

    return (
        incorrect_count,
        correct_count
    )


def print_fold_distribution(
    fold_number,
    train_indices,
    test_indices,
    labels,
    participant_ids
):
    """Print train and test distribution."""
    train_labels = labels[
        train_indices
    ]

    test_labels = labels[
        test_indices
    ]

    (
        train_incorrect,
        train_correct
    ) = count_labels(
        train_labels
    )

    (
        test_incorrect,
        test_correct
    ) = count_labels(
        test_labels
    )

    print()
    print("=" * 60)
    print(f"FOLD {fold_number}")
    print("=" * 60)

    print(
        f"Training samples: "
        f"{len(train_indices)}"
    )

    print(
        f"  Incorrect: "
        f"{train_incorrect}"
    )

    print(
        f"  Correct: "
        f"{train_correct}"
    )

    print(
        f"Test samples: "
        f"{len(test_indices)}"
    )

    print(
        f"  Incorrect: "
        f"{test_incorrect}"
    )

    print(
        f"  Correct: "
        f"{test_correct}"
    )

    print(
        "\nTest samples by participant:"
    )

    for participant_id in sorted(
        np.unique(
            participant_ids
        )
    ):
        participant_mask = (
            participant_ids[
                test_indices
            ]
            == participant_id
        )

        participant_labels = (
            test_labels[
                participant_mask
            ]
        )

        (
            incorrect_count,
            correct_count
        ) = count_labels(
            participant_labels
        )

        print(
            f"  {participant_id}: "
            f"{incorrect_count} incorrect, "
            f"{correct_count} correct"
        )


def calculate_class_weights(
    dataset,
    device
):
    """Calculate GCN class weights."""
    incorrect_count = 0
    correct_count = 0

    for graph in dataset:
        label = int(
            graph.y.item()
        )

        if label == 0:
            incorrect_count += 1
        else:
            correct_count += 1

    total_count = (
        incorrect_count
        + correct_count
    )

    incorrect_weight = (
        total_count
        / (2.0 * incorrect_count)
    )

    correct_weight = (
        total_count
        / (2.0 * correct_count)
    )

    return torch.tensor(
        [
            incorrect_weight,
            correct_weight
        ],
        dtype=torch.float32,
        device=device
    )


def train_gcn_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device
):
    """Train GCN for one epoch."""
    model.train()

    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch in loader:
        batch = batch.to(
            device
        )

        optimizer.zero_grad()

        logits = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )

        labels = batch.y.view(-1)

        loss = criterion(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        batch_size = (
            batch.num_graphs
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        correct_predictions += int(
            (
                predictions
                == labels
            )
            .sum()
            .item()
        )

        total_samples += batch_size

    average_loss = (
        total_loss
        / total_samples
    )

    accuracy = (
        correct_predictions
        / total_samples
    )

    return (
        average_loss,
        accuracy
    )


def predict_gcn(
    model,
    loader,
    device
):
    """Generate GCN predictions."""
    model.eval()

    labels = []
    predictions = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(
                device
            )

            logits = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )

            batch_predictions = (
                torch.argmax(
                    logits,
                    dim=1
                )
            )

            batch_labels = (
                batch.y.view(-1)
            )

            labels.extend(
                batch_labels
                .cpu()
                .numpy()
                .tolist()
            )

            predictions.extend(
                batch_predictions
                .cpu()
                .numpy()
                .tolist()
            )

    return (
        labels,
        predictions
    )


def save_random_forest_model(
    model,
    feature_names,
    fold_number
):
    """Save Random Forest model."""
    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        MODEL_OUTPUT_DIR
        / (
            f"ap_chagi_random_forest_"
            f"cv_fold_{fold_number}.joblib"
        )
    )

    joblib.dump(
        {
            "model": model,
            "feature_names":
                feature_names,
            "fold":
                fold_number
        },
        output_path
    )

    print(
        f"Random Forest saved to: "
        f"{output_path}"
    )


def save_gcn_model(
    model,
    fold_number
):
    """Save GCN model."""
    MODEL_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        MODEL_OUTPUT_DIR
        / (
            f"ap_chagi_gcn_"
            f"cv_fold_{fold_number}.pt"
        )
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),
            "fold":
                fold_number,
            "input_features": 4,
            "hidden_channels": 64,
            "num_classes": 2
        },
        output_path
    )

    print(
        f"GCN saved to: "
        f"{output_path}"
    )


def train_random_forest_fold(
    features,
    labels,
    feature_names,
    train_indices,
    test_indices,
    fold_number
):
    """Train Random Forest for one fold."""
    train_features = (
        features[
            train_indices
        ]
    )

    train_labels = (
        labels[
            train_indices
        ]
    )

    test_features = (
        features[
            test_indices
        ]
    )

    test_labels = (
        labels[
            test_indices
        ]
    )

    model = create_random_forest()

    model.fit(
        train_features,
        train_labels
    )

    predictions = model.predict(
        test_features
    )

    metrics = calculate_metrics(
        test_labels,
        predictions
    )

    save_random_forest_model(
        model,
        feature_names,
        fold_number
    )

    return metrics


def train_gcn_fold(
    graph_dataset,
    graph_lookup,
    sample_names,
    train_indices,
    test_indices,
    fold_number,
    device
):
    """Train GCN for one fold."""
    set_random_seed(
        RANDOM_SEED
    )

    train_sample_names = (
        sample_names[
            train_indices
        ]
    )

    test_sample_names = (
        sample_names[
            test_indices
        ]
    )

    graph_train_indices = [
        graph_lookup[
            sample_name
        ]["index"]
        for sample_name
        in train_sample_names
    ]

    graph_test_indices = [
        graph_lookup[
            sample_name
        ]["index"]
        for sample_name
        in test_sample_names
    ]

    train_dataset = Subset(
        graph_dataset,
        graph_train_indices
    )

    test_dataset = Subset(
        graph_dataset,
        graph_test_indices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    model = ApChagiGCN().to(
        device
    )

    class_weights = (
        calculate_class_weights(
            train_dataset,
            device
        )
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    for epoch in range(
        1,
        NUM_EPOCHS + 1
    ):
        (
            loss,
            train_accuracy
        ) = train_gcn_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == NUM_EPOCHS
        ):
            print(
                f"Epoch "
                f"{epoch:03d}/"
                f"{NUM_EPOCHS} | "
                f"Loss: "
                f"{loss:.4f} | "
                f"Accuracy: "
                f"{train_accuracy:.3f}"
            )

    (
        test_labels,
        predictions
    ) = predict_gcn(
        model,
        test_loader,
        device
    )

    metrics = calculate_metrics(
        test_labels,
        predictions
    )

    save_gcn_model(
        model,
        fold_number
    )

    return metrics


def print_metrics(
    model_name,
    metrics
):
    """Print model metrics."""
    print(
        f"\n{model_name}"
    )

    print(
        f"  Accuracy:  "
        f"{metrics['accuracy']:.3f}"
    )

    print(
        f"  Precision: "
        f"{metrics['precision']:.3f}"
    )

    print(
        f"  Recall:    "
        f"{metrics['recall']:.3f}"
    )

    print(
        f"  F1:        "
        f"{metrics['f1']:.3f}"
    )

    print(
        "  Confusion matrix:"
    )

    print(
        metrics[
            "confusion_matrix"
        ]
    )


def calculate_average(
    results,
    model_name,
    metric_name
):
    """Calculate average metric."""
    values = [
        result[
            metric_name
        ]
        for result in results
        if (
            result["model"]
            == model_name
        )
    ]

    return float(
        np.mean(
            values
        )
    )


def save_results(results):
    """Save cross-validation results."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "fold",
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]

    with open(
        RESULTS_OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "fold":
                        result["fold"],
                    "model":
                        result["model"],
                    "accuracy":
                        result["accuracy"],
                    "precision":
                        result["precision"],
                    "recall":
                        result["recall"],
                    "f1":
                        result["f1"]
                }
            )

    print(
        f"\nResults saved to: "
        f"{RESULTS_OUTPUT_PATH}"
    )


def save_comparison_plot(results):
    """Save average model comparison plot."""
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1"
    ]

    display_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]

    gcn_values = [
        calculate_average(
            results,
            "GCN",
            metric
        )
        for metric in metric_names
    ]

    rf_values = [
        calculate_average(
            results,
            "Random Forest",
            metric
        )
        for metric in metric_names
    ]

    x_positions = np.arange(
        len(metric_names)
    )

    width = 0.35

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    gcn_bars = axis.bar(
        x_positions - width / 2,
        gcn_values,
        width,
        label="GCN"
    )

    rf_bars = axis.bar(
        x_positions + width / 2,
        rf_values,
        width,
        label="Random Forest"
    )

    axis.bar_label(
        gcn_bars,
        fmt="%.3f",
        padding=3
    )

    axis.bar_label(
        rf_bars,
        fmt="%.3f",
        padding=3
    )

    axis.set_ylabel(
        "Score"
    )

    axis.set_title(
        "Average 5-Fold Cross-Validation Results"
    )

    axis.set_xticks(
        x_positions
    )

    axis.set_xticklabels(
        display_names
    )

    axis.set_ylim(
        0.0,
        1.0
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.3
    )

    figure.tight_layout()

    figure.savefig(
        COMPARISON_PLOT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(
        figure
    )

    print(
        f"Comparison plot saved to: "
        f"{COMPARISON_PLOT_PATH}"
    )


def run_cross_validation():
    """Run five-fold cross-validation."""
    set_random_seed(
        RANDOM_SEED
    )

    (
        features,
        labels,
        feature_names,
        sample_names
    ) = load_random_forest_dataset()

    participant_ids = (
        get_participant_ids(
            sample_names
        )
    )

    graph_dataset = (
        ApChagiGraphDataset()
    )

    graph_lookup = validate_datasets(
        graph_dataset,
        sample_names,
        labels,
        participant_ids
    )

    folds = create_folds(
        participant_ids,
        labels
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    print(
        f"Samples: {len(labels)}"
    )

    print(
        f"Folds: {NUM_FOLDS}"
    )

    all_results = []

    for (
        fold_number,
        (
            train_indices,
            test_indices
        )
    ) in enumerate(
        folds,
        start=1
    ):
        print_fold_distribution(
            fold_number,
            train_indices,
            test_indices,
            labels,
            participant_ids
        )

        rf_metrics = (
            train_random_forest_fold(
                features,
                labels,
                feature_names,
                train_indices,
                test_indices,
                fold_number
            )
        )

        gcn_metrics = (
            train_gcn_fold(
                graph_dataset,
                graph_lookup,
                sample_names,
                train_indices,
                test_indices,
                fold_number,
                device
            )
        )

        print_metrics(
            "Random Forest",
            rf_metrics
        )

        print_metrics(
            "GCN",
            gcn_metrics
        )

        all_results.append(
            {
                "fold":
                    fold_number,
                "model":
                    "Random Forest",
                **rf_metrics
            }
        )

        all_results.append(
            {
                "fold":
                    fold_number,
                "model":
                    "GCN",
                **gcn_metrics
            }
        )

    print()
    print("=" * 60)
    print(
        "5-FOLD CROSS-VALIDATION SUMMARY"
    )
    print("=" * 60)

    for model_name in [
        "GCN",
        "Random Forest"
    ]:
        print(
            f"\n{model_name}"
        )

        for metric_name in [
            "accuracy",
            "precision",
            "recall",
            "f1"
        ]:
            average = (
                calculate_average(
                    all_results,
                    model_name,
                    metric_name
                )
            )

            print(
                f"  "
                f"{metric_name.capitalize()}: "
                f"{average:.3f}"
            )

    save_results(
        all_results
    )

    save_comparison_plot(
        all_results
    )


if __name__ == "__main__":
    run_cross_validation()