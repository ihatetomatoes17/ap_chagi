# Ap Chagi Movement Classification

Bachelor thesis project for classification of correct and incorrect
Ap Chagi executions using a Graph Convolutional Network (GCN).

## Pipeline

1. Pose keypoint extraction using MediaPipe Pose Landmarker
2. Keypoint preprocessing
3. Movement feature extraction
4. Spatio-temporal graph construction
5. GCN training and evaluation
6. Comparison with Random Forest

## Dataset

The dataset contains 90 videos:
- 45 correct executions
- 45 incorrect executions
- 3 participants

The dataset is not included in this repository.

## Models

MediaPipe Pose Landmarker requires:

`models/pose_landmarker_full.task`

The model file is not included in the repository.

## Evaluation

Both GCN and Random Forest models were evaluated using
5-fold cross-validation.