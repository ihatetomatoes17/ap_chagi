# Ap Chagi

Analysis of correct and incorrect Ap Chagi executions using Graph Convolutional Networks (GCN), with comparison to a Random Forest model.

## Pipeline

## Pipeline

1. Pose keypoint extraction using MediaPipe Pose Landmarker
2. Keypoint preprocessing
3. GCN pipeline:
   - Spatio-temporal graph construction
   - GCN training and evaluation
4. Random Forest pipeline:
   - Movement feature extraction
   - Random Forest training and evaluation
5. Comparison of GCN and Random Forest results

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