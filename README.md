# EMG Gesture Classification: A Deep Learning Approach to Biological Signal Processing

## Overview
This project implements a comprehensive Data Science pipeline for the classification of Electromyography (EMG) signals. The objective is to decode motor intent from neuromuscular activity by transforming high-dimensional time-series data into spectral feature maps for Convolutional Neural Network (CNN) modeling.

## Data Science Pipeline

### 1. Feature Engineering: Spectral Transformation
Raw EMG signals are non-stationary and stochastic. To extract meaningful features, we employ a frequency-domain transformation:
- **Preprocessing**: Signal normalization (z-score) and rectification to stabilize variance.
- **Windowing Strategy**: 200ms segmentation with 50% overlap to preserve temporal dependencies while increasing the dataset size for model robustness.
- **Spectrogram Generation**: Transformation of 1D signals into 2D log-mel spectrograms (64 mel-bins). This maps biological activity into a representation that highlights frequency-modulated patterns unique to specific gesture classes.

### 2. Dataset Management & Balancing
A critical challenge in biological signal classification is class imbalance and sample scarcity.
- **Dynamic Dataset Wrapper**: Custom PyTorch `SpectrogramDataset` implementation for efficient memory management and real-time data loading.
- **Class Imbalance Mitigation**: Implementation of **Inverse Frequency Weighting** within the `nn.CrossEntropyLoss` function. This ensures that the model does not overfit to the majority class (e.g., 'Rest') and maintains high sensitivity to rarer gesture events.
- **Validation Strategy**: Stratified 80/20 train-test split with fixed seeding for reproducibility.

## Neural Architecture Design
The core model is a custom ConvNet designed for low-latency inference on time-series spectral features.

### Architecture Specifications:
- **Feature Extractor**: Three convolutional layers with increasing filter depth (32, 64, 128).
- **Regularization**: Batch Normalization and 0.3 Dropout to prevent overfitting on small-scale biological datasets.
- **Adaptive Pooling**: Global Average Pooling (GAP) is used to ensure the model is agnostic to varying window lengths and to reduce the total parameter count to ~155k, optimizing for edge-device deployment.

## Statistical Evaluation
Model performance is evaluated using metrics that go beyond simple accuracy to account for class distribution and classification rigor.

### Key Performance Indicators:
| Metric | Result | Context |
|--------|--------|---------|
| **Validation Accuracy** | 83% | Overall classification success across all states. |
| **Macro F1-Score** | 0.74 | Robustness indicator across imbalanced classes. |
| **Model Size** | 0.60 MB | Efficient memory footprint for embedded systems. |

## Execution Guide

### Dependency Installation:
```bash
pip install -r requirements.txt
```

### Pipeline Deployment:
```bash
python train.py
```

## Repository Structure
- `datasets/`: Feature mapping and dataset normalization logic.
- `models/`: CNN architecture and training optimization loops.
- `transforms/`: Signal-to-spectral engineering utilities.
- `train.py`: Main execution script for the end-to-end Data Science pipeline.
