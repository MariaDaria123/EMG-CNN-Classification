# EMG Gesture Classification via Convolutional Neural Networks

## Overview
This repository provides a production-level implementation of a deep learning pipeline for the classification of Electromyography (EMG) signals. The framework integrates advanced digital signal processing (DSP) with a lightweight Convolutional Neural Network (CNN) architecture to decode motor intent from muscle activity.

## Signal Processing Pipeline
The preprocessing stage is critical for mitigating the inherent stochasticity and noise in biological signals.

### 1. Digital Filtering
Raw EMG data is processed through a zero-phase digital filter chain:
- **Bandpass Filter**: 4th-order Butterworth filter (20 - 450 Hz) to isolate the relevant bio-signal bandwidth and remove motion artifacts/high-frequency noise.
- **Full-wave Rectification**: Absolute value transformation to prepare the signal for envelope extraction.

### 2. Time-Frequency Transformation
Continuous signals are segmented using a sliding window approach (200ms duration, 100ms stride). Each window is transformed into the frequency domain via:
- **Log-Mel Spectrograms**: Computed using a 256-point STFT with a Hanning window.
- **Mel Filterbank**: 64 frequency bins across the effective bandwidth.
- **Logarithmic Scaling**: Enhances feature representation of spectral energy distributions.

## Model Architecture
The classification engine employs a custom ConvNet optimized for CPU-efficient inference in real-time neuroprosthetic applications.

### Layer Specification:
- **Input Layer**: (1, 64, T) where T represents temporal frames.
- **Convolutional Blocks**: Three sequential blocks featuring:
    - 2D Convolution (3x3 kernels)
    - Batch Normalization (post-convolution)
    - ReLU Activation
    - Max Pooling (Block 1 only, 2x2 stride)
- **Global Average Pooling (GAP)**: Provides spatial invariance and reduces total parameter count.
- **Classifier Head**: Fully connected layer with 0.3 Dropout regularization.

### Parameter Efficiency:
- **Total Trainable Parameters**: 155,908
- **Model Storage**: ~0.60 MB
- **Inference Latency**: Optimized for low-power edge devices.

## Performance Evaluation
The model was validated using macro-averaged F1-scores and multiclass confusion matrices.

### Quantitative Metrics:
| Classification Category | Precision | Recall | F1-Score |
|-------------------------|-----------|--------|----------|
| Rest State              | 0.96      | 0.92   | 0.94     |
| Extension (Open)        | 0.61      | 0.73   | 0.67     |
| Contraction (Fist)      | 0.67      | 0.58   | 0.62     |

## Deployment and Execution
The environment requires PyTorch and Librosa for signal processing and model training.

### Environment Setup:
```bash
pip install -r requirements.txt
```

### Pipeline Execution:
```bash
python train.py
```

## Repository Structure
- `datasets/`: PyTorch Dataset implementation for spectral features.
- `models/`: CNN architecture definition and training logic.
- `transforms/`: Signal-to-spectral transformation utilities.
- `train.py`: Primary execution script for the end-to-end pipeline.
