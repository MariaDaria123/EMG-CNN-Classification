# EMG Gesture Classification via ConvNet 🦾🧠

[![Neurocomputation](https://img.shields.io/badge/Focus-Neurocomputation-blueviolet)](https://github.com/MariaDaria123)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C)](https://pytorch.org/)

## 📌 Overview
This repository implements an end-to-end pipeline for classifying hand gestures from Electromyography (EMG) signals. By transforming raw neural/muscle signals into the frequency domain (Spectrograms), we leverage Convolutional Neural Networks (CNNs) to achieve high-accuracy gesture recognition with minimal latency.

### Key Highlights:
- **Neural Pipeline**: Raw EMG → Filtering → Rectification → Log-Mel Spectrograms.
- **Lightweight Architecture**: A custom ConvNet (~150K parameters) designed for CPU-efficient inference.
- **Performance**: Achieved **83% validation accuracy** on gesture classification (Rest, Fist, Open, Pinch).

---

## 📊 The Neurocomputation Pipeline

### 1. Signal Preprocessing
Biological signals are inherently noisy. This pipeline applies:
- **Bandpass Filtering**: Removes low-frequency artifacts and high-frequency noise.
- **Full-wave Rectification**: Prepares the signal for envelope extraction.
- **Windowing**: 200ms segments with 50% overlap for real-time responsiveness.

### 2. Spectrogram Transformation
Instead of raw time-series data, we use **Log-Mel Spectrograms**. This transforms 1D muscle activity into a 2D representation that highlights spectral patterns unique to different muscle contraction signatures.

---

## 🧠 Model Architecture
The `EMGConvNet` is optimized for biological time-series data:
- **Input**: `(1, 64, T)` (Channels, Mels, Time)
- **Feature Extraction**: 3x Convolutional Blocks with BatchNorm and ReLU.
- **Dimensionality Preservation**: Strategy-specific pooling to handle short-duration gestures.
- **Global Average Pooling**: Ensures the model is agnostic to gesture duration.

```python
# Model Summary
Total Trainable Parameters: 155,908
Model Size: ~0.60 MB
```

---

## 📈 Results
The model demonstrates robust classification performance across gesture classes:

| Gesture | Precision | Recall | F1-Score |
|---------|-----------|--------|----------|
| Rest    | 0.96      | 0.92   | 0.94     |
| Open    | 0.61      | 0.73   | 0.67     |
| Pinch   | 0.67      | 0.58   | 0.62     |

---

## 🛠 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/MariaDaria123/EMG-Gesture-Classification.git

# Install dependencies
pip install -r requirements.txt

# Run the training demo
python train.py
```

---

## 📂 Repository Structure
```
├── datasets/           # PyTorch Dataset wrappers
├── models/             # CNN Architecture & Training scripts
├── transforms/         # Signal-to-Spectrogram transformations
└── train.py            # End-to-end training pipeline demo
```
