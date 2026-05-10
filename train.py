"""
EMG Gesture Classification - Training Demo

This script demonstrates the complete neurocomputation pipeline:
1. Synthetic EMG signal generation
2. Digital Signal Processing (Filtering & Rectification)
3. Spectrogram Transformation
4. CNN Model Training and Evaluation
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.metrics import classification_report, confusion_matrix

from models.model_cnn import EMGConvNet
from datasets.cnn_dataset import SpectrogramDataset, train_val_split
from transforms.spectrograms import batch_compute_spectrograms

# --- CONFIGURATION ---
FS = 1000  # Sampling rate in Hz
WINDOW_MS = 200
HOP_MS = 100
EPOCHS = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# --- 1. SYNTHETIC DATA GENERATION ---
def generate_synthetic_emg(duration_s=60):
    """Generates synthetic EMG-like signals for 4 gestures."""
    n_samples = int(duration_s * FS)
    t = np.linspace(0, duration_s, n_samples)
    
    # Base noise
    emg = 0.05 * np.random.randn(n_samples)
    labels = []
    
    # Add gesture "bursts"
    gestures = ['rest', 'fist', 'open', 'pinch']
    block_len = int(5 * FS)  # 5 seconds per gesture
    
    for i in range(0, n_samples, block_len):
        g_idx = (i // block_len) % len(gestures)
        gesture = gestures[g_idx]
        
        if gesture != 'rest':
            # Simulating muscle contraction with modulated noise
            emg[i:i+block_len] *= 5.0 
            # Add some frequency characteristics
            f_center = 100 if gesture == 'fist' else 200
            emg[i:i+block_len] += 0.2 * np.sin(2 * np.pi * f_center * t[i:i+block_len])
            
        labels.extend([gesture] * min(block_len, n_samples - i))
        
    return emg, labels

# --- 2. SIGNAL PROCESSING ---
def preprocess_signal(emg):
    """Applies Bandpass filter and Rectification."""
    # Bandpass 20-450Hz
    nyq = 0.5 * FS
    low = 20 / nyq
    high = 450 / nyq
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, emg)
    
    # Rectification
    rectified = np.abs(filtered)
    
    return rectified

# --- 3. WINDOWING ---
def window_data(emg, labels):
    win_samples = int(WINDOW_MS * FS / 1000)
    hop_samples = int(HOP_MS * FS / 1000)
    
    windows = []
    win_labels = []
    
    for i in range(0, len(emg) - win_samples, hop_samples):
        windows.append(emg[i:i+win_samples])
        # Use the label of the last sample in window
        win_labels.append(labels[i + win_samples - 1])
        
    return np.array(windows), win_labels

def main():
    print("🚀 Starting EMG CNN Pipeline Demo...")
    
    # Step 1 & 2: Generate and Preprocess
    raw_emg, raw_labels = generate_synthetic_emg(60)
    processed_emg = preprocess_signal(raw_emg)
    
    # Step 3: Window and Transform
    windows, labels = window_data(processed_emg, raw_labels)
    spectrograms = batch_compute_spectrograms(windows, fs=FS)
    print(f"📊 Dataset prepared: {spectrograms.shape[0]} spectrograms of shape {spectrograms.shape[1:]}")
    
    # Step 4: Dataset & DataLoader
    dataset = SpectrogramDataset(spectrograms, labels)
    train_ds, val_ds = train_val_split(dataset, train_ratio=0.8, seed=42)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    # Step 5: Training
    model = EMGConvNet(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\n🧠 Training CNN Model...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_specs, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_specs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {total_loss/len(train_loader):.4f}")
        
    # Step 6: Evaluation
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for batch_specs, batch_labels in val_loader:
            outputs = model(batch_specs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.numpy())
            all_true.extend(batch_labels.numpy())
            
    print("\n📈 Final Evaluation:")
    print(classification_report(all_true, all_preds, target_names=dataset.DEFAULT_LABEL_MAP.keys()))

if __name__ == "__main__":
    main()
