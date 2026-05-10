"""
PyTorch Dataset for EMG Spectrograms.

This module wraps spectrogram arrays and labels into a PyTorch Dataset
for training CNN models on EMG gesture classification.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple


class SpectrogramDataset(Dataset):
    """
    PyTorch Dataset for spectrogram-based EMG classification.
    
    Parameters
    ----------
    spectrograms : np.ndarray
        Array of shape (N, n_mels, time_frames) containing log-mel spectrograms.
    labels : List[str]
        List of length N containing gesture labels.
    label_map : dict, optional
        Mapping from label string to integer index. If None, uses default.
    transform : callable, optional
        Optional transform to apply to spectrograms.
    """
    
    DEFAULT_LABEL_MAP = {
        'rest': 0,
        'fist': 1,
        'open': 2,
        'pinch': 3
    }
    
    def __init__(
        self,
        spectrograms: np.ndarray,
        labels: List[str],
        label_map: Optional[dict] = None,
        transform: Optional[callable] = None
    ):
        """Initialize the dataset."""
        assert len(spectrograms) == len(labels), \
            f"Mismatch: {len(spectrograms)} spectrograms but {len(labels)} labels"
        
        self.spectrograms = spectrograms.astype(np.float32)
        self.labels = labels
        self.transform = transform
        
        # Use provided map or default
        self.label_to_idx = label_map if label_map is not None else self.DEFAULT_LABEL_MAP
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.num_classes = len(self.label_to_idx)
        
        # Convert string labels to indices
        self.label_indices = [self.label_to_idx[label] for label in labels]
    
    def __len__(self) -> int:
        return len(self.spectrograms)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spec = self.spectrograms[idx]
        if self.transform is not None:
            spec = self.transform(spec)
        
        # Add channel dimension: (H, W) -> (1, H, W)
        spec = np.expand_dims(spec, axis=0)
        
        spec_tensor = torch.from_numpy(spec).float()
        label_tensor = torch.tensor(self.label_indices[idx], dtype=torch.long)
        
        return spec_tensor, label_tensor

    def compute_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced datasets."""
        from collections import Counter
        counts = Counter(self.labels)
        total = len(self.labels)
        weights = torch.zeros(self.num_classes)
        for label, count in counts.items():
            if label in self.label_to_idx:
                weights[self.label_to_idx[label]] = total / (self.num_classes * count)
        return weights


def train_val_split(
    dataset: SpectrogramDataset,
    train_ratio: float = 0.8,
    seed: Optional[int] = None
) -> Tuple[SpectrogramDataset, SpectrogramDataset]:
    """Split dataset into training and validation sets."""
    if seed is not None:
        np.random.seed(seed)
    
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    split_idx = int(n_samples * train_ratio)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    
    train_ds = SpectrogramDataset(
        dataset.spectrograms[train_idx],
        [dataset.labels[i] for i in train_idx],
        label_map=dataset.label_to_idx,
        transform=dataset.transform
    )
    
    val_ds = SpectrogramDataset(
        dataset.spectrograms[val_idx],
        [dataset.labels[i] for i in val_idx],
        label_map=dataset.label_to_idx,
        transform=None
    )
    
    return train_ds, val_ds
