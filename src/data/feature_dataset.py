"""
Dataset loaders for pre-extracted video features.

Why pre-extracted features?
- Running a full video encoder (I3D, CLIP) on every training iteration is too slow.
- Standard practice: extract features ONCE, save them, then train on features.
- UCF-Crime I3D features are publicly available (~4GB).

This module loads those cached features and handles:
- Padding/truncating sequences to fixed length
- Train/test splitting
- Video-level weak labels (normal=0, anomaly=1)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class VideoFeatureDataset(Dataset):
    """
    Loads pre-extracted video features for anomaly detection.

    Each video is a sequence of clip-level feature vectors:
        features: (T, D) where T = number of clips, D = feature dimension
        label:    0 (normal) or 1 (anomaly)

    For weakly supervised training, we only have VIDEO-LEVEL labels,
    not frame/clip-level labels.

    Args:
        feature_dir:  Path to directory containing .npy feature files.
        annotation_file: Path to text file listing video names and labels.
        max_seq_len:  Maximum number of clips per video (pad/truncate).
        feature_dim:  Expected feature dimension (for validation).
        is_test:      If True, load test split; else train split.
    """

    def __init__(
        self,
        feature_dir: str,
        annotation_file: str,
        max_seq_len: int = 200,
        feature_dim: int = 2048,
        is_test: bool = False,
    ):
        self.feature_dir = Path(feature_dir)
        self.max_seq_len = max_seq_len
        self.feature_dim = feature_dim
        self.is_test = is_test

        # Parse annotation file
        self.videos, self.labels = self._parse_annotations(annotation_file)
        logger.info(
            f"Loaded {len(self.videos)} videos "
            f"({sum(self.labels)} anomaly, {len(self.labels) - sum(self.labels)} normal)"
        )

    def _parse_annotations(self, annotation_file: str) -> Tuple[List[str], List[int]]:
        """
        Parse annotation file. Expected format (one per line):
            video_name label
        Where label is 0 (normal) or 1 (anomaly).

        OR for UCF-Crime format:
            AnomalyType/video_name.mp4  label  start_frame  end_frame
        """
        videos = []
        labels = []
        ann_path = Path(annotation_file)

        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}\n"
                f"Run scripts/download_features.py first!"
            )

        with open(ann_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    video_name = parts[0]
                    label = int(parts[1])
                    videos.append(video_name)
                    labels.append(label)

        return videos, labels

    def _load_features(self, video_name: str) -> np.ndarray:
        """
        Load feature file for a single video.
        Tries multiple naming conventions.
        """
        # Try different file patterns
        candidates = [
            self.feature_dir / f"{video_name}.npy",
            self.feature_dir / f"{video_name.replace('.mp4', '')}.npy",
            self.feature_dir / f"{video_name.replace('/', '_')}.npy",
            self.feature_dir / video_name / "features.npy",
        ]

        for path in candidates:
            if path.exists():
                features = np.load(str(path))
                return features

        # If no file found, create zeros (graceful fallback for testing)
        logger.warning(f"Feature file not found for {video_name}, using zeros")
        return np.zeros((self.max_seq_len, self.feature_dim), dtype=np.float32)

    def _pad_or_truncate(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ensure features have shape (max_seq_len, feature_dim).
        Returns (padded_features, mask) where mask[i]=1 for real clips.
        """
        T, D = features.shape
        mask = np.zeros(self.max_seq_len, dtype=np.float32)

        if T >= self.max_seq_len:
            # Truncate: uniformly sample max_seq_len clips
            indices = np.linspace(0, T - 1, self.max_seq_len, dtype=int)
            padded = features[indices]
            mask[:] = 1.0
        else:
            # Pad with zeros
            padded = np.zeros((self.max_seq_len, D), dtype=np.float32)
            padded[:T] = features
            mask[:T] = 1.0

        return padded, mask

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                "features":   (max_seq_len, feature_dim) float tensor
                "mask":       (max_seq_len,) binary mask (1=real, 0=padding)
                "label":      scalar int tensor (0=normal, 1=anomaly)
                "video_name": string (for logging/debugging)
        """
        video_name = self.videos[idx]
        label = self.labels[idx]

        # Load and process features
        features = self._load_features(video_name)

        # Ensure correct dimension
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[1] != self.feature_dim:
            logger.warning(
                f"Feature dim mismatch for {video_name}: "
                f"expected {self.feature_dim}, got {features.shape[1]}"
            )

        padded_features, mask = self._pad_or_truncate(features)

        return {
            "features": torch.FloatTensor(padded_features),
            "mask": torch.FloatTensor(mask),
            "label": torch.LongTensor([label]),
            "video_name": video_name,
        }


def create_dataloaders(
    feature_dir: str,
    train_annotation: str,
    test_annotation: str,
    max_seq_len: int = 200,
    feature_dim: int = 2048,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.

    Args:
        feature_dir:      Directory with .npy feature files.
        train_annotation: Path to train annotation file.
        test_annotation:  Path to test annotation file.
        max_seq_len:      Max clips per video.
        feature_dim:      Feature vector dimension.
        batch_size:       Batch size for training.
        num_workers:      Parallel data loading workers.

    Returns:
        (train_loader, test_loader) tuple.
    """
    # Custom collate to handle string video names
    def collate_fn(batch):
        return {
            "features": torch.stack([item["features"] for item in batch]),
            "mask": torch.stack([item["mask"] for item in batch]),
            "label": torch.cat([item["label"] for item in batch]),
            "video_name": [item["video_name"] for item in batch],
        }

    train_dataset = VideoFeatureDataset(
        feature_dir=feature_dir,
        annotation_file=train_annotation,
        max_seq_len=max_seq_len,
        feature_dim=feature_dim,
        is_test=False,
    )

    test_dataset = VideoFeatureDataset(
        feature_dir=feature_dir,
        annotation_file=test_annotation,
        max_seq_len=max_seq_len,
        feature_dim=feature_dim,
        is_test=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, test_loader
