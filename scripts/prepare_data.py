#!/usr/bin/env python3
"""
==============================================================================
Data Preparation Script
==============================================================================

This script does two things:

1. CREATES SYNTHETIC DATA for testing the pipeline immediately
   (so you can verify everything works BEFORE downloading large datasets)

2. PROVIDES INSTRUCTIONS for downloading real datasets

Run this FIRST after environment setup:
    python scripts/prepare_data.py --mode synthetic

Then, when ready for real data:
    python scripts/prepare_data.py --mode download_instructions
==============================================================================
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def create_synthetic_dataset(
    output_dir: str = "data/features/synthetic",
    num_normal: int = 100,
    num_anomaly: int = 100,
    seq_length: int = 200,
    feature_dim: int = 2048,
    seed: int = 42,
):
    """
    Create a synthetic dataset for pipeline testing.

    Normal videos:  features drawn from N(0, 1)
    Anomaly videos: features drawn from N(0, 1) with a "spike" region
                    in the middle where features shift to N(2, 1.5)

    This mimics the real scenario where anomaly videos contain mostly
    normal content with a segment of anomalous activity.
    """
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Creating SYNTHETIC dataset for pipeline testing")
    logger.info(f"  Normal videos:  {num_normal}")
    logger.info(f"  Anomaly videos: {num_anomaly}")
    logger.info(f"  Sequence length: {seq_length}")
    logger.info(f"  Feature dim:     {feature_dim}")
    logger.info(f"  Output dir:      {output_dir}")
    logger.info("=" * 60)

    train_annotations = []
    test_annotations = []

    # ─── Generate Normal Videos ──────────────────────────────────
    for i in range(num_normal):
        video_name = f"Normal_{i:04d}"
        features = np.random.randn(seq_length, feature_dim).astype(np.float32)
        np.save(output_dir / f"{video_name}.npy", features)

        # 80% train, 20% test
        if i < int(num_normal * 0.8):
            train_annotations.append(f"{video_name} 0")
        else:
            test_annotations.append(f"{video_name} 0")

    logger.info(f"  Created {num_normal} normal video features")

    # ─── Generate Anomaly Videos ─────────────────────────────────
    for i in range(num_anomaly):
        video_name = f"Anomaly_{i:04d}"
        features = np.random.randn(seq_length, feature_dim).astype(np.float32)

        # Insert anomalous segment (shifted distribution)
        anomaly_start = np.random.randint(seq_length // 4, seq_length // 2)
        anomaly_length = np.random.randint(seq_length // 8, seq_length // 4)
        anomaly_end = min(anomaly_start + anomaly_length, seq_length)

        # Anomalous features: shifted mean + higher variance
        features[anomaly_start:anomaly_end] = (
            np.random.randn(anomaly_end - anomaly_start, feature_dim).astype(np.float32) * 1.5 + 2.0
        )

        np.save(output_dir / f"{video_name}.npy", features)

        if i < int(num_anomaly * 0.8):
            train_annotations.append(f"{video_name} 1")
        else:
            test_annotations.append(f"{video_name} 1")

    logger.info(f"  Created {num_anomaly} anomaly video features")

    # ─── Write Annotation Files ──────────────────────────────────
    ann_dir = PROJECT_ROOT / "data" / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)

    train_path = ann_dir / "synthetic_train.txt"
    test_path = ann_dir / "synthetic_test.txt"

    # Shuffle annotations
    np.random.shuffle(train_annotations)
    np.random.shuffle(test_annotations)

    with open(train_path, "w") as f:
        f.write("# video_name label\n")
        f.write("\n".join(train_annotations) + "\n")

    with open(test_path, "w") as f:
        f.write("# video_name label\n")
        f.write("\n".join(test_annotations) + "\n")

    logger.info(f"  Train annotations: {train_path} ({len(train_annotations)} videos)")
    logger.info(f"  Test annotations:  {test_path} ({len(test_annotations)} videos)")
    logger.info("")
    logger.info("SYNTHETIC DATA READY!")
    logger.info("You can now run: python scripts/train.py --dataset synthetic")


def print_download_instructions():
    """Print detailed instructions for downloading real datasets."""
    instructions = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                    REAL DATASET DOWNLOAD INSTRUCTIONS                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

We use PRE-EXTRACTED I3D features (not raw videos) to keep compute manageable.
Several research groups have made these features publicly available.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATASET 1: UCF-Crime (Primary Dataset)
───────────────────────────────────────
Description: 1,900 real-world surveillance videos, 13 anomaly types.
Size:        I3D features ~4 GB

Option A — From the RTFM paper (Tian et al., ICCV 2021):
  URL: https://github.com/tianyu0207/RTFM
  Features: I3D features, 10-crop, 2048-d
  Steps:
    1. Go to the GitHub link above
    2. Find the Google Drive link for UCF-Crime I3D features
    3. Download the .zip file
    4. Extract to: data/features/ucf_crime/

Option B — From the VadCLIP paper:
  URL: https://github.com/nwpu-zxr/VadCLIP
  Features: CLIP ViT-B/16 features, 512-d
  Steps:
    1. Go to the GitHub link above
    2. Download CLIP features from the provided link
    3. Extract to: data/features/ucf_crime_clip/

After downloading, your directory should look like:
  data/features/ucf_crime/
    ├── Abuse001_x264.npy
    ├── Abuse002_x264.npy
    ├── ...
    └── Normal_Videos_event001_x264.npy

Annotations:
  Download train/test split files from the RTFM repo.
  Place in: data/annotations/
    ├── ucf_crime_train.txt
    └── ucf_crime_test.txt

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATASET 2: XD-Violence (Secondary Dataset)
──────────────────────────────────────────
Description: 4,754 untrimmed videos with audio, 6 anomaly types.
Size:        I3D features ~6 GB

Source: https://roc-ng.github.io/XD-Violence/
Steps:
  1. Visit the link above
  2. Download I3D features (Google Drive link on the page)
  3. Extract to: data/features/xd_violence/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATASET 3: ShanghaiTech (Supplementary Dataset)
───────────────────────────────────────────────
Description: 437 campus surveillance videos, simpler anomalies.
Size:        I3D features ~1 GB

Source: https://github.com/tianyu0207/RTFM
Steps:
  Same as UCF-Crime — the RTFM repo has features for all three datasets.
  Extract to: data/features/shanghaitech/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IMPORTANT NOTES:
  - Start with UCF-Crime. It's the most important benchmark.
  - Use the SYNTHETIC dataset first to verify the pipeline works.
  - If Google Drive links are dead, check the Issues tab on each GitHub repo
    for alternative download links.
  - Feature files should be .npy format with shape (T, 2048) for I3D
    or (T, 512) for CLIP features.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    print(instructions)


def main():
    parser = argparse.ArgumentParser(description="CausalVAD Data Preparation")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["synthetic", "download_instructions", "both"],
        default="both",
        help="What to do: create synthetic data, show download instructions, or both",
    )
    parser.add_argument("--num_normal", type=int, default=100)
    parser.add_argument("--num_anomaly", type=int, default=100)
    parser.add_argument("--seq_length", type=int, default=200)
    parser.add_argument("--feature_dim", type=int, default=2048)

    args = parser.parse_args()

    if args.mode in ("synthetic", "both"):
        create_synthetic_dataset(
            num_normal=args.num_normal,
            num_anomaly=args.num_anomaly,
            seq_length=args.seq_length,
            feature_dim=args.feature_dim,
        )

    if args.mode in ("download_instructions", "both"):
        print_download_instructions()


if __name__ == "__main__":
    main()
