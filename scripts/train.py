#!/usr/bin/env python3
"""
==============================================================================
CausalVAD - Main Training Script
==============================================================================

Usage:
    # Train on synthetic data (for testing pipeline):
    python scripts/train.py --dataset synthetic

    # Train on UCF-Crime:
    python scripts/train.py --dataset ucf_crime

    # Train with custom settings:
    python scripts/train.py --dataset synthetic --epochs 30 --batch_size 16

    # Ablation: no causal mask
    python scripts/train.py --dataset synthetic --no_causal_mask

    # Ablation: no temporal decay
    python scripts/train.py --dataset synthetic --no_temporal_decay
==============================================================================
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.causal_vad import CausalVAD, CausalVADLoss
from src.data.feature_dataset import create_dataloaders
from src.training.trainer import Trainer
from src.utils.device import get_device
from src.evaluation.metrics import compute_all_metrics, generate_evaluation_plots


def setup_logging(log_dir: str):
    """Configure logging to both console and file."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file)),
        ],
    )
    return log_file


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_dataset_paths(dataset_name: str) -> dict:
    """Get paths for each supported dataset."""
    datasets = {
        "synthetic": {
            "feature_dir": str(PROJECT_ROOT / "data/features/synthetic"),
            "train_annotation": str(PROJECT_ROOT / "data/annotations/synthetic_train.txt"),
            "test_annotation": str(PROJECT_ROOT / "data/annotations/synthetic_test.txt"),
            "feature_dim": 2048,
        },
        "ucf_crime": {
            "feature_dir": str(PROJECT_ROOT / "data/features/ucf_crime"),
            "train_annotation": str(PROJECT_ROOT / "data/annotations/ucf_crime_train.txt"),
            "test_annotation": str(PROJECT_ROOT / "data/annotations/ucf_crime_test.txt"),
            "feature_dim": 2048,
        },
        "ucf_crime_clip": {
            "feature_dir": str(PROJECT_ROOT / "data/features/ucf_crime_clip"),
            "train_annotation": str(PROJECT_ROOT / "data/annotations/ucf_crime_train.txt"),
            "test_annotation": str(PROJECT_ROOT / "data/annotations/ucf_crime_test.txt"),
            "feature_dim": 512,
        },
        "ucf_crime_clip_flat": {
            "feature_dir": str(PROJECT_ROOT / "data/features/ucf_crime_clip_flat"),
            "train_annotation": str(PROJECT_ROOT / "data/annotations/ucf_crime_clip_train.txt"),
            "test_annotation": str(PROJECT_ROOT / "data/annotations/ucf_crime_clip_test.txt"),
            "feature_dim": 512,
        },
        "ucf_merged": {
            "feature_dir": str(PROJECT_ROOT / "data/features/ucf_crime_merged"),
            "train_annotation": str(PROJECT_ROOT / "data/annotations/ucf_merged_train.txt"),
            "test_annotation": str(PROJECT_ROOT / "data/annotations/ucf_merged_test.txt"),
            "feature_dim": 512,
        },
        "ucf_standard": {
            "feature_dir": str(PROJECT_ROOT / "data/features/ucf_crime_merged"),
            "train_annotation": str(PROJECT_ROOT / "data/annotations/ucf_standard_train.txt"),
            "test_annotation": str(PROJECT_ROOT / "data/annotations/ucf_standard_test.txt"),
            "feature_dim": 512,
        },
        "xd_violence": {
            "feature_dir": str(PROJECT_ROOT / "data/features/xd_violence"),
            "train_annotation": str(PROJECT_ROOT / "data/annotations/xd_violence_train.txt"),
            "test_annotation": str(PROJECT_ROOT / "data/annotations/xd_violence_test.txt"),
            "feature_dim": 2048,
        },
        "shanghaitech": {
            "feature_dir": str(PROJECT_ROOT / "data/features/shanghaitech"),
            "train_annotation": str(PROJECT_ROOT / "data/annotations/shanghaitech_train.txt"),
            "test_annotation": str(PROJECT_ROOT / "data/annotations/shanghaitech_test.txt"),
            "feature_dim": 2048,
        },
    }

    if dataset_name not in datasets:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(datasets.keys())}"
        )

    return datasets[dataset_name]


def main():
    parser = argparse.ArgumentParser(description="CausalVAD Training")

    # Dataset
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic","ucf_merged", "ucf_crime","ucf_standard", "ucf_crime_clip",
                                 "ucf_crime_clip_flat", "xd_violence", "shanghaitech"])

    # Model architecture
    parser.add_argument("--model_dim", type=int, default=512)
    parser.add_argument("--num_prompts", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=200)

    # Causal components (for ablation studies)
    parser.add_argument("--no_causal_mask", action="store_true",
                        help="Disable causal attention mask (ablation)")
    parser.add_argument("--no_temporal_decay", action="store_true",
                        help="Disable temporal decay bias (ablation)")
    parser.add_argument("--decay_type", type=str, default="exponential",
                        choices=["exponential", "linear", "learned"])
    parser.add_argument("--decay_rate", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=2)

    # System
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    # Experiment naming
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name for organizing outputs")

    args = parser.parse_args()

    # ─── Setup ───────────────────────────────────────────────────
    set_seed(args.seed)
    device = get_device(args.device)

    # Create experiment name
    if args.exp_name is None:
        components = [args.dataset]
        if args.no_causal_mask:
            components.append("no_causal")
        if args.no_temporal_decay:
            components.append("no_decay")
        components.append(time.strftime("%m%d_%H%M"))
        args.exp_name = "_".join(components)

    exp_dir = PROJECT_ROOT / "outputs" / args.exp_name
    log_file = setup_logging(str(exp_dir / "logs"))

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info(f"CausalVAD Training - Experiment: {args.exp_name}")
    logger.info("=" * 60)
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Device: {device}")
    logger.info(f"Log file: {log_file}")

    # ─── Dataset ─────────────────────────────────────────────────
    dataset_cfg = get_dataset_paths(args.dataset)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Feature dir: {dataset_cfg['feature_dir']}")

    # Verify data exists
    if not Path(dataset_cfg["feature_dir"]).exists():
        logger.error(
            f"Feature directory not found: {dataset_cfg['feature_dir']}\n"
            f"Run: python scripts/prepare_data.py --mode synthetic"
        )
        sys.exit(1)

    train_loader, test_loader = create_dataloaders(
        feature_dir=dataset_cfg["feature_dir"],
        train_annotation=dataset_cfg["train_annotation"],
        test_annotation=dataset_cfg["test_annotation"],
        max_seq_len=args.max_seq_len,
        feature_dim=dataset_cfg["feature_dim"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Test batches:  {len(test_loader)}")

    # ─── Model ───────────────────────────────────────────────────
    model = CausalVAD(
        input_dim=dataset_cfg["feature_dim"],
        model_dim=args.model_dim,
        num_prompts=args.num_prompts,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        use_causal_mask=not args.no_causal_mask,
        use_temporal_decay=not args.no_temporal_decay,
        decay_type=args.decay_type,
        decay_rate=args.decay_rate,
    ).to(device)

    criterion = CausalVADLoss().to(device)

    # ─── Optimizer & Scheduler ───────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )

    # ─── Trainer ─────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=str(exp_dir / "checkpoints"),
        log_dir=str(exp_dir / "logs" / "tensorboard"),
        early_stopping_patience=args.patience,
    )

    # ─── Train ───────────────────────────────────────────────────
    results = trainer.fit(
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
    )

    # ─── Final Evaluation ────────────────────────────────────────
    logger.info("Running final evaluation with best model...")
    trainer.load_checkpoint(str(exp_dir / "checkpoints" / "best_model.pt"))
    final_metrics = trainer.evaluate(test_loader)

    logger.info("=" * 60)
    logger.info("FINAL RESULTS")
    logger.info("=" * 60)
    logger.info(f"  AUC:       {final_metrics['auc']:.4f}")
    logger.info(f"  AP:        {final_metrics['ap']:.4f}")
    logger.info(f"  F1:        {final_metrics['f1']:.4f}")
    logger.info(f"  FAR:       {final_metrics['far']:.4f}")
    logger.info(f"  Threshold: {final_metrics['threshold']:.4f}")
    logger.info("=" * 60)

    # ─── Generate Plots ──────────────────────────────────────────
    logger.info("Generating evaluation plots...")

    # We need raw scores for plotting - run evaluation again to collect them
    model.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            output = model(features, mask)
            all_scores.append(output["video_score"].cpu().numpy())
            all_labels.append(batch["label"].numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    generate_evaluation_plots(
        labels=labels,
        scores=scores,
        save_dir=str(exp_dir / "figures"),
        prefix=args.exp_name,
    )

    logger.info(f"All outputs saved to: {exp_dir}")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
