"""
==============================================================================
Evaluation Metrics for Video Anomaly Detection
==============================================================================

Computes standard metrics:
  - AUC: Area Under ROC Curve (primary metric for VAD)
  - AP:  Average Precision
  - F1:  F1 Score at optimal threshold
  - FAR: False Alarm Rate at optimal threshold

Also generates visualizations for the paper.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    confusion_matrix,
)
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def compute_all_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        labels: (N,) ground truth labels (0 or 1)
        scores: (N,) predicted anomaly scores (0 to 1)
        threshold: Classification threshold. If None, finds optimal.

    Returns:
        Dict with "auc", "ap", "f1", "far", "threshold" keys.
    """
    # Ensure numpy arrays
    labels = np.asarray(labels).flatten()
    scores = np.asarray(scores).flatten()

    assert len(labels) == len(scores), (
        f"Label/score length mismatch: {len(labels)} vs {len(scores)}"
    )

    # Handle edge cases
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        logger.warning("Only one class present in labels — metrics may be unreliable")
        return {"auc": 0.5, "ap": 0.5, "f1": 0.0, "far": 0.0, "threshold": 0.5}

    # ─── AUC (Area Under ROC Curve) ──────────────────────────────
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.5
        logger.warning("AUC computation failed, defaulting to 0.5")

    # ─── AP (Average Precision) ──────────────────────────────────
    try:
        ap = average_precision_score(labels, scores)
    except ValueError:
        ap = 0.5
        logger.warning("AP computation failed, defaulting to 0.5")

    # ─── Find Optimal Threshold ──────────────────────────────────
    if threshold is None:
        threshold = find_optimal_threshold(labels, scores)

    # ─── F1 Score ────────────────────────────────────────────────
    predictions = (scores >= threshold).astype(int)
    f1 = f1_score(labels, predictions, zero_division=0)

    # ─── False Alarm Rate ────────────────────────────────────────
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    far = fp / max(fp + tn, 1)

    return {
        "auc": float(auc),
        "ap": float(ap),
        "f1": float(f1),
        "far": float(far),
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def find_optimal_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    Find the threshold that maximizes F1 score.

    Uses the ROC curve to test multiple thresholds efficiently.
    """
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Compute F1 at each threshold
    best_f1 = 0.0
    best_threshold = 0.5

    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        current_f1 = f1_score(labels, preds, zero_division=0)
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = thresh

    return best_threshold


def generate_evaluation_plots(
    labels: np.ndarray,
    scores: np.ndarray,
    save_dir: str = "outputs/figures",
    prefix: str = "eval",
):
    """
    Generate publication-quality evaluation plots.

    Creates:
      1. ROC Curve
      2. Precision-Recall Curve
      3. Score Distribution
      4. Temporal Score Visualization (if clip-level scores provided)
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend

    from pathlib import Path
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Use publication-quality settings
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "serif",
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.figsize": (6, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })

    metrics = compute_all_metrics(labels, scores)

    # ─── 1. ROC Curve ────────────────────────────────────────────
    fig, ax = plt.subplots()
    fpr, tpr, _ = roc_curve(labels, scores)
    ax.plot(fpr, tpr, color="#2E75B6", linewidth=2,
            label=f"CausalVAD (AUC = {metrics['auc']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.savefig(save_dir / f"{prefix}_roc_curve.png")
    fig.savefig(save_dir / f"{prefix}_roc_curve.pdf")
    plt.close(fig)
    logger.info(f"Saved ROC curve to {save_dir}/{prefix}_roc_curve.png")

    # ─── 2. Precision-Recall Curve ───────────────────────────────
    fig, ax = plt.subplots()
    precision, recall, _ = precision_recall_curve(labels, scores)
    ax.plot(recall, precision, color="#E74C3C", linewidth=2,
            label=f"CausalVAD (AP = {metrics['ap']:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.savefig(save_dir / f"{prefix}_pr_curve.png")
    fig.savefig(save_dir / f"{prefix}_pr_curve.pdf")
    plt.close(fig)

    # ─── 3. Score Distribution ───────────────────────────────────
    fig, ax = plt.subplots()
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    ax.hist(normal_scores, bins=50, alpha=0.6, color="#2ECC71",
            label="Normal", density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.6, color="#E74C3C",
            label="Anomaly", density=True)
    ax.axvline(x=metrics["threshold"], color="k", linestyle="--",
               linewidth=1.5, label=f"Threshold = {metrics['threshold']:.3f}")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(save_dir / f"{prefix}_score_dist.png")
    fig.savefig(save_dir / f"{prefix}_score_dist.pdf")
    plt.close(fig)

    logger.info(f"All evaluation plots saved to {save_dir}/")
    return metrics


def generate_comparison_table(
    results: Dict[str, Dict[str, float]],
    save_path: str = "outputs/tables/comparison.txt",
):
    """
    Generate a comparison table (for the paper).

    Args:
        results: Dict mapping method names to their metrics.
                 e.g., {"CausalVAD": {"auc": 0.88, "ap": 0.85}, ...}

    Creates a formatted table suitable for LaTeX conversion.
    """
    from pathlib import Path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 70)
    lines.append(f"{'Method':<25} {'AUC':>8} {'AP':>8} {'F1':>8} {'FAR':>8}")
    lines.append("-" * 70)

    for method, metrics in results.items():
        lines.append(
            f"{method:<25} "
            f"{metrics.get('auc', 0):.4f}  "
            f"{metrics.get('ap', 0):.4f}  "
            f"{metrics.get('f1', 0):.4f}  "
            f"{metrics.get('far', 0):.4f}"
        )

    lines.append("=" * 70)

    table_str = "\n".join(lines)
    with open(save_path, "w") as f:
        f.write(table_str)

    print(table_str)
    return table_str
