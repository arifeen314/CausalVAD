#!/usr/bin/env python3
"""
==============================================================================
CausalVAD — Attention Heatmap & Temporal Score Visualization
==============================================================================

Generates publication-quality figures:
  1. Temporal anomaly score plots (score over time for sample videos)
  2. Attention heatmaps (which clips the model attends to)

These serve as visual proof of the "temporal interpretability" claim.

Usage:
    python scripts/visualize.py --checkpoint outputs/standard_split/checkpoints/best_model.pt --dataset ucf_standard
==============================================================================
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.causal_vad import CausalVAD
from src.data.feature_dataset import VideoFeatureDataset
from src.utils.device import get_device


def extract_attention_and_scores(model, features, mask, device):
    """Extract attention weights and anomaly scores from the model."""
    model.eval()
    B, T, D_in = features.shape

    with torch.no_grad():
        features_dev = features.to(device)
        mask_dev = mask.to(device) if mask is not None else None

        # Get anomaly scores
        output = model(features_dev, mask_dev)
        clip_scores = output["clip_scores"].cpu().numpy()
        video_score = output["video_score"].cpu().numpy()

        # Extract attention weights from first transformer layer
        projected = model.feature_proj(features_dev)
        P = model.causal_prompt.num_prompts
        prompts = model.causal_prompt.prompt_tokens.expand(B, -1, -1)

        proj_feat = model.causal_prompt.input_proj(projected)
        proj_feat = model.causal_prompt.input_norm(proj_feat)

        combined = torch.cat([prompts, proj_feat], dim=1)
        combined = combined + model.causal_prompt.positional_encoding[:, :P + T, :]

        # Get attention from first layer
        layer = model.causal_prompt.transformer_layers[0]
        normed = layer.norm1(combined)
        attn = layer.attn
        L = combined.shape[1]

        Q = attn.q_proj(normed).view(B, L, attn.num_heads, attn.head_dim).transpose(1, 2)
        K = attn.k_proj(normed).view(B, L, attn.num_heads, attn.head_dim).transpose(1, 2)

        scores_attn = torch.matmul(Q, K.transpose(-2, -1)) * attn.scale

        if attn.use_causal_mask:
            scores_attn = scores_attn + attn.causal_mask[:L, :L].unsqueeze(0).unsqueeze(0)
        if attn.use_temporal_decay:
            decay = attn.temporal_decay(L)
            scores_attn = scores_attn + decay.unsqueeze(0).unsqueeze(0)

        weights = torch.softmax(scores_attn, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)

        # Average across heads: (B, num_heads, L, L) -> (B, L, L)
        avg_weights = weights.mean(dim=1).cpu().numpy()

    return clip_scores, video_score, avg_weights


def plot_temporal_scores(anomaly_scores_list, normal_scores_list,
                         anomaly_names, normal_names, save_path):
    """Plot anomaly scores over time for sample videos."""
    plt.rcParams.update({
        "font.size": 11, "font.family": "serif",
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"
    })

    n_anomaly = len(anomaly_scores_list)
    n_normal = len(normal_scores_list)
    n_total = n_anomaly + n_normal

    fig, axes = plt.subplots(n_total, 1, figsize=(10, 2.2 * n_total), sharex=False)
    if n_total == 1:
        axes = [axes]

    # Plot anomaly videos
    for idx in range(n_anomaly):
        ax = axes[idx]
        scores = anomaly_scores_list[idx]
        T = len(scores)
        x = np.arange(T)

        ax.fill_between(x, scores, alpha=0.3, color="#E74C3C")
        ax.plot(x, scores, color="#E74C3C", linewidth=1.5)
        ax.set_ylabel("Score", fontsize=10)
        name = anomaly_names[idx].replace("_x264", "")
        ax.set_title(f"{name} (Anomaly)", fontsize=11, fontweight="bold", color="#C0392B")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    # Plot normal videos
    for idx in range(n_normal):
        ax = axes[n_anomaly + idx]
        scores = normal_scores_list[idx]
        T = len(scores)
        x = np.arange(T)

        ax.fill_between(x, scores, alpha=0.3, color="#2ECC71")
        ax.plot(x, scores, color="#27AE60", linewidth=1.5)
        ax.set_ylabel("Score", fontsize=10)
        name = normal_names[idx].replace("_x264", "")
        ax.set_title(f"{name} (Normal)", fontsize=11, fontweight="bold", color="#27AE60")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    axes[-1].set_xlabel("Clip Index (Time)", fontsize=11)
    plt.tight_layout()
    fig.savefig(save_path)
    fig.savefig(str(save_path).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  Saved temporal scores: {save_path}")


def plot_attention_heatmap(attn_weights, num_prompts, video_name, label,
                           clip_scores, save_path):
    """Plot attention heatmap for a single video."""
    plt.rcParams.update({
        "font.size": 11, "font.family": "serif",
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"
    })

    P = num_prompts
    # Get feature-to-feature attention (skip prompt rows/cols)
    feat_attn = attn_weights[P:, P:]

    # Limit to first 80 clips for readability
    max_clips = min(80, feat_attn.shape[0], len(clip_scores))
    feat_attn = feat_attn[:max_clips, :max_clips]
    scores = clip_scores[:max_clips]

    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[1, 4],
                           hspace=0.05, wspace=0.05)

    # Top: anomaly scores
    ax_scores = fig.add_subplot(gs[0, 0])
    color = "#E74C3C" if label == 1 else "#2ECC71"
    ax_scores.fill_between(np.arange(max_clips), scores, alpha=0.3, color=color)
    ax_scores.plot(np.arange(max_clips), scores, color=color, linewidth=1.5)
    ax_scores.set_xlim(0, max_clips - 1)
    ax_scores.set_ylim(-0.05, 1.05)
    ax_scores.set_ylabel("Score", fontsize=9)
    ax_scores.set_xticks([])
    label_text = "Anomaly" if label == 1 else "Normal"
    name = video_name.replace("_x264", "")
    ax_scores.set_title(f"{name} ({label_text})", fontsize=11, fontweight="bold")
    ax_scores.grid(True, alpha=0.2)

    # Main: attention heatmap
    ax_heatmap = fig.add_subplot(gs[1, 0])
    im = ax_heatmap.imshow(feat_attn, aspect="auto", cmap="YlOrRd",
                            interpolation="nearest", origin="lower")
    ax_heatmap.set_xlabel("Key (attending to)", fontsize=10)
    ax_heatmap.set_ylabel("Query (attending from)", fontsize=10)

    # Colorbar
    ax_cb = fig.add_subplot(gs[1, 1])
    plt.colorbar(im, cax=ax_cb, label="Attention weight")

    fig.savefig(save_path)
    fig.savefig(str(save_path).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  Saved attention heatmap: {save_path}")


def plot_combined_figure(anomaly_data, normal_data, num_prompts, save_path):
    """
    Create a single combined figure for the paper showing:
    - Top row: anomaly video (scores + heatmap)
    - Bottom row: normal video (scores + heatmap)
    """
    plt.rcParams.update({
        "font.size": 10, "font.family": "serif",
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"
    })

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    P = num_prompts
    max_clips = 60

    for row, (scores, attn, name, label) in enumerate([anomaly_data, normal_data]):
        color = "#E74C3C" if label == 1 else "#27AE60"
        label_text = "Anomaly" if label == 1 else "Normal"
        clean_name = name.replace("_x264", "")
        clip_count = min(max_clips, len(scores))

        # Left: temporal scores
        ax1 = fig.add_subplot(gs[row, 0])
        x = np.arange(clip_count)
        ax1.fill_between(x, scores[:clip_count], alpha=0.3, color=color)
        ax1.plot(x, scores[:clip_count], color=color, linewidth=1.5)
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_xlabel("Clip index (time)")
        ax1.set_ylabel("Anomaly score")
        ax1.set_title(f"{clean_name} ({label_text}) - Temporal scores",
                      fontsize=11, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

        # Right: attention heatmap
        ax2 = fig.add_subplot(gs[row, 1])
        feat_attn = attn[P:P+clip_count, P:P+clip_count]
        im = ax2.imshow(feat_attn, aspect="auto", cmap="YlOrRd",
                        interpolation="nearest", origin="lower")
        ax2.set_xlabel("Key position")
        ax2.set_ylabel("Query position")
        ax2.set_title(f"{clean_name} ({label_text}) - Causal attention",
                      fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.savefig(save_path)
    fig.savefig(str(save_path).replace(".png", ".pdf"))
    plt.close(fig)
    print(f"  Saved combined figure: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="CausalVAD Visualization")
    parser.add_argument("--checkpoint", type=str,
                        default="outputs/standard_split/checkpoints/best_model.pt")
    parser.add_argument("--dataset", type=str, default="ucf_standard")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of anomaly/normal videos to visualize")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations")
    args = parser.parse_args()

    device = get_device("auto")
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  CausalVAD — Attention Visualization")
    print("=" * 60)

    # Load model
    print(f"\n  Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(PROJECT_ROOT / args.checkpoint, map_location=device)

    model = CausalVAD(
        input_dim=512, model_dim=256, num_prompts=8,
        num_layers=2, num_heads=8, max_seq_len=200,
        use_causal_mask=True, use_temporal_decay=True,
        decay_type="exponential", decay_rate=0.1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Model loaded (epoch {checkpoint['epoch']})")

    # Dataset paths
    dataset_paths = {
        "ucf_standard": {
            "feature_dir": str(PROJECT_ROOT / "data/features/ucf_crime_merged"),
            "annotation": str(PROJECT_ROOT / "data/annotations/ucf_standard_test.txt"),
        },
        "ucf_merged": {
            "feature_dir": str(PROJECT_ROOT / "data/features/ucf_crime_merged"),
            "annotation": str(PROJECT_ROOT / "data/annotations/ucf_merged_test.txt"),
        },
    }

    cfg = dataset_paths.get(args.dataset, dataset_paths["ucf_standard"])

    # Load test dataset
    test_dataset = VideoFeatureDataset(
        feature_dir=cfg["feature_dir"],
        annotation_file=cfg["annotation"],
        max_seq_len=200, feature_dim=512, is_test=True,
    )

    # Find good anomaly and normal examples
    print(f"\n  Finding sample videos ({args.num_samples} each)...")
    anomaly_indices = []
    normal_indices = []

    for idx in range(len(test_dataset)):
        item = test_dataset[idx]
        if item["label"].item() == 1 and len(anomaly_indices) < args.num_samples:
            anomaly_indices.append(idx)
        elif item["label"].item() == 0 and len(normal_indices) < args.num_samples:
            normal_indices.append(idx)
        if len(anomaly_indices) >= args.num_samples and len(normal_indices) >= args.num_samples:
            break

    # Process anomaly videos
    print(f"\n  Processing {len(anomaly_indices)} anomaly videos...")
    anomaly_scores_list = []
    anomaly_names = []
    anomaly_attn_list = []
    anomaly_labels = []

    for idx in anomaly_indices:
        item = test_dataset[idx]
        features = item["features"].unsqueeze(0)
        mask = item["mask"].unsqueeze(0)

        clip_scores, video_score, attn_weights = extract_attention_and_scores(
            model, features, mask, device
        )

        real_len = int(mask.sum().item())
        anomaly_scores_list.append(clip_scores[0, :real_len])
        anomaly_names.append(item["video_name"])
        anomaly_attn_list.append(attn_weights[0])
        anomaly_labels.append(1)
        print(f"    {item['video_name']}: score={video_score[0]:.4f}, len={real_len}")

    # Process normal videos
    print(f"\n  Processing {len(normal_indices)} normal videos...")
    normal_scores_list = []
    normal_names = []
    normal_attn_list = []
    normal_labels = []

    for idx in normal_indices:
        item = test_dataset[idx]
        features = item["features"].unsqueeze(0)
        mask = item["mask"].unsqueeze(0)

        clip_scores, video_score, attn_weights = extract_attention_and_scores(
            model, features, mask, device
        )

        real_len = int(mask.sum().item())
        normal_scores_list.append(clip_scores[0, :real_len])
        normal_names.append(item["video_name"])
        normal_attn_list.append(attn_weights[0])
        normal_labels.append(0)
        print(f"    {item['video_name']}: score={video_score[0]:.4f}, len={real_len}")

    # Generate plots
    print(f"\n  Generating figures...")

    # 1. Temporal score plots
    plot_temporal_scores(
        anomaly_scores_list, normal_scores_list,
        anomaly_names, normal_names,
        str(output_dir / "temporal_scores.png")
    )

    # 2. Individual attention heatmaps
    for i, idx in enumerate(anomaly_indices):
        plot_attention_heatmap(
            anomaly_attn_list[i], 8, anomaly_names[i], 1,
            anomaly_scores_list[i],
            str(output_dir / f"attention_anomaly_{i}.png")
        )

    for i, idx in enumerate(normal_indices):
        plot_attention_heatmap(
            normal_attn_list[i], 8, normal_names[i], 0,
            normal_scores_list[i],
            str(output_dir / f"attention_normal_{i}.png")
        )

    # 3. Combined figure for paper (1 anomaly + 1 normal)
    anomaly_data = (anomaly_scores_list[0], anomaly_attn_list[0], anomaly_names[0], 1)
    normal_data = (normal_scores_list[0], normal_attn_list[0], normal_names[0], 0)

    plot_combined_figure(
        anomaly_data, normal_data, 8,
        str(output_dir / "paper_figure_attention.png")
    )

    print(f"\n{'=' * 60}")
    print(f"  ALL VISUALIZATIONS SAVED TO: {output_dir}")
    print(f"{'=' * 60}")
    print(f"\n  Files generated:")
    print(f"    temporal_scores.png        — Score over time (all samples)")
    print(f"    attention_anomaly_0.png    — Heatmap for anomaly video")
    print(f"    attention_normal_0.png     — Heatmap for normal video")
    print(f"    paper_figure_attention.png — Combined figure for paper")
    print(f"\n  Upload paper_figure_attention.png to Overleaf as Figure 3")


if __name__ == "__main__":
    main()
