"""
==============================================================================
CausalVAD — Full Model
==============================================================================

Integrates all components:
  1. Feature projection (maps I3D features to VLM-compatible dimension)
  2. CausalTemporalPrompt (the core innovation)
  3. Anomaly classifier (per-clip anomaly scores)
  4. Loss computation (MIL-based weakly supervised loss)

Training paradigm: Multiple Instance Learning (MIL)
  - Video = "bag" of clip instances
  - Video-level label (normal/anomaly) is known
  - Clip-level labels are NOT known
  - Key insight: in an anomaly video, at least some clips are anomalous
  - We select top-k highest scoring clips from each video and train on those

==============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import logging

from .causal_prompt import CausalTemporalPrompt

logger = logging.getLogger(__name__)


class AnomalyClassifier(nn.Module):
    """
    Per-clip anomaly scoring head.

    Takes temporally-enriched features and produces a scalar anomaly
    score for each clip (0 = definitely normal, 1 = definitely anomalous).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        current_dim = input_dim

        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        # Final layer: single score per clip
        layers.append(nn.Linear(current_dim, 1))

        self.classifier = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, D)
        Returns:
            scores: (B, T) — anomaly score per clip
        """
        scores = self.classifier(features).squeeze(-1)
        return torch.sigmoid(scores)


class CausalVAD(nn.Module):
    """
    ╔══════════════════════════════════════════════════════════════════╗
    ║  CausalVAD — Complete Model                                    ║
    ║                                                                ║
    ║  Causal Temporal Prompting for Explainable Video Anomaly       ║
    ║  Detection via Compact Vision-Language Models                  ║
    ╚══════════════════════════════════════════════════════════════════╝

    Pipeline:
      Input Features (B, T, D_in)
        → Feature Projection (B, T, D_model)
        → CausalTemporalPrompt (B, T, D_model)
        → AnomalyClassifier (B, T) per-clip scores
        → MIL Aggregation → video-level prediction

    Args:
        input_dim:       Raw feature dimension (e.g., 2048 for I3D).
        model_dim:       Internal model dimension (e.g., 512).
        num_prompts:     Number of learnable prompt tokens.
        num_layers:      Number of causal transformer layers.
        num_heads:       Number of attention heads.
        max_seq_len:     Max clips per video.
        dropout:         Dropout rate.
        use_causal_mask: Enable causal attention masking.
        use_temporal_decay: Enable temporal decay bias.
        decay_type:      "exponential" | "linear" | "learned"
        decay_rate:      Temporal decay rate.
        topk_ratio:      Fraction of clips to select for MIL loss.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        model_dim: int = 512,
        num_prompts: int = 8,
        num_layers: int = 2,
        num_heads: int = 8,
        max_seq_len: int = 200,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
        use_temporal_decay: bool = True,
        decay_type: str = "exponential",
        decay_rate: float = 0.1,
        classifier_hidden: int = 256,
        classifier_layers: int = 2,
        classifier_dropout: float = 0.3,
        topk_ratio: float = 0.1,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.topk_ratio = topk_ratio
        self.max_seq_len = max_seq_len

        # ─── Feature Projection ──────────────────────────────────────
        # Map I3D (2048-d) features to model dimension (512-d)
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ─── Causal Temporal Prompt Module ───────────────────────────
        self.causal_prompt = CausalTemporalPrompt(
            feature_dim=model_dim,
            num_prompts=num_prompts,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            use_causal_mask=use_causal_mask,
            use_temporal_decay=use_temporal_decay,
            decay_type=decay_type,
            decay_rate=decay_rate,
        )

        # ─── Anomaly Classifier ──────────────────────────────────────
        self.classifier = AnomalyClassifier(
            input_dim=model_dim,
            hidden_dim=classifier_hidden,
            num_layers=classifier_layers,
            dropout=classifier_dropout,
        )

        # Log total parameters
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"CausalVAD: {total:,} total params, {trainable:,} trainable")

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: (B, T, D_in) — raw video features
            mask:     (B, T) — padding mask

        Returns:
            dict with:
                "clip_scores":   (B, T) — per-clip anomaly scores
                "video_score":   (B,)   — video-level anomaly score
                "temporal_features": (B, T, D) — enriched features
                "prompt_features":   (B, P, D) — processed prompts
        """
        B, T, D_in = features.shape

        # Step 1: Project features to model dimension
        projected = self.feature_proj(features)  # (B, T, model_dim)

        # Step 2: Causal temporal prompting
        temporal_features, prompt_features = self.causal_prompt(projected, mask)

        # Step 3: Per-clip anomaly classification
        clip_scores = self.classifier(temporal_features)  # (B, T)

        # Step 4: Apply mask to scores (zero out padding positions)
        if mask is not None:
            clip_scores = clip_scores * mask

        # Step 5: Video-level score via top-k aggregation
        # Select the top-k% highest scoring clips and average them
        k = max(1, int(T * self.topk_ratio))
        topk_scores, _ = torch.topk(clip_scores, k=k, dim=1)
        video_score = topk_scores.mean(dim=1)  # (B,)

        return {
            "clip_scores": clip_scores,
            "video_score": video_score,
            "temporal_features": temporal_features,
            "prompt_features": prompt_features,
        }


class CausalVADLoss(nn.Module):
    """
    Loss function for CausalVAD with three components:

    1. MIL RANKING LOSS (primary):
       For each pair of (anomaly_video, normal_video):
         max(0, 1 - score(anomaly_top_k) + score(normal_top_k))
       Ensures anomaly videos score higher than normal videos.

    2. TEMPORAL SMOOTHNESS LOSS:
       Penalizes large score differences between adjacent clips:
         sum(|score[t] - score[t-1]|)
       Anomalies should have smooth temporal boundaries, not random spikes.

    3. SPARSITY LOSS:
       Penalizes having too many clips flagged as anomalous:
         mean(scores)
       Most clips, even in anomaly videos, should be normal.
    """

    def __init__(
        self,
        mil_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        sparsity_weight: float = 0.01,
        margin: float = 1.0,
        topk_ratio: float = 0.1,
    ):
        super().__init__()
        self.mil_weight = mil_weight
        self.smoothness_weight = smoothness_weight
        self.sparsity_weight = sparsity_weight
        self.margin = margin
        self.topk_ratio = topk_ratio

    def forward(
        self,
        clip_scores: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            clip_scores: (B, T) per-clip anomaly scores
            labels:      (B,)   video-level labels (0=normal, 1=anomaly)
            mask:        (B, T) padding mask

        Returns:
            dict with "total", "mil", "smoothness", "sparsity" losses
        """
        B, T = clip_scores.shape

        # Apply mask
        if mask is not None:
            clip_scores = clip_scores * mask

        # ─── Top-k Selection ─────────────────────────────────────────
        k = max(1, int(T * self.topk_ratio))
        topk_scores, _ = torch.topk(clip_scores, k=k, dim=1)
        video_scores = topk_scores.mean(dim=1)  # (B,)

        # ─── 1. MIL Ranking Loss ─────────────────────────────────────
        # Separate anomaly and normal video scores
        anomaly_mask = (labels == 1)
        normal_mask = (labels == 0)

        anomaly_scores = video_scores[anomaly_mask]
        normal_scores = video_scores[normal_mask]

        mil_loss = torch.tensor(0.0, device=clip_scores.device)

        if len(anomaly_scores) > 0 and len(normal_scores) > 0:
            # Pairwise ranking: anomaly should score higher than normal
            # Broadcast: (num_anomaly, 1) - (1, num_normal)
            a = anomaly_scores.unsqueeze(1)
            n = normal_scores.unsqueeze(0)
            mil_loss = F.relu(self.margin - a + n).mean()

        # ─── 2. Temporal Smoothness Loss ─────────────────────────────
        # Penalize abrupt score changes between adjacent clips
        diff = clip_scores[:, 1:] - clip_scores[:, :-1]
        smoothness_loss = (diff ** 2).mean()

        # ─── 3. Sparsity Loss ────────────────────────────────────────
        # Encourage most clip scores to be low
        sparsity_loss = clip_scores.mean()

        # ─── Combined Loss ───────────────────────────────────────────
        total = (
            self.mil_weight * mil_loss
            + self.smoothness_weight * smoothness_loss
            + self.sparsity_weight * sparsity_loss
        )

        return {
            "total": total,
            "mil": mil_loss,
            "smoothness": smoothness_loss,
            "sparsity": sparsity_loss,
        }


def build_model(
    input_dim: int = 2048,
    model_dim: int = 512,
    num_prompts: int = 8,
    num_layers: int = 2,
    num_heads: int = 8,
    max_seq_len: int = 200,
    use_causal_mask: bool = True,
    use_temporal_decay: bool = True,
    decay_type: str = "exponential",
    decay_rate: float = 0.1,
    device: str = "cpu",
) -> Tuple[CausalVAD, CausalVADLoss]:
    """
    Convenience function to build model + loss.

    Example:
        >>> model, criterion = build_model(device="cuda")
        >>> output = model(features, mask)
        >>> losses = criterion(output["clip_scores"], labels, mask)
    """
    model = CausalVAD(
        input_dim=input_dim,
        model_dim=model_dim,
        num_prompts=num_prompts,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
        use_causal_mask=use_causal_mask,
        use_temporal_decay=use_temporal_decay,
        decay_type=decay_type,
        decay_rate=decay_rate,
    ).to(device)

    criterion = CausalVADLoss().to(device)

    return model, criterion
