"""
==============================================================================
Causal Temporal Prompt Module — Core Research Contribution
==============================================================================

This module implements the causal temporal prompting mechanism for
video anomaly detection. It is the KEY NOVELTY of the CausalVAD paper.

The idea:
  Existing VLM-based VAD methods (VERA, VadCLIP) treat video clips
  independently or use simple temporal aggregation. They ignore that
  anomalies UNFOLD OVER TIME — a person walking normally becomes an
  anomaly when they suddenly change direction toward a restricted area.

  CausalVAD addresses this with three innovations:

  1. LEARNABLE SOFT PROMPTS: Continuous prompt vectors prepended to
     feature sequences, trained via backpropagation to capture
     anomaly-relevant patterns.

  2. CAUSAL ATTENTION MASK: Prevents future information leakage.
     When evaluating whether clip t is anomalous, the model can only
     attend to clips 0..t (not t+1..T). This forces causal reasoning:
     "given what happened so far, is this abnormal?"

  3. TEMPORAL DECAY WEIGHTING: Recent clips are weighted more heavily
     than distant ones. An exponentially decaying attention bias
     captures the intuition that the immediate temporal context is
     most relevant for anomaly judgment.

Architecture:
  Input:  (B, T, D) — batch of video feature sequences
  ┌──────────────────────────────────────┐
  │  Learnable Prompt Tokens  (P, D)     │
  │  + Input Features          (T, D)    │
  │  = Combined Sequence       (P+T, D)  │
  └──────────────┬───────────────────────┘
                 │
  ┌──────────────▼───────────────────────┐
  │  Causal Temporal Transformer         │
  │  - Multi-head self-attention         │
  │  - Causal mask (lower triangular)    │
  │  - Temporal decay bias               │
  │  - LayerNorm + FFN                   │
  └──────────────┬───────────────────────┘
                 │
  ┌──────────────▼───────────────────────┐
  │  Output: Temporally-aware features   │
  │  (B, T, D) — enriched with causal   │
  │  temporal context from prompts       │
  └──────────────────────────────────────┘

==============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TemporalDecayBias(nn.Module):
    """
    Generates a temporal decay bias matrix for attention scores.

    The bias encourages the model to attend more strongly to recent
    clips than distant ones. This captures the intuition that
    temporally proximate events are more relevant for anomaly judgment.

    Three modes:
      - "exponential": bias[i,j] = -rate * |i - j|
      - "linear":      bias[i,j] = -rate * |i - j| / max_len
      - "learned":     bias = learnable parameter matrix
    """

    def __init__(
        self,
        max_len: int,
        decay_type: str = "exponential",
        decay_rate: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.decay_type = decay_type
        self.decay_rate = decay_rate

        if decay_type == "learned":
            # Fully learnable bias — model discovers its own temporal pattern
            self.bias = nn.Parameter(torch.zeros(max_len, max_len))
            nn.init.normal_(self.bias, std=0.02)
        else:
            # Pre-computed bias based on temporal distance
            positions = torch.arange(max_len).float()
            # distance[i,j] = |i - j|
            distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()

            if decay_type == "exponential":
                bias = -decay_rate * distance
            elif decay_type == "linear":
                bias = -decay_rate * distance / max_len
            else:
                raise ValueError(f"Unknown decay type: {decay_type}")

            # Register as buffer (moves to device with model but not a parameter)
            self.register_buffer("bias", bias)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Returns temporal decay bias for the given sequence length.

        Args:
            seq_len: Actual sequence length (may be < max_len).

        Returns:
            (seq_len, seq_len) bias tensor to add to attention scores.
        """
        return self.bias[:seq_len, :seq_len]


class CausalTemporalAttention(nn.Module):
    """
    Multi-head self-attention with:
      1. Causal mask (can only attend to past + current, not future)
      2. Temporal decay bias (recent clips weighted more)
      3. Padding mask support

    This is the CORE MECHANISM of CausalVAD.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_len: int = 210,
        use_causal_mask: bool = True,
        decay_type: str = "exponential",
        decay_rate: float = 0.1,
        use_temporal_decay: bool = True,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_causal_mask = use_causal_mask
        self.use_temporal_decay = use_temporal_decay

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Temporal decay bias
        if use_temporal_decay:
            self.temporal_decay = TemporalDecayBias(
                max_len=max_len,
                decay_type=decay_type,
                decay_rate=decay_rate,
            )

        # Pre-compute causal mask
        if use_causal_mask:
            # Lower triangular mask: position i can attend to positions 0..i
            causal = torch.tril(torch.ones(max_len, max_len))
            # Convert to additive mask: 0 for allowed, -inf for blocked
            causal_mask = causal.log()  # 0 -> -inf, 1 -> 0
            self.register_buffer("causal_mask", causal_mask)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input sequence
            padding_mask: (B, L) with 1 for real tokens, 0 for padding

        Returns:
            (B, L, D) attended sequence
        """
        B, L, D = x.shape

        # Project to Q, K, V
        Q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (B, num_heads, L, head_dim)

        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # Shape: (B, num_heads, L, L)

        # Apply causal mask (prevents attending to future)
        if self.use_causal_mask:
            attn_scores = attn_scores + self.causal_mask[:L, :L].unsqueeze(0).unsqueeze(0)

        # Apply temporal decay bias (discounts distant past)
        if self.use_temporal_decay:
            decay_bias = self.temporal_decay(L)
            attn_scores = attn_scores + decay_bias.unsqueeze(0).unsqueeze(0)

        # Apply padding mask
        if padding_mask is not None:
            # padding_mask: (B, L) -> (B, 1, 1, L) for broadcasting
            pad_mask = padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(pad_mask == 0, float("-inf"))

        # Softmax + dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)  # handle all-masked rows
        attn_weights = self.attn_dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)
        # Shape: (B, num_heads, L, head_dim)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out


class CausalTransformerBlock(nn.Module):
    """
    Single transformer block with causal temporal attention.

    Architecture:
        x -> LayerNorm -> CausalTemporalAttention -> + residual
          -> LayerNorm -> FFN                      -> + residual
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 210,
        use_causal_mask: bool = True,
        decay_type: str = "exponential",
        decay_rate: float = 0.1,
        use_temporal_decay: bool = True,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalTemporalAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            max_len=max_len,
            use_causal_mask=use_causal_mask,
            decay_type=decay_type,
            decay_rate=decay_rate,
            use_temporal_decay=use_temporal_decay,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
            padding_mask: (B, L)
        Returns:
            (B, L, D)
        """
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), padding_mask=padding_mask)
        # FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class CausalTemporalPrompt(nn.Module):
    """
    ╔══════════════════════════════════════════════════════════════════╗
    ║  CAUSAL TEMPORAL PROMPT MODULE — The Core Innovation           ║
    ╚══════════════════════════════════════════════════════════════════╝

    This module prepends learnable prompt tokens to the input feature
    sequence, then processes the combined sequence through a causal
    temporal transformer.

    The prompts learn to:
      - Encode what "normal" patterns look like (context)
      - Capture anomaly-detection-specific temporal patterns
      - Provide a learned prior that conditions attention

    The causal transformer ensures:
      - No future information leakage (clip t can't see clip t+1)
      - Temporal decay biases attention toward recent context
      - Output features are temporally-aware and causally valid

    Args:
        feature_dim:     Input feature dimension (must match VLM features).
        num_prompts:     Number of learnable prompt tokens to prepend.
        num_layers:      Number of causal transformer layers.
        num_heads:       Number of attention heads.
        dropout:         Dropout rate.
        max_seq_len:     Maximum input sequence length.
        use_causal_mask: Whether to apply causal attention masking.
        use_temporal_decay: Whether to apply temporal decay bias.
        decay_type:      Type of temporal decay ("exponential"|"linear"|"learned").
        decay_rate:      Decay rate parameter.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        num_prompts: int = 8,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 200,
        use_causal_mask: bool = True,
        use_temporal_decay: bool = True,
        decay_type: str = "exponential",
        decay_rate: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prompts = num_prompts
        self.max_seq_len = max_seq_len

        # ─── Learnable Prompt Tokens ─────────────────────────────────
        # These are the "soft prompts" that get prepended to every video
        # They learn anomaly-detection-specific context during training
        self.prompt_tokens = nn.Parameter(
            torch.randn(1, num_prompts, feature_dim) * 0.02
        )

        # ─── Positional Encoding ─────────────────────────────────────
        # Sinusoidal positional encoding for the full sequence
        # (prompts + video clips)
        total_len = num_prompts + max_seq_len
        pe = self._create_sinusoidal_pe(total_len, feature_dim)
        self.register_buffer("positional_encoding", pe)

        # ─── Input Projection ────────────────────────────────────────
        # If input features have different dim than model, project them
        self.input_proj = nn.Linear(feature_dim, feature_dim)
        self.input_norm = nn.LayerNorm(feature_dim)

        # ─── Causal Transformer Layers ───────────────────────────────
        self.transformer_layers = nn.ModuleList([
            CausalTransformerBlock(
                d_model=feature_dim,
                num_heads=num_heads,
                ffn_dim=feature_dim * 4,
                dropout=dropout,
                max_len=total_len,
                use_causal_mask=use_causal_mask,
                decay_type=decay_type,
                decay_rate=decay_rate,
                use_temporal_decay=use_temporal_decay,
            )
            for _ in range(num_layers)
        ])

        # ─── Output Norm ─────────────────────────────────────────────
        self.output_norm = nn.LayerNorm(feature_dim)

        # Log parameter count
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"CausalTemporalPrompt: {n_params:,} trainable parameters")

    @staticmethod
    def _create_sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            features: (B, T, D) — batch of video feature sequences
            mask:     (B, T) — padding mask (1=real clip, 0=padding)

        Returns:
            output_features: (B, T, D) — temporally-enriched features
                             (prompt tokens are stripped from output)
            prompt_features: (B, P, D) — the processed prompt token features
                             (useful for analysis/explanation)
        """
        B, T, D = features.shape
        P = self.num_prompts

        # Project input features
        features = self.input_proj(features)
        features = self.input_norm(features)

        # Expand prompt tokens for the batch
        # (1, P, D) -> (B, P, D)
        prompts = self.prompt_tokens.expand(B, -1, -1)

        # Concatenate: [prompts | features]
        # Shape: (B, P+T, D)
        combined = torch.cat([prompts, features], dim=1)

        # Add positional encoding
        combined = combined + self.positional_encoding[:, :P + T, :]

        # Create extended mask (prompts are always unmasked)
        if mask is not None:
            # Prompts always visible: ones for prompts, original mask for features
            prompt_mask = torch.ones(B, P, device=mask.device, dtype=mask.dtype)
            extended_mask = torch.cat([prompt_mask, mask], dim=1)
        else:
            extended_mask = None

        # Pass through causal transformer layers
        x = combined
        for layer in self.transformer_layers:
            x = layer(x, padding_mask=extended_mask)

        # Final normalization
        x = self.output_norm(x)

        # Split back into prompt and feature parts
        prompt_features = x[:, :P, :]     # (B, P, D)
        output_features = x[:, P:, :]     # (B, T, D)

        return output_features, prompt_features
