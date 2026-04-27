"""Transformer language model for family-state sequence prediction.

The architecture explicitly includes RoPE, multi-head causal attention,
feedforward blocks, and layer normalization.
"""

from __future__ import annotations

import math


def require_torch():
    """Import torch lazily so non-neural stages still run without it."""

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required for phase_family_ml teacher training. Install requirements-phase-ml.txt.") from exc
    return torch, nn


def build_family_transformer(input_dim: int, horizon: int, num_classes: int, config: dict[str, object]):
    """Build a causal sequence model that predicts all horizons at once."""

    torch, nn = require_torch()

    class CausalSelfAttention(nn.Module):
        def __init__(self, hidden: int, heads: int, dropout: float) -> None:
            super().__init__()
            if hidden % heads != 0:
                raise ValueError("hidden_dim must be divisible by num_heads")
            self.heads = heads
            self.head_dim = hidden // heads
            if self.head_dim % 2 != 0:
                raise ValueError("hidden_dim / num_heads must be even for RoPE")
            self.qkv = nn.Linear(hidden, 3 * hidden)
            self.out = nn.Linear(hidden, hidden)
            self.attn_dropout = nn.Dropout(dropout)
            self.resid_dropout = nn.Dropout(dropout)
            self.rope_theta = float(config.get("rope_theta", 10000.0))

        def _apply_rope(self, q, k):
            seq_len = q.shape[1]
            half = self.head_dim // 2
            positions = torch.arange(seq_len, device=q.device, dtype=torch.float32)
            inv_freq = self.rope_theta ** (-torch.arange(0, half, device=q.device, dtype=torch.float32) / half)
            angles = positions[:, None] * inv_freq[None, :]
            cos = angles.cos()[None, :, None, :].to(q.dtype)
            sin = angles.sin()[None, :, None, :].to(q.dtype)

            def rotate(x):
                x1 = x[..., :half]
                x2 = x[..., half:]
                return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

            return rotate(q), rotate(k)

        def forward(self, x):
            bsz, seq_len, hidden = x.shape
            qkv = self.qkv(x).view(bsz, seq_len, 3, self.heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q, k = self._apply_rope(q, k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(1)
            attn = attn.masked_fill(mask, float("-inf"))
            weight = torch.softmax(attn, dim=-1)
            weight = self.attn_dropout(weight)
            out = torch.matmul(weight, v).transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
            return self.resid_dropout(self.out(out))

    class FeedForward(nn.Module):
        def __init__(self, hidden: int, ff_dim: int, dropout: float) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(hidden, ff_dim),
                nn.GELU(),
                nn.Linear(ff_dim, hidden),
                nn.Dropout(dropout),
            )

        def forward(self, x):
            return self.net(x)

    class Block(nn.Module):
        def __init__(self, hidden: int, heads: int, ff_dim: int, dropout: float) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden)
            self.attn = CausalSelfAttention(hidden, heads, dropout)
            self.norm2 = nn.LayerNorm(hidden)
            self.ffn = FeedForward(hidden, ff_dim, dropout)

        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.ffn(self.norm2(x))
            return x

    class FamilyTransformerLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = int(config.get("hidden_dim", 128))
            heads = int(config.get("num_heads", 4))
            ff_dim = int(config.get("ff_dim", 256))
            dropout = float(config.get("dropout", 0.1))
            self.input_proj = nn.Linear(input_dim, hidden)
            self.dropout = nn.Dropout(dropout)
            self.blocks = nn.ModuleList([Block(hidden, heads, ff_dim, dropout) for _ in range(int(config.get("num_layers", 4)))])
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, horizon * num_classes)
            self.horizon = horizon
            self.num_classes = num_classes

        def forward(self, x):
            hidden = self.dropout(self.input_proj(x))
            for block in self.blocks:
                hidden = block(hidden)
            pooled = self.norm(hidden[:, -1, :])
            logits = self.head(pooled)
            return logits.view(-1, self.horizon, self.num_classes)

    return FamilyTransformerLM()
