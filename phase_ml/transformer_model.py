"""PyTorch language-model-style teacher for phase prediction."""

from __future__ import annotations

import math


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required for phase_ml.train_transformer. Install requirements-phase-ml.txt on the A100 machine.") from exc
    return torch, nn


def build_model(input_dim: int, phase_count: int, config: dict[str, object]):
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
            self.proj = nn.Linear(hidden, hidden)
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
            batch, seq_len, hidden = x.shape
            qkv = self.qkv(x).view(batch, seq_len, 3, self.heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q, k = self._apply_rope(q, k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(causal_mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            weights = self.attn_dropout(weights)
            out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, seq_len, hidden)
            return self.resid_dropout(self.proj(out))

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

    class PhaseLMBlock(nn.Module):
        def __init__(self, hidden: int, heads: int, ff_dim: int, dropout: float) -> None:
            super().__init__()
            self.attn_norm = nn.LayerNorm(hidden)
            self.attn = CausalSelfAttention(hidden, heads, dropout)
            self.ffn_norm = nn.LayerNorm(hidden)
            self.ffn = FeedForward(hidden, ff_dim, dropout)

        def forward(self, x):
            x = x + self.attn(self.attn_norm(x))
            x = x + self.ffn(self.ffn_norm(x))
            return x

    class PhaseLanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = int(config.get("hidden_dim", 128))
            heads = int(config.get("num_heads", 4))
            ff_dim = int(config.get("ff_dim", 4 * hidden))
            dropout = float(config.get("dropout", 0.1))
            self.input_proj = nn.Linear(input_dim, hidden)
            self.dropout = nn.Dropout(dropout)
            self.blocks = nn.ModuleList(
                [PhaseLMBlock(hidden, heads, ff_dim, dropout) for _ in range(int(config.get("num_layers", 4)))]
            )
            self.final_norm = nn.LayerNorm(hidden)
            self.next_phase = nn.Linear(hidden, phase_count)
            self.phase_change = nn.Linear(hidden, 1)

        def forward(self, x):
            hidden = self.dropout(self.input_proj(x))
            for block in self.blocks:
                hidden = block(hidden)
            pooled = self.final_norm(hidden[:, -1, :])
            return self.next_phase(pooled), self.phase_change(pooled).squeeze(-1)

    return PhaseLanguageModel()
