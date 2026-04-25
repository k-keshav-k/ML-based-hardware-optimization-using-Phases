"""PyTorch transformer teacher for phase prediction."""

from __future__ import annotations


def require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required for phase_ml.train_transformer. Install requirements-phase-ml.txt on the A100 machine.") from exc
    return torch, nn


def build_model(input_dim: int, phase_count: int, config: dict[str, object]):
    torch, nn = require_torch()

    class PhaseTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = int(config.get("hidden_dim", 128))
            self.input_proj = nn.Linear(input_dim, hidden)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=int(config.get("num_heads", 4)),
                dim_feedforward=int(config.get("ff_dim", 256)),
                dropout=float(config.get("dropout", 0.1)),
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(config.get("num_layers", 4)))
            self.next_phase = nn.Linear(hidden, phase_count)
            self.phase_change = nn.Linear(hidden, 1)

        def forward(self, x):
            encoded = self.encoder(self.input_proj(x))
            pooled = encoded[:, -1, :]
            return self.next_phase(pooled), self.phase_change(pooled).squeeze(-1)

    return PhaseTransformer()
