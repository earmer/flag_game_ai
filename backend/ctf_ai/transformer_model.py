from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def require_torch() -> "object":
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch not installed; Transformer model unavailable") from exc
    return torch, nn


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    num_entity_types: int = 6
    feature_dim: int = 8  # must match encoding.py
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.1
    num_roles: int = 5


def build_model(cfg: Optional[TransformerConfig] = None) -> "object":
    torch, nn = require_torch()
    cfg = cfg or TransformerConfig()

    class SmallPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.type_emb = nn.Embedding(cfg.num_entity_types, cfg.d_model)
            self.feat_proj = nn.Linear(cfg.feature_dim, cfg.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.nhead,
                dim_feedforward=cfg.dim_feedforward,
                dropout=cfg.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Linear(cfg.d_model, cfg.num_roles),
            )

        def forward(self, type_ids: "torch.Tensor", features: "torch.Tensor", padding_mask: "torch.Tensor") -> "torch.Tensor":
            # type_ids: (B,T), features: (B,T,F), padding_mask: (B,T) True=PAD
            x = self.type_emb(type_ids) + self.feat_proj(features)
            h = self.encoder(x, src_key_padding_mask=padding_mask)
            return self.head(h)  # (B,T,num_roles)

    return SmallPolicy()

