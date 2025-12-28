from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from _import_bootstrap import get_geometry

Geometry = get_geometry()


Token = Tuple[int, List[float]]  # (type_id, features)


ENTITY_TYPES: Dict[str, int] = {
    "global": 0,
    "my_player": 1,
    "opp_player": 2,
    "opp_flag": 3,  # pickup-able only
    "my_target": 4,
    "my_prison": 5,
}


def _pos(entity: Mapping[str, Any]) -> Tuple[int, int]:
    return (int(entity.get("posX", 0)), int(entity.get("posY", 0)))


def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _min_dist(src: Tuple[int, int], targets: Sequence[Tuple[int, int]]) -> float:
    if not targets:
        return 999.0
    return float(min(_manhattan(src, t) for t in targets))


@dataclass(frozen=True, slots=True)
class EncodedBatch:
    type_ids: "object"  # torch.Tensor (B,T) long, but torch is optional
    features: "object"  # torch.Tensor (B,T,F) float
    padding_mask: "object"  # torch.Tensor (B,T) bool; True for PAD
    my_player_token_indices: Tuple[int, ...]  # indices into T


def encode_status_for_team(
    status_req: Mapping[str, Any],
    geometry: Geometry,
    *,
    max_tokens: int = 32,
) -> Tuple[List[int], List[List[float]], List[bool], Tuple[int, ...]]:
    """Encode a single status payload into Transformer-friendly tokens.

    Returns: (type_ids, features, padding_mask, my_player_token_indices).
    - `padding_mask[t] == True` means PAD (ignored by attention).
    """
    tokens: List[Token] = []

    my_players = list(status_req.get("myteamPlayer") or [])
    opp_players = list(status_req.get("opponentPlayer") or [])
    opp_flags = [f for f in (status_req.get("opponentFlag") or []) if bool(f.get("canPickup"))]

    my_targets = [geometry.normalize_pos(t) for t in geometry.my_targets]
    my_prisons = [geometry.normalize_pos(p) for p in geometry.my_prisons]
    my_target_center = my_targets[0]
    my_prison_center = my_prisons[0] if my_prisons else my_target_center

    # Global token
    num_my_prisoners = sum(1 for p in my_players if bool(p.get("inPrison")))
    num_opp_prisoners = sum(1 for p in opp_players if bool(p.get("inPrison")))
    g = [
        float(status_req.get("myteamScore", 0)),
        float(status_req.get("opponentScore", 0)),
        float(num_my_prisoners),
        float(num_opp_prisoners),
        float(len(opp_flags)),
        float(geometry.width),
        float(geometry.height),
    ]
    tokens.append((ENTITY_TYPES["global"], g))

    # My players (keep indices for outputs)
    my_idx: List[int] = []
    enemy_flag_positions = [geometry.normalize_pos(_pos(f)) for f in opp_flags]
    opp_positions = [
        geometry.normalize_pos(_pos(p))
        for p in opp_players
        if not bool(p.get("inPrison"))
    ]
    for p in my_players:
        pos = geometry.normalize_pos(_pos(p))
        feats = [
            float(pos[0]) / max(1.0, float(geometry.width - 1)),
            float(pos[1]) / max(1.0, float(geometry.height - 1)),
            1.0 if bool(p.get("hasFlag")) else 0.0,
            1.0 if bool(p.get("inPrison")) else 0.0,
            _min_dist(pos, enemy_flag_positions) / float(geometry.width + geometry.height),
            _min_dist(pos, [my_target_center]) / float(geometry.width + geometry.height),
            _min_dist(pos, [my_prison_center]) / float(geometry.width + geometry.height),
            _min_dist(pos, opp_positions) / float(geometry.width + geometry.height),
        ]
        my_idx.append(len(tokens))
        tokens.append((ENTITY_TYPES["my_player"], feats))

    # Opp players
    for p in opp_players:
        pos = geometry.normalize_pos(_pos(p))
        feats = [
            float(pos[0]) / max(1.0, float(geometry.width - 1)),
            float(pos[1]) / max(1.0, float(geometry.height - 1)),
            1.0 if bool(p.get("hasFlag")) else 0.0,
            1.0 if bool(p.get("inPrison")) else 0.0,
        ]
        tokens.append((ENTITY_TYPES["opp_player"], feats))

    # Opp flags (pickup-able only)
    for f in opp_flags:
        pos = geometry.normalize_pos(_pos(f))
        feats = [
            float(pos[0]) / max(1.0, float(geometry.width - 1)),
            float(pos[1]) / max(1.0, float(geometry.height - 1)),
        ]
        tokens.append((ENTITY_TYPES["opp_flag"], feats))

    # My target / prison centers as tokens
    tokens.append(
        (
            ENTITY_TYPES["my_target"],
            [
                float(my_target_center[0]) / max(1.0, float(geometry.width - 1)),
                float(my_target_center[1]) / max(1.0, float(geometry.height - 1)),
            ],
        )
    )
    tokens.append(
        (
            ENTITY_TYPES["my_prison"],
            [
                float(my_prison_center[0]) / max(1.0, float(geometry.width - 1)),
                float(my_prison_center[1]) / max(1.0, float(geometry.height - 1)),
            ],
        )
    )

    # Pad / truncate
    tokens = tokens[:max_tokens]
    type_ids = [t[0] for t in tokens]
    feats = [t[1] for t in tokens]
    padding_mask = [False for _ in tokens]
    while len(type_ids) < max_tokens:
        type_ids.append(0)
        feats.append([0.0] * len(feats[0]))
        padding_mask.append(True)

    my_player_indices = tuple(i for i in my_idx if i < max_tokens)
    return type_ids, feats, padding_mask, my_player_indices


def to_torch_batch(
    encoded: Sequence[Tuple[List[int], List[List[float]], List[bool], Tuple[int, ...]]]
) -> EncodedBatch:
    """Convert encoded items into torch tensors (requires torch at runtime)."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch not installed; cannot build tensors") from exc

    if not encoded:
        raise ValueError("Empty batch")

    max_tokens = len(encoded[0][0])
    feat_dim = len(encoded[0][1][0])
    for type_ids, feats, mask, _idx in encoded:
        if len(type_ids) != max_tokens or len(mask) != max_tokens:
            raise ValueError("Inconsistent token length in batch")
        if any(len(f) != feat_dim for f in feats):
            raise ValueError("Inconsistent feature dim in batch")

    type_tensor = torch.tensor([e[0] for e in encoded], dtype=torch.long)
    feat_tensor = torch.tensor([e[1] for e in encoded], dtype=torch.float32)
    pad_mask = torch.tensor([e[2] for e in encoded], dtype=torch.bool)
    my_player_idx = encoded[0][3]
    return EncodedBatch(
        type_ids=type_tensor,
        features=feat_tensor,
        padding_mask=pad_mask,
        my_player_token_indices=my_player_idx,
    )
