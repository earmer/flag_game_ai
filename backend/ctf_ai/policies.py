from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

try:
    from lib.game_engine import GameMap  # type: ignore
    from lib.tree_features import Geometry  # type: ignore
except Exception:  # pragma: no cover
    from backend.lib.game_engine import GameMap  # type: ignore
    from backend.lib.tree_features import Geometry  # type: ignore

from .constants import Roles
from .encoding import encode_status_for_team, to_torch_batch
from .transformer_model import TransformerConfig, build_model


Action = str
Pos = Tuple[int, int]


@dataclass(slots=True)
class MacroParams:
    # Role biases added to logits (can be evolved via SA-GA).
    steal_bias: float = 0.0
    return_bias: float = 0.0
    rescue_bias: float = 0.0
    chase_bias: float = 0.0
    defend_bias: float = 0.0
    # Safety: how much to treat opponents as obstacles in enemy half.
    avoid_opponents: bool = True

    def to_vec(self) -> list[float]:
        return [
            float(self.steal_bias),
            float(self.return_bias),
            float(self.rescue_bias),
            float(self.chase_bias),
            float(self.defend_bias),
        ]

    @staticmethod
    def from_json(path: str) -> "MacroParams":
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return MacroParams(
            steal_bias=float(data.get("steal_bias", 0.0)),
            return_bias=float(data.get("return_bias", 0.0)),
            rescue_bias=float(data.get("rescue_bias", 0.0)),
            chase_bias=float(data.get("chase_bias", 0.0)),
            defend_bias=float(data.get("defend_bias", 0.0)),
            avoid_opponents=bool(data.get("avoid_opponents", True)),
        )


class HeuristicPolicy:
    """Baseline: closest enemy flag; avoid opponents in enemy territory; return when carrying.

    This mirrors `backend/pick_flag_ai.py` in a dependency-light, reusable way.
    """

    def __init__(self) -> None:
        self.world = GameMap(show_gap_in_msec=10_000.0)
        self.player_to_flag: Dict[str, Pos] = {}
        self.my_side_is_left: Optional[bool] = None

    def start(self, init_req: Mapping[str, Any]) -> None:
        self.world.init(dict(init_req))
        targets = list(self.world.list_targets(mine=True))
        self.my_side_is_left = self.world.is_on_left(targets[0]) if targets else True
        self.player_to_flag = {}

    def act(self, status_req: Mapping[str, Any]) -> Dict[str, Action]:
        if not self.world.update(dict(status_req)):
            return {}

        my_players = self.world.list_players(mine=True, inPrison=False, hasFlag=None)
        opponents = self.world.list_players(mine=False, inPrison=False, hasFlag=None)
        enemy_flags = self.world.list_flags(mine=False, canPickup=True)
        my_targets = list(self.world.list_targets(mine=True))

        active_names = {p["name"] for p in my_players if not p["hasFlag"]}
        self.player_to_flag = {n: pos for n, pos in self.player_to_flag.items() if n in active_names}

        for p in my_players:
            if p["hasFlag"] or p["inPrison"]:
                continue
            if p["name"] in self.player_to_flag:
                continue
            if enemy_flags:
                f = min(enemy_flags, key=lambda fl: abs(fl["posX"] - p["posX"]) + abs(fl["posY"] - p["posY"]))
                self.player_to_flag[p["name"]] = (int(f["posX"]), int(f["posY"]))

        moves: Dict[str, Action] = {}
        for p in my_players:
            pos = (int(p["posX"]), int(p["posY"]))
            if p["hasFlag"]:
                dest = my_targets[0]
            else:
                dest = self.player_to_flag.get(p["name"])
                if not dest:
                    continue
            is_safe = self.world.is_on_left(pos) == bool(self.my_side_is_left)
            blockers = [] if is_safe else [(int(o["posX"]), int(o["posY"])) for o in opponents]
            path = self.world.route_to(pos, dest, extra_obstacles=blockers)
            if len(path) > 1:
                moves[p["name"]] = self.world.get_direction(pos, path[1])
        return moves


class TransformerMacroPolicy:
    """Small Transformer -> macro-role per player -> BFS to concrete action.

    If torch isn't installed (or no model is provided), it falls back to heuristics.
    """

    def __init__(
        self,
        *,
        model_path: str = "",
        params: Optional[MacroParams] = None,
        max_tokens: int = 32,
        device: str = "cpu",
    ) -> None:
        self.model_path = model_path
        self.params = params or MacroParams()
        self.max_tokens = int(max_tokens)
        self.device = device

        self.geometry: Optional[Geometry] = None
        self.world = GameMap(show_gap_in_msec=10_000.0)
        self.fallback = HeuristicPolicy()
        self._model: Optional[object] = None

    def start(self, init_req: Mapping[str, Any]) -> None:
        self.geometry = Geometry.from_init(init_req)
        self.world.init(dict(init_req))
        self.fallback.start(init_req)

        if not self.model_path:
            self._model = None
            return

        try:
            import torch  # type: ignore
        except Exception:
            self._model = None
            return

        model = build_model(TransformerConfig())
        state = torch.load(self.model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        model.eval()
        model.to(self.device)
        self._model = model

    def act(self, status_req: Mapping[str, Any]) -> Dict[str, Action]:
        if self.geometry is None or self._model is None:
            return self.fallback.act(status_req)
        if not self.world.update(dict(status_req)):
            return {}

        # Encode + forward
        encoded = encode_status_for_team(status_req, self.geometry, max_tokens=self.max_tokens)
        batch = to_torch_batch([encoded])

        import torch  # type: ignore

        model = self._model
        logits = model(batch.type_ids.to(self.device), batch.features.to(self.device), batch.padding_mask.to(self.device))
        # (1,T,R)
        role_bias = torch.tensor(self.params.to_vec(), dtype=logits.dtype, device=logits.device).view(1, 1, -1)
        logits = logits + role_bias

        my_indices = batch.my_player_token_indices
        if not my_indices:
            return {}

        # Build destinations + convert to actions
        # IMPORTANT: keep the same player ordering as the status payload (encoding uses this order).
        my_players = list(status_req.get("myteamPlayer") or [])
        opponents = self.world.list_players(mine=False, inPrison=False, hasFlag=None)
        enemy_flags = self.world.list_flags(mine=False, canPickup=True)
        my_targets = list(self.world.list_targets(mine=True))
        my_prisons = list(self.world.list_prisons(mine=True))

        # Simple per-tick assignment: predicted stealers take distinct nearest flags.
        available_flags = {(int(f["posX"]), int(f["posY"])) for f in enemy_flags}
        moves: Dict[str, Action] = {}
        for token_i, p in zip(my_indices, my_players):
            name = str(p.get("name", "") or "")
            pos = (int(p.get("posX", 0)), int(p.get("posY", 0)))
            if not name or bool(p.get("inPrison")):
                continue

            # Hard override: carrying -> return (keeps training stable)
            if bool(p.get("hasFlag")) and my_targets:
                dest = my_targets[0]
            else:
                role = int(torch.argmax(logits[0, token_i]).item())
                if role == Roles.RESCUE and my_prisons:
                    dest = next(iter(my_prisons))
                elif role == Roles.CHASE:
                    carriers = [o for o in opponents if bool(o.get("hasFlag")) and not bool(o.get("inPrison"))]
                    if carriers:
                        c = min(carriers, key=lambda o: abs(int(o["posX"]) - pos[0]) + abs(int(o["posY"]) - pos[1]))
                        dest = (int(c["posX"]), int(c["posY"]))
                    else:
                        dest = my_targets[0] if my_targets else pos
                elif role == Roles.DEFEND and my_targets:
                    dest = my_targets[0]
                else:
                    # STEAL (default)
                    if available_flags:
                        dest = min(available_flags, key=lambda f: abs(f[0] - pos[0]) + abs(f[1] - pos[1]))
                        available_flags.discard(dest)
                    else:
                        dest = my_targets[0] if my_targets else pos

            # Safety: avoid opponents only when in enemy half (same as baseline)
            blockers = []
            if self.params.avoid_opponents:
                my_side_is_left = self.world.is_on_left(my_targets[0]) if my_targets else True
                is_safe = self.world.is_on_left(pos) == bool(my_side_is_left)
                blockers = [] if is_safe else [(int(o["posX"]), int(o["posY"])) for o in opponents]

            path = self.world.route_to(pos, dest, extra_obstacles=blockers)
            if len(path) > 1:
                moves[name] = self.world.get_direction(pos, path[1])
        return moves
