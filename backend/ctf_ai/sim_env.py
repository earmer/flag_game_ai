from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Action = str  # "", "up", "down", "left", "right"
Pos = Tuple[int, int]


def _create3x3(center_x: int, center_y: int) -> List[Pos]:
    return [
        (center_x - 1, center_y - 1),
        (center_x, center_y - 1),
        (center_x + 1, center_y - 1),
        (center_x - 1, center_y),
        (center_x, center_y),
        (center_x + 1, center_y),
        (center_x - 1, center_y + 1),
        (center_x, center_y + 1),
        (center_x + 1, center_y + 1),
    ]


def _in_bounds(width: int, height: int, pos: Pos) -> bool:
    x, y = pos
    return 0 <= x < width and 0 <= y < height


def _move(pos: Pos, action: Action) -> Pos:
    x, y = pos
    if action == "up":
        return x, y - 1
    if action == "down":
        return x, y + 1
    if action == "left":
        return x - 1, y
    if action == "right":
        return x + 1, y
    return x, y


def _manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _bfs_next_step(width: int, height: int, blocked: set[Pos], src: Pos, dst: Pos) -> Optional[Pos]:
    if src == dst:
        return None
    if src in blocked or dst in blocked:
        return None

    from collections import deque

    q = deque([src])
    prev: Dict[Pos, Pos] = {}
    seen = {src}

    while q:
        cur = q.popleft()
        if cur == dst:
            break
        for act in ("up", "down", "left", "right"):
            nxt = _move(cur, act)
            if not _in_bounds(width, height, nxt):
                continue
            if nxt in blocked:
                continue
            if nxt in seen:
                continue
            seen.add(nxt)
            prev[nxt] = cur
            q.append(nxt)

    if dst not in seen:
        return None

    cur = dst
    while prev.get(cur) != src:
        cur = prev[cur]
    return cur


def _dir(src: Pos, nxt: Optional[Pos]) -> Action:
    if nxt is None:
        return ""
    dx = nxt[0] - src[0]
    dy = nxt[1] - src[1]
    if dx == 1:
        return "right"
    if dx == -1:
        return "left"
    if dy == 1:
        return "down"
    if dy == -1:
        return "up"
    return ""


@dataclass(slots=True)
class PlayerState:
    name: str
    team: str  # "L" or "R"
    pos: Pos
    has_flag: bool = False
    in_prison: bool = False
    in_prison_time_left_ms: int = 0
    in_prison_duration_ms: int = 20_000

    def status(self) -> Dict[str, object]:
        x, y = self.pos
        return {
            "name": self.name,
            "team": self.team,
            "hasFlag": bool(self.has_flag),
            "posX": int(x),
            "posY": int(y),
            "inPrison": bool(self.in_prison),
            "inPrisonTimeLeft": int(self.in_prison_time_left_ms),
            "inPrisonDuration": int(self.in_prison_duration_ms),
        }


@dataclass(slots=True)
class FlagState:
    team: str  # "L" or "R"
    pos: Pos
    can_pickup: bool = True

    def status(self) -> Dict[str, object]:
        x, y = self.pos
        return {"canPickup": bool(self.can_pickup), "posX": int(x), "posY": int(y)}


@dataclass(frozen=True, slots=True)
class MapSpec:
    width: int
    height: int
    blocked: frozenset[Pos]


@dataclass(slots=True)
class TeamStatic:
    target_tiles: List[Pos]
    prison_tiles: List[Pos]


@dataclass(slots=True)
class EpisodeResult:
    winner: Optional[str]  # "L"/"R"/None
    l_score: int
    r_score: int
    steps: int


class CTFSim:
    """A small discrete simulator mirroring the frontend rules closely enough for self-play training.

    Notes:
    - Discrete time: one decision = one tile move.
    - Capture is approximated: if opposing players end a step on the same tile.
    - Prison timers use a fixed dt_ms per step (default 600ms).
    """

    def __init__(
        self,
        *,
        width: int = 20,
        height: int = 20,
        num_players: int = 3,
        num_flags: int = 9,
        use_random_flags: bool = True,
        dt_ms: int = 600,
        seed: Optional[int] = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.num_players = int(num_players)
        self.num_flags = int(num_flags)
        self.use_random_flags = bool(use_random_flags)
        self.dt_ms = int(dt_ms)
        self.rng = random.Random(seed)

        self.map: Optional[MapSpec] = None
        self.players: List[PlayerState] = []
        self.flags: List[FlagState] = []
        self.l_static: Optional[TeamStatic] = None
        self.r_static: Optional[TeamStatic] = None
        self.l_score: int = 0
        self.r_score: int = 0
        self.step_count: int = 0
        self.done: bool = False

    def reset(self) -> None:
        self.step_count = 0
        self.done = False
        self.l_score = 0
        self.r_score = 0

        blocked = set()
        # Boundary walls (match frontend)
        for x in range(self.width):
            blocked.add((x, 0))
            blocked.add((x, self.height - 1))
        for y in range(self.height):
            blocked.add((0, y))
            blocked.add((self.width - 1, y))

        obstacles1: List[Pos] = []
        for _ in range(8):
            while True:
                x = self.rng.randint(4, self.width - 5)
                y = self.rng.randint(1, self.height - 2)
                if (x, y) not in blocked and (x, y) not in obstacles1:
                    obstacles1.append((x, y))
                    break

        obstacles2: List[Pos] = []
        for _ in range(4):
            while True:
                x = self.rng.randint(4, self.width - 5)
                y = self.rng.randint(1, self.height - 3)
                if (x, y) in obstacles1:
                    continue
                if (x, y + 1) in obstacles1:
                    continue
                if (x, y - 1) in obstacles2:
                    continue
                if (x, y) in obstacles2:
                    continue
                obstacles2.append((x, y))
                break

        for pos in obstacles1:
            blocked.add(pos)
        for pos in obstacles2:
            blocked.add(pos)
            blocked.add((pos[0], pos[1] + 1))

        self.map = MapSpec(width=self.width, height=self.height, blocked=frozenset(blocked))

        center_y = self.height // 2
        self.l_static = TeamStatic(
            target_tiles=_create3x3(2, center_y),
            prison_tiles=_create3x3(2, self.height - 3),
        )
        self.r_static = TeamStatic(
            target_tiles=_create3x3(self.width - 3, center_y),
            prison_tiles=_create3x3(self.width - 3, self.height - 3),
        )

        # Flags
        self.flags = []
        if self.use_random_flags:
            l_flags: List[Pos] = []
            while len(l_flags) < self.num_flags:
                x = self.rng.randint(2, (self.width // 2) - 1)
                y = self.rng.randint(1, self.height - 3)
                if (x, y) in blocked:
                    continue
                if (x, y) in l_flags:
                    continue
                l_flags.append((x, y))
            r_flags: List[Pos] = []
            while len(r_flags) < self.num_flags:
                x = self.rng.randint(self.width // 2, self.width - 2)
                y = self.rng.randint(1, self.height - 3)
                if (x, y) in blocked:
                    continue
                if (x, y) in r_flags:
                    continue
                r_flags.append((x, y))
        else:
            l_flags = [(1, i + 1) for i in range(self.num_flags)]
            r_flags = [(self.width - 2, i + 1) for i in range(self.num_flags)]

        for pos in l_flags:
            self.flags.append(FlagState(team="L", pos=pos, can_pickup=True))
        for pos in r_flags:
            self.flags.append(FlagState(team="R", pos=pos, can_pickup=True))

        # Players
        self.players = []
        if self.use_random_flags:
            l_px = 1
            r_px = self.width - 2
        else:
            l_px = 2
            r_px = self.width - 3

        for i in range(self.num_players):
            self.players.append(PlayerState(name=f"L{i}", team="L", pos=(l_px, i + 1)))
        for i in range(self.num_players):
            self.players.append(PlayerState(name=f"R{i}", team="R", pos=(r_px, i + 1)))

    def _static(self, team: str) -> TeamStatic:
        if team == "L":
            assert self.l_static is not None
            return self.l_static
        assert self.r_static is not None
        return self.r_static

    def _score(self, team: str) -> int:
        return self.l_score if team == "L" else self.r_score

    def _set_score(self, team: str, score: int) -> None:
        if team == "L":
            self.l_score = score
        else:
            self.r_score = score

    def _middle_line_x(self) -> float:
        return self.width / 2.0

    def status(self, my_team: str) -> Dict[str, object]:
        assert my_team in ("L", "R")
        opp_team = "R" if my_team == "L" else "L"
        my_players = [p.status() for p in self.players if p.team == my_team]
        opp_players = [p.status() for p in self.players if p.team == opp_team]
        my_flags = [f.status() for f in self.flags if f.team == my_team]
        opp_flags = [f.status() for f in self.flags if f.team == opp_team]
        my_static = self._static(my_team)
        opp_static = self._static(opp_team)

        return {
            "action": "status",
            "time": float(self.step_count * self.dt_ms),
            "myteamPlayer": my_players,
            "myteamFlag": my_flags,
            "myteamScore": int(self._score(my_team)),
            "opponentPlayer": opp_players,
            "opponentFlag": opp_flags,
            "opponentScore": int(self._score(opp_team)),
            # Not in frontend status payload, but handy for training/debug:
            "_myteamTarget": [{"x": x, "y": y} for x, y in my_static.target_tiles],
            "_myteamPrison": [{"x": x, "y": y} for x, y in my_static.prison_tiles],
            "_opponentTarget": [{"x": x, "y": y} for x, y in opp_static.target_tiles],
            "_opponentPrison": [{"x": x, "y": y} for x, y in opp_static.prison_tiles],
        }

    def init_payload(self, my_team: str) -> Dict[str, object]:
        assert self.map is not None
        my_static = self._static(my_team)
        opp_team = "R" if my_team == "L" else "L"
        opp_static = self._static(opp_team)
        blocked = list(self.map.blocked)
        # In the real init payload, walls/obstacles are separated. For training, we only need blocked.
        return {
            "action": "init",
            "map": {
                "width": int(self.map.width),
                "height": int(self.map.height),
                "walls": [{"x": x, "y": y} for x, y in blocked],
                "obstacles": [],
            },
            "numPlayers": int(self.num_players),
            "numFlags": int(self.num_flags),
            "myteamName": my_team,
            "myteamPrison": [{"x": x, "y": y} for x, y in my_static.prison_tiles],
            "myteamTarget": [{"x": x, "y": y} for x, y in my_static.target_tiles],
            "opponentPrison": [{"x": x, "y": y} for x, y in opp_static.prison_tiles],
            "opponentTarget": [{"x": x, "y": y} for x, y in opp_static.target_tiles],
        }

    def step(self, l_actions: Mapping[str, Action], r_actions: Mapping[str, Action]) -> None:
        if self.done:
            return
        assert self.map is not None

        # 1) Update prison timers and move players (one tile max)
        blocked = set(self.map.blocked)
        name_to_action: Dict[str, Action] = {}
        name_to_action.update(l_actions)
        name_to_action.update(r_actions)

        for p in self.players:
            if p.in_prison:
                p.in_prison_time_left_ms -= self.dt_ms
                if p.in_prison_time_left_ms <= 0:
                    p.in_prison = False
                    p.in_prison_time_left_ms = 0
                continue
            act = name_to_action.get(p.name, "")
            nxt = _move(p.pos, act)
            if not _in_bounds(self.width, self.height, nxt):
                continue
            if nxt in blocked:
                continue
            p.pos = nxt

        # 2) Prison rescue (any free teammate steps onto own prison zone)
        for team in ("L", "R"):
            prison = set(self._static(team).prison_tiles)
            rescuer_present = any((p.team == team and (not p.in_prison) and p.pos in prison) for p in self.players)
            if rescuer_present:
                for p in self.players:
                    if p.team == team and p.in_prison:
                        p.in_prison = False
                        p.in_prison_time_left_ms = 0

        # 3) Flag pickup
        for p in self.players:
            if p.in_prison or p.has_flag:
                continue
            for f in list(self.flags):
                if f.team == p.team:
                    continue
                if not f.can_pickup:
                    continue
                if f.pos == p.pos:
                    p.has_flag = True
                    self.flags.remove(f)
                    break

        # 4) Capture (if opposing players overlap after movement)
        # In the frontend, capture depends on which half the collision occurs in.
        middle_line = self._middle_line_x()
        l_free = [p for p in self.players if p.team == "L" and not p.in_prison]
        r_free = [p for p in self.players if p.team == "R" and not p.in_prison]

        overlapped: List[Tuple[PlayerState, PlayerState]] = []
        r_by_pos: Dict[Pos, List[PlayerState]] = {}
        for rp in r_free:
            r_by_pos.setdefault(rp.pos, []).append(rp)
        for lp in l_free:
            for rp in r_by_pos.get(lp.pos, []):
                overlapped.append((lp, rp))

        for lp, rp in overlapped:
            if lp.in_prison or rp.in_prison:
                continue
            x, _y = lp.pos
            left_half = float(x) < middle_line
            caught = rp if left_half else lp
            self._send_to_prison(caught)

        # 5) Score (deliver flag on target zone)
        for p in self.players:
            if p.in_prison or not p.has_flag:
                continue
            target = set(self._static(p.team).target_tiles)
            if p.pos in target:
                p.has_flag = False
                self._set_score(p.team, self._score(p.team) + 1)
                # Mimic frontend: spawn an un-pickable enemy flag in the scoring team's target area.
                enemy_team = "R" if p.team == "L" else "L"
                spawn = self._find_free_tile_for_flag(self._static(p.team).target_tiles, team=enemy_team, can_pickup=False)
                if spawn is not None:
                    self.flags.append(FlagState(team=enemy_team, pos=spawn, can_pickup=False))

        # 6) Check termination
        self.step_count += 1
        if self.l_score >= self.num_flags:
            self.done = True
        elif self.r_score >= self.num_flags:
            self.done = True

    def result(self) -> EpisodeResult:
        winner: Optional[str] = None
        if self.l_score > self.r_score:
            winner = "L"
        elif self.r_score > self.l_score:
            winner = "R"
        return EpisodeResult(winner=winner, l_score=self.l_score, r_score=self.r_score, steps=self.step_count)

    def best_step_towards(self, src: Pos, dst: Pos, extra_blocked: Optional[Iterable[Pos]] = None) -> Action:
        assert self.map is not None
        blocked = set(self.map.blocked)
        if extra_blocked:
            blocked |= set(extra_blocked)
        nxt = _bfs_next_step(self.width, self.height, blocked, src, dst)
        return _dir(src, nxt)

    def nearest_enemy_flag(self, team: str, src: Pos) -> Optional[FlagState]:
        enemy = "R" if team == "L" else "L"
        candidates = [f for f in self.flags if f.team == enemy and f.can_pickup]
        if not candidates:
            return None
        return min(candidates, key=lambda f: _manhattan(src, f.pos))

    def nearest_enemy_carrier_in_half(self, team: str, half: str) -> Optional[PlayerState]:
        assert half in ("my", "enemy")
        enemy = "R" if team == "L" else "L"
        middle_line = self._middle_line_x()

        def in_my_half(pos: Pos) -> bool:
            x, _y = pos
            if team == "L":
                return float(x) < middle_line
            return float(x) >= middle_line

        want_my = half == "my"
        carriers = [
            p
            for p in self.players
            if p.team == enemy and (not p.in_prison) and p.has_flag and (in_my_half(p.pos) == want_my)
        ]
        if not carriers:
            return None
        return carriers[0]

    def _send_to_prison(self, player: PlayerState) -> None:
        # Drop carried flag at capture tile (as pickup-able flag for the flag's owner)
        if player.has_flag:
            drop_team = player.team  # player is carrying opponent flag
            flag_team = "R" if drop_team == "L" else "L"
            self.flags.append(FlagState(team=flag_team, pos=player.pos, can_pickup=True))
            player.has_flag = False

        prison_tiles = self._static(player.team).prison_tiles
        spot = self._find_available_prison_tile(prison_tiles, team=player.team)
        if spot is None:
            spot = prison_tiles[0]
        player.pos = spot
        player.in_prison = True
        player.in_prison_time_left_ms = player.in_prison_duration_ms

    def _find_available_prison_tile(self, prison_tiles: Sequence[Pos], team: str) -> Optional[Pos]:
        occupied = {p.pos for p in self.players if p.team == team and p.in_prison}
        for tile in prison_tiles:
            if tile not in occupied:
                return tile
        return None

    def _find_free_tile_for_flag(self, tiles: Sequence[Pos], *, team: str, can_pickup: bool) -> Optional[Pos]:
        occupied = {(f.team, f.pos, f.can_pickup) for f in self.flags}
        for t in tiles:
            if (team, t, can_pickup) not in occupied:
                return t
        return None

