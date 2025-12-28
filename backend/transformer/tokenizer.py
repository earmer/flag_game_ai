from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple
from collections import deque

try:
    from backend.lib.tree_features import Geometry
    from backend.ctf_ai.encoding import encode_status_for_team, to_torch_batch, EncodedBatch
except ImportError:
    from lib.tree_features import Geometry
    from ctf_ai.encoding import encode_status_for_team, to_torch_batch, EncodedBatch


@dataclass
class GameState:
    """完整的游戏状态快照"""
    timestamp: float  # 游戏时间(ms)

    # 地图静态信息
    width: int
    height: int
    walls: set[Tuple[int, int]]
    my_team: str  # "L" or "R"
    my_prisons: List[Tuple[int, int]]
    my_targets: List[Tuple[int, int]]
    opp_prisons: List[Tuple[int, int]]
    opp_targets: List[Tuple[int, int]]

    # 动态实体
    my_players: List[Dict[str, Any]]
    opp_players: List[Dict[str, Any]]
    my_flags: List[Dict[str, Any]]
    opp_flags: List[Dict[str, Any]]

    # 分数
    my_score: int
    opp_score: int

    # 缓存的位置信息
    _my_player_positions: Optional[Dict[str, Tuple[int, int]]] = None
    _opp_player_positions: Optional[Dict[str, Tuple[int, int]]] = None

    def get_my_player_pos(self, name: str) -> Optional[Tuple[int, int]]:
        """获取我方玩家位置"""
        if self._my_player_positions is None:
            self._my_player_positions = {
                p["name"]: (p["posX"], p["posY"])
                for p in self.my_players
            }
        return self._my_player_positions.get(name)

    def get_pickable_opp_flags(self) -> List[Dict[str, Any]]:
        """获取可拾取的敌方旗帜"""
        return [f for f in self.opp_flags if f.get("canPickup", False)]

    def get_free_my_players(self) -> List[Dict[str, Any]]:
        """获取未被囚禁的我方玩家"""
        return [p for p in self.my_players if not p.get("inPrison", False)]


class StateManager:
    """游戏状态管理器 - 连接WebSocket和AI决策"""

    def __init__(self, history_size: int = 10):
        self.current_state: Optional[GameState] = None
        self.history: deque[GameState] = deque(maxlen=history_size)
        self.initialized: bool = False

        # 静态地图信息(init时设置)
        self.width: int = 0
        self.height: int = 0
        self.walls: set[Tuple[int, int]] = set()
        self.my_team: str = ""
        self.my_prisons: List[Tuple[int, int]] = []
        self.my_targets: List[Tuple[int, int]] = []
        self.opp_prisons: List[Tuple[int, int]] = []
        self.opp_targets: List[Tuple[int, int]] = []

    def handle_init(self, init_req: Dict[str, Any]) -> None:
        """处理初始化请求"""
        map_data = init_req["map"]
        self.width = map_data["width"]
        self.height = map_data["height"]

        # 合并walls和obstacles
        self.walls = {
            (w["x"], w["y"])
            for w in (map_data.get("walls", []) + map_data.get("obstacles", []))
        }

        self.my_team = init_req.get("myteamName", "")
        self.my_prisons = [(p["x"], p["y"]) for p in init_req.get("myteamPrison", [])]
        self.my_targets = [(t["x"], t["y"]) for t in init_req.get("myteamTarget", [])]
        self.opp_prisons = [(p["x"], p["y"]) for p in init_req.get("opponentPrison", [])]
        self.opp_targets = [(t["x"], t["y"]) for t in init_req.get("opponentTarget", [])]

        self.initialized = True

    def handle_status(self, status_req: Dict[str, Any]) -> GameState:
        """处理状态更新请求,返回新的GameState"""
        if not self.initialized:
            raise RuntimeError("Must call handle_init() first")

        state = GameState(
            timestamp=status_req.get("time", 0.0),
            width=self.width,
            height=self.height,
            walls=self.walls,
            my_team=self.my_team,
            my_prisons=self.my_prisons,
            my_targets=self.my_targets,
            opp_prisons=self.opp_prisons,
            opp_targets=self.opp_targets,
            my_players=status_req.get("myteamPlayer", []),
            opp_players=status_req.get("opponentPlayer", []),
            my_flags=status_req.get("myteamFlag", []),
            opp_flags=status_req.get("opponentFlag", []),
            my_score=status_req.get("myteamScore", 0),
            opp_score=status_req.get("opponentScore", 0),
        )

        # 保存到历史
        if self.current_state is not None:
            self.history.append(self.current_state)
        self.current_state = state

        return state

    def get_state_for_encoding(self) -> Dict[str, Any]:
        """
        将当前GameState转换为encoding.py需要的格式
        这是关键的转换接口!
        """
        if self.current_state is None:
            raise RuntimeError("No current state available")

        state = self.current_state

        # 构造encoding.encode_status_for_team()需要的字典格式
        return {
            "time": state.timestamp,
            "myteamPlayer": state.my_players,
            "opponentPlayer": state.opp_players,
            "myteamFlag": state.my_flags,
            "opponentFlag": state.opp_flags,
            "myteamScore": state.my_score,
            "opponentScore": state.opp_score,
        }


class TokenConverter:
    """Token转换器 - 封装encoding.py的调用"""

    def __init__(self, max_tokens: int = 32):
        self.max_tokens = max_tokens

    def state_to_tokens(
        self,
        state_dict: Dict[str, Any],
        geometry: Geometry
    ) -> Tuple[List[int], List[List[float]], List[bool], Tuple[int, ...]]:
        """
        将状态字典转换为tokens

        这是对encoding.encode_status_for_team()的封装
        """
        return encode_status_for_team(
            status_req=state_dict,
            geometry=geometry,
            max_tokens=self.max_tokens
        )

    def tokens_to_batch(
        self,
        encoded_list: List[Tuple[List[int], List[List[float]], List[bool], Tuple[int, ...]]]
    ) -> EncodedBatch:
        """将多个编码结果转换为batch"""
        return to_torch_batch(encoded_list)
