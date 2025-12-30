"""
基准测试模块 - 与基础固定规则AI对战评估

提供基准测试功能，用于评估进化AI相对于基础AI的性能。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from game_interface import PolicyAgent


# ============ 基础启发式AI ============

class HeuristicAgent(PolicyAgent):
    """
    基础启发式AI（基于pick_flag_ai.py的逻辑）

    策略：
    1. 为每个没有旗帜的玩家随机分配一个敌方旗帜
    2. 持有旗帜时前往己方目标点
    3. 没有旗帜时前往分配的敌方旗帜
    4. 在敌方领土时避开敌人
    """

    def __init__(self, team: str):
        super().__init__(team)
        self.player_to_flag_assignments: Dict[str, Tuple[int, int]] = {}

    def reset(self) -> None:
        """重置智能体状态"""
        self.player_to_flag_assignments = {}

    def select_actions(
        self,
        status_dict: Dict[str, Any],
        geometry: Any
    ) -> Dict[str, str]:
        """根据当前状态选择动作"""
        actions = {}

        my_players = status_dict.get("myteamPlayer", [])
        opp_players = status_dict.get("opponentPlayer", [])
        opp_flags = status_dict.get("opponentFlag", [])
        my_target = status_dict.get("myteamTarget", [(0, 0)])[0]

        # 获取可拾取的敌方旗帜
        enemy_flags = [
            f for f in opp_flags
            if f.get("canPickup", True)
        ]

        # 确定己方是左侧还是右侧
        my_side_is_left = self._is_on_left(my_target, geometry)

        # 清理无效的旗帜分配
        active_player_names = {
            p["name"] for p in my_players
            if not p.get("hasFlag", False) and not p.get("inPrison", False)
        }
        self.player_to_flag_assignments = {
            name: pos for name, pos in self.player_to_flag_assignments.items()
            if name in active_player_names
        }

        # 为没有旗帜的玩家分配旗帜
        if enemy_flags:
            for p in my_players:
                if not p.get("hasFlag", False) and not p.get("inPrison", False):
                    if p["name"] not in self.player_to_flag_assignments:
                        f = random.choice(enemy_flags)
                        self.player_to_flag_assignments[p["name"]] = (f["posX"], f["posY"])

        # 为每个玩家计算动作
        for p in my_players:
            if p.get("inPrison", False):
                actions[p["name"]] = ""
                continue

            curr_pos = (p["posX"], p["posY"])

            # 确定目标
            if p.get("hasFlag", False):
                dest = my_target
            elif p["name"] in self.player_to_flag_assignments:
                dest = self.player_to_flag_assignments[p["name"]]
            else:
                actions[p["name"]] = ""
                continue

            # 确定是否在安全区
            is_safe = self._is_on_left(curr_pos, geometry) == my_side_is_left

            # 计算移动方向（简单版本：直接朝目标移动）
            move = self._get_direction_to(curr_pos, dest, opp_players if not is_safe else [])
            actions[p["name"]] = move

        return actions

    def _is_on_left(self, pos: Tuple[int, int], geometry: Any) -> bool:
        """判断位置是否在左侧"""
        if geometry is None:
            return pos[0] < 10  # 默认假设地图宽度20
        mid_x = geometry.width // 2
        return pos[0] < mid_x

    def _get_direction_to(
        self,
        curr: Tuple[int, int],
        dest: Tuple[int, int],
        blockers: List[Dict[str, Any]]
    ) -> str:
        """计算从当前位置到目标的移动方向（简单版本）"""
        dx = dest[0] - curr[0]
        dy = dest[1] - curr[1]

        # 获取阻挡位置
        blocker_positions = {(b["posX"], b["posY"]) for b in blockers}

        # 优先移动方向
        moves = []
        if abs(dx) >= abs(dy):
            if dx > 0:
                moves = ["right", "up", "down", "left"]
            elif dx < 0:
                moves = ["left", "up", "down", "right"]
            else:
                moves = ["up", "down", "left", "right"]
        else:
            if dy > 0:
                moves = ["down", "up", "left", "right"]
            elif dy < 0:
                moves = ["up", "down", "left", "right"]
            else:
                moves = ["up", "down", "left", "right"]

        # 检查每个方向是否被阻挡
        direction_deltas = {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0)
        }

        for move in moves:
            delta = direction_deltas[move]
            new_pos = (curr[0] + delta[0], curr[1] + delta[1])
            if new_pos not in blocker_positions:
                return move

        # 所有方向都被阻挡，原地不动
        return ""


# ============ 基准测试结果 ============

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    generation: int
    win_rate: float           # 胜率
    draw_rate: float          # 平局率
    loss_rate: float          # 负率
    avg_score_diff: float     # 平均得分差
    avg_steps: float          # 平均游戏步数
    left_wins: int            # 作为左侧的胜场
    right_wins: int           # 作为右侧的胜场
    total_games: int          # 总对局数


# ============ 基准测试函数 ============

def run_benchmark(
    transformer_agent: PolicyAgent,
    num_games: int = 100,
    max_steps: int = 1000,
    seed: Optional[int] = None
) -> BenchmarkResult:
    """
    运行基准测试：Transformer AI vs 基础启发式AI

    Args:
        transformer_agent: Transformer智能体
        num_games: 总对局数（一半作为左侧，一半作为右侧）
        max_steps: 单局最大步数
        seed: 随机种子

    Returns:
        BenchmarkResult对象
    """
    from sim_env import CTFGameSimulator, Geometry

    if seed is not None:
        random.seed(seed)

    # 创建基础AI
    heuristic_l = HeuristicAgent("L")
    heuristic_r = HeuristicAgent("R")

    # 统计
    left_wins = 0    # Transformer作为左侧的胜场
    right_wins = 0   # Transformer作为右侧的胜场
    draws = 0
    total_score_diff = 0.0
    total_steps = 0

    games_per_side = num_games // 2

    # Transformer作为左侧
    for _ in range(games_per_side):
        result = _run_single_game(
            l_agent=transformer_agent,
            r_agent=heuristic_r,
            max_steps=max_steps
        )
        if result["winner"] == "L":
            left_wins += 1
        elif result["winner"] is None:
            draws += 1
        total_score_diff += result["l_score"] - result["r_score"]
        total_steps += result["steps"]

    # Transformer作为右侧
    for _ in range(games_per_side):
        result = _run_single_game(
            l_agent=heuristic_l,
            r_agent=transformer_agent,
            max_steps=max_steps
        )
        if result["winner"] == "R":
            right_wins += 1
        elif result["winner"] is None:
            draws += 1
        total_score_diff += result["r_score"] - result["l_score"]
        total_steps += result["steps"]

    total_games = games_per_side * 2
    wins = left_wins + right_wins
    losses = total_games - wins - draws

    return BenchmarkResult(
        generation=0,  # 由调用者设置
        win_rate=wins / total_games if total_games > 0 else 0.0,
        draw_rate=draws / total_games if total_games > 0 else 0.0,
        loss_rate=losses / total_games if total_games > 0 else 0.0,
        avg_score_diff=total_score_diff / total_games if total_games > 0 else 0.0,
        avg_steps=total_steps / total_games if total_games > 0 else 0.0,
        left_wins=left_wins,
        right_wins=right_wins,
        total_games=total_games
    )


def _run_single_game(
    l_agent: PolicyAgent,
    r_agent: PolicyAgent,
    max_steps: int = 1000
) -> Dict[str, Any]:
    """运行单局游戏"""
    from sim_env import CTFGameSimulator, Geometry

    # 创建游戏模拟器
    geometry = Geometry()
    sim = CTFGameSimulator(geometry)

    # 重置智能体
    l_agent.reset()
    r_agent.reset()

    # 运行游戏
    for step in range(max_steps):
        status_l = sim.status("L")
        status_r = sim.status("R")

        # 检查游戏是否结束
        if sim.is_game_over():
            break

        # 获取动作
        l_actions = l_agent.select_actions(status_l, geometry)
        r_actions = r_agent.select_actions(status_r, geometry)

        # 执行动作
        sim.step(l_actions, r_actions)

    # 获取最终状态
    final_status = sim.status("L")
    l_score = final_status.get("myteamScore", 0)
    r_score = final_status.get("opponentScore", 0)

    # 判断胜负
    if l_score > r_score:
        winner = "L"
    elif r_score > l_score:
        winner = "R"
    else:
        winner = None

    return {
        "winner": winner,
        "l_score": l_score,
        "r_score": r_score,
        "steps": step + 1
    }
