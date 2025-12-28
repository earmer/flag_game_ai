from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GameStateSnapshot:
    """游戏状态快照（用于奖励计算）"""

    # 时间
    timestamp: float

    # 玩家状态
    my_players: List[Dict[str, Any]]    # 我方玩家
    opp_players: List[Dict[str, Any]]   # 敌方玩家

    # 旗帜状态
    my_flags: List[Dict[str, Any]]
    opp_flags: List[Dict[str, Any]]

    # 分数
    my_score: int
    opp_score: int

    # 游戏状态
    game_over: bool
    winner: Optional[str]  # "my_team", "opp_team", "draw"

    # 衍生信息（缓存）
    _my_prisoners: Optional[int] = None
    _opp_prisoners: Optional[int] = None

    def get_my_prisoners_count(self) -> int:
        if self._my_prisoners is None:
            self._my_prisoners = sum(1 for p in self.my_players if p.get('inPrison'))
        return self._my_prisoners

    def get_opp_prisoners_count(self) -> int:
        if self._opp_prisoners is None:
            self._opp_prisoners = sum(1 for p in self.opp_players if p.get('inPrison'))
        return self._opp_prisoners


@dataclass
class RewardInfo:
    """奖励详细信息"""
    total: float                        # 总奖励
    sparse: float                       # 稀疏奖励部分
    dense: float                        # 密集奖励部分
    shaping: float                      # 塑形奖励部分
    breakdown: Dict[str, Any]           # 详细分解


# ============ 稀疏奖励计算器 ============

class SparseRewardCalculator:
    """稀疏奖励计算器 - 只在关键事件发生时给予奖励"""

    def __init__(self):
        # 奖励权重
        self.win_reward = 1000.0
        self.loss_penalty = -500.0
        self.flag_captured_reward = 200.0
        self.flag_lost_penalty = -200.0
        self.teammate_freed_reward = 50.0

    def calculate(
        self,
        current_state: GameStateSnapshot,
        prev_state: GameStateSnapshot
    ) -> Tuple[float, Dict[str, float]]:
        """计算稀疏奖励"""
        reward = 0.0
        breakdown = {}

        # 1. 游戏结束奖励
        if current_state.game_over:
            if current_state.winner == "my_team":
                reward += self.win_reward
                breakdown['win'] = self.win_reward
            elif current_state.winner == "opp_team":
                reward += self.loss_penalty
                breakdown['loss'] = self.loss_penalty
            else:
                breakdown['draw'] = 0.0

        # 2. 捕获旗帜奖励
        flags_captured = current_state.my_score - prev_state.my_score
        if flags_captured > 0:
            flag_reward = flags_captured * self.flag_captured_reward
            reward += flag_reward
            breakdown['flags_captured'] = flag_reward

        # 3. 丢失旗帜惩罚
        flags_lost = current_state.opp_score - prev_state.opp_score
        if flags_lost > 0:
            flag_penalty = flags_lost * self.flag_lost_penalty
            reward += flag_penalty
            breakdown['flags_lost'] = flag_penalty

        # 4. 救出队友奖励
        prisoners_freed = (
            prev_state.get_my_prisoners_count() -
            current_state.get_my_prisoners_count()
        )
        if prisoners_freed > 0:
            freed_reward = prisoners_freed * self.teammate_freed_reward
            reward += freed_reward
            breakdown['teammates_freed'] = freed_reward

        return reward, breakdown


# ============ 密集奖励计算器 ============

class DenseRewardCalculator:
    """密集奖励计算器 - 每步都提供反馈"""

    def __init__(self):
        # 基础行为奖励
        self.move_reward = 0.1
        self.approach_target_reward = 2.0
        self.approach_flag_reward = 1.0
        self.pickup_flag_reward = 50.0
        self.tag_enemy_reward = 10.0
        self.tagged_penalty = -30.0
        self.safe_with_flag_reward = 0.5

        # 团队协作奖励
        self.optimal_distance_min = 3
        self.optimal_distance_max = 8
        self.teamwork_reward = 0.3
        self.clustering_penalty = -0.1

    def _manhattan_distance(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int]
    ) -> int:
        """计算曼哈顿距离"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_player_pos(self, player: Dict[str, Any]) -> Tuple[int, int]:
        """获取玩家位置"""
        return (player['posX'], player['posY'])

    def _find_nearest_flag(
        self,
        player_pos: Tuple[int, int],
        flags: List[Dict[str, Any]]
    ) -> Optional[Tuple[int, int]]:
        """找到最近的可拾取旗帜"""
        pickable_flags = [f for f in flags if f.get('canPickup', False)]
        if not pickable_flags:
            return None

        flag_positions = [self._get_player_pos(f) for f in pickable_flags]
        nearest = min(flag_positions, key=lambda fp: self._manhattan_distance(player_pos, fp))
        return nearest

    def _get_nearest_teammate_distance(
        self,
        player_id: int,
        all_players: List[Dict[str, Any]]
    ) -> float:
        """获取到最近队友的距离"""
        player_pos = self._get_player_pos(all_players[player_id])

        distances = []
        for i, teammate in enumerate(all_players):
            if i != player_id and not teammate.get('inPrison'):
                teammate_pos = self._get_player_pos(teammate)
                dist = self._manhattan_distance(player_pos, teammate_pos)
                distances.append(dist)

        return min(distances) if distances else 999.0

    def _count_enemies_tagged(
        self,
        current_state: GameStateSnapshot,
        prev_state: GameStateSnapshot
    ) -> int:
        """统计新被标记的敌人数量"""
        prev_prisoners = prev_state.get_opp_prisoners_count()
        curr_prisoners = current_state.get_opp_prisoners_count()
        return max(0, curr_prisoners - prev_prisoners)

    def calculate(
        self,
        current_state: GameStateSnapshot,
        prev_state: GameStateSnapshot,
        my_target_pos: Tuple[int, int]
    ) -> Tuple[float, Dict[str, float]]:
        """计算整个团队的密集奖励"""
        total_reward = 0.0
        total_breakdown = {}

        # 计算每个玩家的奖励
        for player_id in range(len(current_state.my_players)):
            player = current_state.my_players[player_id]
            prev_player = prev_state.my_players[player_id]

            player_pos = self._get_player_pos(player)
            prev_pos = self._get_player_pos(prev_player)

            # 1. 移动奖励
            if player_pos != prev_pos:
                total_reward += self.move_reward
                total_breakdown['move'] = total_breakdown.get('move', 0.0) + self.move_reward

            # 2. 接近目标奖励
            if player.get('hasFlag', False):
                dist_to_target = self._manhattan_distance(player_pos, my_target_pos)
                prev_dist = self._manhattan_distance(prev_pos, my_target_pos)

                if dist_to_target < prev_dist:
                    total_reward += self.approach_target_reward
                    total_breakdown['approach_target'] = total_breakdown.get('approach_target', 0.0) + self.approach_target_reward
            else:
                # 无旗时：接近敌方旗帜
                nearest_flag = self._find_nearest_flag(player_pos, current_state.opp_flags)
                if nearest_flag:
                    dist = self._manhattan_distance(player_pos, nearest_flag)
                    prev_nearest = self._find_nearest_flag(prev_pos, prev_state.opp_flags)
                    if prev_nearest:
                        prev_dist = self._manhattan_distance(prev_pos, prev_nearest)
                        if dist < prev_dist:
                            total_reward += self.approach_flag_reward
                            total_breakdown['approach_flag'] = total_breakdown.get('approach_flag', 0.0) + self.approach_flag_reward

            # 3. 拾取旗帜奖励
            if player.get('hasFlag') and not prev_player.get('hasFlag'):
                total_reward += self.pickup_flag_reward
                total_breakdown['pickup_flag'] = total_breakdown.get('pickup_flag', 0.0) + self.pickup_flag_reward

            # 4. 被标记惩罚
            if player.get('inPrison') and not prev_player.get('inPrison'):
                total_reward += self.tagged_penalty
                total_breakdown['tagged'] = total_breakdown.get('tagged', 0.0) + self.tagged_penalty

            # 5. 安全区域奖励
            if player.get('hasFlag') and player.get('isSafe', False):
                total_reward += self.safe_with_flag_reward
                total_breakdown['safe_with_flag'] = total_breakdown.get('safe_with_flag', 0.0) + self.safe_with_flag_reward

            # 6. 团队协作奖励
            teammate_dist = self._get_nearest_teammate_distance(player_id, current_state.my_players)
            if self.optimal_distance_min <= teammate_dist <= self.optimal_distance_max:
                total_reward += self.teamwork_reward
                total_breakdown['teamwork'] = total_breakdown.get('teamwork', 0.0) + self.teamwork_reward
            elif teammate_dist < 2:
                total_reward += self.clustering_penalty
                total_breakdown['clustering'] = total_breakdown.get('clustering', 0.0) + self.clustering_penalty

        # 标记敌人奖励（团队级别）
        enemies_tagged = self._count_enemies_tagged(current_state, prev_state)
        if enemies_tagged > 0:
            tag_reward = enemies_tagged * self.tag_enemy_reward
            total_reward += tag_reward
            total_breakdown['enemies_tagged'] = tag_reward

        return total_reward, total_breakdown


# ============ 课程学习调度器 ============

class CurriculumScheduler:
    """课程学习调度器 - 动态调整稀疏奖励和密集奖励的混合比例"""

    def __init__(self):
        # 阶段划分
        self.stage1_end = 10      # Gen 0-10: 密集为主
        self.stage2_end = 25      # Gen 11-25: 线性过渡
        self.stage3_end = 40      # Gen 26-40: 稀疏为主
        # Gen 41+: 纯稀疏

    def get_weights(self, generation: int) -> Tuple[float, float]:
        """获取当前世代的奖励权重 (dense_weight, sparse_weight)"""
        if generation <= self.stage1_end:
            return 0.8, 0.2
        elif generation <= self.stage2_end:
            progress = (generation - self.stage1_end) / (self.stage2_end - self.stage1_end)
            dense_weight = 0.8 - 0.7 * progress
            sparse_weight = 0.2 + 0.7 * progress
            return dense_weight, sparse_weight
        elif generation <= self.stage3_end:
            return 0.1, 0.9
        else:
            return 0.0, 1.0

    def get_stage(self, generation: int) -> int:
        """获取当前训练阶段"""
        if generation <= self.stage1_end:
            return 1
        elif generation <= self.stage2_end:
            return 2
        elif generation <= self.stage3_end:
            return 3
        else:
            return 4


# ============ 自适应奖励系统 ============

class AdaptiveRewardSystem:
    """自适应奖励系统 - 整合稀疏奖励、密集奖励和课程学习"""

    def __init__(self):
        self.sparse_calculator = SparseRewardCalculator()
        self.dense_calculator = DenseRewardCalculator()
        self.curriculum = CurriculumScheduler()

        self.current_generation = 0
        self.dense_weight = 0.8
        self.sparse_weight = 0.2

    def reset_for_generation(self, generation: int) -> None:
        """为新世代重置奖励系统"""
        self.current_generation = generation
        self.dense_weight, self.sparse_weight = self.curriculum.get_weights(generation)

        stage = self.curriculum.get_stage(generation)
        print(f"Generation {generation} - Stage {stage}: "
              f"Dense={self.dense_weight:.1%}, Sparse={self.sparse_weight:.1%}")

    def calculate_reward(
        self,
        current_state: GameStateSnapshot,
        prev_state: GameStateSnapshot,
        my_target_pos: Tuple[int, int]
    ) -> RewardInfo:
        """计算混合奖励"""
        # 1. 计算稀疏奖励
        sparse_reward, sparse_breakdown = self.sparse_calculator.calculate(
            current_state, prev_state
        )

        # 2. 计算密集奖励
        dense_reward, dense_breakdown = self.dense_calculator.calculate(
            current_state, prev_state, my_target_pos
        )

        # 3. 混合奖励
        total_reward = (
            self.dense_weight * dense_reward +
            self.sparse_weight * sparse_reward
        )

        # 4. 构建详细信息
        breakdown = {
            'sparse': sparse_breakdown,
            'dense': dense_breakdown,
            'weights': {
                'dense_weight': self.dense_weight,
                'sparse_weight': self.sparse_weight
            }
        }

        return RewardInfo(
            total=total_reward,
            sparse=sparse_reward,
            dense=dense_reward,
            shaping=0.0,
            breakdown=breakdown
        )
