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
        my_target_pos: Tuple[int, int],
        generation: int = 0
    ) -> Tuple[float, Dict[str, float], List[Dict[str, float]]]:
        """计算整个团队的密集奖励，包含每个玩家的分解（随训练衰减）"""
        total_reward = 0.0
        total_breakdown = {}
        player_breakdowns = []

        # Per-player detail decays: Gen 0=1.0, Gen 40+=0.0
        detail_strength = max(0.0, 1.0 - generation / 40)

        # 计算每个玩家的奖励
        for player_id in range(len(current_state.my_players)):
            player = current_state.my_players[player_id]
            prev_player = prev_state.my_players[player_id]
            player_breakdown = {}

            player_pos = self._get_player_pos(player)
            prev_pos = self._get_player_pos(prev_player)

            # 1. 移动奖励
            if player_pos != prev_pos:
                total_reward += self.move_reward
                total_breakdown['move'] = total_breakdown.get('move', 0.0) + self.move_reward
                player_breakdown['move'] = self.move_reward * detail_strength

            # 2. 接近目标奖励
            if player.get('hasFlag', False):
                dist_to_target = self._manhattan_distance(player_pos, my_target_pos)
                prev_dist = self._manhattan_distance(prev_pos, my_target_pos)

                if dist_to_target < prev_dist:
                    total_reward += self.approach_target_reward
                    total_breakdown['approach_target'] = total_breakdown.get('approach_target', 0.0) + self.approach_target_reward
                    player_breakdown['approach_target'] = self.approach_target_reward * detail_strength
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
                            player_breakdown['approach_flag'] = self.approach_flag_reward * detail_strength

            # 3. 拾取旗帜奖励
            if player.get('hasFlag') and not prev_player.get('hasFlag'):
                total_reward += self.pickup_flag_reward
                total_breakdown['pickup_flag'] = total_breakdown.get('pickup_flag', 0.0) + self.pickup_flag_reward
                player_breakdown['pickup_flag'] = self.pickup_flag_reward * detail_strength

            # 4. 被标记惩罚
            if player.get('inPrison') and not prev_player.get('inPrison'):
                total_reward += self.tagged_penalty
                total_breakdown['tagged'] = total_breakdown.get('tagged', 0.0) + self.tagged_penalty
                player_breakdown['tagged'] = self.tagged_penalty * detail_strength

            # 5. 安全区域奖励
            if player.get('hasFlag') and player.get('isSafe', False):
                total_reward += self.safe_with_flag_reward
                total_breakdown['safe_with_flag'] = total_breakdown.get('safe_with_flag', 0.0) + self.safe_with_flag_reward
                player_breakdown['safe_with_flag'] = self.safe_with_flag_reward * detail_strength

            # 6. 团队协作奖励
            teammate_dist = self._get_nearest_teammate_distance(player_id, current_state.my_players)
            if self.optimal_distance_min <= teammate_dist <= self.optimal_distance_max:
                total_reward += self.teamwork_reward
                total_breakdown['teamwork'] = total_breakdown.get('teamwork', 0.0) + self.teamwork_reward
                player_breakdown['teamwork'] = self.teamwork_reward * detail_strength
            elif teammate_dist < 2:
                total_reward += self.clustering_penalty
                total_breakdown['clustering'] = total_breakdown.get('clustering', 0.0) + self.clustering_penalty
                player_breakdown['clustering'] = self.clustering_penalty * detail_strength

            # Append player breakdown
            player_breakdowns.append(player_breakdown)

        # 标记敌人奖励（团队级别）
        enemies_tagged = self._count_enemies_tagged(current_state, prev_state)
        if enemies_tagged > 0:
            tag_reward = enemies_tagged * self.tag_enemy_reward
            total_reward += tag_reward
            total_breakdown['enemies_tagged'] = tag_reward

        return total_reward, total_breakdown, player_breakdowns


# ============ 课程学习调度器 ============

class CurriculumScheduler:
    """课程学习调度器 - 动态调整稀疏奖励和密集奖励的混合比例

    新增：基于平局率的自适应切换，只有当平局率<90%时才开始增加稀疏奖励
    """

    def __init__(self):
        # 阶段划分（基于世代）
        self.stage1_end = 50      # Gen 0-50: 密集为主（延长探索期）
        self.stage2_end = 100     # Gen 51-100: 线性过渡
        self.stage3_end = 150     # Gen 101-150: 稀疏为主
        # Gen 151+: 纯稀疏

        # 平局率阈值：只有平局率低于此值才开始增加稀疏奖励
        self.draw_rate_threshold = 0.90  # 90%平局率阈值
        self.current_draw_rate = 1.0     # 当前平局率（初始假设100%）
        self.sparse_enabled = False      # 是否启用稀疏奖励增加

    def update_draw_rate(self, draw_rate: float) -> None:
        """更新当前平局率"""
        self.current_draw_rate = draw_rate
        # 只有当平局率低于阈值时才启用稀疏奖励增加
        if draw_rate < self.draw_rate_threshold:
            if not self.sparse_enabled:
                print(f"[Curriculum] 平局率 {draw_rate:.1%} < {self.draw_rate_threshold:.0%}，启用稀疏奖励增加")
            self.sparse_enabled = True

    def get_weights(self, generation: int) -> Tuple[float, float]:
        """获取当前世代的奖励权重 (dense_weight, sparse_weight)

        如果平局率仍然>=90%，则保持密集奖励为主，不增加稀疏奖励
        """
        # 如果平局率仍然很高，保持密集奖励为主
        if not self.sparse_enabled:
            return 0.8, 0.2

        # 平局率已降低，按世代进行课程学习
        if generation <= self.stage1_end:
            # Stage 1 (Gen 0-50): 密集奖励为主，快速学习
            return 0.8, 0.2
        elif generation <= self.stage2_end:
            # Stage 2 (Gen 51-100): 线性过渡从(0.8, 0.2)到(0.1, 0.9)
            progress = (generation - self.stage1_end) / (self.stage2_end - self.stage1_end)
            dense_weight = 0.8 - 0.7 * progress  # 0.8 → 0.1
            sparse_weight = 0.2 + 0.7 * progress  # 0.2 → 0.9
            return dense_weight, sparse_weight
        elif generation <= self.stage3_end:
            # Stage 3 (Gen 101-150): 稀疏奖励为主，优化目标
            return 0.1, 0.9
        else:
            # Stage 4 (Gen 151+): 纯稀疏奖励，完全自主学习
            return 0.0, 1.0

    def get_stage(self, generation: int) -> int:
        """获取当前训练阶段"""
        if not self.sparse_enabled:
            return 0  # 预热阶段（平局率仍然很高）
        if generation <= self.stage1_end:
            return 1
        elif generation <= self.stage2_end:
            return 2
        elif generation <= self.stage3_end:
            return 3
        else:
            return 4


# ============ 奖励塑形 ============

class RewardShaping:
    """奖励塑形 - 通过领域知识引导早期学习，随训练衰减"""

    def __init__(self, generation: int):
        # 塑形强度衰减: 从Gen 0的0.3线性衰减到Gen 30的0.0
        # Gen 0: 0.3 (强引导)
        # Gen 15: 0.15 (中等引导)
        # Gen 30+: 0.0 (无引导，完全自主)
        self.shaping_strength = max(0.0, 0.3 - generation / 100)

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_player_pos(self, player: Dict[str, Any]) -> Tuple[int, int]:
        return (player['posX'], player['posY'])

    def calculate_shaping_reward(
        self,
        current_state: GameStateSnapshot,
        prev_state: GameStateSnapshot,
        my_target_pos: Tuple[int, int],
        my_prison_pos: Tuple[int, int]
    ) -> Tuple[float, Dict[str, float]]:
        """计算团队塑形奖励 (A+B+D options)"""
        if self.shaping_strength < 0.01:
            return 0.0, {}

        reward = 0.0
        breakdown = {}

        for player_id, player in enumerate(current_state.my_players):
            if player.get('inPrison', False):
                continue

            player_pos = self._get_player_pos(player)
            prev_player = prev_state.my_players[player_id]
            prev_pos = self._get_player_pos(prev_player)

            # === Option A: Distance-Based ===
            # A1: Approach enemy flag when not carrying
            if not player.get('hasFlag', False):
                pickable_flags = [f for f in current_state.opp_flags if f.get('canPickup', False)]
                if pickable_flags:
                    flag_positions = [(f['posX'], f['posY']) for f in pickable_flags]
                    curr_dist = min(self._manhattan_distance(player_pos, fp) for fp in flag_positions)
                    prev_dist = min(self._manhattan_distance(prev_pos, fp) for fp in flag_positions)
                    if curr_dist < prev_dist:
                        reward += 0.5
                        breakdown['approach_flag'] = breakdown.get('approach_flag', 0) + 0.5

            # A2: Approach target when carrying flag
            if player.get('hasFlag', False):
                curr_dist = self._manhattan_distance(player_pos, my_target_pos)
                prev_dist = self._manhattan_distance(prev_pos, my_target_pos)
                if curr_dist < prev_dist:
                    reward += 1.0
                    breakdown['approach_target'] = breakdown.get('approach_target', 0) + 1.0

            # === Option B: Exploration-Based ===
            # B1: Movement reward (penalize staying still)
            if player_pos != prev_pos:
                reward += 0.2
                breakdown['movement'] = breakdown.get('movement', 0) + 0.2

        # B2: Team spread bonus (avoid clustering)
        active_players = [p for p in current_state.my_players if not p.get('inPrison', False)]
        if len(active_players) >= 2:
            positions = [self._get_player_pos(p) for p in active_players]
            min_dist = float('inf')
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    d = self._manhattan_distance(positions[i], positions[j])
                    min_dist = min(min_dist, d)
            if min_dist >= 4:  # Good spread
                reward += 0.3
                breakdown['team_spread'] = 0.3
            elif min_dist <= 1:  # Too clustered
                reward -= 0.2
                breakdown['clustering_penalty'] = -0.2

        # === Option D: Event-Anticipation ===
        for player_id, player in enumerate(current_state.my_players):
            if player.get('inPrison', False):
                continue
            player_pos = self._get_player_pos(player)

            # D1: Flag pickup potential (1-3 steps from flag)
            if not player.get('hasFlag', False):
                pickable = [f for f in current_state.opp_flags if f.get('canPickup', False)]
                if pickable:
                    flag_pos = [(f['posX'], f['posY']) for f in pickable]
                    nearest = min(self._manhattan_distance(player_pos, fp) for fp in flag_pos)
                    if 1 <= nearest <= 3:
                        reward += 0.4
                        breakdown['flag_potential'] = breakdown.get('flag_potential', 0) + 0.4

            # D2: Tag potential (near vulnerable enemies)
            vulnerable = [p for p in current_state.opp_players
                         if not p.get('inPrison', False) and not p.get('isSafe', True)]
            if vulnerable:
                enemy_pos = [self._get_player_pos(p) for p in vulnerable]
                nearest = min(self._manhattan_distance(player_pos, ep) for ep in enemy_pos)
                if 1 <= nearest <= 2:
                    reward += 0.3
                    breakdown['tag_potential'] = breakdown.get('tag_potential', 0) + 0.3

            # D3: Rescue potential (within 2 cells of prison when allies captured)
            if current_state.get_my_prisoners_count() > 0:
                dist_to_prison = self._manhattan_distance(player_pos, my_prison_pos)
                if dist_to_prison <= 2:
                    reward += 0.8
                    breakdown['rescue_potential'] = breakdown.get('rescue_potential', 0) + 0.8

        # Apply decay
        scaled_reward = reward * self.shaping_strength
        scaled_breakdown = {k: v * self.shaping_strength for k, v in breakdown.items()}
        return scaled_reward, scaled_breakdown


# ============ 自适应奖励系统 ============

class AdaptiveRewardSystem:
    """自适应奖励系统 - 整合稀疏奖励、密集奖励和课程学习"""

    def __init__(self):
        self.sparse_calculator = SparseRewardCalculator()
        self.dense_calculator = DenseRewardCalculator()
        self.curriculum = CurriculumScheduler()
        self.shaping: Optional[RewardShaping] = None

        self.current_generation = 0
        self.dense_weight = 0.8
        self.sparse_weight = 0.2

    def reset_for_generation(self, generation: int) -> None:
        """为新世代重置奖励系统"""
        self.current_generation = generation
        self.dense_weight, self.sparse_weight = self.curriculum.get_weights(generation)
        self.shaping = RewardShaping(generation)

        stage = self.curriculum.get_stage(generation)
        stage_name = "预热" if stage == 0 else f"Stage {stage}"
        print(f"Generation {generation} - {stage_name}: "
              f"Dense={self.dense_weight:.1%}, Sparse={self.sparse_weight:.1%}, "
              f"Shaping={self.shaping.shaping_strength:.2f}, "
              f"SparseEnabled={self.curriculum.sparse_enabled}")

    def update_draw_rate(self, draw_rate: float) -> None:
        """更新平局率，用于自适应奖励切换"""
        self.curriculum.update_draw_rate(draw_rate)

    def calculate_reward(
        self,
        current_state: GameStateSnapshot,
        prev_state: GameStateSnapshot,
        my_target_pos: Tuple[int, int],
        my_prison_pos: Tuple[int, int] = (0, 0)
    ) -> RewardInfo:
        """计算混合奖励"""
        # 1. 计算稀疏奖励
        sparse_reward, sparse_breakdown = self.sparse_calculator.calculate(
            current_state, prev_state
        )

        # 2. 计算密集奖励
        dense_reward, dense_breakdown, player_breakdowns = self.dense_calculator.calculate(
            current_state, prev_state, my_target_pos, self.current_generation
        )

        # 3. 计算塑形奖励
        shaping_reward = 0.0
        shaping_breakdown = {}
        if self.shaping is not None:
            shaping_reward, shaping_breakdown = self.shaping.calculate_shaping_reward(
                current_state, prev_state, my_target_pos, my_prison_pos
            )

        # 4. 混合奖励
        total_reward = (
            self.dense_weight * dense_reward +
            self.sparse_weight * sparse_reward +
            shaping_reward
        )

        # 5. 构建详细信息
        breakdown = {
            'sparse': sparse_breakdown,
            'dense': dense_breakdown,
            'shaping': shaping_breakdown,
            'player_breakdowns': player_breakdowns,
            'weights': {
                'dense_weight': self.dense_weight,
                'sparse_weight': self.sparse_weight
            }
        }

        return RewardInfo(
            total=total_reward,
            sparse=sparse_reward,
            dense=dense_reward,
            shaping=shaping_reward,
            breakdown=breakdown
        )
