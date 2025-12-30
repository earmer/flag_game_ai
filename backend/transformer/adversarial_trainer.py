"""
adversarial_trainer.py

对抗训练引擎 - 负责种群内个体的对战配对、并行执行、统计收集和适应度更新
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from population import Individual


# ============================================================
# 配对策略模块
# ============================================================

class MatchupStrategy(ABC):
    """对战配对策略抽象基类"""

    @abstractmethod
    def create_matchups(
        self,
        population: List[Individual],
        generation: int
    ) -> List[Tuple[Individual, Individual]]:
        """
        创建对战配对列表

        Args:
            population: 种群列表
            generation: 当前世代数

        Returns:
            配对列表 [(ind1, ind2), ...]
        """
        pass


class RoundRobinStrategy(MatchupStrategy):
    """循环赛配对策略 - 每个个体与所有其他个体对战一次"""

    def create_matchups(
        self,
        population: List[Individual],
        generation: int
    ) -> List[Tuple[Individual, Individual]]:
        """
        生成循环赛配对

        对于N个个体，生成 N*(N-1)/2 场比赛
        """
        matchups = []
        n = len(population)

        for i in range(n):
            for j in range(i + 1, n):
                matchups.append((population[i], population[j]))

        return matchups


class TournamentStrategy(MatchupStrategy):
    """锦标赛配对策略 - 每个个体随机对战K次"""

    def __init__(self, games_per_individual: int = 4):
        """
        Args:
            games_per_individual: 每个个体参与的对战次数
        """
        self.games_per_individual = games_per_individual

    def create_matchups(
        self,
        population: List[Individual],
        generation: int
    ) -> List[Tuple[Individual, Individual]]:
        """
        生成锦标赛配对

        策略：
        1. 为每个个体随机选择K个对手
        2. 避免重复配对（A vs B 和 B vs A 只保留一个）
        """
        matchups = []
        paired = set()

        for individual in population:
            # 获取可选对手（排除自己）
            available_opponents = [
                opp for opp in population
                if opp.id != individual.id
            ]

            # 随机选择对手
            num_games = min(self.games_per_individual, len(available_opponents))
            opponents = random.sample(available_opponents, num_games)

            for opponent in opponents:
                # 避免重复配对
                pair_key = tuple(sorted([individual.id, opponent.id]))
                if pair_key not in paired:
                    matchups.append((individual, opponent))
                    paired.add(pair_key)

        return matchups


class AdaptiveMatchupStrategy(MatchupStrategy):
    """自适应配对策略 - 根据四阶段训练流程自动选择策略"""

    def __init__(
        self,
        round_robin_until: int = 20,
        tournament_games: int = 8
    ):
        """
        Args:
            round_robin_until: 阶段1结束世代（循环赛）
            tournament_games: 锦标赛每个体对战次数
        """
        self.stage1_end = 50      # Gen 0-50: 高强度循环赛
        self.stage2_end = 100     # Gen 51-100: 混合模式
        self.stage3_end = 150     # Gen 101-150: 锦标赛为主
        # Gen 151+: 精英对抗

        self.round_robin = RoundRobinStrategy()
        self.tournament = TournamentStrategy(tournament_games)

    def create_matchups(
        self,
        population: List[Individual],
        generation: int
    ) -> List[Tuple[Individual, Individual]]:
        """
        根据四阶段训练流程选择配对策略

        - Stage 1 (Gen 0-50): 循环赛（充分评估）
        - Stage 2 (Gen 51-100): 混合模式（50%循环赛+锦标赛）
        - Stage 3 (Gen 101-150): 锦标赛为主
        - Stage 4 (Gen 151+): 精英对抗（前4名循环赛）
        """
        if generation <= self.stage1_end:
            # Stage 1: 完整循环赛
            return self.round_robin.create_matchups(population, generation)
        elif generation <= self.stage2_end:
            # Stage 2: 混合模式
            return self._mixed_matchups(population, generation)
        elif generation <= self.stage3_end:
            # Stage 3: 锦标赛
            return self.tournament.create_matchups(population, generation)
        else:
            # Stage 4: 精英对抗
            return self._elite_matchups(population, generation)

    def _mixed_matchups(
        self,
        population: List[Individual],
        generation: int
    ) -> List[Tuple[Individual, Individual]]:
        """混合模式：循环赛+锦标赛"""
        # 前4名进行循环赛
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        top_half = sorted_pop[:len(sorted_pop)//2]
        bottom_half = sorted_pop[len(sorted_pop)//2:]

        matchups = []
        # 前半部分循环赛
        for i in range(len(top_half)):
            for j in range(i + 1, len(top_half)):
                matchups.append((top_half[i], top_half[j]))

        # 后半部分锦标赛
        matchups.extend(self.tournament.create_matchups(bottom_half, generation))

        return matchups

    def _elite_matchups(
        self,
        population: List[Individual],
        generation: int
    ) -> List[Tuple[Individual, Individual]]:
        """精英对抗：前4名高强度循环赛"""
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        elites = sorted_pop[:4]
        challengers = sorted_pop[4:]

        matchups = []
        # 精英之间多轮循环赛
        for _ in range(3):  # 3轮
            for i in range(len(elites)):
                for j in range(i + 1, len(elites)):
                    matchups.append((elites[i], elites[j]))

        # 挑战者与精英对战
        for challenger in challengers:
            for elite in elites[:2]:  # 与前2名精英对战
                matchups.append((challenger, elite))

        return matchups


# ============================================================
# 对战执行模块
# ============================================================

@dataclass
class GameResult:
    """单场对战结果"""

    # 参赛个体
    agent_l_id: str
    agent_r_id: str

    # 胜负信息
    winner: Optional[str]  # "L" / "R" / None (平局)
    l_score: int
    r_score: int

    # 游戏统计
    steps: int
    duration_ms: float

    # 奖励信息
    l_total_reward: float
    r_total_reward: float
    l_reward_breakdown: Dict[str, float]
    r_reward_breakdown: Dict[str, float]

    # 详细统计
    l_flags_captured: int
    r_flags_captured: int
    l_enemies_tagged: int
    r_enemies_tagged: int
    l_avg_survival_rate: float
    r_avg_survival_rate: float


def run_single_game(
    individual_l: Individual,
    individual_r: Individual,
    reward_system: 'AdaptiveRewardSystem',
    max_steps: int = 1000,
    temperature: float = 1.0,
    seed: Optional[int] = None
) -> GameResult:
    """
    执行单场对战

    Args:
        individual_l: L队个体
        individual_r: R队个体
        reward_system: 奖励系统实例
        max_steps: 最大步数
        temperature: 动作采样温度
        seed: 随机种子（可选）

    Returns:
        GameResult对象
    """
    # 导入依赖模块
    from _import_bootstrap import get_geometry
    from sim_env import CTFSim
    from game_interface import GameInterface, TransformerAgent

    Geometry = get_geometry()

    # 1. 创建游戏环境
    if seed is None:
        seed = int(time.time() * 1000) % 1000000

    sim = CTFSim(width=20, height=20, seed=seed)
    sim.reset()

    # 2. 创建Geometry
    init_payload = sim.init_payload("L")
    geometry = Geometry.from_init(init_payload)

    # 3. 创建智能体
    agent_l = TransformerAgent(
        model=individual_l.model,
        team="L",
        temperature=temperature
    )

    agent_r = TransformerAgent(
        model=individual_r.model,
        team="R",
        temperature=temperature
    )

    # 4. 运行对战
    interface = GameInterface(sim, geometry, reward_system, max_steps=max_steps)
    episode_result = interface.run_episode(
        agent_l,
        agent_r,
        record_trajectory=False  # 训练时不记录轨迹
    )

    # 5. 转换为GameResult
    game_result = GameResult(
        agent_l_id=individual_l.id,
        agent_r_id=individual_r.id,
        winner=episode_result.winner,
        l_score=episode_result.l_score,
        r_score=episode_result.r_score,
        steps=episode_result.steps,
        duration_ms=episode_result.duration_ms,
        l_total_reward=episode_result.l_total_reward,
        r_total_reward=episode_result.r_total_reward,
        l_reward_breakdown=episode_result.l_reward_breakdown,
        r_reward_breakdown=episode_result.r_reward_breakdown,
        l_flags_captured=episode_result.l_flags_captured,
        r_flags_captured=episode_result.r_flags_captured,
        l_enemies_tagged=episode_result.l_enemies_tagged,
        r_enemies_tagged=episode_result.r_enemies_tagged,
        l_avg_survival_rate=episode_result.l_avg_survival_rate,
        r_avg_survival_rate=episode_result.r_avg_survival_rate
    )

    return game_result


# ============================================================
# 并行执行模块
# ============================================================

class ParallelGameExecutor:
    """并行游戏执行器"""

    def __init__(self, num_workers: int = 4):
        """
        Args:
            num_workers: 并行工作进程数（默认4）
        """
        self.num_workers = num_workers

    def execute_matchups(
        self,
        matchups: List[Tuple[Individual, Individual]],
        reward_system: 'AdaptiveRewardSystem',
        max_steps: int = 1000,
        temperature: float = 1.0,
        show_progress: bool = True
    ) -> List[GameResult]:
        """
        并行执行所有对战

        Args:
            matchups: 对战配对列表
            reward_system: 奖励系统
            max_steps: 单场最大步数
            temperature: 采样温度
            show_progress: 是否显示进度条

        Returns:
            所有对战结果列表
        """
        results = []
        total_games = len(matchups)

        if show_progress:
            print(f"开始执行 {total_games} 场对战 (并行度: {self.num_workers})")

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            future_to_matchup = {
                executor.submit(
                    run_single_game,
                    ind_l,
                    ind_r,
                    reward_system,
                    max_steps,
                    temperature,
                    seed=None
                ): (ind_l, ind_r)
                for ind_l, ind_r in matchups
            }

            # 收集结果
            completed = 0
            for future in as_completed(future_to_matchup):
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1

                    if show_progress:
                        print(f"进度: {completed}/{total_games} "
                              f"({100*completed/total_games:.1f}%)", end='\r')

                except Exception as e:
                    ind_l, ind_r = future_to_matchup[future]
                    print(f"\n警告: 对战失败 (ID {ind_l.id} vs {ind_r.id}): {e}")

        if show_progress:
            print()  # 换行

        return results


# ============================================================
# 适应度更新模块
# ============================================================

def update_fitness_from_results(
    population: List[Individual],
    results: List[GameResult]
) -> None:
    """
    根据对战结果更新种群适应度

    Args:
        population: 种群列表
        results: 所有对战结果

    Note:
        直接修改Individual对象的fitness属性
    """
    # 1. 创建ID到个体的映射
    id_to_individual = {ind.id: ind for ind in population}

    # 2. 重置临时统计
    for ind in population:
        ind.epoch_wins = 0
        ind.epoch_losses = 0
        ind.epoch_draws = 0
        ind.epoch_total_reward = 0.0
        ind.epoch_games_played = 0

    # 3. 累积统计
    for result in results:
        ind_l = id_to_individual[result.agent_l_id]
        ind_r = id_to_individual[result.agent_r_id]

        # 更新对战次数
        ind_l.epoch_games_played += 1
        ind_r.epoch_games_played += 1

        # 更新胜负
        if result.winner == "L":
            ind_l.epoch_wins += 1
            ind_r.epoch_losses += 1
        elif result.winner == "R":
            ind_r.epoch_wins += 1
            ind_l.epoch_losses += 1
        else:
            ind_l.epoch_draws += 1
            ind_r.epoch_draws += 1

        # 累积奖励
        ind_l.epoch_total_reward += result.l_total_reward
        ind_r.epoch_total_reward += result.r_total_reward

        # 累积其他统计
        ind_l.flags_captured += result.l_flags_captured
        ind_r.flags_captured += result.r_flags_captured

    # 4. 计算适应度
    for ind in population:
        if ind.epoch_games_played == 0:
            continue

        # 胜率奖励
        win_rate = ind.epoch_wins / ind.epoch_games_played
        fitness = win_rate * 100.0

        # 平均奖励
        avg_reward = ind.epoch_total_reward / ind.epoch_games_played
        fitness += avg_reward * 0.1

        # 更新个体适应度
        ind.fitness = fitness
        ind.wins += ind.epoch_wins
        ind.losses += ind.epoch_losses


# ============================================================
# 主训练接口
# ============================================================

class AdversarialTrainer:
    """对抗训练器 - 统一管理对战流程"""

    def __init__(
        self,
        matchup_strategy: MatchupStrategy,
        reward_system: 'AdaptiveRewardSystem',
        num_workers: int = 4,
        max_steps: int = 1000,
        temperature: float = 1.0
    ):
        """
        Args:
            matchup_strategy: 配对策略
            reward_system: 奖励系统
            num_workers: 并行工作进程数
            max_steps: 单场最大步数
            temperature: 采样温度
        """
        self.matchup_strategy = matchup_strategy
        self.reward_system = reward_system
        self.executor = ParallelGameExecutor(num_workers)
        self.max_steps = max_steps
        self.temperature = temperature

    def train_epoch(
        self,
        population: List[Individual],
        generation: int
    ) -> Dict[str, Any]:
        """
        执行一轮对抗训练

        Args:
            population: 种群列表
            generation: 当前世代数

        Returns:
            训练统计信息字典
        """
        # 1. 创建配对
        matchups = self.matchup_strategy.create_matchups(population, generation)
        print(f"世代 {generation}: 创建 {len(matchups)} 场对战")

        # 2. 并行执行对战
        results = self.executor.execute_matchups(
            matchups,
            self.reward_system,
            self.max_steps,
            self.temperature
        )

        # 3. 更新适应度
        update_fitness_from_results(population, results)

        # 4. 收集统计信息
        stats = self._collect_statistics(population, results)

        return stats

    def _collect_statistics(
        self,
        population: List[Individual],
        results: List[GameResult]
    ) -> Dict[str, Any]:
        """收集训练统计信息"""
        if len(results) == 0:
            return {
                "num_games": 0,
                "avg_steps": 0,
                "avg_duration_ms": 0,
                "l_wins": 0,
                "r_wins": 0,
                "draws": 0,
                "best_fitness": 0,
                "avg_fitness": 0,
                "worst_fitness": 0
            }

        stats = {
            "num_games": len(results),
            "avg_steps": sum(r.steps for r in results) / len(results),
            "avg_duration_ms": sum(r.duration_ms for r in results) / len(results),
            "l_wins": sum(1 for r in results if r.winner == "L"),
            "r_wins": sum(1 for r in results if r.winner == "R"),
            "draws": sum(1 for r in results if r.winner is None),
            "best_fitness": max(ind.fitness for ind in population),
            "avg_fitness": sum(ind.fitness for ind in population) / len(population),
            "worst_fitness": min(ind.fitness for ind in population)
        }
        return stats

