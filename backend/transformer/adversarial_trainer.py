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
        tournament_games: int = 8,
        stage_boundaries: Optional[List[int]] = None
    ):
        """
        Args:
            round_robin_until: 阶段1结束世代（循环赛）
            tournament_games: 锦标赛每个体对战次数
            stage_boundaries: 阶段边界列表 [stage1_end, stage2_end, stage3_end]
                             默认 [50, 100, 150]，可配置为 [160, 300, 600, 800]
        """
        # 参数化阶段边界（支持自定义配置）
        if stage_boundaries is None:
            stage_boundaries = [50, 100, 150]  # 默认边界（向后兼容）

        self.stage1_end = stage_boundaries[0]  # Gen 0-stage1_end: 高强度循环赛
        self.stage2_end = stage_boundaries[1]  # Gen stage1_end+1 - stage2_end: 混合模式
        self.stage3_end = stage_boundaries[2]  # Gen stage2_end+1 - stage3_end: 锦标赛为主
        # Gen stage3_end+1+: 精英对抗

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
    seed: Optional[int] = None,
    use_fixed_flags: bool = False,
    device: Optional[Any] = None
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
        use_fixed_flags: 是否使用固定旗帜位置
        device: torch.device 设备（可选）

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

    if use_fixed_flags:
        # 使用固定旗帜位置（默认垂直线布局）
        sim = CTFSim(width=20, height=20, seed=seed, use_random_flags=False)
    else:
        # 使用随机旗帜位置
        sim = CTFSim(width=20, height=20, seed=seed)
    sim.reset()

    # 2. 创建Geometry
    init_payload = sim.init_payload("L")
    geometry = Geometry.from_init(init_payload)

    # 3. 创建智能体
    agent_l = TransformerAgent(
        model=individual_l.model,
        team="L",
        temperature=temperature,
        device=device
    )

    agent_r = TransformerAgent(
        model=individual_r.model,
        team="R",
        temperature=temperature,
        device=device
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
        show_progress: bool = True,
        fixed_flag_indices: Optional[range] = None,
        device: Optional[Any] = None
    ) -> List[GameResult]:
        """
        并行执行所有对战

        Args:
            matchups: 对战配对列表
            reward_system: 奖励系统
            max_steps: 单场最大步数
            temperature: 采样温度
            show_progress: 是否显示进度条
            device: torch.device 设备（可选）

        Returns:
            所有对战结果列表
        """
        results = []
        total_games = len(matchups)

        if show_progress:
            print(f"开始执行 {total_games} 场对战 (并行度: {self.num_workers})")

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            future_to_matchup = {}
            for idx, (ind_l, ind_r) in enumerate(matchups):
                use_fixed_flags = (fixed_flag_indices is not None and
                                   idx in fixed_flag_indices)

                future = executor.submit(
                    run_single_game,
                    ind_l,
                    ind_r,
                    reward_system,
                    max_steps,
                    temperature,
                    seed=None,
                    use_fixed_flags=use_fixed_flags,
                    device=device
                )
                future_to_matchup[future] = (ind_l, ind_r)

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
        # 使用.get()处理HoF对手（不在种群中）
        ind_l = id_to_individual.get(result.agent_l_id)
        ind_r = id_to_individual.get(result.agent_r_id)

        # 更新L侧个体（如果在种群中）
        if ind_l:
            ind_l.epoch_games_played += 1
            if result.winner == "L":
                ind_l.epoch_wins += 1
            elif result.winner == "R":
                ind_l.epoch_losses += 1
            else:
                ind_l.epoch_draws += 1
            ind_l.epoch_total_reward += result.l_total_reward
            ind_l.flags_captured += result.l_flags_captured

        # 更新R侧个体（如果在种群中，HoF对手会被跳过）
        if ind_r:
            ind_r.epoch_games_played += 1
            if result.winner == "R":
                ind_r.epoch_wins += 1
            elif result.winner == "L":
                ind_r.epoch_losses += 1
            else:
                ind_r.epoch_draws += 1
            ind_r.epoch_total_reward += result.r_total_reward
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
        temperature: float = 1.0,
        hof: Optional['HallOfFame'] = None,
        hof_sample_rate: float = 0.0,
        round_per_game: int = 1,
        fixed_flag_ratio: float = 0.1,
        use_fixed_flags: bool = True,
        device: Optional[Any] = None
    ):
        """
        Args:
            matchup_strategy: 配对策略
            reward_system: 奖励系统
            num_workers: 并行工作进程数
            max_steps: 单场最大步数
            temperature: 采样温度
            hof: Hall of Fame对象（可选）
            hof_sample_rate: HoF对手采样率（0.0-1.0）
            round_per_game: 每个配对的对战轮次（默认1轮）
            fixed_flag_ratio: 固定旗帜游戏比例（0.0-1.0）
            use_fixed_flags: 是否启用固定旗帜游戏
            device: torch.device 设备（可选）
        """
        self.matchup_strategy = matchup_strategy
        self.reward_system = reward_system
        self.executor = ParallelGameExecutor(num_workers)
        self.max_steps = max_steps
        self.temperature = temperature
        self.hof = hof
        self.hof_sample_rate = hof_sample_rate
        self.round_per_game = round_per_game
        self.fixed_flag_ratio = fixed_flag_ratio
        self.use_fixed_flags = use_fixed_flags
        self.device = device

    def _create_agent_from_state(self, state_dict: Dict, team: str) -> 'TransformerAgent':
        """
        从HoF的state_dict创建TransformerAgent

        Args:
            state_dict: 模型的state_dict
            team: 队伍标识 ("L" or "R")

        Returns:
            TransformerAgent实例
        """
        from game_interface import TransformerAgent
        from transformer_model import CTFTransformer, CTFTransformerConfig

        # 创建新模型并加载权重
        config = CTFTransformerConfig()
        model = CTFTransformer(config)
        model.load_state_dict(state_dict)
        model.eval()  # 设置为评估模式

        # 创建智能体
        agent = TransformerAgent(
            model=model,
            team=team,
            temperature=self.temperature
        )

        return agent

    def _create_fixed_flag_matchups(
        self,
        population: List[Individual],
        generation: int
    ) -> List[Tuple[Individual, Individual]]:
        """
        创建固定旗帜对战配对

        策略：
        - 每个个体至少参与一场固定旗帜游戏
        - 如果比例允许，进行循环赛

        Args:
            population: 种群列表
            generation: 当前世代数

        Returns:
            固定旗帜对战配对列表
        """
        if not self.use_fixed_flags or self.fixed_flag_ratio <= 0:
            return []

        # 计算固定旗帜游戏数量
        # 基于当前世代的总对战数估算
        base_matchups = len(population) * (len(population) - 1) // 2  # 循环赛数量
        num_fixed_games = max(1, int(base_matchups * self.fixed_flag_ratio))

        # 创建配对：优先让每个个体都参与
        matchups = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                matchups.append((population[i], population[j]))
                if len(matchups) >= num_fixed_games:
                    break
            if len(matchups) >= num_fixed_games:
                break

        return matchups

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

        # 2. 支持多轮对战：扩展配对列表
        if self.round_per_game > 1:
            expanded_matchups = []
            for ind_l, ind_r in matchups:
                for _ in range(self.round_per_game):
                    expanded_matchups.append((ind_l, ind_r))
            matchups = expanded_matchups
            print(f"  └─ 多轮对战: 每个配对 {self.round_per_game} 轮 → 总计 {len(matchups)} 场对战")

        # 3. HoF对手采样（如果启用）
        hof_matchups_count = 0
        if self.hof and self.hof_sample_rate > 0 and not self.hof.is_empty():
            import random
            from transformer_model import CTFTransformer, CTFTransformerConfig

            modified_matchups = []
            for ind_l, ind_r in matchups:
                # 按概率替换R侧对手为HoF成员
                if random.random() < self.hof_sample_rate:
                    hof_member = self.hof.sample_opponent()
                    if hof_member:
                        # 创建临时Individual对象（只用于对战）
                        hof_config = CTFTransformerConfig()
                        hof_model = CTFTransformer(hof_config)
                        hof_model.load_state_dict(hof_member['model_state_dict'])
                        hof_individual = Individual(
                            id=f"hof_stage{hof_member['stage']}_gen{hof_member['generation']}",
                            model=hof_model,
                            generation=hof_member['generation']
                        )
                        modified_matchups.append((ind_l, hof_individual))
                        hof_matchups_count += 1
                    else:
                        modified_matchups.append((ind_l, ind_r))
                else:
                    modified_matchups.append((ind_l, ind_r))

            matchups = modified_matchups
            if hof_matchups_count > 0:
                print(f"  └─ HoF采样: {hof_matchups_count}/{len(matchups)} 场对战使用HoF对手")

        # 3.5 固定旗帜对战注入
        fixed_flag_matchups = self._create_fixed_flag_matchups(population, generation)
        num_fixed_games = len(fixed_flag_matchups)
        fixed_flag_indices = None

        if num_fixed_games > 0:
            fixed_flag_indices = range(len(matchups), len(matchups) + num_fixed_games)
            matchups.extend(fixed_flag_matchups)
            print(f"  └─ 固定旗帜测试: {num_fixed_games} 场对战 ({num_fixed_games/len(matchups)*100:.1f}%)")

        # 4. 并行执行对战
        results = self.executor.execute_matchups(
            matchups,
            self.reward_system,
            self.max_steps,
            self.temperature,
            fixed_flag_indices=fixed_flag_indices,
            device=self.device
        )

        # 5. 更新适应度
        update_fitness_from_results(population, results)

        # 6. 收集统计信息
        stats = self._collect_statistics(population, results, hof_matchups_count)

        return stats

    def _collect_statistics(
        self,
        population: List[Individual],
        results: List[GameResult],
        hof_matchups_count: int = 0
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
                "worst_fitness": 0,
                "hof_matchups": 0
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
            "worst_fitness": min(ind.fitness for ind in population),
            "hof_matchups": hof_matchups_count
        }
        return stats

