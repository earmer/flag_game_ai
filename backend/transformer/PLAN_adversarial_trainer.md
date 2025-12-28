# adversarial_trainer.py 实现计划

## 一、核心职责

对抗训练引擎 - 负责种群内个体的对战配对、并行执行、统计收集和适应度更新。

---

## 二、主要类和函数设计

### 2.1 MatchupStrategy 类

**职责**：定义对战配对策略

```python
from abc import ABC, abstractmethod
from typing import List, Tuple
from population import Individual

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
```

---

### 2.2 RoundRobinStrategy 类

**职责**：循环赛配对（早期世代使用）

```python
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

        Example:
            population = [A, B, C, D]
            matchups = [(A,B), (A,C), (A,D), (B,C), (B,D), (C,D)]
        """
        matchups = []
        n = len(population)

        for i in range(n):
            for j in range(i + 1, n):
                matchups.append((population[i], population[j]))

        return matchups
```

---

### 2.3 TournamentStrategy 类

**职责**：锦标赛配对（中后期世代使用）

```python
import random

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
            # 获取可选对手（排除自己和已配对的）
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
```

---

### 2.4 AdaptiveMatchupStrategy 类

**职责**：自适应配对策略（根据世代自动切换）

```python
class AdaptiveMatchupStrategy(MatchupStrategy):
    """自适应配对策略 - 根据世代自动选择策略"""

    def __init__(
        self,
        round_robin_until: int = 10,
        tournament_games: int = 4
    ):
        """
        Args:
            round_robin_until: 在此世代之前使用循环赛
            tournament_games: 锦标赛每个体对战次数
        """
        self.round_robin_until = round_robin_until
        self.round_robin = RoundRobinStrategy()
        self.tournament = TournamentStrategy(tournament_games)

    def create_matchups(
        self,
        population: List[Individual],
        generation: int
    ) -> List[Tuple[Individual, Individual]]:
        """
        根据世代选择策略

        - Gen 0-10: 循环赛（充分评估）
        - Gen 11+: 锦标赛（加速训练）
        """
        if generation <= self.round_robin_until:
            return self.round_robin.create_matchups(population, generation)
        else:
            return self.tournament.create_matchups(population, generation)
```

---

## 三、对战执行模块

### 3.1 GameResult 数据类

```python
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class GameResult:
    """单场对战结果"""

    # 参赛个体
    agent_l_id: int
    agent_r_id: int

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
```

---

### 3.2 run_single_game 函数

```python
import time
from ctf_ai.sim_env import CTFSim
from lib.tree_features import Geometry
from reward_system import AdaptiveRewardSystem
from game_interface import GameInterface, TransformerAgent

def run_single_game(
    individual_l: Individual,
    individual_r: Individual,
    reward_system: AdaptiveRewardSystem,
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
    # 1. 创建游戏环境
    if seed is None:
        seed = int(time.time() * 1000) % 1000000

    sim = CTFSim(width=20, height=20, seed=seed)
    sim.reset()

    # 2. 创建Geometry
    init_payload = sim.init_payload("L")
    geometry = Geometry(
        my_side_is_left=True,
        my_targets=init_payload["myteamTarget"],
        opp_targets=init_payload["opponentTarget"],
        my_prison=init_payload["myteamPrison"],
        opp_prison=init_payload["opponentPrison"],
        walls=init_payload["walls"]
    )

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
```

---

## 四、并行执行模块

### 4.1 ParallelGameExecutor 类

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List
import multiprocessing as mp

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
        reward_system: AdaptiveRewardSystem,
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
```

---

