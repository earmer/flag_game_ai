# adversarial_trainer.py 实现计划 (续)

## 五、适应度更新模块

### 5.1 update_fitness_from_results 函数

```python
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
```

---

## 六、主训练接口

### 6.1 AdversarialTrainer 类

```python
class AdversarialTrainer:
    """对抗训练器 - 统一管理对战流程"""

    def __init__(
        self,
        matchup_strategy: MatchupStrategy,
        reward_system: AdaptiveRewardSystem,
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
```

---

## 七、测试代码

### 7.1 test_matchup_strategies

```python
def test_matchup_strategies():
    """测试配对策略"""
    from population import Population, PopulationConfig
    from transformer_model import build_ctf_transformer, CTFTransformerConfig

    # 创建测试种群
    config = PopulationConfig(population_size=4)
    model_config = CTFTransformerConfig(d_model=64, num_layers=1)
    population = Population(config, model_config)

    # 测试循环赛
    round_robin = RoundRobinStrategy()
    matchups = round_robin.create_matchups(population.individuals, 0)
    print(f"循环赛配对数: {len(matchups)} (预期: 6)")
    assert len(matchups) == 6

    # 测试锦标赛
    tournament = TournamentStrategy(games_per_individual=2)
    matchups = tournament.create_matchups(population.individuals, 0)
    print(f"锦标赛配对数: {len(matchups)} (预期: ~4)")

    # 测试自适应策略
    adaptive = AdaptiveMatchupStrategy(round_robin_until=5)
    matchups_early = adaptive.create_matchups(population.individuals, 3)
    matchups_late = adaptive.create_matchups(population.individuals, 15)
    print(f"早期配对数: {len(matchups_early)}, 后期配对数: {len(matchups_late)}")

    print("✓ 配对策略测试通过")
```

### 7.2 test_adversarial_trainer

```python
def test_adversarial_trainer():
    """测试对抗训练器"""
    from population import Population, PopulationConfig
    from transformer_model import CTFTransformerConfig
    from reward_system import AdaptiveRewardSystem

    # 创建种群
    config = PopulationConfig(population_size=4)
    model_config = CTFTransformerConfig(d_model=64, num_layers=1)
    population = Population(config, model_config)

    # 创建训练器
    strategy = AdaptiveMatchupStrategy(round_robin_until=2)
    reward_system = AdaptiveRewardSystem()
    reward_system.reset_for_generation(0)

    trainer = AdversarialTrainer(
        matchup_strategy=strategy,
        reward_system=reward_system,
        num_workers=2,
        max_steps=100,  # 快速测试
        temperature=1.0
    )

    # 执行一轮训练
    print("开始测试对抗训练...")
    stats = trainer.train_epoch(population.individuals, generation=0)

    # 验证结果
    print(f"对战场数: {stats['num_games']}")
    print(f"平均步数: {stats['avg_steps']:.1f}")
    print(f"最佳适应度: {stats['best_fitness']:.2f}")
    print(f"平均适应度: {stats['avg_fitness']:.2f}")

    assert stats['num_games'] > 0
    assert stats['best_fitness'] >= 0

    print("✓ 对抗训练器测试通过")
```

---

## 八、实现注意事项

### 8.1 并行化陷阱

⚠️ **问题**：PyTorch模型在多进程间传递可能出错

**解决方案**：
1. 使用 `torch.multiprocessing` 替代 `multiprocessing`
2. 确保模型在 `spawn` 模式下可序列化
3. 或者使用线程池（ThreadPoolExecutor）替代进程池

```python
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
```

### 8.2 内存管理

⚠️ **问题**：并行执行大量游戏可能耗尽内存

**解决方案**：
1. 限制并行度（num_workers=4）
2. 训练时不记录轨迹（record_trajectory=False）
3. 及时释放游戏环境

### 8.3 随机种子

⚠️ **重要**：确保每场游戏使用不同的随机种子

```python
seed = int(time.time() * 1000) % 1000000 + game_id
```

### 8.4 适应度计算

⚠️ **关键**：适应度必须反映真实实力

当前公式：
```
fitness = win_rate * 100 + avg_reward * 0.1
```

可选改进：
- 考虑对手强度（Elo评分系统）
- 加入旗帜捕获奖励
- 惩罚超时游戏

---

## 九、文件结构

```
adversarial_trainer.py
├── MatchupStrategy (抽象基类)
│   ├── RoundRobinStrategy
│   ├── TournamentStrategy
│   └── AdaptiveMatchupStrategy
├── GameResult (数据类)
├── run_single_game (函数)
├── ParallelGameExecutor (类)
├── update_fitness_from_results (函数)
├── AdversarialTrainer (主类)
└── 测试函数
```

---

## 十、预估代码量

- MatchupStrategy 及子类: ~150行
- GameResult + run_single_game: ~120行
- ParallelGameExecutor: ~80行
- update_fitness_from_results: ~60行
- AdversarialTrainer: ~100行
- 测试代码: ~100行

**总计**: ~610行

---

## 十一、依赖关系

```
adversarial_trainer.py
├── population.py (Individual, Population)
├── reward_system.py (AdaptiveRewardSystem)
├── game_interface.py (GameInterface, TransformerAgent)
├── transformer_model.py (CTFTransformer)
├── ctf_ai.sim_env (CTFSim)
└── lib.tree_features (Geometry)
```

**所有依赖已实现，可直接开发。**
