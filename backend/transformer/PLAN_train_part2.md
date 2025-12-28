# train.py 实现计划 (续1)

## 五、主训练循环

### 5.1 EvolutionaryTrainer 类

```python
class EvolutionaryTrainer:
    """进化训练器 - 整合所有模块"""

    def __init__(self, config: TrainingConfig):
        """
        Args:
            config: 训练配置
        """
        self.config = config

        # 设置随机种子
        if config.seed is not None:
            self._set_seed(config.seed)

        # 初始化组件
        self.logger = TrainingLogger(config.log_dir, config.experiment_name)
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir,
            config.keep_best_n,
            config.experiment_name
        )

        # 创建种群
        from population import Population, PopulationConfig
        from transformer_model import CTFTransformerConfig

        pop_config = PopulationConfig(
            population_size=config.population_size,
            elite_size=config.elite_size,
            tournament_size=config.tournament_size
        )

        model_config = CTFTransformerConfig(
            d_model=config.d_model,
            num_layers=config.num_layers,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        )

        self.population = Population(pop_config, model_config)
        self.logger.log_message(f"种群已创建: {config.population_size} 个体")

        # 创建奖励系统
        from reward_system import AdaptiveRewardSystem
        self.reward_system = AdaptiveRewardSystem()

        # 创建对抗训练器
        from adversarial_trainer import AdversarialTrainer, AdaptiveMatchupStrategy

        matchup_strategy = AdaptiveMatchupStrategy(
            round_robin_until=config.round_robin_until,
            tournament_games=config.tournament_games
        )

        self.adversarial_trainer = AdversarialTrainer(
            matchup_strategy=matchup_strategy,
            reward_system=self.reward_system,
            num_workers=config.num_workers,
            max_steps=config.max_game_steps,
            temperature=config.action_temperature
        )

        # 遗传算法参数
        from genetic_ops import AnnealingScheduler
        self.annealing = AnnealingScheduler(
            initial_temp=config.initial_temperature,
            min_temp=config.min_temperature,
            cooling_rate=config.cooling_rate
        )

        self.current_generation = 0
        self.best_fitness_ever = float('-inf')

    def _set_seed(self, seed: int) -> None:
        """设置随机种子"""
        import random
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train(self, resume_from: Optional[str] = None) -> None:
        """
        执行完整训练流程

        Args:
            resume_from: 检查点路径（可选，用于恢复训练）
        """
        # 恢复训练
        if resume_from:
            self._resume_from_checkpoint(resume_from)

        self.logger.log_message("=" * 60)
        self.logger.log_message("开始进化训练")
        self.logger.log_message("=" * 60)
        self.logger.log_message(f"配置: {self.config}")

        # 主循环
        for generation in range(self.current_generation, self.config.num_generations):
            self.current_generation = generation

            # 获取当前温度
            temperature = self.annealing.get_temperature(generation)

            self.logger.log_message(f"\n{'='*60}")
            self.logger.log_message(f"世代 {generation}/{self.config.num_generations}")
            self.logger.log_message(f"温度: {temperature:.4f}")
            self.logger.log_message(f"{'='*60}")

            # 1. 重置奖励系统
            self.reward_system.reset_for_generation(generation)

            # 2. 对抗训练
            stats = self.adversarial_trainer.train_epoch(
                self.population.individuals,
                generation
            )

            # 3. 记录日志
            self.logger.log_generation(generation, temperature, stats)

            # 4. 保存检查点
            is_best = stats['best_fitness'] > self.best_fitness_ever
            if is_best:
                self.best_fitness_ever = stats['best_fitness']

            if generation % self.config.save_every == 0 or is_best:
                self.checkpoint_manager.save_checkpoint(
                    generation,
                    self.population,
                    temperature,
                    stats,
                    is_best=is_best
                )

            # 5. 遗传演化（最后一代不演化）
            if generation < self.config.num_generations - 1:
                self._evolve_population(temperature)

        # 训练结束
        self.logger.log_message("\n" + "=" * 60)
        self.logger.log_message("训练完成！")
        self.logger.log_message(f"最佳适应度: {self.best_fitness_ever:.2f}")
        self.logger.log_message("=" * 60)
        self.logger.close()

    def _evolve_population(self, temperature: float) -> None:
        """执行遗传演化"""
        from genetic_ops import evolve_generation

        self.logger.log_message("开始遗传演化...")

        new_population = evolve_generation(
            population=self.population.individuals,
            elite_size=self.config.elite_size,
            tournament_size=self.config.tournament_size,
            crossover_alpha=self.config.crossover_alpha,
            mutation_rate=self.config.mutation_rate,
            temperature=temperature
        )

        self.population.individuals = new_population
        self.logger.log_message("✓ 遗传演化完成")

    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """从检查点恢复训练"""
        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)

        self.current_generation = checkpoint['generation'] + 1
        self.best_fitness_ever = checkpoint['best_fitness']
        self.population.load_state_dict(checkpoint['population_state'])

        self.logger.log_message(f"从世代 {checkpoint['generation']} 恢复训练")
```

---

## 六、命令行接口

### 6.1 parse_arguments 函数

```python
import argparse

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="CTF Transformer 进化训练"
    )

    # 配置文件
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='配置文件路径 (JSON格式)'
    )

    # 恢复训练
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='检查点路径（恢复训练）'
    )

    # 快速测试模式
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='快速测试模式（小规模训练）'
    )

    # 覆盖配置参数
    parser.add_argument('--population-size', type=int, help='种群大小')
    parser.add_argument('--num-generations', type=int, help='世代数')
    parser.add_argument('--num-workers', type=int, help='并行工作进程数')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--experiment-name', type=str, help='实验名称')

    return parser.parse_args()
```

### 6.2 main 函数

```python
def main():
    """主函数"""
    args = parse_arguments()

    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        config = TrainingConfig()

    # 快速测试模式
    if args.quick_test:
        config.population_size = 4
        config.num_generations = 5
        config.max_game_steps = 100
        config.num_workers = 2
        config.experiment_name = "quick_test"
        print("⚡ 快速测试模式")

    # 覆盖配置参数
    if args.population_size:
        config.population_size = args.population_size
    if args.num_generations:
        config.num_generations = args.num_generations
    if args.num_workers:
        config.num_workers = args.num_workers
    if args.seed:
        config.seed = args.seed
    if args.experiment_name:
        config.experiment_name = args.experiment_name

    # 创建训练器
    trainer = EvolutionaryTrainer(config)

    # 开始训练
    try:
        trainer.train(resume_from=args.resume)
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        trainer.logger.log_message("训练被用户中断")
        trainer.logger.close()
    except Exception as e:
        print(f"\n训练出错: {e}")
        trainer.logger.log_message(f"训练出错: {e}")
        trainer.logger.close()
        raise


if __name__ == "__main__":
    main()
```

---
