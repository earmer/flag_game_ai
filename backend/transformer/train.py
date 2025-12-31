"""
train.py

主训练脚本 - 整合所有模块，实现完整的进化训练流程
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
import json
import os
import shutil
from pathlib import Path

# 4阶段训练相关导入
from stage_config import (
    TrainingStage, StageConfig,
    create_stage_configs, create_quick_test_configs,
    calculate_win_rate_with_ci
)
from hall_of_fame import HallOfFame

from _import_bootstrap import TORCH_AVAILABLE, NUMPY_AVAILABLE

if not TORCH_AVAILABLE:
    raise RuntimeError("PyTorch is required for training. Install with: pip install torch")

import torch


# ============================================================
# 配置管理
# ============================================================

@dataclass
class TrainingConfig:
    """训练配置"""

    # 种群参数
    population_size: int = 8
    elite_size: int = 2
    tournament_size: int = 3

    # 遗传参数
    crossover_alpha: float = 0.5
    mutation_rate: float = 0.1
    initial_temperature: float = 1.0
    min_temperature: float = 0.1
    cooling_rate: float = 0.99  # 放缓退火速度（原0.95过快）

    # 训练参数
    num_generations: int = 200   # 增加到200代（原50代，支持100000场对战）
    max_game_steps: int = 500    # 修复: 1000 -> 500 (2:30限制，每步0.3秒)
    action_temperature: float = 1.0

    # 对抗训练参数
    round_robin_until: int = 20  # 延长循环赛阶段（原10代）
    tournament_games: int = 8    # 增加对局数量（原4场）
    num_workers: int = 4

    # 模型参数
    d_model: int = 128
    num_layers: int = 2
    nhead: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1

    # 检查点参数
    checkpoint_dir: str = "checkpoints"
    save_every: int = 5
    keep_best_n: int = 3

    # 日志参数
    log_dir: str = "logs"
    log_every: int = 1

    # 随机种子
    seed: Optional[int] = None

    # 实验名称
    experiment_name: str = "ctf_evolution"


def load_config(config_path: Optional[str] = None) -> TrainingConfig:
    """
    加载训练配置

    Args:
        config_path: 配置文件路径（JSON格式）

    Returns:
        TrainingConfig对象
    """
    if config_path is None:
        return TrainingConfig()

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    return TrainingConfig(**config_dict)


def save_config(config: TrainingConfig, save_path: str) -> None:
    """保存配置到JSON文件"""
    config_dict = asdict(config)

    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)


# ============================================================
# 检查点管理
# ============================================================

class CheckpointManager:
    """检查点管理器"""

    def __init__(
        self,
        checkpoint_dir: str,
        keep_best_n: int = 3,
        experiment_name: str = "experiment"
    ):
        """
        Args:
            checkpoint_dir: 检查点保存目录
            keep_best_n: 保留最佳的N个检查点
            experiment_name: 实验名称
        """
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best_n = keep_best_n
        self.best_checkpoints = []  # [(fitness, path), ...]

    def save_checkpoint(
        self,
        generation: int,
        population: 'Population',
        temperature: float,
        stats: Dict[str, Any],
        is_best: bool = False
    ) -> str:
        """
        保存检查点

        Args:
            generation: 当前世代
            population: 种群对象
            temperature: 当前温度
            stats: 统计信息
            is_best: 是否为最佳检查点

        Returns:
            检查点文件路径
        """
        # 生成文件名
        if is_best:
            filename = f"best_gen_{generation}"
        else:
            filename = f"checkpoint_gen_{generation}"

        checkpoint_dir = self.checkpoint_dir / filename
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 保存种群
        population.save_population(str(checkpoint_dir / "population"))

        # 保存检查点元数据
        checkpoint_meta = {
            'generation': generation,
            'temperature': temperature,
            'stats': stats,
            'best_fitness': stats['best_fitness']
        }

        meta_path = checkpoint_dir / "checkpoint_meta.json"
        with open(meta_path, 'w') as f:
            json.dump(checkpoint_meta, f, indent=2)

        print(f"✓ 检查点已保存: {checkpoint_dir}")

        # 更新最佳检查点列表
        if is_best:
            self._update_best_checkpoints(stats['best_fitness'], checkpoint_dir)

        return str(checkpoint_dir)

    def _update_best_checkpoints(self, fitness: float, filepath: Path) -> None:
        """更新最佳检查点列表"""
        self.best_checkpoints.append((fitness, filepath))
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)

        # 删除多余的检查点
        if len(self.best_checkpoints) > self.keep_best_n:
            _, old_path = self.best_checkpoints.pop()
            if old_path.exists():
                # FIX: Use shutil.rmtree() for directories instead of unlink()
                try:
                    shutil.rmtree(old_path)
                    print(f"删除旧检查点: {old_path}")
                except Exception as e:
                    print(f"警告: 无法删除检查点 {old_path}: {e}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载检查点"""
        checkpoint_dir = Path(checkpoint_path)

        # 加载元数据
        meta_path = checkpoint_dir / "checkpoint_meta.json"
        with open(meta_path, 'r') as f:
            checkpoint_meta = json.load(f)

        print(f"✓ 检查点已加载: {checkpoint_path}")
        return checkpoint_meta

    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新的检查点路径"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_gen_*"))
        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)


# ============================================================
# 日志系统
# ============================================================

import csv
import time
from datetime import datetime


class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, log_dir: str, experiment_name: str):
        """
        Args:
            log_dir: 日志保存目录
            experiment_name: 实验名称
        """
        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 创建CSV日志文件
        self.csv_path = self.log_dir / "training_log.csv"
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)

        # 写入表头
        self.csv_writer.writerow([
            'generation', 'timestamp', 'temperature',
            'num_games', 'avg_steps', 'avg_duration_ms',
            'best_fitness', 'avg_fitness', 'worst_fitness',
            'l_wins', 'r_wins', 'draws', 'draw_rate',
            'avg_flags_captured', 'sparse_enabled'
        ])

        # 创建文本日志文件
        self.txt_path = self.log_dir / "training.log"
        self.txt_file = open(self.txt_path, 'w')

        self.start_time = time.time()

    def log_generation(
        self,
        generation: int,
        temperature: float,
        stats: Dict[str, Any]
    ) -> None:
        """记录世代信息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 计算平局率
        total_games = stats['num_games']
        draw_rate = stats['draws'] / total_games if total_games > 0 else 0.0

        # 写入CSV
        self.csv_writer.writerow([
            generation,
            timestamp,
            f"{temperature:.4f}",
            stats['num_games'],
            f"{stats['avg_steps']:.1f}",
            f"{stats['avg_duration_ms']:.1f}",
            f"{stats['best_fitness']:.2f}",
            f"{stats['avg_fitness']:.2f}",
            f"{stats['worst_fitness']:.2f}",
            stats['l_wins'],
            stats['r_wins'],
            stats['draws'],
            f"{draw_rate:.2%}",
            f"{stats.get('avg_flags_captured', 0):.2f}",
            stats.get('sparse_enabled', False)
        ])
        self.csv_file.flush()

        # 写入文本日志
        elapsed = time.time() - self.start_time
        log_msg = (
            f"[世代 {generation:3d}] "
            f"温度={temperature:.3f} | "
            f"对局={stats['num_games']:3d} "
            f"(平均{stats['avg_steps']:.0f}步/{stats['avg_duration_ms']/1000:.1f}秒) | "
            f"适应度: {stats['best_fitness']:6.2f} / "
            f"{stats['avg_fitness']:6.2f} / "
            f"{stats['worst_fitness']:6.2f} | "
            f"胜场: L={stats['l_wins']:3d} R={stats['r_wins']:3d} "
            f"平={stats['draws']:2d} ({draw_rate:.0%}) | "
            f"世代训练时长: {elapsed:.1f}秒"
        )

        print(log_msg)
        self.txt_file.write(log_msg + '\n')
        self.txt_file.flush()

    def log_message(self, message: str) -> None:
        """记录自定义消息"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        self.txt_file.write(log_msg + '\n')
        self.txt_file.flush()

    def close(self) -> None:
        """关闭日志文件"""
        self.csv_file.close()
        self.txt_file.close()


# ============================================================
# 主训练循环
# ============================================================

import random
import numpy as np


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

        self.population = Population(pop_config)
        self.population.initialize_random(model_config)
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
        from genetic_ops import AnnealingScheduler, AnnealingConfig

        annealing_config = AnnealingConfig(
            initial_temperature=config.initial_temperature,
            min_temperature=config.min_temperature,
            cooling_rate=config.cooling_rate
        )
        self.annealing = AnnealingScheduler(annealing_config)

        self.current_generation = 0
        self.best_fitness_ever = float('-inf')

    def _set_seed(self, seed: int) -> None:
        """设置随机种子"""
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
        self.logger.log_message(f"配置: population_size={self.config.population_size}, "
                                f"generations={self.config.num_generations}")

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

            # 2.5 更新平局率到奖励系统（用于自适应奖励切换）
            draw_rate = stats['draws'] / stats['num_games'] if stats['num_games'] > 0 else 1.0
            self.reward_system.update_draw_rate(draw_rate)
            stats['sparse_enabled'] = self.reward_system.curriculum.sparse_enabled

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

            # 6. 定期基准测试（每10代）
            if generation % 10 == 0:
                self._run_benchmark(generation)

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

        new_individuals = evolve_generation(
            population=self.population,
            temperature=temperature,
            crossover_alpha=self.config.crossover_alpha,
            mutation_rate=self.config.mutation_rate
        )

        self.population.individuals = new_individuals
        self.logger.log_message("✓ 遗传演化完成")

    def _run_benchmark(self, generation: int) -> None:
        """运行基准测试"""
        try:
            from benchmark import run_benchmark, BenchmarkResult
            from game_interface import TransformerAgent

            self.logger.log_message(f"\n--- 基准测试 (Gen {generation}) ---")

            # 获取最优个体
            self.population.sort_by_fitness()
            best_individual = self.population.individuals[0]

            # 创建Transformer智能体
            agent = TransformerAgent(
                team="L",
                model=best_individual.model,
                temperature=0.5  # 较低温度，更确定性
            )

            # 运行基准测试（20场快速测试）
            result = run_benchmark(
                transformer_agent=agent,
                num_games=20,
                max_steps=self.config.max_game_steps
            )
            result.generation = generation

            # 记录结果
            self.logger.log_message(
                f"基准测试结果: 胜率={result.win_rate:.1%}, "
                f"平局率={result.draw_rate:.1%}, "
                f"L胜={result.left_wins}, R胜={result.right_wins}"
            )

            # 检查阶段升级条件
            self._check_stage_upgrade(generation, result)

        except Exception as e:
            self.logger.log_message(f"基准测试失败: {e}")

    def _check_stage_upgrade(self, generation: int, benchmark_result) -> None:
        """检查是否满足阶段升级条件"""
        stage = self.reward_system.curriculum.get_stage(generation)
        win_rate = benchmark_result.win_rate

        if stage == 1 and win_rate >= 0.50:
            self.logger.log_message(f"✓ Stage 1升级条件满足: 胜率{win_rate:.1%} >= 50%")
        elif stage == 2 and win_rate >= 0.85:
            self.logger.log_message(f"✓ Stage 2升级条件满足: 胜率{win_rate:.1%} >= 85%")
        elif stage == 3 and win_rate >= 0.95:
            self.logger.log_message(f"✓ Stage 3升级条件满足: 胜率{win_rate:.1%} >= 95%")

    def _resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """从检查点恢复训练"""
        from population import Population

        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)

        self.current_generation = checkpoint['generation'] + 1
        self.best_fitness_ever = checkpoint['best_fitness']

        # 加载种群
        checkpoint_dir = Path(checkpoint_path)
        population_dir = checkpoint_dir / "population"
        self.population = Population.load_population(
            str(population_dir),
            self.population.config
        )

        self.logger.log_message(f"从世代 {checkpoint['generation']} 恢复训练")


# ============================================================
# 4阶段训练器
# ============================================================

class StagedEvolutionaryTrainer:
    """4阶段进化训练器 - 支持阶段晋级和HoF对手采样"""

    def __init__(
        self,
        stage_configs: Dict[TrainingStage, StageConfig],
        experiment_name: str = "staged_training",
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        seed: Optional[int] = None
    ):
        """
        Args:
            stage_configs: 4阶段配置字典
            experiment_name: 实验名称
            checkpoint_dir: 检查点目录
            log_dir: 日志目录
            seed: 随机种子
        """
        self.stage_configs = stage_configs
        self.experiment_name = experiment_name
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        # 设置随机种子
        if seed is not None:
            self._set_seed(seed)

        # 初始化日志和检查点管理
        self.logger = TrainingLogger(log_dir, experiment_name)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir, keep_best_n=5, experiment_name=experiment_name
        )

        # 初始化HoF
        self.hof = HallOfFame(max_size=10)

        # 模型配置（用于种群调整）
        from transformer_model import CTFTransformerConfig
        self.model_config = CTFTransformerConfig()

        # 当前状态
        self.current_stage = TrainingStage.FOUNDATION
        self.current_generation = 0
        self.best_fitness_ever = float('-inf')

        # 延迟初始化的组件
        self.population = None
        self.reward_system = None
        self.adversarial_trainer = None
        self.annealing = None

    def _set_seed(self, seed: int) -> None:
        """设置随机种子"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _initialize_for_stage(self, stage: TrainingStage) -> None:
        """为指定阶段初始化或更新组件"""
        config = self.stage_configs[stage]
        self.current_stage = stage

        self.logger.log_message(f"\n{'='*60}")
        self.logger.log_message(f"初始化阶段 {stage.value}: {config.name}")
        self.logger.log_message(f"{'='*60}")

        # 1. 初始化或调整种群
        from population import Population, PopulationConfig

        if self.population is None:
            # 首次初始化
            pop_config = PopulationConfig(
                population_size=config.population_size,
                elite_size=config.elite_size,
                tournament_size=config.tournament_size
            )
            self.population = Population(pop_config, self.model_config)
            self.logger.log_message(f"种群已创建: {config.population_size} 个体")
        else:
            # 调整种群大小
            self.population.resize(config.population_size, self.model_config)
            self.population.config.elite_size = config.elite_size
            self.population.config.tournament_size = config.tournament_size

        # 2. 初始化或更新奖励系统
        from reward_system import AdaptiveRewardSystem

        if self.reward_system is None:
            self.reward_system = AdaptiveRewardSystem()

        # 设置阶段奖励权重
        self.reward_system.curriculum.set_weights(
            config.dense_weight, config.sparse_weight
        )
        self.logger.log_message(
            f"奖励权重: dense={config.dense_weight:.1f}, "
            f"sparse={config.sparse_weight:.1f}"
        )

        # 3. 创建对抗训练器（每阶段重新创建以更新HoF采样率）
        from adversarial_trainer import AdversarialTrainer, AdaptiveMatchupStrategy

        # 阶段边界配置
        stage_boundaries = [
            self.stage_configs[TrainingStage.FOUNDATION].end_gen,
            self.stage_configs[TrainingStage.OPTIMIZATION].end_gen,
            self.stage_configs[TrainingStage.COMPETITION].end_gen
        ]

        matchup_strategy = AdaptiveMatchupStrategy(
            round_robin_until=20,
            tournament_games=config.games_per_individual,
            stage_boundaries=stage_boundaries
        )

        self.adversarial_trainer = AdversarialTrainer(
            matchup_strategy=matchup_strategy,
            reward_system=self.reward_system,
            num_workers=config.num_workers,
            max_steps=1000,
            temperature=1.0,
            hof=self.hof,
            hof_sample_rate=config.hof_sample_rate,
            round_per_game=config.round_per_game
        )
        self.logger.log_message(f"HoF采样率: {config.hof_sample_rate:.0%}")
        self.logger.log_message(f"多轮对战: 每个配对 {config.round_per_game} 轮")

        # 4. 初始化退火调度器
        from genetic_ops import AnnealingScheduler, AnnealingConfig

        annealing_config = AnnealingConfig(
            initial_temperature=config.initial_temperature,
            min_temperature=config.min_temperature,
            cooling_rate=config.cooling_rate
        )
        self.annealing = AnnealingScheduler(annealing_config)

        self.logger.log_message(f"阶段 {stage.value} 初始化完成")

    def train(self) -> None:
        """执行4阶段训练流程"""
        self.logger.log_message("=" * 60)
        self.logger.log_message("开始4阶段进化训练")
        self.logger.log_message("=" * 60)

        # 遍历4个阶段
        for stage in TrainingStage:
            config = self.stage_configs[stage]

            self.logger.log_message(f"\n{'#'*60}")
            self.logger.log_message(f"# 阶段 {stage.value}: {config.name}")
            self.logger.log_message(f"# 世代范围: {config.start_gen} - {config.end_gen}")
            self.logger.log_message(f"# 种群大小: {config.population_size}")
            self.logger.log_message(f"{'#'*60}")

            # 初始化阶段
            self._initialize_for_stage(stage)

            # 训练当前阶段
            advanced = self._train_stage(stage)

            if advanced:
                # 保存冠军到HoF
                self._save_champion_to_hof(stage)
                self.logger.log_message(f"✓ 阶段 {stage.value} 完成，晋级到下一阶段")
            else:
                self.logger.log_message(f"✗ 阶段 {stage.value} 未能晋级，训练结束")
                break

        # 训练结束
        self.logger.log_message("\n" + "=" * 60)
        self.logger.log_message("4阶段训练完成！")
        self.logger.log_message(f"最佳适应度: {self.best_fitness_ever:.2f}")
        self.logger.log_message(f"HoF成员数: {len(self.hof)}")
        self.logger.log_message("=" * 60)

        # 保存HoF
        hof_path = Path(self.log_dir) / self.experiment_name / "hof"
        self.hof.save(str(hof_path))

        self.logger.close()

    def _train_stage(self, stage: TrainingStage) -> bool:
        """
        训练单个阶段

        修复: 使用 max_generations 作为循环上限，而非 end_gen

        Args:
            stage: 当前阶段

        Returns:
            是否成功晋级
        """
        config = self.stage_configs[stage]
        stage_start_gen = config.start_gen

        # 使用 max_generations 作为循环上限（修复原 bug）
        for generations_in_stage in range(1, config.max_generations + 1):
            gen = stage_start_gen + generations_in_stage - 1
            self.current_generation = gen

            # 获取温度（相对于阶段内的世代数）
            temperature = self.annealing.get_temperature(generations_in_stage - 1)

            self.logger.log_message(f"\n--- 世代 {gen} (阶段内第{generations_in_stage}代) ---")

            # 1. 重置奖励系统
            self.reward_system.reset_for_generation(gen)

            # 2. 对抗训练
            stats = self.adversarial_trainer.train_epoch(
                self.population.individuals, gen
            )

            # 3. 更新平局率
            draw_rate = stats['draws'] / stats['num_games'] if stats['num_games'] > 0 else 1.0
            self.reward_system.update_draw_rate(draw_rate)

            # 4. 记录日志
            self.logger.log_generation(gen, temperature, stats)

            # 5. 更新最佳适应度
            if stats['best_fitness'] > self.best_fitness_ever:
                self.best_fitness_ever = stats['best_fitness']

            # 6. 保存检查点
            if generations_in_stage % config.benchmark_interval == 0:
                self.checkpoint_manager.save_checkpoint(
                    gen, self.population, temperature, stats,
                    is_best=(stats['best_fitness'] == self.best_fitness_ever)
                )

            # 7. 检查晋级条件（达到最小世代数后）
            if generations_in_stage >= config.min_generations:
                if generations_in_stage % config.benchmark_interval == 0:
                    advanced = self._check_advancement(stage, gen)
                    if advanced:
                        return True

            # 8. 遗传演化
            self._evolve_population(temperature)

        # 达到最大世代数，强制结束阶段
        self.logger.log_message(
            f"达到最大世代数 {config.max_generations}，强制结束阶段"
        )
        return config.auto_advance

    def _check_advancement(self, stage: TrainingStage, generation: int) -> bool:
        """
        检查是否满足晋级条件

        Args:
            stage: 当前阶段
            generation: 当前世代

        Returns:
            是否可以晋级
        """
        config = self.stage_configs[stage]

        # 自动晋级模式（快速测试）
        if config.auto_advance:
            self.logger.log_message("自动晋级模式，跳过胜率检查")
            return True

        self.logger.log_message(f"\n--- 基准测试 (Gen {generation}) ---")

        try:
            from benchmark import run_benchmark
            from game_interface import TransformerAgent

            # 获取最优个体
            self.population.sort_by_fitness()
            best_individual = self.population.individuals[0]

            # 创建智能体
            agent = TransformerAgent(
                team="L",
                model=best_individual.model,
                temperature=0.5
            )

            # 运行基准测试
            result = run_benchmark(
                transformer_agent=agent,
                num_games=config.min_benchmark_games,
                max_steps=1000
            )

            # 计算置信区间
            win_rate, lower, upper = calculate_win_rate_with_ci(
                result.left_wins, result.total_games
            )

            self.logger.log_message(
                f"基准测试: 胜率={win_rate:.1%} "
                f"(95% CI: [{lower:.1%}, {upper:.1%}]), "
                f"门槛={config.min_win_rate:.0%}"
            )

            # 检查下界是否达标
            if lower >= config.min_win_rate:
                self.logger.log_message(
                    f"✓ 晋级条件满足: 下界{lower:.1%} >= {config.min_win_rate:.0%}"
                )
                return True
            else:
                self.logger.log_message(
                    f"✗ 晋级条件未满足: 下界{lower:.1%} < {config.min_win_rate:.0%}"
                )
                return False

        except Exception as e:
            self.logger.log_message(f"基准测试失败: {e}")
            return False

    def _save_champion_to_hof(self, stage: TrainingStage) -> None:
        """保存当前阶段冠军到HoF"""
        self.population.sort_by_fitness()
        best = self.population.individuals[0]

        # 计算胜率
        win_rate = best.win_rate() if best.games_played > 0 else 0.0

        # 保存到HoF
        self.hof.add_champion(
            model_state_dict=best.model.state_dict(),
            stage=stage.value,
            generation=self.current_generation,
            win_rate=win_rate,
            metadata={
                'fitness': best.fitness,
                'wins': best.wins,
                'losses': best.losses,
                'flags_captured': best.flags_captured
            }
        )

        self.logger.log_message(
            f"冠军已保存到HoF: Stage {stage.value}, "
            f"Gen {self.current_generation}, WR {win_rate:.1%}"
        )

    def _evolve_population(self, temperature: float) -> None:
        """执行遗传演化"""
        from genetic_ops import evolve_generation

        config = self.stage_configs[self.current_stage]

        new_individuals = evolve_generation(
            population=self.population,
            temperature=temperature,
            crossover_alpha=config.crossover_alpha,
            mutation_rate=config.mutation_rate
        )

        self.population.individuals = new_individuals


# ============================================================
# 命令行接口
# ============================================================

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

    # 4阶段快速测试模式
    parser.add_argument(
        '--quick-test-staged',
        action='store_true',
        help='4阶段快速测试模式（完整流程验证，约5分钟）'
    )

    # 4阶段完整训练模式
    parser.add_argument(
        '--staged',
        action='store_true',
        help='4阶段完整训练模式'
    )

    # 覆盖配置参数
    parser.add_argument('--population-size', type=int, help='种群大小')
    parser.add_argument('--num-generations', type=int, help='世代数')
    parser.add_argument('--num-workers', type=int, help='并行工作进程数')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--experiment-name', type=str, help='实验名称')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    # 4阶段快速测试模式
    if args.quick_test_staged:
        print("⚡ 4阶段快速测试模式")
        stage_configs = create_quick_test_configs()
        trainer = StagedEvolutionaryTrainer(
            stage_configs=stage_configs,
            experiment_name="quick_test_staged",
            seed=args.seed
        )
        try:
            trainer.train()
        except KeyboardInterrupt:
            print("\n训练被用户中断")
            trainer.logger.close()
        except Exception as e:
            print(f"\n训练出错: {e}")
            trainer.logger.close()
            raise
        return

    # 4阶段完整训练模式
    if args.staged:
        print("🚀 4阶段完整训练模式")
        stage_configs = create_stage_configs()
        experiment_name = args.experiment_name or "staged_training"
        trainer = StagedEvolutionaryTrainer(
            stage_configs=stage_configs,
            experiment_name=experiment_name,
            seed=args.seed
        )
        try:
            trainer.train()
        except KeyboardInterrupt:
            print("\n训练被用户中断")
            trainer.logger.close()
        except Exception as e:
            print(f"\n训练出错: {e}")
            trainer.logger.close()
            raise
        return

    # 原有训练模式（向后兼容）
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

