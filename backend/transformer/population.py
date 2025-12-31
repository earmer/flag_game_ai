from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from _import_bootstrap import NUMPY_AVAILABLE

if NUMPY_AVAILABLE:
    import numpy as np

import torch

from transformer_model import CTFTransformer, CTFTransformerConfig, build_ctf_transformer


@dataclass
class Individual:
    """AI个体 - 封装一个Transformer模型及其统计信息"""

    # 基本信息
    id: str                              # 唯一标识符
    model: CTFTransformer                # Transformer策略网络
    generation: int                      # 出生世代

    # 适应度统计
    fitness: float = 0.0                 # 当前适应度评分
    lifetime_fitness: float = 0.0        # 累计适应度

    # 对战统计
    wins: int = 0                        # 胜场数
    losses: int = 0                      # 负场数
    draws: int = 0                       # 平局数
    games_played: int = 0                # 总对战场数

    # 游戏表现统计
    flags_captured: int = 0              # 捕获旗帜总数
    flags_lost: int = 0                  # 丢失旗帜总数
    enemies_tagged: int = 0              # 标记敌人总数
    times_tagged: int = 0                # 被标记次数
    teammates_freed: int = 0             # 救出队友次数

    # 临时世代统计（每代重置）
    epoch_wins: int = 0                  # 本代胜场数
    epoch_losses: int = 0                # 本代负场数
    epoch_draws: int = 0                 # 本代平局数
    epoch_total_reward: float = 0.0      # 本代累计奖励
    epoch_games_played: int = 0          # 本代对战场数

    # 世代信息
    age: int = 0                         # 存活世代数
    parent_ids: List[str] = field(default_factory=list)  # 父代ID

    def win_rate(self) -> float:
        """计算胜率"""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played

    def update_stats(self, game_result: Dict[str, Any]) -> None:
        """更新个体统计信息"""
        self.games_played += 1

        if game_result.get('won', False):
            self.wins += 1
        elif game_result.get('draw', False):
            self.draws += 1
        else:
            self.losses += 1

        self.flags_captured += game_result.get('flags_captured', 0)
        self.flags_lost += game_result.get('flags_lost', 0)
        self.enemies_tagged += game_result.get('enemies_tagged', 0)
        self.times_tagged += game_result.get('times_tagged', 0)
        self.teammates_freed += game_result.get('teammates_freed', 0)

    def clone(self, new_id: str, generation: int) -> 'Individual':
        """克隆个体（深拷贝模型）"""
        new_model = copy.deepcopy(self.model)
        return Individual(
            id=new_id,
            model=new_model,
            generation=generation,
            parent_ids=[self.id]
        )


@dataclass
class PopulationConfig:
    """种群配置"""
    population_size: int = 8             # 种群大小
    elite_size: int = 2                  # 精英保留数量
    tournament_size: int = 3             # 锦标赛选择大小


class Population:
    """种群管理器"""

    def __init__(self, config: PopulationConfig, model_config: Optional[CTFTransformerConfig] = None, device: Optional[torch.device] = None):
        self.config = config
        self.individuals: List[Individual] = []
        self.current_generation: int = 0
        self.history: List[Dict[str, Any]] = []
        self.best_individual: Optional[Individual] = None
        self.best_fitness: float = float('-inf')

        # 设备配置
        self.device = device if device is not None else torch.device("cpu")

        # Auto-initialize if model_config provided
        if model_config is not None:
            self.initialize_random(model_config)

    def initialize_random(self, model_config: CTFTransformerConfig) -> None:
        """随机初始化种群"""
        self.individuals = []
        for i in range(self.config.population_size):
            model = build_ctf_transformer(model_config)
            # 将模型移动到指定设备
            model = model.to(self.device)
            individual = Individual(
                id=f"gen0_ind{i}",
                model=model,
                generation=0
            )
            self.individuals.append(individual)

        print(f"✓ Initialized population with {len(self.individuals)} individuals on {self.device}")

    def sort_by_fitness(self) -> None:
        """按适应度降序排序"""
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)

        # 更新最佳个体
        if self.individuals and self.individuals[0].fitness > self.best_fitness:
            self.best_fitness = self.individuals[0].fitness
            self.best_individual = self.individuals[0]

    def get_top_k(self, k: int) -> List[Individual]:
        """获取前k个个体"""
        return self.individuals[:k]

    def get_elites(self) -> List[Individual]:
        """获取精英个体"""
        return self.get_top_k(self.config.elite_size)

    def get_statistics(self) -> Dict[str, Any]:
        """获取种群统计信息"""
        if not self.individuals:
            return {}

        if NUMPY_AVAILABLE:
            fitnesses = np.array([ind.fitness for ind in self.individuals])
            win_rates = np.array([ind.win_rate() for ind in self.individuals])

            return {
                'generation': self.current_generation,
                'population_size': len(self.individuals),
                'fitness': {
                    'mean': float(np.mean(fitnesses)),
                    'std': float(np.std(fitnesses)),
                    'min': float(np.min(fitnesses)),
                    'max': float(np.max(fitnesses)),
                    'best_ever': self.best_fitness
                },
                'win_rate': {
                    'mean': float(np.mean(win_rates)),
                    'std': float(np.std(win_rates)),
                    'best': float(np.max(win_rates))
                },
                'games_played': sum(ind.games_played for ind in self.individuals),
                'total_wins': sum(ind.wins for ind in self.individuals)
            }
        else:
            # Fallback without numpy
            fitnesses = [ind.fitness for ind in self.individuals]
            win_rates = [ind.win_rate() for ind in self.individuals]

            return {
                'generation': self.current_generation,
                'population_size': len(self.individuals),
                'fitness': {
                    'mean': sum(fitnesses) / len(fitnesses),
                    'min': min(fitnesses),
                    'max': max(fitnesses),
                    'best_ever': self.best_fitness
                },
                'win_rate': {
                    'mean': sum(win_rates) / len(win_rates),
                    'best': max(win_rates)
                },
                'games_played': sum(ind.games_played for ind in self.individuals),
                'total_wins': sum(ind.wins for ind in self.individuals)
            }

    def advance_generation(self, new_individuals: List[Individual]) -> None:
        """进入下一代"""
        self.current_generation += 1

        # 保存当前代统计
        stats = self.get_statistics()
        self.history.append(stats)

        # 更新种群
        self.individuals = new_individuals

        # 增加个体年龄
        for ind in self.individuals:
            if ind.generation < self.current_generation:
                ind.age += 1

        print(f"✓ Advanced to generation {self.current_generation}")

    def reset_epoch_stats(self) -> None:
        """重置本轮统计（保留lifetime统计）"""
        for ind in self.individuals:
            ind.fitness = 0.0

    def resize(self, new_size: int, model_config: CTFTransformerConfig) -> None:
        """
        动态调整种群大小（用于4阶段训练）

        Args:
            new_size: 新的种群大小
            model_config: 模型配置（用于创建新个体）
        """
        current_size = len(self.individuals)

        if new_size == current_size:
            return  # 无需调整

        if new_size > current_size:
            # 扩大种群：保留所有现有个体 + 随机初始化新个体
            num_new = new_size - current_size
            print(f"[Population] 扩大种群: {current_size} → {new_size} (+{num_new}个新个体)")

            for i in range(num_new):
                model = build_ctf_transformer(model_config)
                individual = Individual(
                    id=f"gen{self.current_generation}_expand{i}",
                    model=model,
                    generation=self.current_generation
                )
                self.individuals.append(individual)

        else:
            # 缩小种群：按适应度排序，保留前N个
            num_removed = current_size - new_size
            print(f"[Population] 缩小种群: {current_size} → {new_size} (-{num_removed}个个体)")

            # 排序并保留最优个体
            self.sort_by_fitness()
            self.individuals = self.individuals[:new_size]

        # 更新配置
        self.config.population_size = new_size
        print(f"✓ 种群大小已调整为 {new_size}")

    def print_summary(self) -> None:
        """打印种群摘要"""
        stats = self.get_statistics()
        if not stats:
            print("Empty population")
            return

        print(f"\n{'='*60}")
        print(f"Generation {self.current_generation} Summary")
        print(f"{'='*60}")
        print(f"Population Size: {stats['population_size']}")
        print(f"Fitness - Mean: {stats['fitness']['mean']:.2f}, "
              f"Max: {stats['fitness']['max']:.2f}, "
              f"Best Ever: {stats['fitness']['best_ever']:.2f}")
        print(f"Win Rate - Mean: {stats['win_rate']['mean']:.2%}, "
              f"Best: {stats['win_rate']['best']:.2%}")
        print(f"Total Games: {stats['games_played']}")
        print(f"{'='*60}\n")

    def save_population(self, save_dir: str) -> None:
        """保存整个种群"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存每个个体
        for ind in self.individuals:
            model_path = os.path.join(save_dir, f"{ind.id}.pth")
            ind.model.save_checkpoint(model_path)

        # 保存种群元数据
        metadata = {
            'generation': self.current_generation,
            'individuals': [
                {
                    'id': ind.id,
                    'fitness': ind.fitness,
                    'wins': ind.wins,
                    'losses': ind.losses,
                    'generation': ind.generation
                }
                for ind in self.individuals
            ],
            'history': self.history
        }

        with open(os.path.join(save_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✓ Population saved to {save_dir}")

    @classmethod
    def load_population(cls, load_dir: str, config: PopulationConfig) -> 'Population':
        """加载保存的种群"""
        # 加载元数据
        with open(os.path.join(load_dir, 'metadata.json'), 'r') as f:
            metadata = json.load(f)

        population = cls(config)
        population.current_generation = metadata['generation']
        population.history = metadata['history']

        # 加载每个个体
        for ind_meta in metadata['individuals']:
            model_path = os.path.join(load_dir, f"{ind_meta['id']}.pth")
            model = CTFTransformer.load_checkpoint(model_path)

            individual = Individual(
                id=ind_meta['id'],
                model=model,
                generation=ind_meta['generation'],
                fitness=ind_meta['fitness'],
                wins=ind_meta['wins'],
                losses=ind_meta['losses']
            )
            population.individuals.append(individual)

        print(f"✓ Population loaded from {load_dir}")
        return population
