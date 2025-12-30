from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from _import_bootstrap import NUMPY_AVAILABLE, TORCH_AVAILABLE
from population import Individual

if NUMPY_AVAILABLE:
    import numpy as np

if TORCH_AVAILABLE:
    import torch


@dataclass
class AnnealingConfig:
    """退火配置"""
    initial_temperature: float = 1.0     # 初始温度
    min_temperature: float = 0.1         # 最小温度
    cooling_rate: float = 0.95           # 冷却率（每代降温）
    cooling_schedule: str = "exponential"  # 冷却策略


class AnnealingScheduler:
    """退火温度调度器"""

    def __init__(self, config: AnnealingConfig):
        self.config = config
        self.current_temperature = config.initial_temperature

    def get_temperature(self, generation: int) -> float:
        """获取当前世代的温度"""
        if self.config.cooling_schedule == "exponential":
            temp = self.config.initial_temperature * (self.config.cooling_rate ** generation)
        elif self.config.cooling_schedule == "linear":
            temp = self.config.initial_temperature - generation * 0.02
        else:
            raise ValueError(f"Unknown schedule: {self.config.cooling_schedule}")

        return max(temp, self.config.min_temperature)

    def update(self, generation: int) -> float:
        """更新并返回温度"""
        self.current_temperature = self.get_temperature(generation)
        return self.current_temperature


# ============ 选择算子 ============

def tournament_selection(
    population: List[Individual],
    tournament_size: int = 3,
    temperature: float = 1.0
) -> Individual:
    """
    锦标赛选择（改进版：归一化适应度以增强选择压力）

    Args:
        population: 种群列表
        tournament_size: 锦标赛大小
        temperature: 温度（影响选择压力）

    Returns:
        选中的个体
    """
    # 随机选择tournament_size个个体
    tournament = random.sample(population, min(tournament_size, len(population)))

    # 温度影响选择：温度高时更随机，温度低时更确定
    if temperature > 0.3 and NUMPY_AVAILABLE:
        # 使用归一化softmax概率选择
        fitnesses = np.array([ind.fitness for ind in tournament])

        # 归一化适应度到[0, 1]范围，放大差异
        f_min, f_max = np.min(fitnesses), np.max(fitnesses)
        if f_max - f_min > 1e-6:
            # 归一化后乘以10，增强选择压力
            normalized = (fitnesses - f_min) / (f_max - f_min) * 10.0
        else:
            # 适应度相同，均匀选择
            normalized = np.ones_like(fitnesses)

        # 应用温度缩放
        scaled = normalized / temperature
        # 避免数值溢出
        scaled = scaled - np.max(scaled)
        probs = np.exp(scaled)
        probs = probs / np.sum(probs)

        winner = np.random.choice(tournament, p=probs)
    else:
        # 低温：直接选择最优
        winner = max(tournament, key=lambda ind: ind.fitness)

    return winner


# ============ 交叉算子 ============

def crossover_average(
    parent1: Individual,
    parent2: Individual,
    alpha: float = 0.5,
    new_id: str = "",
    generation: int = 0
) -> Individual:
    """
    权重平均交叉

    child = alpha * parent1 + (1-alpha) * parent2

    Args:
        parent1: 父代1
        parent2: 父代2
        alpha: 混合系数（0.5表示均匀混合）
        new_id: 新个体ID
        generation: 世代编号

    Returns:
        子代个体
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available for crossover")

    import copy

    # 创建新模型
    child_model = copy.deepcopy(parent1.model)

    # 获取父代参数
    state1 = parent1.model.state_dict()
    state2 = parent2.model.state_dict()

    # 混合参数
    child_state = {}
    for key in state1.keys():
        child_state[key] = alpha * state1[key] + (1 - alpha) * state2[key]

    # 加载到子代模型
    child_model.load_state_dict(child_state)

    # 创建子代个体
    child = Individual(
        id=new_id,
        model=child_model,
        generation=generation,
        parent_ids=[parent1.id, parent2.id]
    )

    return child


# ============ 变异算子 ============

def mutate_gaussian(
    individual: Individual,
    temperature: float,
    mutation_rate: float = 0.1,
    noise_scale: float = 0.01
) -> Individual:
    """
    高斯噪声变异（温度控制）

    Args:
        individual: 待变异个体
        temperature: 当前温度（控制噪声强度）
        mutation_rate: 变异概率（每个参数被变异的概率）
        noise_scale: 基础噪声尺度

    Returns:
        变异后的个体（原地修改）
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not available for mutation")

    state = individual.model.state_dict()

    for key in state.keys():
        # 跳过非浮点参数
        if not state[key].dtype.is_floating_point:
            continue

        # 以mutation_rate概率变异每个参数
        if random.random() < mutation_rate:
            # 噪声强度随温度变化
            effective_scale = noise_scale * temperature
            noise = torch.randn_like(state[key]) * effective_scale
            state[key] = state[key] + noise

    individual.model.load_state_dict(state)
    return individual


# ============ 完整遗传流程 ============

def evolve_generation(
    population,  # Population对象
    temperature: float,
    crossover_alpha: float = 0.5,
    mutation_rate: float = 0.1
) -> List[Individual]:
    """
    执行一代遗传演化

    流程:
    1. 精英保留
    2. 锦标赛选择父代
    3. 交叉生成子代
    4. 变异

    Returns:
        新一代种群
    """
    config = population.config
    current_gen = population.current_generation + 1

    # 1. 精英保留
    population.sort_by_fitness()
    elites = population.get_elites()
    new_population = [elite.clone(f"gen{current_gen}_elite{i}", current_gen)
                      for i, elite in enumerate(elites)]

    # 2. 生成剩余个体
    offspring_count = config.population_size - len(new_population)

    for i in range(offspring_count):
        # 选择父代
        parent1 = tournament_selection(
            population.individuals,
            config.tournament_size,
            temperature
        )
        parent2 = tournament_selection(
            population.individuals,
            config.tournament_size,
            temperature
        )

        # 交叉
        child_id = f"gen{current_gen}_ind{len(new_population)}"
        child = crossover_average(
            parent1, parent2,
            alpha=crossover_alpha,
            new_id=child_id,
            generation=current_gen
        )

        # 变异
        child = mutate_gaussian(
            child,
            temperature=temperature,
            mutation_rate=mutation_rate
        )

        new_population.append(child)

    return new_population
