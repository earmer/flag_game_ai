"""
stage_config.py

4阶段训练配置系统
- TrainingStage: 训练阶段枚举
- StageConfig: 单阶段配置
- 配置生成函数
"""

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Dict, Any, List
import math


# ============================================================
# 训练阶段枚举
# ============================================================

class TrainingStage(IntEnum):
    """训练阶段枚举"""
    FOUNDATION = 1      # Gen 0-160: 基础技能
    OPTIMIZATION = 2    # Gen 161-300: 对抗成型
    COMPETITION = 3     # Gen 301-600: 精英竞争
    MASTERY = 4        # Gen 601-800: 持续优化


# ============================================================
# 阶段配置
# ============================================================

@dataclass
class StageConfig:
    """单个阶段的配置"""

    # 基本信息
    stage: TrainingStage
    name: str

    # 世代范围
    start_gen: int
    end_gen: int
    min_generations: int        # 最小世代数（下界）
    max_generations: int        # 最大世代数（上界，3×最小）

    # 种群参数
    population_size: int
    elite_size: int
    tournament_size: int

    # 对局参数
    games_per_individual: int   # 每个体每代对局数
    max_game_steps: int         # 每局最大步数

    # 遗传参数
    initial_temperature: float
    min_temperature: float          # 阶段最低温度
    cooling_rate: float
    mutation_rate: float

    # 奖励参数
    dense_weight: float
    sparse_weight: float

    # 以下为有默认值的参数
    crossover_alpha: float = 0.5
    round_per_game: int = 1     # 每个配对的对战轮次（默认1轮）

    # HoF参数
    hof_sample_rate: float = 0.0  # HoF对手采样率

    # 评测参数
    benchmark_interval: int = 20    # 每N代评测一次
    min_win_rate: float = 0.70      # 晋级所需胜率下界
    min_benchmark_games: int = 100  # 评测最小场次
    auto_advance: bool = False      # 自动晋级（快速测试用）

    # 并行参数
    num_workers: int = 8

    # 固定旗帜游戏配置
    use_fixed_flag_games: bool = True
    fixed_flag_game_ratio: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'stage': self.stage.value,
            'name': self.name,
            'start_gen': self.start_gen,
            'end_gen': self.end_gen,
            'min_generations': self.min_generations,
            'max_generations': self.max_generations,
            'population_size': self.population_size,
            'elite_size': self.elite_size,
            'tournament_size': self.tournament_size,
            'games_per_individual': self.games_per_individual,
            'max_game_steps': self.max_game_steps,
            'round_per_game': self.round_per_game,
            'initial_temperature': self.initial_temperature,
            'min_temperature': self.min_temperature,
            'cooling_rate': self.cooling_rate,
            'mutation_rate': self.mutation_rate,
            'crossover_alpha': self.crossover_alpha,
            'dense_weight': self.dense_weight,
            'sparse_weight': self.sparse_weight,
            'hof_sample_rate': self.hof_sample_rate,
            'benchmark_interval': self.benchmark_interval,
            'min_win_rate': self.min_win_rate,
            'min_benchmark_games': self.min_benchmark_games,
            'auto_advance': self.auto_advance,
            'num_workers': self.num_workers,
            'use_fixed_flag_games': self.use_fixed_flag_games,
            'fixed_flag_game_ratio': self.fixed_flag_game_ratio
        }


# ============================================================
# 置信区间计算
# ============================================================

def calculate_win_rate_with_ci(
    wins: int,
    total: int,
    confidence: float = 0.95
) -> tuple[float, float, float]:
    """
    计算胜率及其置信区间（简化版正态近似）

    Args:
        wins: 胜场数
        total: 总场数
        confidence: 置信度（默认95%）

    Returns:
        (win_rate, lower_bound, upper_bound)
    """
    if total == 0:
        return 0.0, 0.0, 0.0

    p = wins / total

    # 标准误
    se = math.sqrt(p * (1 - p) / total)

    # z值（95%置信度）
    z = 1.96 if confidence == 0.95 else 1.645

    lower = max(0.0, p - z * se)
    upper = min(1.0, p + z * se)

    return p, lower, upper


# ============================================================
# 配置生成函数
# ============================================================

def create_stage_configs() -> Dict[TrainingStage, StageConfig]:
    """
    创建4个阶段的配置

    阶段边界：160/300/600/800代
    种群规模：8/24/64/32

    Returns:
        阶段配置字典
    """
    configs = {}

    # Stage 1: Foundation (基础技能)
    configs[TrainingStage.FOUNDATION] = StageConfig(
        stage=TrainingStage.FOUNDATION,
        name="基础技能",
        start_gen=0,
        end_gen=160,
        min_generations=160,
        max_generations=480,  # 3×160
        population_size=8,
        elite_size=2,
        tournament_size=3,
        games_per_individual=24,
        max_game_steps=500,  # 基础阶段使用较短步数，加快迭代
        round_per_game=5,  # 每个配对3轮对战
        initial_temperature=1.5,
        min_temperature=0.1,
        cooling_rate=0.98,
        mutation_rate=0.15,
        crossover_alpha=0.5,
        dense_weight=0.8,
        sparse_weight=0.2,
        hof_sample_rate=0.0,  # Stage 1不使用HoF
        benchmark_interval=20,
        min_win_rate=0.70,
        min_benchmark_games=100,
        num_workers=8
    )

    # Stage 2: Optimization (对抗成型)
    configs[TrainingStage.OPTIMIZATION] = StageConfig(
        stage=TrainingStage.OPTIMIZATION,
        name="对抗成型",
        start_gen=161,
        end_gen=300,
        min_generations=140,  # 300-161+1
        max_generations=420,  # 3×140
        population_size=24,
        elite_size=6,
        tournament_size=4,
        games_per_individual=24,
        max_game_steps=500,  # 统一使用500步
        round_per_game=3,  # 每个配对2轮对战
        initial_temperature=1.0,
        min_temperature=0.15,
        cooling_rate=0.99,
        mutation_rate=0.12,
        crossover_alpha=0.5,
        dense_weight=0.5,
        sparse_weight=0.5,
        hof_sample_rate=0.2,  # 20% HoF采样
        benchmark_interval=20,
        min_win_rate=0.60,
        min_benchmark_games=100,
        num_workers=12
    )

    # Stage 3: Competition (精英竞争)
    configs[TrainingStage.COMPETITION] = StageConfig(
        stage=TrainingStage.COMPETITION,
        name="精英竞争",
        start_gen=301,
        end_gen=600,
        min_generations=300,  # 600-301+1
        max_generations=900,  # 3×300
        population_size=64,
        elite_size=16,
        tournament_size=5,
        games_per_individual=24,
        max_game_steps=500,  # 统一使用500步
        round_per_game=3,  # 每个配对1轮对战
        initial_temperature=0.8,
        min_temperature=0.2,
        cooling_rate=0.995,
        mutation_rate=0.08,
        crossover_alpha=0.5,
        dense_weight=0.2,
        sparse_weight=0.8,
        hof_sample_rate=0.3,  # 30% HoF采样
        benchmark_interval=30,
        min_win_rate=0.55,
        min_benchmark_games=150,
        num_workers=16
    )

    # Stage 4: Mastery (持续优化)
    configs[TrainingStage.MASTERY] = StageConfig(
        stage=TrainingStage.MASTERY,
        name="持续优化",
        start_gen=601,
        end_gen=800,
        min_generations=200,  # 800-601+1
        max_generations=600,  # 3×200
        population_size=32,
        elite_size=8,
        tournament_size=6,
        games_per_individual=24,
        max_game_steps=500,  # 统一使用500步
        round_per_game=3,  # 每个配对1轮对战
        initial_temperature=0.5,
        min_temperature=0.25,
        cooling_rate=0.998,
        mutation_rate=0.05,
        crossover_alpha=0.5,
        dense_weight=0.0,
        sparse_weight=1.0,
        hof_sample_rate=0.4,  # 40% HoF采样
        benchmark_interval=50,
        min_win_rate=0.50,
        min_benchmark_games=200,
        num_workers=20
    )

    return configs


def create_quick_test_configs() -> Dict[TrainingStage, StageConfig]:
    """
    创建快速测试配置（5分钟内完成）

    配置：4个体×3代×4场
    特点：运行基准测试但自动晋级（不检查胜率门槛）

    Returns:
        快速测试阶段配置字典
    """
    configs = {}

    # 所有阶段使用相同的快速测试配置
    for stage in TrainingStage:
        configs[stage] = StageConfig(
            stage=stage,
            name=f"快速测试-{stage.name}",
            start_gen=0,
            end_gen=3,
            min_generations=1,
            max_generations=9,
            population_size=3,
            elite_size=1,
            tournament_size=2,
            games_per_individual=4,
            max_game_steps=100,  # 快速测试使用100步
            round_per_game=2,
            initial_temperature=1.0,
            min_temperature=0.1,
            cooling_rate=0.95,
            mutation_rate=0.1,
            crossover_alpha=0.5,
            dense_weight=0.5,
            sparse_weight=0.5,
            hof_sample_rate=0.0 if stage == TrainingStage.FOUNDATION else 0.2,
            benchmark_interval=2,
            min_win_rate=0.50,
            min_benchmark_games=20,
            auto_advance=True,  # 关键：自动晋级
            num_workers=4
        )

    return configs


