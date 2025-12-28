# train.py 实现计划

## 一、核心职责

主训练脚本 - 整合所有模块，实现完整的进化训练流程，包括：
- 种群初始化
- 世代循环
- 对抗训练
- 遗传演化
- 检查点管理
- 日志记录
- 可视化

---

## 二、配置管理

### 2.1 TrainingConfig 数据类

```python
from dataclasses import dataclass, field
from typing import Optional

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
    cooling_rate: float = 0.95

    # 训练参数
    num_generations: int = 50
    max_game_steps: int = 1000
    action_temperature: float = 1.0

    # 对抗训练参数
    round_robin_until: int = 10
    tournament_games: int = 4
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
```

### 2.2 配置加载函数

```python
import json
from pathlib import Path

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
    config_dict = {
        k: v for k, v in config.__dict__.items()
        if not k.startswith('_')
    }

    with open(save_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
```

---

## 三、检查点管理

### 3.1 CheckpointManager 类

```python
import os
import torch
from typing import List, Dict, Any
from pathlib import Path

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
        checkpoint = {
            'generation': generation,
            'temperature': temperature,
            'stats': stats,
            'population_state': population.get_state_dict(),
            'best_fitness': stats['best_fitness']
        }

        # 生成文件名
        if is_best:
            filename = f"best_gen_{generation}.pth"
        else:
            filename = f"checkpoint_gen_{generation}.pth"

        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)

        print(f"✓ 检查点已保存: {filepath}")

        # 更新最佳检查点列表
        if is_best:
            self._update_best_checkpoints(stats['best_fitness'], filepath)

        return str(filepath)

    def _update_best_checkpoints(self, fitness: float, filepath: Path) -> None:
        """更新最佳检查点列表"""
        self.best_checkpoints.append((fitness, filepath))
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)

        # 删除多余的检查点
        if len(self.best_checkpoints) > self.keep_best_n:
            _, old_path = self.best_checkpoints.pop()
            if old_path.exists():
                old_path.unlink()
                print(f"删除旧检查点: {old_path}")

    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path)
        print(f"✓ 检查点已加载: {checkpoint_path}")
        return checkpoint

    def get_latest_checkpoint(self) -> Optional[str]:
        """获取最新的检查点路径"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_gen_*.pth"))
        if not checkpoints:
            return None

        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)
```

---

## 四、日志系统

### 4.1 TrainingLogger 类

```python
import csv
import time
from datetime import datetime
from typing import Dict, Any, List

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
            'l_wins', 'r_wins', 'draws'
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
            stats['draws']
        ])
        self.csv_file.flush()

        # 写入文本日志
        elapsed = time.time() - self.start_time
        log_msg = (
            f"[Gen {generation:3d}] "
            f"T={temperature:.3f} | "
            f"Games={stats['num_games']:3d} | "
            f"Fitness: {stats['best_fitness']:6.2f} / "
            f"{stats['avg_fitness']:6.2f} / "
            f"{stats['worst_fitness']:6.2f} | "
            f"Wins: L={stats['l_wins']:3d} R={stats['r_wins']:3d} "
            f"D={stats['draws']:2d} | "
            f"Time: {elapsed:.1f}s"
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
```

---
