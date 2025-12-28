# train.py 实现计划 (续2)

## 七、使用示例

### 7.1 基础使用

```bash
# 使用默认配置训练
python train.py

# 使用自定义配置文件
python train.py --config configs/my_config.json

# 快速测试模式
python train.py --quick-test

# 指定实验名称
python train.py --experiment-name "exp_001"

# 设置随机种子（可复现）
python train.py --seed 42
```

### 7.2 恢复训练

```bash
# 从检查点恢复
python train.py --resume checkpoints/exp_001/checkpoint_gen_25.pth

# 从最佳检查点恢复
python train.py --resume checkpoints/exp_001/best_gen_30.pth
```

### 7.3 自定义配置

```bash
# 覆盖特定参数
python train.py --population-size 16 --num-generations 100 --num-workers 8
```

### 7.4 配置文件示例

创建 `configs/default.json`:

```json
{
  "population_size": 8,
  "elite_size": 2,
  "tournament_size": 3,
  "crossover_alpha": 0.5,
  "mutation_rate": 0.1,
  "initial_temperature": 1.0,
  "min_temperature": 0.1,
  "cooling_rate": 0.95,
  "num_generations": 50,
  "max_game_steps": 1000,
  "action_temperature": 1.0,
  "round_robin_until": 10,
  "tournament_games": 4,
  "num_workers": 4,
  "d_model": 128,
  "num_layers": 2,
  "nhead": 4,
  "dim_feedforward": 256,
  "dropout": 0.1,
  "checkpoint_dir": "checkpoints",
  "save_every": 5,
  "keep_best_n": 3,
  "log_dir": "logs",
  "log_every": 1,
  "seed": 42,
  "experiment_name": "ctf_evolution"
}
```

---

## 八、输出示例

### 8.1 控制台输出

```
============================================================
开始进化训练
============================================================
配置: TrainingConfig(population_size=8, num_generations=50, ...)
种群已创建: 8 个体

============================================================
世代 0/50
温度: 1.0000
============================================================
世代 0: 创建 28 场对战
开始执行 28 场对战 (并行度: 4)
进度: 28/28 (100.0%)
开始遗传演化...
✓ 遗传演化完成
[Gen   0] T=1.000 | Games= 28 | Fitness: 120.45 /  85.32 /  45.20 | Wins: L= 15 R= 12 D=  1 | Time: 45.2s
✓ 检查点已保存: checkpoints/ctf_evolution/best_gen_0.pth

============================================================
世代 1/50
温度: 0.9500
============================================================
...
```

### 8.2 日志文件

**training.log**:
```
[2025-12-28 10:30:15] 种群已创建: 8 个体
[Gen   0] T=1.000 | Games= 28 | Fitness: 120.45 /  85.32 /  45.20 | Wins: L= 15 R= 12 D=  1 | Time: 45.2s
[Gen   1] T=0.950 | Games= 28 | Fitness: 135.20 /  92.15 /  50.30 | Wins: L= 16 R= 11 D=  1 | Time: 90.5s
...
```

**training_log.csv**:
```csv
generation,timestamp,temperature,num_games,avg_steps,avg_duration_ms,best_fitness,avg_fitness,worst_fitness,l_wins,r_wins,draws
0,2025-12-28 10:30:15,1.0000,28,450.5,15234.2,120.45,85.32,45.20,15,12,1
1,2025-12-28 10:31:00,0.9500,28,465.3,15890.1,135.20,92.15,50.30,16,11,1
...
```

---

## 九、实现注意事项

### 9.1 内存管理

⚠️ **问题**：长时间训练可能导致内存泄漏

**解决方案**：
```python
import gc
import torch

# 在每个世代结束后
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### 9.2 异常处理

⚠️ **重要**：确保异常时保存检查点

```python
try:
    trainer.train()
except KeyboardInterrupt:
    # 用户中断，保存紧急检查点
    checkpoint_manager.save_checkpoint(
        current_generation,
        population,
        temperature,
        stats,
        is_best=False
    )
except Exception as e:
    # 其他异常，记录并保存
    logger.log_message(f"错误: {e}")
    checkpoint_manager.save_checkpoint(...)
    raise
```

### 9.3 进度监控

⚠️ **建议**：使用 `tqdm` 显示进度条

```python
from tqdm import tqdm

for generation in tqdm(range(num_generations), desc="训练进度"):
    ...
```

### 9.4 分布式训练（可选）

⚠️ **高级**：支持多机训练

```python
# 使用 Ray 或 Dask 进行分布式计算
import ray

@ray.remote
def run_game_remote(ind_l, ind_r, ...):
    return run_single_game(ind_l, ind_r, ...)

# 并行执行
futures = [run_game_remote.remote(...) for matchup in matchups]
results = ray.get(futures)
```

---

## 十、测试代码

### 10.1 test_training_config

```python
def test_training_config():
    """测试配置管理"""
    # 创建配置
    config = TrainingConfig(
        population_size=4,
        num_generations=10,
        experiment_name="test"
    )

    # 保存配置
    save_config(config, "test_config.json")

    # 加载配置
    loaded_config = load_config("test_config.json")

    assert loaded_config.population_size == 4
    assert loaded_config.num_generations == 10

    print("✓ 配置管理测试通过")
```

### 10.2 test_checkpoint_manager

```python
def test_checkpoint_manager():
    """测试检查点管理"""
    from population import Population, PopulationConfig
    from transformer_model import CTFTransformerConfig

    # 创建种群
    pop_config = PopulationConfig(population_size=4)
    model_config = CTFTransformerConfig(d_model=64, num_layers=1)
    population = Population(pop_config, model_config)

    # 创建检查点管理器
    manager = CheckpointManager("test_checkpoints", keep_best_n=2)

    # 保存检查点
    stats = {'best_fitness': 100.0, 'avg_fitness': 80.0}
    path = manager.save_checkpoint(0, population, 1.0, stats, is_best=True)

    # 加载检查点
    checkpoint = manager.load_checkpoint(path)

    assert checkpoint['generation'] == 0
    assert checkpoint['stats']['best_fitness'] == 100.0

    print("✓ 检查点管理测试通过")
```

### 10.3 test_full_training

```python
def test_full_training():
    """测试完整训练流程（小规模）"""
    config = TrainingConfig(
        population_size=4,
        num_generations=2,
        max_game_steps=50,
        num_workers=2,
        experiment_name="test_training"
    )

    trainer = EvolutionaryTrainer(config)

    try:
        trainer.train()
        print("✓ 完整训练测试通过")
    except Exception as e:
        print(f"✗ 训练失败: {e}")
        raise
```

---

## 十一、性能优化建议

### 11.1 并行度调优

```python
# 根据CPU核心数自动设置
import os
num_workers = max(1, os.cpu_count() - 2)
```

### 11.2 批量处理

```python
# 批量执行游戏，减少进程创建开销
batch_size = 10
for i in range(0, len(matchups), batch_size):
    batch = matchups[i:i+batch_size]
    results.extend(executor.execute_matchups(batch))
```

### 11.3 模型编译（PyTorch 2.0+）

```python
# 使用 torch.compile 加速推理
if hasattr(torch, 'compile'):
    model = torch.compile(model)
```

---

## 十二、文件结构

```
train.py
├── TrainingConfig (数据类)
├── load_config / save_config (函数)
├── CheckpointManager (类)
├── TrainingLogger (类)
├── EvolutionaryTrainer (主类)
├── parse_arguments (函数)
├── main (函数)
└── 测试函数
```

---

## 十三、预估代码量

- TrainingConfig: ~50行
- 配置管理: ~40行
- CheckpointManager: ~120行
- TrainingLogger: ~100行
- EvolutionaryTrainer: ~200行
- 命令行接口: ~80行
- 测试代码: ~100行

**总计**: ~690行

---

## 十四、依赖关系

```
train.py
├── population.py (Population, PopulationConfig)
├── transformer_model.py (CTFTransformer, CTFTransformerConfig)
├── reward_system.py (AdaptiveRewardSystem)
├── adversarial_trainer.py (AdversarialTrainer, AdaptiveMatchupStrategy)
├── genetic_ops.py (evolve_generation, AnnealingScheduler)
└── 标准库 (argparse, json, csv, pathlib, etc.)
```

**所有依赖已实现，可直接开发。**

---

## 十五、实现顺序建议

1. **第一步**: TrainingConfig + 配置管理（简单）
2. **第二步**: CheckpointManager（核心）
3. **第三步**: TrainingLogger（核心）
4. **第四步**: EvolutionaryTrainer（主类）
5. **第五步**: 命令行接口（简单）
6. **第六步**: 测试并验证

---

## 十六、预期训练时间

### 16.1 单场游戏

- 平均步数: 500步
- 单步耗时: ~30ms
- 单场耗时: ~15秒

### 16.2 单世代

- 种群大小: 8
- 循环赛: 28场
- 并行度: 4
- 世代耗时: ~105秒（1.75分钟）

### 16.3 完整训练

- 世代数: 50
- 总耗时: ~88分钟（1.5小时）
- 加速后: ~30-40分钟（优化并行度）

---

## 十七、后续扩展

### 17.1 可视化

```python
# 使用 matplotlib 绘制训练曲线
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("logs/ctf_evolution/training_log.csv")
plt.plot(df['generation'], df['best_fitness'], label='Best')
plt.plot(df['generation'], df['avg_fitness'], label='Average')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend()
plt.savefig('training_curve.png')
```

### 17.2 超参数搜索

```python
# 使用 Optuna 进行超参数优化
import optuna

def objective(trial):
    config = TrainingConfig(
        mutation_rate=trial.suggest_float('mutation_rate', 0.05, 0.2),
        cooling_rate=trial.suggest_float('cooling_rate', 0.90, 0.98),
        ...
    )
    trainer = EvolutionaryTrainer(config)
    trainer.train()
    return trainer.best_fitness_ever

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)
```

### 17.3 对战可视化

```python
# 保存对战回放
def save_game_replay(episode_result, filepath):
    replay = {
        'trajectory': episode_result.trajectory,
        'winner': episode_result.winner,
        'scores': (episode_result.l_score, episode_result.r_score)
    }
    with open(filepath, 'w') as f:
        json.dump(replay, f)
```

---

**实现日期**: 2025-12-28
**状态**: 📋 待实现
