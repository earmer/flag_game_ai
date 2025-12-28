# 种群-对抗学习-退火遗传-Transformer 对局AI系统设计

## 1. 系统总体架构

四层混合架构：

```
┌─────────────────────────────────────────────────────────┐
│           Population Manager (种群管理器)                │
│  - 维护8个AI个体（每个个体是一个Transformer模型）       │
│  - 跟踪适应度评分和世代演化                              │
└─────────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────┐
│      Adversarial Training Engine (对抗训练引擎)          │
│  - 自我对弈：种群内个体互相对战                          │
│  - 对抗学习：通过竞争提升策略                            │
└─────────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────┐
│   Simulated Annealing GA (退火遗传算法)                  │
│  - 选择：锦标赛选择（k=3）                               │
│  - 交叉：模型参数混合（alpha=0.5）                       │
│  - 变异：温度控制的噪声注入                              │
└─────────────────────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────────────────────┐
│      Transformer Policy Network (策略网络)               │
│  - 输入：堆叠状态矩阵 (20×20(map)×60通道) [3 ticks × 20] │
│  - 输出：3个玩家的动作概率分布 (5个动作)                │
└─────────────────────────────────────────────────────────┘
```

**核心优势**：
- 种群多样性：避免陷入局部最优
- 对抗学习：通过竞争自然涌现复杂策略
- 退火遗传：平衡探索与利用
- Transformer：捕捉全局空间关系

---

## 2. 种群管理模块

### 2.1 种群配置

```python
POPULATION_CONFIG = {
    "population_size": 8,        # 种群大小
    "elite_size": 2,             # 精英保留数量（前25%）
    "tournament_size": 3,        # 锦标赛选择大小
    "initial_temperature": 1.0,  # 初始退火温度
    "min_temperature": 0.1,      # 最小温度
    "cooling_rate": 0.95,        # 冷却率（每代降温5%）
    "generations": 50            # 总代数
}
```

### 2.2 个体表示

```python
class Individual:
    def __init__(self, model_id, transformer_model):
        self.id = model_id
        self.model = transformer_model  # Transformer策略网络
        self.fitness = 0.0              # 适应度评分
        self.wins = 0                   # 胜场数
        self.losses = 0                 # 负场数
        self.flags_captured = 0         # 捕获旗帜数
        self.age = 0                    # 世代年龄
```

### 2.3 适应度评估

多维度评分系统：

```python
def calculate_fitness(game_result):
    fitness = 0.0

    # 1. 胜负奖励（主要指标）
    if game_result['won']:
        fitness += 100.0
    else:
        fitness -= 50.0

    # 2. 旗帜捕获奖励
    fitness += game_result['flags_captured'] * 20.0

    # 3. 存活时间奖励
    fitness += game_result['survival_rate'] * 10.0

    # 4. 标记敌人奖励
    fitness += game_result['enemies_tagged'] * 5.0

    # 5. 时间惩罚（鼓励快速获胜）
    fitness -= game_result['game_duration'] * 0.1

    return fitness
```

---

## 3. 对抗学习训练流程

### 3.1 对战配对策略

**循环赛 + 锦标赛混合模式**：

- **早期世代（0-10代）**：循环赛，每个个体与所有其他个体对战一次
- **中后期（11-50代）**：锦标赛，每个个体随机对战3-5次

### 3.2 自我对弈训练循环

```python
def adversarial_training_epoch(population, game_engine):
    # 1. 重置临时统计
    for individual in population:
        individual.epoch_wins = 0
        individual.epoch_fitness = 0.0

    # 2. 创建对战配对
    matchups = create_matchups(population, current_generation)

    # 3. 执行所有对战（可并行）
    for agent1, agent2 in matchups:
        result = run_game(agent1, agent2, game_engine)
        update_individual_stats(agent1, result['team_L'])
        update_individual_stats(agent2, result['team_R'])

    # 4. 计算平均适应度
    for individual in population:
        individual.fitness = individual.epoch_fitness / max(1, games_played)

    return results
```

---

## 4. 退火遗传算法

### 4.1 选择算子（锦标赛选择）

```python
def tournament_selection(population, tournament_size=3):
    tournament = random.sample(population, tournament_size)
    winner = max(tournament, key=lambda ind: ind.fitness)
    return winner
```

### 4.2 交叉算子（权重平均）

```python
def crossover(parent1, parent2, alpha=0.5):
    child_model = create_new_transformer()
    state1 = parent1.model.state_dict()
    state2 = parent2.model.state_dict()

    child_state = {}
    for key in state1.keys():
        child_state[key] = alpha * state1[key] + (1 - alpha) * state2[key]

    child_model.load_state_dict(child_state)
    return Individual(new_id(), child_model)
```

### 4.3 变异算子（温度控制）

**关键创新：噪声强度随温度降低而减小**

```python
def mutate(individual, temperature, mutation_rate=0.1):
    state = individual.model.state_dict()

    for key in state.keys():
        if random.random() < mutation_rate:
            noise_scale = temperature * 0.01
            noise = torch.randn_like(state[key]) * noise_scale
            state[key] = state[key] + noise

    individual.model.load_state_dict(state)
    return individual
```

### 4.4 退火调度

```python
def update_temperature(initial_temp, generation, cooling_rate):
    return initial_temp * (cooling_rate ** generation)
```

**温度演化曲线**：
- Gen 0: 1.00
- Gen 10: 0.60
- Gen 25: 0.28
- Gen 40: 0.13
- Gen 50: 0.08

---

## 5. Transformer模型架构

### 5.1 轻量级Vision Transformer

```python
class CTFTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 卷积编码器（降维）
        # 输入通道为 60 (20 channels * 3 history ticks)
        self.conv_encoder = nn.Sequential(
            nn.Conv2d(60, 64, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 20x20 -> 10x10
        )

        # 2. 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, 128))

        # 3. Transformer编码器（2层）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # 4. 多智能体输出头（3个玩家）
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 5)  # 5个动作: up/down/left/right/stay
            ) for _ in range(3)
        ])

    def forward(self, x):
        # x: (batch, 60, 20, 20)
        features = self.conv_encoder(x)  # (batch, 128, 10, 10)
        features = features.flatten(2).transpose(1, 2)  # (batch, 100, 128)
        features = features + self.pos_embedding
        context = self.transformer(features)  # (batch, 100, 128)
        global_context = context.mean(dim=1)  # (batch, 128)
        actions = [head(global_context) for head in self.action_heads]
        return actions  # List of 3 tensors: (batch, 5)
```

**模型规模**：
- 参数量：约500K
- 输入：(batch, 20, 20, 20) - 20通道游戏状态矩阵
- 输出：3个玩家的动作logits，每个(batch, 5)

---

## 6. 动态演化奖励系统

### 6.1 稀疏奖励（Sparse Rewards）

**特点**：只在关键事件发生时给予奖励

```python
class SparseRewardCalculator:
    def calculate(self, game_state, prev_state, team):
        reward = 0.0

        # 1. 游戏胜利/失败
        if game_state['game_over']:
            reward += 1000.0 if won else -500.0

        # 2. 捕获旗帜
        flags_captured = game_state['my_flags_in_target'] - prev_state['my_flags_in_target']
        reward += flags_captured * 200.0

        # 3. 丢失旗帜
        flags_lost = game_state['opp_flags_in_target'] - prev_state['opp_flags_in_target']
        reward -= flags_lost * 200.0

        # 4. 救出队友
        prisoners_freed = prev_state['num_my_prisoners'] - game_state['num_my_prisoners']
        reward += prisoners_freed * 50.0

        return reward
```

**优点**：目标明确，不会误导
**缺点**：学习困难，早期几乎无反馈

### 6.2 密集奖励（Dense Rewards）

**特点**：每步都提供反馈

```python
class DenseRewardCalculator:
    def calculate(self, game_state, prev_state, team, player_id):
        reward = 0.0
        player = game_state['players'][player_id]
        prev_player = prev_state['players'][player_id]

        # 1. 移动奖励（鼓励探索）
        if player['pos'] != prev_player['pos']:
            reward += 0.1

        # 2. 接近目标奖励
        if player['has_flag']:
            # 持旗时：接近家目标
            dist_to_target = distance(player['pos'], game_state['my_target'])
            prev_dist = distance(prev_player['pos'], game_state['my_target'])
            reward += 2.0 if dist_to_target < prev_dist else -1.0
        else:
            # 无旗时：接近敌方旗帜
            nearest_flag = find_nearest_flag(player['pos'], game_state['opp_flags'])
            if nearest_flag:
                dist = distance(player['pos'], nearest_flag)
                prev_dist = distance(prev_player['pos'], nearest_flag)
                reward += 1.0 if dist < prev_dist else 0.0

        # 3. 拾取旗帜奖励
        if player['has_flag'] and not prev_player['has_flag']:
            reward += 50.0

        # 4. 标记敌人奖励
        enemies_tagged = count_enemies_tagged(game_state, prev_state, team)
        reward += enemies_tagged * 10.0

        # 5. 被标记惩罚
        if player['in_prison'] and not prev_player['in_prison']:
            reward -= 30.0

        # 6. 安全区域奖励（持旗时）
        if player['has_flag'] and player['is_safe']:
            reward += 0.5

        # 7. 团队协作奖励（改进版）
        # 保持适当距离：不太近（避免聚堆）也不太远（保持支援）
        teammate_dist = get_nearest_teammate_distance(player, game_state['players'])
        if 3 <= teammate_dist <= 8:  # 最佳距离范围
            reward += 0.3
        elif teammate_dist < 2:  # 太近，聚堆
            reward -= 0.1

        return reward
```

**优点**：学习快速，早期有效
**缺点**：可能过度拟合短期行为

### 6.3 课程学习奖励演化

**核心创新**：奖励函数随训练进度动态调整

```python
class CurriculumRewardScheduler:
    """
    训练分为4个阶段：
    Stage 1 (Gen 0-10):   密集奖励80% + 稀疏奖励20%
    Stage 2 (Gen 11-25):  线性过渡（80%→30% 密集）
    Stage 3 (Gen 26-40):  密集奖励10% + 稀疏奖励90%
    Stage 4 (Gen 41-50):  纯稀疏奖励100%
    """

    def _get_weights(self, generation):
        if generation <= 10:
            return 0.8, 0.2  # (dense_weight, sparse_weight)
        elif generation <= 25:
            progress = (generation - 10) / 15
            dense_weight = 0.8 - 0.5 * progress
            sparse_weight = 0.2 + 0.5 * progress
            return dense_weight, sparse_weight
        elif generation <= 40:
            return 0.1, 0.9
        else:
            return 0.0, 1.0
```

**演化曲线**：

```
密集奖励权重
1.0 |████████╗
0.8 |        ║
    |        ╚═══════╗
0.5 |                ║
    |                ╚═══════════╗
0.2 |                            ║
0.0 |____________________________╚═══════════
    0    10    25    40    50  (Generation)
```

### 6.4 奖励塑形（Reward Shaping）

**目的**：通过领域知识引导学习

```python
class RewardShaping:
    def __init__(self, generation):
        # 塑形强度随训练衰减
        self.shaping_strength = max(0.0, 1.0 - generation / 30)

    def calculate_shaping_reward(self, game_state, team, player_id):
        if self.shaping_strength < 0.01:
            return 0.0

        reward = 0.0
        player = game_state['players'][player_id]

        # 1. 角色分工奖励
        if player_id == 0:  # 攻击手
            if not player['is_safe']:
                reward += 1.0
            if player['has_flag']:
                reward += 2.0
        elif player_id == 1:  # 防守者
            if player['is_safe']:
                reward += 1.0
            # 接近入侵者
            intruders = [p for p in game_state['opp_players'] if not p['is_safe']]
            if intruders:
                nearest_dist = min(distance(player['pos'], i['pos']) for i in intruders)
                reward += max(0, 2.0 - nearest_dist * 0.1)
        elif player_id == 2:  # 支援
            # 接近持旗队友
            teammates_with_flag = [p for p in game_state['my_players'] if p['has_flag']]
            if teammates_with_flag:
                nearest_dist = min(distance(player['pos'], t['pos']) for t in teammates_with_flag)
                reward += max(0, 1.5 - nearest_dist * 0.1)

        # 2. 位置价值奖励
        # 中线控制
        mid_x = game_state['map_width'] // 2
        if abs(player['pos'][0] - mid_x) < 3:
            reward += 0.5

        # 监狱附近（救援）
        if game_state['num_my_prisoners'] > 0:
            if distance(player['pos'], game_state['my_prison']) < 5:
                reward += 1.0

        # 3. 时机奖励
        score_diff = game_state['my_flags_in_target'] - game_state['opp_flags_in_target']
        if score_diff > 0:  # 领先时防守
            if player['is_safe']:
                reward += 0.5
        else:  # 落后时进攻
            if not player['is_safe']:
                reward += 0.5

        return reward * self.shaping_strength
```

**塑形强度衰减**：
- Gen 0: 1.00 (强引导)
- Gen 15: 0.50
- Gen 30: 0.00 (完全自主)

### 6.5 完整奖励系统

```python
class AdaptiveRewardSystem:
    def __init__(self):
        self.dense_calc = DenseRewardCalculator()
        self.sparse_calc = SparseRewardCalculator()
        self.curriculum = CurriculumRewardScheduler()
        self.shaping = None

    def reset_for_generation(self, generation):
        self.shaping = RewardShaping(generation)
        self.current_generation = generation

    def calculate_step_reward(self, game_state, prev_state, team, player_id):
        # 1. 基础奖励（密集+稀疏混合）
        base_reward, info = self.curriculum.get_reward(
            game_state, prev_state, team, player_id,
            self.current_generation
        )

        # 2. 塑形奖励
        shaping_reward = self.shaping.calculate_shaping_reward(
            game_state, team, player_id
        )

        # 3. 总奖励
        total_reward = base_reward + shaping_reward

        return total_reward, {
            'dense': info['dense'],
            'sparse': info['sparse'],
            'shaping': shaping_reward,
            'total': total_reward
        }
```

---

## 7. 完整训练Pipeline

### 7.1 主训练循环

```python
def train_population(config):
    # 1. 初始化
    population = initialize_population(config['population_size'])
    reward_system = AdaptiveRewardSystem()
    temperature = config['initial_temperature']

    # 2. 世代循环
    for generation in range(config['generations']):
        print(f"Generation {generation}, Temperature: {temperature:.3f}")

        # 重置奖励系统
        reward_system.reset_for_generation(generation)

        # 对抗训练
        adversarial_training_epoch(population, game_engine, reward_system)

        # 排序种群
        population.sort(key=lambda ind: ind.fitness, reverse=True)

        # 保存最佳个体
        save_checkpoint(population[0], f"gen_{generation}_best.pth")

        # 精英保留
        elites = population[:config['elite_size']]

        # 生成新一代
        new_population = elites.copy()
        while len(new_population) < config['population_size']:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            child = crossover(parent1, parent2)
            child = mutate(child, temperature)
            new_population.append(child)

        population = new_population

        # 降低温度
        temperature = update_temperature(
            config['initial_temperature'],
            generation,
            config['cooling_rate']
        )

        log_generation_stats(generation, population)

    return population[0]
```

### 7.2 初始化策略

**迁移学习 + 随机初始化混合**：

```python
def initialize_population(size):
    population = []

    # 策略1: 从决策树AI迁移学习（50%）
    if os.path.exists('tree_models/tree_l.json'):
        tree_data = collect_tree_rollouts(num_games=100)
        base_model = pretrain_from_tree(tree_data)

        for i in range(size // 2):
            model = clone_and_perturb(base_model, noise_scale=0.1)
            population.append(Individual(i, model))

    # 策略2: 随机初始化（50%，保持多样性）
    for i in range(size // 2, size):
        model = CTFTransformer()
        population.append(Individual(i, model))

    return population
```

---

## 8. 超参数推荐

```python
RECOMMENDED_CONFIG = {
    # 种群参数
    "population_size": 8,
    "elite_size": 2,
    "tournament_size": 3,

    # 遗传参数
    "crossover_alpha": 0.5,
    "mutation_rate": 0.1,
    "initial_temperature": 1.0,
    "min_temperature": 0.1,
    "cooling_rate": 0.95,

    # 训练参数
    "generations": 50,
    "games_per_matchup": 1,
    "max_game_steps": 1000,

    # 模型参数
    "history_len": 3,
    "d_model": 128,
    "d_model": 128,
    "nhead": 4,
    "num_layers": 2,
    "dropout": 0.1,

    # 奖励参数
    "reward_gamma": 0.99,  # 折扣因子
    "reward_clip": 10.0,   # 奖励裁剪范围
}
```

---

## 9. 实现路线图

### 阶段1：基础框架（1-2天）
- [ ] 实现 `CTFTransformer` 模型
- [ ] 实现游戏接口封装
- [ ] 测试单个模型运行游戏

### 阶段2：种群管理（1天）
- [ ] 实现 `Individual` 类
- [ ] 实现种群初始化
- [ ] 实现适应度计算

### 阶段3：遗传算子（1天）
- [ ] 实现选择、交叉、变异
- [ ] 实现退火温度调度
- [ ] 单元测试

### 阶段4：奖励系统（1-2天）
- [ ] 实现稀疏奖励计算器
- [ ] 实现密集奖励计算器
- [ ] 实现课程学习调度器
- [ ] 实现奖励塑形机制

### 阶段5：对抗训练（2-3天）
- [ ] 实现对战配对逻辑
- [ ] 实现并行游戏执行
- [ ] 集成所有模块

### 阶段6：训练和调优（持续）
- [ ] 运行完整训练
- [ ] 调整超参数
- [ ] 性能优化

---

## 10. 预期效果

- **训练时间**：8个体×50代×平均20场/代 = 约8000场游戏
- **单场游戏**：约30秒（1000步上限）
- **总训练时间**：约67小时（可并行加速到20小时）
- **性能提升**：预期在20代后超越决策树AI

---

## 11. 关键技术要点

### 11.1 状态表示转换

```python
from collections import deque
import numpy as np
import torch

class StateStacker:
    def __init__(self, history_len=3, channels=20):
        self.history_len = history_len
        self.channels = channels
        # 使用 deque 存储最近的 history_len 帧
        self.buffer = deque(maxlen=history_len)
        self.converter = CTFMatrixConverter()

    def reset(self, init_req):
        """游戏开始时重置，并初始化静态地图"""
        self.converter.initialize_static_map(init_req)
        self.buffer.clear()

    def get_stacked_tensor(self, status_req, team):
        # 1. 获取当前帧矩阵 (20, 20, 20) -> (20, 20, 20)
        # 注意：通常 PyTorch 需要 (Channels, H, W)，确保 converter 输出或后续转置正确
        # 假设 converter 输出为 (20, 20, 20) [C, H, W]
        current_matrix = self.converter.convert_to_matrix(status_req, team)
        
        # 2. 冷启动处理：如果 buffer 为空，复制当前帧填充
        if len(self.buffer) == 0:
            for _ in range(self.history_len):
                self.buffer.append(current_matrix)
        else:
            self.buffer.append(current_matrix)

        # 3. 堆叠通道
        # list of (20, 20, 20) -> np.concatenate -> (60, 20, 20)
        stacked_state = np.concatenate(list(self.buffer), axis=0)
        
        # 4. 转为 Tensor 并增加 Batch 维度 -> (1, 60, 20, 20)
        tensor = torch.FloatTensor(stacked_state).unsqueeze(0)
        return tensor
```

### 11.2 动作采样

```python
def sample_action(logits, allowed_actions, temperature=1.0):
    probs = F.softmax(logits / temperature, dim=-1)

    # 过滤非法动作
    mask = torch.zeros_like(probs)
    for action_idx in allowed_actions:
        mask[action_idx] = 1.0
    probs = probs * mask
    probs = probs / probs.sum()

    action_idx = torch.multinomial(probs, 1).item()
    return ACTION_MAP[action_idx]
```

### 11.3 并行游戏执行

```python
from concurrent.futures import ProcessPoolExecutor

def parallel_games(matchups, num_workers=4):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(run_game, a1, a2) for a1, a2 in matchups]
        results = [f.result() for f in futures]
    return results
```

---

## 12. 项目文件结构

```
CTF/backend/
├── evolutionary_ai/
│   ├── __init__.py
│   ├── transformer_model.py      # Transformer模型定义
│   ├── population.py              # 种群管理
│   ├── genetic_ops.py             # 遗传算子
│   ├── reward_system.py           # 奖励系统
│   ├── adversarial_trainer.py    # 对抗训练引擎
│   ├── game_interface.py         # 游戏接口封装
│   └── train.py                  # 主训练脚本
├── transformer/
│   ├── checkpoints/               # 模型检查点
│   ├── logs/                      # 训练日志
│   └── CLAUDE.MD                  # 本文档
└── lib/
    ├── game_engine.py             # 现有游戏引擎
    ├── matrix_util.py             # 矩阵转换工具
    └── ...
```

---

## 13. 快速启动

### 测试Transformer模型
```bash
python evolutionary_ai/test_transformer.py
```

### 单场游戏测试
```bash
python evolutionary_ai/test_single_game.py
```

### 小规模训练（快速验证）
```bash
python evolutionary_ai/train.py --population 4 --generations 10
```

### 完整训练
```bash
python evolutionary_ai/train.py --config recommended_config.json
```

---

## 14. 核心创新点

1. **退火控制变异强度**：早期大胆探索，后期精细优化
2. **对抗驱动进化**：无需人工标注，自然涌现策略
3. **课程学习奖励**：密集→稀疏渐进过渡，平衡学习速度和策略质量
4. **奖励塑形衰减**：初期强引导，后期完全自主
5. **轻量级Transformer**：快速训练，适合遗传算法
6. **迁移学习初始化**：从决策树AI获取先验知识
7. **团队协作距离控制**：保持3-8格最佳距离，避免聚堆也不失联

