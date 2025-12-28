# game_interface.py 实现计划

## 一、核心职责

将 **sim_env.py**、**encoding.py**、**reward_system.py** 三个模块串联起来，提供完整的对战执行接口。

---

## 二、主要类和函数设计

### 2.1 GameInterface 类

**职责**：封装单局游戏的完整执行流程

```python
class GameInterface:
    """游戏接口 - 执行单局对战并收集训练数据"""

    def __init__(
        self,
        sim: CTFSim,
        geometry: Geometry,
        reward_system: AdaptiveRewardSystem,
        max_tokens: int = 32,
        max_steps: int = 1000
    ):
        """
        Args:
            sim: CTFSim游戏模拟器实例
            geometry: Geometry对象（用于encoding）
            reward_system: 奖励系统实例
            max_tokens: token序列最大长度
            max_steps: 单局最大步数
        """
        pass

    def run_episode(
        self,
        agent_l: PolicyAgent,
        agent_r: PolicyAgent,
        record_trajectory: bool = True
    ) -> EpisodeResult:
        """
        执行一局完整对战

        Args:
            agent_l: L队智能体
            agent_r: R队智能体
            record_trajectory: 是否记录完整轨迹

        Returns:
            EpisodeResult: 对战结果（包含奖励、统计等）
        """
        pass
```

---

### 2.2 PolicyAgent 抽象类

**职责**：定义智能体接口，支持多种策略类型

```python
class PolicyAgent(ABC):
    """策略智能体抽象基类"""

    @abstractmethod
    def select_actions(
        self,
        status_dict: Dict[str, Any],
        geometry: Geometry
    ) -> Dict[str, str]:
        """
        根据当前状态选择动作

        Args:
            status_dict: sim_env.status()的输出
            geometry: Geometry对象

        Returns:
            动作字典 {"L0": "up", "L1": "right", "L2": ""}
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """重置智能体状态（用于新局开始）"""
        pass
```

---

### 2.3 TransformerAgent 类

**职责**：基于Transformer模型的智能体实现

```python
class TransformerAgent(PolicyAgent):
    """Transformer策略智能体"""

    def __init__(
        self,
        model: CTFTransformer,
        team: str,
        max_tokens: int = 32,
        temperature: float = 1.0
    ):
        """
        Args:
            model: CTFTransformer模型
            team: 队伍名称 "L" or "R"
            max_tokens: token序列最大长度
            temperature: 动作采样温度
        """
        pass

    def select_actions(
        self,
        status_dict: Dict[str, Any],
        geometry: Geometry
    ) -> Dict[str, str]:
        """
        使用Transformer模型选择动作

        流程:
        1. status_dict -> encode_status_for_team() -> tokens
        2. tokens -> model.forward() -> action_logits
        3. action_logits -> sample_actions() -> action_dict
        """
        pass
```

---

### 2.4 EpisodeResult 数据类

**职责**：存储单局对战的完整结果

```python
@dataclass
class EpisodeResult:
    """单局对战结果"""

    # 基础信息
    winner: Optional[str]           # "L"/"R"/None
    l_score: int
    r_score: int
    steps: int
    duration_ms: float

    # 奖励信息
    l_total_reward: float
    r_total_reward: float
    l_reward_breakdown: Dict[str, float]
    r_reward_breakdown: Dict[str, float]

    # 统计信息
    l_flags_captured: int
    r_flags_captured: int
    l_enemies_tagged: int
    r_enemies_tagged: int
    l_avg_survival_rate: float
    r_avg_survival_rate: float

    # 轨迹数据（可选）
    trajectory: Optional[List[StepRecord]] = None
```

---

### 2.5 StepRecord 数据类

**职责**：记录单步的完整信息

```python
@dataclass
class StepRecord:
    """单步记录"""

    step: int

    # 状态
    state_before: Dict[str, Any]
    state_after: Dict[str, Any]

    # 动作
    l_actions: Dict[str, str]
    r_actions: Dict[str, str]

    # 奖励
    l_reward: float
    r_reward: float
    l_reward_info: RewardInfo
    r_reward_info: RewardInfo
```

---

## 三、辅助函数

### 3.1 动作采样函数

```python
def sample_actions(
    action_logits: torch.Tensor,
    player_names: List[str],
    temperature: float = 1.0,
    action_vocab: List[str] = ["", "up", "down", "left", "right"]
) -> Dict[str, str]:
    """
    从模型输出的logits采样动作

    Args:
        action_logits: (num_players, num_actions) 动作logits
        player_names: 玩家名称列表
        temperature: 采样温度（>1更随机，<1更确定）
        action_vocab: 动作词汇表

    Returns:
        动作字典 {"L0": "up", "L1": "right", ...}
    """
    pass
```

### 3.2 状态转换函数

```python
def status_to_game_state_snapshot(
    status_dict: Dict[str, Any],
    team: str
) -> GameStateSnapshot:
    """
    将sim_env.status()转换为reward_system需要的GameStateSnapshot

    Args:
        status_dict: sim_env.status()的输出
        team: 队伍名称 "L" or "R"

    Returns:
        GameStateSnapshot对象
    """
    pass
```

---

## 四、实现注意事项

### 4.1 接口兼容性

⚠️ **关键**：确保数据格式在各模块间正确传递

1. **sim_env.status() → encoding.encode_status_for_team()**
   - ✅ 已验证兼容：sim_env的status输出包含encoding需要的所有字段
   - ⚠️ 注意：encoding需要 `_myteamTarget` 和 `_myteamPrison` 字段（sim_env已提供）

2. **sim_env.status() → GameStateSnapshot**
   - ⚠️ 需要手动转换字段名：
     - `myteamPlayer` → `my_players`
     - `opponentPlayer` → `opp_players`
     - `myteamFlag` → `my_flags`
     - 等等

3. **action_logits → sim_env.step()**
   - ⚠️ sim_env.step()需要两个字典：`l_actions`, `r_actions`
   - ⚠️ 动作格式：`{"L0": "up", "L1": "right", "L2": ""}`

### 4.2 状态管理

⚠️ **关键**：正确维护前后状态用于奖励计算

```python
# 正确的流程
prev_state = status_to_game_state_snapshot(sim.status("L"), "L")
sim.step(l_actions, r_actions)
current_state = status_to_game_state_snapshot(sim.status("L"), "L")
reward_info = reward_system.calculate_reward(current_state, prev_state, my_target_pos)
```

### 4.3 奖励计算

⚠️ **关键**：每步都要计算双方的奖励

1. **L队奖励**：
   ```python
   l_state_prev = status_to_game_state_snapshot(prev_status, "L")
   l_state_curr = status_to_game_state_snapshot(curr_status, "L")
   l_reward_info = reward_system.calculate_reward(l_state_curr, l_state_prev, l_target_pos)
   ```

2. **R队奖励**：
   ```python
   r_state_prev = status_to_game_state_snapshot(prev_status, "R")
   r_state_curr = status_to_game_state_snapshot(curr_status, "R")
   r_reward_info = reward_system.calculate_reward(r_state_curr, r_state_prev, r_target_pos)
   ```

⚠️ **注意**：同一个sim.status()要从两个视角转换（L和R）

### 4.4 目标位置获取

⚠️ **关键**：my_target_pos是奖励计算的必需参数

```python
# 从geometry获取目标位置
if team == "L":
    my_target_pos = geometry.my_targets[0] if geometry.my_side_is_left else geometry.opp_targets[0]
else:
    my_target_pos = geometry.opp_targets[0] if geometry.my_side_is_left else geometry.my_targets[0]
```

### 4.5 游戏结束判断

⚠️ **关键**：正确判断游戏结束条件

```python
# sim_env的结束条件
done = (
    sim.done or                    # sim内部判断
    sim.step_count >= max_steps or # 超时
    sim.l_score >= 3 or            # L队获胜
    sim.r_score >= 3               # R队获胜
)
```

### 4.6 轨迹记录

⚠️ **性能**：轨迹记录会占用大量内存

- ✅ 训练时：`record_trajectory=False`（只需要最终统计）
- ✅ 评估时：`record_trajectory=True`（用于可视化和分析）
- ⚠️ 如果记录轨迹，考虑只保存关键信息，避免深拷贝整个状态

### 4.7 随机性控制

⚠️ **重要**：确保可复现性

```python
# 在run_episode开始时
sim.reset()  # 会使用sim初始化时的seed
agent_l.reset()
agent_r.reset()
```

### 4.8 异常处理

⚠️ **鲁棒性**：处理可能的异常情况

1. **模型推理失败**：
   - 捕获torch异常
   - 返回默认动作（stay）

2. **非法动作**：
   - sim_env会自动忽略非法动作
   - 无需额外处理

3. **超时**：
   - 达到max_steps时强制结束
   - 判定为平局

---

## 五、测试策略

### 5.1 单元测试

```python
def test_game_interface():
    """测试GameInterface基本功能"""
    # 1. 创建模拟器和智能体
    sim = CTFSim(width=20, height=20, seed=42)
    geometry = create_geometry_from_sim(sim)
    reward_system = AdaptiveRewardSystem()

    # 2. 创建随机智能体
    agent_l = RandomAgent("L")
    agent_r = RandomAgent("R")

    # 3. 运行对战
    interface = GameInterface(sim, geometry, reward_system)
    result = interface.run_episode(agent_l, agent_r)

    # 4. 验证结果
    assert result.winner in ["L", "R", None]
    assert result.steps > 0
    assert result.l_score >= 0
    assert result.r_score >= 0
```

### 5.2 集成测试

```python
def test_transformer_agent():
    """测试TransformerAgent与GameInterface集成"""
    # 1. 创建Transformer模型
    config = CTFTransformerConfig(d_model=64, num_layers=1)
    model = build_ctf_transformer(config)

    # 2. 创建智能体
    agent_l = TransformerAgent(model, "L")
    agent_r = TransformerAgent(model, "R")

    # 3. 运行对战
    sim = CTFSim(seed=42)
    geometry = create_geometry_from_sim(sim)
    reward_system = AdaptiveRewardSystem()
    interface = GameInterface(sim, geometry, reward_system)

    result = interface.run_episode(agent_l, agent_r)

    # 4. 验证
    assert result is not None
    print(f"Winner: {result.winner}, Steps: {result.steps}")
```

---

## 六、实现顺序建议

1. **第一步**：实现 `EpisodeResult` 和 `StepRecord` 数据类（简单）
2. **第二步**：实现辅助函数 `sample_actions()` 和 `status_to_game_state_snapshot()`
3. **第三步**：实现 `PolicyAgent` 抽象类和 `RandomAgent`（用于测试）
4. **第四步**：实现 `GameInterface.run_episode()`（核心）
5. **第五步**：实现 `TransformerAgent`
6. **第六步**：编写测试并验证

---

## 七、预估代码量

- `EpisodeResult` + `StepRecord`: ~50行
- 辅助函数: ~80行
- `PolicyAgent` + `RandomAgent`: ~40行
- `GameInterface`: ~150行
- `TransformerAgent`: ~100行
- 测试代码: ~100行

**总计**: ~520行（包含注释和文档字符串）

---

## 八、依赖关系

```
game_interface.py
├── sim_env.py (CTFSim)
├── encoding.py (encode_status_for_team, to_torch_batch)
├── reward_system.py (AdaptiveRewardSystem, GameStateSnapshot)
├── transformer_model.py (CTFTransformer)
└── tree_features.py (Geometry)
```

**注意**：所有依赖都已实现，无需额外开发。
