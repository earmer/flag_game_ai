# 训练推理可视化交互界面设计

## 📋 功能列表总览

### 模块1：训练控制中心 (Training Control Center)

#### 1.1 训练启动与配置
- ✅ 交互式超参数配置面板（滑块、下拉框）
- ✅ 预设配置模板（快速测试/标准训练/深度训练）
- ✅ 一键启动/暂停/恢复/停止训练
- ✅ 检查点管理（保存/加载/回滚）

#### 1.2 实时训练监控
- ✅ 当前世代进度条（带ETA预估）
- ✅ 实时适应度曲线（最佳/平均/最差）
- ✅ 温度衰减曲线
- ✅ 奖励权重演化曲线（密集/稀疏/塑形）
- ✅ 种群多样性指标

#### 1.3 训练日志
- ✅ 实时滚动日志窗口
- ✅ 关键事件高亮（新纪录、异常等）
- ✅ 日志过滤和搜索

---

### 模块2：游戏可视化 (Game Visualization)

#### 2.1 实时对局播放器
- ✅ 2D游戏地图渲染（matplotlib/plotly）
- ✅ 玩家移动轨迹动画
- ✅ 旗帜状态实时更新
- ✅ 播放控制（播放/暂停/快进/慢放）
- ✅ 时间轴拖动

#### 2.2 对局信息面板
- ✅ 当前比分显示
- ✅ 玩家状态表格（位置/持旗/监狱）
- ✅ 关键事件时间线
- ✅ 双方策略热力图

#### 2.3 多对局对比
- ✅ 并排显示多场游戏
- ✅ 同步播放控制
- ✅ 性能指标对比

---

### 模块3：种群分析 (Population Analytics)

#### 3.1 种群概览
- ✅ 个体适应度排行榜
- ✅ 世代演化树状图
- ✅ 基因多样性热力图
- ✅ 精英个体历史追踪

#### 3.2 个体详情
- ✅ 选择个体查看详细统计
- ✅ 胜率/捕旗数/存活率等指标
- ✅ 对战记录表
- ✅ 策略特征分析

#### 3.3 遗传操作可视化
- ✅ 交叉操作动画
- ✅ 变异前后参数对比
- ✅ 选择压力分析

---

### 模块4：奖励系统分析 (Reward Analysis)

#### 4.1 奖励分解
- ✅ 密集/稀疏/塑形奖励分别显示
- ✅ 单步奖励时间序列
- ✅ 累积奖励曲线
- ✅ 奖励分布直方图

#### 4.2 奖励演化
- ✅ 不同世代的奖励对比
- ✅ 课程学习权重变化
- ✅ 塑形强度衰减曲线

#### 4.3 奖励调试
- ✅ 自定义奖励函数测试
- ✅ 奖励敏感性分析
- ✅ A/B测试对比

---

### 模块5：模型探索与调试 (Model Exploration)

#### 5.1 模型推理测试
- ✅ 加载任意检查点
- ✅ 单步推理可视化
- ✅ 注意力权重热力图
- ✅ 动作概率分布

#### 5.2 对抗测试
- ✅ 选择两个模型对战
- ✅ 实时观看对局
- ✅ 策略差异分析

#### 5.3 模型诊断
- ✅ 参数分布统计
- ✅ 梯度流分析
- ✅ 激活值可视化
- ✅ 性能瓶颈分析

---

## 🎨 技术栈选择

```python
VISUALIZATION_STACK = {
    "静态图表": "matplotlib + seaborn",
    "交互图表": "plotly",
    "实时更新": "IPython.display + ipywidgets",
    "游戏渲染": "matplotlib.animation",
    "控制面板": "ipywidgets",
    "进度条": "tqdm.notebook",
    "数据处理": "pandas + numpy",
    "异步处理": "threading + asyncio"
}
```

**依赖安装**：
```bash
pip install jupyter ipywidgets matplotlib seaborn plotly pandas tqdm
jupyter nbextension enable --py widgetsnbextension
```

---

## 🏗️ 设计思路

### 核心设计原则

#### 原则1：模块化设计
- 每个功能独立封装为类
- 可复用的组件放在 `utils/` 目录
- 清晰的接口定义
- 便于单元测试和维护

#### 原则2：实时响应
- 使用异步更新避免阻塞UI
- 进度条显示长时间操作
- 后台线程处理训练任务
- 定期刷新显示（避免过于频繁）

#### 原则3：用户友好
- 预设配置快速开始
- 工具提示和帮助文档
- 错误处理和友好提示
- 支持撤销和重置操作

#### 原则4：性能优化
- 数据采样避免过载（如只显示最近100代）
- 懒加载大型数据（按需加载检查点）
- 缓存计算结果（避免重复计算）
- 使用生成器处理大数据集

---

## 📁 项目文件结构

```
CTF/backend/transformer/
├── notebooks/
│   ├── 1_training_dashboard.ipynb      # 训练控制和监控主界面
│   ├── 2_game_viewer.ipynb             # 游戏对局回放和分析
│   ├── 3_population_explorer.ipynb    # 种群演化探索
│   ├── 4_reward_analyzer.ipynb        # 奖励系统分析和调试
│   ├── 5_model_inspector.ipynb        # 模型诊断和对比
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py           # 可视化组件
│       ├── widgets.py                 # 交互控件
│       ├── data_loader.py             # 数据加载工具
│       └── game_renderer.py           # 游戏渲染器
├── checkpoints/                        # 模型检查点
├── logs/                               # 训练日志
└── VISUALIZATION_DESIGN.md             # 本文档
```

---

## 🔧 核心组件设计

### 1. 训练仪表盘 (TrainingDashboard)

**功能**：实时监控训练进度和关键指标

**核心方法**：
```python
class TrainingDashboard:
    def __init__(self):
        # 初始化图表和数据缓冲区

    def update(self, generation_data):
        # 更新显示（每代调用一次）

    def save_snapshot(self, filename):
        # 保存当前状态快照
```

**显示内容**：
- 左上：适应度曲线（最佳/平均/最差）
- 右上：温度衰减曲线
- 左下：种群多样性指标
- 右下：奖励权重演化

---

### 2. 游戏可视化器 (GameVisualizer)

**功能**：渲染游戏对局，支持回放和分析

**核心方法**：
```python
class GameVisualizer:
    def __init__(self, game_width=20, game_height=20):
        # 初始化渲染器

    def render_frame(self, game_state):
        # 渲染单帧

    def play_animation(self, game_history):
        # 播放完整对局动画

    def export_video(self, filename):
        # 导出为视频文件
```

**渲染元素**：
- 地图：墙壁、障碍物
- 玩家：圆形，颜色区分队伍
- 旗帜：星形标记
- 目标区域：半透明圆圈
- 监狱：灰色方块
- 轨迹：虚线显示移动路径

---

### 3. 训练控制面板 (TrainingControlPanel)

**功能**：配置训练参数，控制训练流程

**核心控件**：
```python
class TrainingControlPanel:
    # 种群参数
    - pop_size: IntSlider (4-16)
    - elite_size: IntSlider (1-4)
    - generations: IntSlider (10-100)

    # 遗传参数
    - mutation_rate: FloatSlider (0.0-0.5)
    - init_temp: FloatSlider (0.5-2.0)
    - cooling_rate: FloatSlider (0.85-0.99)

    # 控制按钮
    - start_btn: 开始训练
    - pause_btn: 暂停训练
    - stop_btn: 停止训练
    - resume_btn: 恢复训练

    # 预设配置
    - preset: Dropdown (快速测试/标准训练/深度训练)
```

**预设配置**：

| 配置名称 | 种群大小 | 代数 | 变异率 | 初始温度 | 冷却率 |
|---------|---------|------|--------|---------|--------|
| 快速测试 | 4 | 10 | 0.15 | 1.5 | 0.90 |
| 标准训练 | 8 | 50 | 0.10 | 1.0 | 0.95 |
| 深度训练 | 12 | 100 | 0.08 | 0.8 | 0.97 |

---

### 4. 模型对比工具 (ModelComparator)

**功能**：对比不同模型的性能

**核心方法**：
```python
class ModelComparator:
    def compare_models(self, model_a, model_b, num_games=5):
        # 运行多场对战

    def display_results(self, results):
        # 显示对比结果

    def analyze_strategies(self, model_a, model_b):
        # 分析策略差异
```

**对比维度**：
- 胜率
- 平均捕旗数
- 平均存活时间
- 标记敌人数
- 游戏时长
- 策略激进度

---

### 5. 奖励函数调试器 (RewardDebugger)

**功能**：测试和调优奖励函数

**核心方法**：
```python
class RewardDebugger:
    def test_reward_config(self, dense_w, sparse_w, shaping_s):
        # 测试自定义奖励配置

    def visualize_rewards(self, game_data):
        # 可视化奖励分解

    def sensitivity_analysis(self):
        # 奖励敏感性分析
```

**可视化内容**：
- 左上：密集奖励时间序列
- 右上：稀疏奖励时间序列
- 左下：塑形奖励时间序列
- 右下：总奖励累积曲线

---

## 🎯 使用流程

### 场景1：快速开始训练

```python
# Cell 1: 导入和初始化
from utils.visualization import TrainingDashboard
from utils.widgets import TrainingControlPanel

dashboard = TrainingDashboard()
control_panel = TrainingControlPanel()

# Cell 2: 显示控制面板
control_panel.display()

# Cell 3: 开始训练（点击按钮后自动执行）
# 训练过程中自动更新 dashboard
```

### 场景2：观看对局回放

```python
# Cell 1: 加载游戏数据
from utils.game_renderer import GameVisualizer

game_data = load_game_history('checkpoints/gen_25_game_3.json')
visualizer = GameVisualizer()

# Cell 2: 播放动画
visualizer.play_animation(game_data)

# Cell 3: 导出视频
visualizer.export_video('best_game.mp4')
```

### 场景3：对比两个模型

```python
# Cell 1: 初始化对比工具
from utils.widgets import ModelComparator

comparator = ModelComparator()

# Cell 2: 显示界面并选择模型
comparator.display()

# Cell 3: 查看详细分析
comparator.analyze_strategies('gen_10_best', 'gen_50_best')
```

### 场景4：调试奖励函数

```python
# Cell 1: 初始化调试器
from utils.widgets import RewardDebugger

debugger = RewardDebugger()

# Cell 2: 显示界面并调整参数
debugger.display()

# Cell 3: 运行敏感性分析
debugger.sensitivity_analysis()
```

---

## 📊 界面布局示意

### 训练仪表盘布局

```
┌─────────────────────────────────────────────────────────────┐
│  🎮 训练控制面板                                             │
│  [预设: 标准训练 ▼] [种群:8] [代数:50] [▶开始] [⏸暂停]     │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────┬──────────────────────────────────┐
│  📈 适应度曲线            │  🌡️ 温度衰减                     │
│  [最佳/平均/最差曲线]     │  [温度随代数变化]                 │
│                          │                                  │
├──────────────────────────┼──────────────────────────────────┤
│  🧬 种群多样性            │  🎯 奖励权重演化                  │
│  [多样性指标曲线]         │  [密集/稀疏/塑形权重]             │
│                          │                                  │
└──────────────────────────┴──────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  📋 训练日志                                                 │
│  [Gen 25] 最佳适应度: 850.3 | 平均: 620.5 | 温度: 0.45      │
│  [Gen 26] 新纪录! 最佳适应度: 920.1 🎉                       │
│  ...                                                        │
└─────────────────────────────────────────────────────────────┘
```

### 游戏可视化布局

```
┌─────────────────────────────────────────────────────────────┐
│  🎮 游戏回放控制                                             │
│  [◀◀] [▶] [⏸] [▶▶] [━━━━━━●━━━━━━━] Step: 245/1000        │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────┬──────────────────────────────────┐
│                          │  📊 对局信息                      │
│                          │  比分: L队 2 - 1 R队              │
│                          │                                  │
│    🗺️ 游戏地图            │  玩家状态:                        │
│    [20x20网格]           │  L1: (5,10) 持旗 ✓               │
│    [玩家/旗帜/墙壁]       │  L2: (8,12) 监狱 ✗               │
│                          │  L3: (3,8)  正常                 │
│                          │                                  │
│                          │  关键事件:                        │
│                          │  [Step 120] L1 捕获旗帜           │
│                          │  [Step 180] L2 被标记             │
│                          │  [Step 245] L1 得分! 🎉          │
└──────────────────────────┴──────────────────────────────────┘
```

---

## 🚀 实现优先级

### Phase 1: 核心功能（1-2周）
- [x] 训练控制面板
- [x] 基础训练仪表盘
- [x] 简单游戏可视化
- [ ] 检查点保存/加载

### Phase 2: 增强功能（1-2周）
- [ ] 游戏回放动画
- [ ] 种群分析工具
- [ ] 奖励分解可视化
- [ ] 模型对比工具

### Phase 3: 高级功能（1-2周）
- [ ] 注意力可视化
- [ ] 策略热力图
- [ ] 敏感性分析
- [ ] 视频导出

### Phase 4: 优化和完善（持续）
- [ ] 性能优化
- [ ] 用户体验改进
- [ ] 文档完善
- [ ] 单元测试

---

## 💡 最佳实践

### 1. 数据管理
- 定期保存检查点（每5代）
- 压缩历史数据（只保留关键帧）
- 使用数据库存储大量对局记录

### 2. 性能优化
- 限制显示的数据点数量（如最近100代）
- 使用降采样显示长时间序列
- 异步加载大型检查点

### 3. 用户体验
- 提供快速预览模式
- 支持键盘快捷键
- 添加工具提示说明
- 错误时给出明确提示

### 4. 调试技巧
- 使用小规模配置快速验证
- 对比不同配置的效果
- 记录异常情况的检查点
- 可视化中间结果

---

**文档版本**: v1.0
**创建日期**: 2025-12-28
**作者**: Claude Sonnet 4.5
