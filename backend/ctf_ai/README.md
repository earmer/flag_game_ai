# CTF：小模型 Transformer + 种群/对抗/退火遗传（简化可行版）

这套实现专门针对本仓库的规则与工程约束：**Transformer 只负责“宏决策”**（角色/目标选择），**微观走格子交给 BFS**（`backend/lib/game_engine.py`），从而做到：
- 模型很小（可 CPU 推理），训练不依赖复杂 RL 框架；
- 对抗学习可用“对手池 + 评估压力”快速迭代；
- 退火遗传只进化低维参数（5 个 role bias），稳定且易复现；
- 真实对局时即使没有 `torch` 也能自动回退到启发式策略。

## 1) 运行 Transformer 后端（可回退）

```bash
python3 backend/pick_flag_transformer_ai.py 8081 --model path/to/model.pt --params path/to/params.json
```

- 没有 `--model` 或未安装 `torch`：自动回退到启发式（类似 `backend/pick_flag_ai.py`）。
- `params.json` 是可选的 role 偏置（用于 SA-GA 调参），示例结构：

```json
{
  "steal_bias": 0.0,
  "return_bias": 0.0,
  "rescue_bias": 0.0,
  "chase_bias": 0.0,
  "defend_bias": 0.0,
  "avoid_opponents": true
}
```

## 2) 退火遗传（SA-GA）调参（头less 模拟器）

```bash
python3 -m backend.ctf_ai.train_sa_ga --generations 30 --population 16 --games-per-individual 8
```

输出在 `backend/ctf_ai_runs/`，每代会写 `gen_XXXX_best_params.json`。

注意：此脚本为了“简单可行”，GA 只进化 role bias（低维参数），并把“对抗性”体现在**对手池**（`opponent_pool`）上。

## 3) “Transformer + 对抗学习”的最省事训练路线（建议）

1. **先用启发式老师做行为克隆（BC）**：让 Transformer 学会基础“偷旗/回家/救人/追击/防守”的分配逻辑。
2. **再引入对手池 + 自对弈**：每轮用当前最强模型对局，收集失败局面，继续训练（你可以把对手池里放不同风格的启发式/历史 checkpoint）。
3. **最后用 SA-GA 调 bias**：让宏策略在不同对手/地图分布上更稳（尤其是 `rescue_bias` 和 `avoid_opponents`）。

如果你希望我把“BC 数据采集 + 训练脚本（PyTorch）”也补齐，我需要你确认：你打算在本机离线跑 headless self-play，还是只依赖真实前端对局日志（WebSocket）来训练？
