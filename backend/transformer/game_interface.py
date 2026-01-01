"""
游戏接口模块 - 连接sim_env、encoding、reward_system

提供完整的对战执行接口，用于训练和评估。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from _import_bootstrap import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


# ============ 数据类定义 ============

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
    l_reward_info: Optional[Any] = None  # RewardInfo
    r_reward_info: Optional[Any] = None


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
    l_reward_breakdown: Dict[str, float] = field(default_factory=dict)
    r_reward_breakdown: Dict[str, float] = field(default_factory=dict)

    # 统计信息
    l_flags_captured: int = 0
    r_flags_captured: int = 0
    l_enemies_tagged: int = 0
    r_enemies_tagged: int = 0
    l_avg_survival_rate: float = 0.0
    r_avg_survival_rate: float = 0.0

    # 轨迹数据（可选）
    trajectory: Optional[List[StepRecord]] = None

    def get_fitness(self, team: str) -> float:
        """计算适应度评分"""
        if team == "L":
            return self.l_total_reward
        else:
            return self.r_total_reward


# ============ 辅助函数 ============

def sample_actions(
    action_logits: Any,  # torch.Tensor or numpy array
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
    if TORCH_AVAILABLE and isinstance(action_logits, torch.Tensor):
        # PyTorch版本
        if temperature != 1.0:
            action_logits = action_logits / temperature

        probs = F.softmax(action_logits, dim=-1)  # (num_players, num_actions)

        # 采样
        action_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (num_players,)
        action_indices = action_indices.cpu().numpy()
    else:
        # NumPy版本
        import numpy as np
        if isinstance(action_logits, torch.Tensor):
            action_logits = action_logits.cpu().numpy()

        if temperature != 1.0:
            action_logits = action_logits / temperature

        # Softmax
        exp_logits = np.exp(action_logits - np.max(action_logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # 采样
        action_indices = np.array([
            np.random.choice(len(action_vocab), p=probs[i])
            for i in range(len(player_names))
        ])

    # 转换为动作字典
    actions = {}
    for name, idx in zip(player_names, action_indices):
        actions[name] = action_vocab[int(idx)]

    return actions


def status_to_game_state_snapshot(
    status_dict: Dict[str, Any],
    team: str
) -> Any:  # GameStateSnapshot
    """
    将sim_env.status()转换为reward_system需要的GameStateSnapshot

    Args:
        status_dict: sim_env.status()的输出
        team: 队伍名称 "L" or "R"

    Returns:
        GameStateSnapshot对象
    """
    from reward_system import GameStateSnapshot

    return GameStateSnapshot(
        timestamp=status_dict.get("time", 0.0),
        my_players=status_dict.get("myteamPlayer", []),
        opp_players=status_dict.get("opponentPlayer", []),
        my_flags=status_dict.get("myteamFlag", []),
        opp_flags=status_dict.get("opponentFlag", []),
        my_score=status_dict.get("myteamScore", 0),
        opp_score=status_dict.get("opponentScore", 0),
        game_over=False,  # 由调用者判断
        winner=None
    )


# ============ 智能体抽象类 ============

class PolicyAgent(ABC):
    """策略智能体抽象基类"""

    def __init__(self, team: str):
        """
        Args:
            team: 队伍名称 "L" or "R"
        """
        self.team = team

    @abstractmethod
    def select_actions(
        self,
        status_dict: Dict[str, Any],
        geometry: Any  # Geometry对象
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

    def reset(self) -> None:
        """重置智能体状态（用于新局开始）"""
        pass


class RandomAgent(PolicyAgent):
    """随机策略智能体（用于测试）"""

    def __init__(self, team: str, action_vocab: List[str] = ["", "up", "down", "left", "right"]):
        super().__init__(team)
        self.action_vocab = action_vocab

    def select_actions(
        self,
        status_dict: Dict[str, Any],
        geometry: Any
    ) -> Dict[str, str]:
        """随机选择动作"""
        import random

        my_players = status_dict.get("myteamPlayer", [])
        actions = {}

        for player in my_players:
            # 跳过被囚禁的玩家
            if player.get("inPrison", False):
                actions[player["name"]] = ""
            else:
                actions[player["name"]] = random.choice(self.action_vocab)

        return actions


class TransformerAgent(PolicyAgent):
    """基于Transformer模型的智能体"""

    def __init__(
        self,
        model: Any,  # CTFTransformer
        team: str,
        max_tokens: int = 32,
        temperature: float = 1.0,
        action_vocab: List[str] = ["", "up", "down", "left", "right"],
        device: Optional[Any] = None
    ):
        """
        Args:
            model: CTFTransformer模型
            team: 队伍名称 "L" or "R"
            max_tokens: token序列最大长度
            temperature: 动作采样温度
            action_vocab: 动作词汇表
            device: torch.device 设备
        """
        super().__init__(team)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.action_vocab = action_vocab

        # 设备配置 - 自动检测最佳设备
        if device is None and TORCH_AVAILABLE:
            from device_utils import get_device
            self.device = get_device(verbose=False)
        else:
            self.device = device

        # 将模型移动到设备 (处理multiprocessing unpickle到CPU的情况)
        if TORCH_AVAILABLE and self.model is not None:
            from device_utils import move_to_device
            self.model = move_to_device(self.model, self.device)

        # 设置为评估模式
        if hasattr(self.model, 'eval'):
            self.model.eval()

    def select_actions(
        self,
        status_dict: Dict[str, Any],
        geometry: Any
    ) -> Dict[str, str]:
        """使用Transformer模型选择动作"""
        try:
            # 1. 导入encoding模块
            from encoding import encode_status_for_team, to_torch_batch

            # 2. 状态编码
            type_ids, features, padding_mask, my_player_indices = encode_status_for_team(
                status_req=status_dict,
                geometry=geometry,
                max_tokens=self.max_tokens
            )

            # 3. 转换为batch
            batch = to_torch_batch([
                (type_ids, features, padding_mask, my_player_indices)
            ])

            # 4. 将batch张量移动到设备
            batch_type_ids = batch.type_ids.to(self.device)
            batch_features = batch.features.to(self.device)
            batch_padding_mask = batch.padding_mask.to(self.device)
            # my_player_token_indices is a tuple of ints (not a tensor), used for indexing
            batch_my_player_indices = batch.my_player_token_indices

            # 5. 模型推理
            with torch.no_grad():
                action_logits = self.model(
                    type_ids=batch_type_ids,
                    features=batch_features,
                    padding_mask=batch_padding_mask,
                    my_player_token_indices=batch_my_player_indices
                )  # (1, num_players, num_actions)

            # 6. 移除batch维度
            action_logits = action_logits.squeeze(0)  # (num_players, num_actions)

            # 7. 获取玩家名称
            my_players = status_dict.get("myteamPlayer", [])
            player_names = [p["name"] for p in my_players]

            # 8. 采样动作
            actions = sample_actions(
                action_logits,
                player_names,
                temperature=self.temperature,
                action_vocab=self.action_vocab
            )

            return actions

        except Exception as e:
            # 出错时返回默认动作（stay）
            print(f"Warning: TransformerAgent.select_actions failed: {e}")
            my_players = status_dict.get("myteamPlayer", [])
            return {p["name"]: "" for p in my_players}


# ============ 游戏接口类 ============

class GameInterface:
    """游戏接口 - 执行单局对战并收集训练数据"""

    def __init__(
        self,
        sim: Any,  # CTFSim
        geometry: Any,  # Geometry
        reward_system: Any,  # AdaptiveRewardSystem
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
        self.sim = sim
        self.geometry = geometry
        self.reward_system = reward_system
        self.max_tokens = max_tokens
        self.max_steps = max_steps

    def _get_target_pos(self, team: str) -> Tuple[int, int]:
        """获取队伍的目标位置

        注意: Geometry对象总是从L队的视角初始化(my_side_is_left=True)
        - my_targets = L队的目标位置
        - opp_targets = R队的目标位置
        - my_prisons = L队的监狱位置
        - opp_prisons = R队的监狱位置
        """
        if team == "L":
            # L队的目标是my_targets
            targets = self.geometry.my_targets if self.geometry.my_side_is_left else self.geometry.opp_targets
        else:
            # R队的目标是opp_targets(从L的视角)
            targets = self.geometry.opp_targets if self.geometry.my_side_is_left else self.geometry.my_targets

        return targets[0] if targets else (0, 0)

    def _get_prison_pos(self, team: str) -> Tuple[int, int]:
        """获取队伍的监狱位置（用于奖励塑形中的救援潜力计算）"""
        if team == "L":
            # L队的监狱在左侧
            prisons = self.geometry.my_prisons if self.geometry.my_side_is_left else self.geometry.opp_prisons
        else:
            # R队的监狱在右侧
            prisons = self.geometry.opp_prisons if self.geometry.my_side_is_left else self.geometry.my_prisons

        return prisons[0] if prisons else (0, 0)

    def _determine_winner(self) -> Optional[str]:
        """判断游戏胜者"""
        if self.sim.l_score >= 3:
            return "L"
        elif self.sim.r_score >= 3:
            return "R"
        elif self.sim.done:
            # 游戏结束但未达到3分，判断分数
            if self.sim.l_score > self.sim.r_score:
                return "L"
            elif self.sim.r_score > self.sim.l_score:
                return "R"
        return None

    def run_episode(
        self,
        agent_l: PolicyAgent,
        agent_r: PolicyAgent,
        record_trajectory: bool = False
    ) -> EpisodeResult:
        """
        执行一局完整对战

        Args:
            agent_l: L队智能体
            agent_r: R队智能体
            record_trajectory: 是否记录完整轨迹

        Returns:
            EpisodeResult: 对战结果
        """
        # 1. 重置游戏和智能体
        self.sim.reset()
        agent_l.reset()
        agent_r.reset()

        # 2. 初始化统计变量
        l_total_reward = 0.0
        r_total_reward = 0.0
        l_reward_breakdown = {}
        r_reward_breakdown = {}
        trajectory = [] if record_trajectory else None

        # 2.1 初始化存活率和标记统计变量
        l_survival_steps = 0
        r_survival_steps = 0
        l_enemies_tagged = 0
        r_enemies_tagged = 0

        # 3. 获取目标位置
        l_target_pos = self._get_target_pos("L")
        r_target_pos = self._get_target_pos("R")

        # 3.1 获取监狱位置（用于奖励塑形）
        l_prison_pos = self._get_prison_pos("L")
        r_prison_pos = self._get_prison_pos("R")

        # 4. 主循环
        for step in range(self.max_steps):
            # 4.1 获取当前状态
            l_status = self.sim.status("L")
            r_status = self.sim.status("R")

            # 4.2 保存前状态（用于奖励计算）
            l_state_before = status_to_game_state_snapshot(l_status, "L")
            r_state_before = status_to_game_state_snapshot(r_status, "R")

            # 4.3 智能体选择动作
            l_actions = agent_l.select_actions(l_status, self.geometry)
            r_actions = agent_r.select_actions(r_status, self.geometry)

            # 4.4 执行动作
            self.sim.step(l_actions, r_actions)

            # 4.5 获取后状态
            l_status_after = self.sim.status("L")
            r_status_after = self.sim.status("R")

            l_state_after = status_to_game_state_snapshot(l_status_after, "L")
            r_state_after = status_to_game_state_snapshot(r_status_after, "R")

            # 4.5.1 统计存活率（每步累计非囚禁玩家数）
            l_alive = sum(1 for p in l_status_after.get("myteamPlayer", []) if not p.get("inPrison", False))
            r_alive = sum(1 for p in r_status_after.get("myteamPlayer", []) if not p.get("inPrison", False))
            l_survival_steps += l_alive
            r_survival_steps += r_alive

            # 4.5.2 统计敌人标记（检测对方玩家新入狱）
            for r_player in r_status_after.get("myteamPlayer", []):
                if r_player.get("inPrison", False):
                    # 检查之前是否不在监狱
                    r_id = r_player.get("id")
                    was_free = not any(
                        p.get("id") == r_id and p.get("inPrison", False)
                        for p in r_status.get("myteamPlayer", [])
                    )
                    if was_free:
                        l_enemies_tagged += 1

            for l_player in l_status_after.get("myteamPlayer", []):
                if l_player.get("inPrison", False):
                    l_id = l_player.get("id")
                    was_free = not any(
                        p.get("id") == l_id and p.get("inPrison", False)
                        for p in l_status.get("myteamPlayer", [])
                    )
                    if was_free:
                        r_enemies_tagged += 1

            # 4.6 计算奖励
            l_reward_info = self.reward_system.calculate_reward(
                l_state_after, l_state_before, l_target_pos, l_prison_pos
            )
            r_reward_info = self.reward_system.calculate_reward(
                r_state_after, r_state_before, r_target_pos, r_prison_pos
            )

            l_total_reward += l_reward_info.total
            r_total_reward += r_reward_info.total

            # 4.7 累积奖励分解
            for key, value in l_reward_info.breakdown.get('dense', {}).items():
                l_reward_breakdown[key] = l_reward_breakdown.get(key, 0.0) + value
            for key, value in r_reward_info.breakdown.get('dense', {}).items():
                r_reward_breakdown[key] = r_reward_breakdown.get(key, 0.0) + value

            # 4.8 记录轨迹（可选）
            if record_trajectory:
                trajectory.append(StepRecord(
                    step=step,
                    state_before=l_status,
                    state_after=l_status_after,
                    l_actions=l_actions,
                    r_actions=r_actions,
                    l_reward=l_reward_info.total,
                    r_reward=r_reward_info.total,
                    l_reward_info=l_reward_info,
                    r_reward_info=r_reward_info
                ))

            # 4.9 检查游戏是否结束
            if self.sim.l_score >= 3 or self.sim.r_score >= 3 or self.sim.done:
                break

        # 5. 收集统计信息
        winner = self._determine_winner()
        final_step = self.sim.step_count

        # 6. 计算额外统计
        l_flags_captured = self.sim.l_score
        r_flags_captured = self.sim.r_score

        # 计算平均存活率（使用循环中累计的数据）
        l_avg_survival = l_survival_steps / max(1, final_step * 3)  # 3 players
        r_avg_survival = r_survival_steps / max(1, final_step * 3)

        # 7. 构造返回结果
        result = EpisodeResult(
            winner=winner,
            l_score=self.sim.l_score,
            r_score=self.sim.r_score,
            steps=final_step,
            duration_ms=float(getattr(self.sim, "sim_time_ms", final_step * self.sim.dt_ms)),
            l_total_reward=l_total_reward,
            r_total_reward=r_total_reward,
            l_reward_breakdown=l_reward_breakdown,
            r_reward_breakdown=r_reward_breakdown,
            l_flags_captured=l_flags_captured,
            r_flags_captured=r_flags_captured,
            l_enemies_tagged=l_enemies_tagged,
            r_enemies_tagged=r_enemies_tagged,
            l_avg_survival_rate=l_avg_survival,
            r_avg_survival_rate=r_avg_survival,
            trajectory=trajectory
        )

        return result


# ============ 测试代码 ============

def test_game_interface():
    """测试GameInterface基本功能"""
    print("\n" + "="*60)
    print("测试 GameInterface")
    print("="*60)

    try:
        # 1. 导入依赖
        from _import_bootstrap import get_geometry
        from sim_env import CTFSim

        Geometry = get_geometry()

        from reward_system import AdaptiveRewardSystem

        # 2. 创建模拟器
        sim = CTFSim(width=20, height=20, num_players=3, seed=42)
        sim.reset()

        # 3. 创建Geometry对象
        init_payload = sim.init_payload("L")
        geometry = Geometry(
            width=init_payload["map"]["width"],
            height=init_payload["map"]["height"],
            my_side_is_left=True,
            left_max_x=(init_payload["map"]["width"] - 1) // 2,
            my_targets=tuple((t["x"], t["y"]) for t in init_payload["myteamTarget"]),
            my_prisons=tuple((p["x"], p["y"]) for p in init_payload["myteamPrison"]),
            opp_targets=tuple((t["x"], t["y"]) for t in init_payload["opponentTarget"]),
            opp_prisons=tuple((p["x"], p["y"]) for p in init_payload["opponentPrison"]),
            blocked=frozenset((w["x"], w["y"]) for w in init_payload["map"]["walls"])
        )

        # 4. 创建奖励系统
        reward_system = AdaptiveRewardSystem()
        reward_system.reset_for_generation(0)

        # 5. 创建随机智能体
        agent_l = RandomAgent("L")
        agent_r = RandomAgent("R")

        # 6. 创建GameInterface并运行对战
        interface = GameInterface(sim, geometry, reward_system, max_steps=100)
        result = interface.run_episode(agent_l, agent_r, record_trajectory=False)

        # 7. 验证结果
        print(f"✓ 对战完成")
        print(f"  胜者: {result.winner}")
        print(f"  比分: L {result.l_score} - {result.r_score} R")
        print(f"  步数: {result.steps}")
        print(f"  L队奖励: {result.l_total_reward:.2f}")
        print(f"  R队奖励: {result.r_total_reward:.2f}")

        assert result.winner in ["L", "R", None]
        assert result.steps > 0
        assert result.l_score >= 0
        assert result.r_score >= 0

        print("✓ 所有断言通过")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transformer_agent():
    """测试TransformerAgent与GameInterface集成"""
    print("\n" + "="*60)
    print("测试 TransformerAgent")
    print("="*60)

    try:
        # 1. 导入依赖
        from _import_bootstrap import get_geometry
        from sim_env import CTFSim

        Geometry = get_geometry()

        from reward_system import AdaptiveRewardSystem
        from transformer_model import CTFTransformerConfig, build_ctf_transformer

        # 2. 创建小型Transformer模型
        config = CTFTransformerConfig(
            d_model=64,
            nhead=4,
            num_layers=1,
            dim_feedforward=128
        )
        model = build_ctf_transformer(config)

        # 3. 创建模拟器和Geometry
        sim = CTFSim(width=20, height=20, num_players=3, seed=42)
        sim.reset()

        init_payload = sim.init_payload("L")
        geometry = Geometry(
            width=init_payload["map"]["width"],
            height=init_payload["map"]["height"],
            my_side_is_left=True,
            left_max_x=(init_payload["map"]["width"] - 1) // 2,
            my_targets=tuple((t["x"], t["y"]) for t in init_payload["myteamTarget"]),
            my_prisons=tuple((p["x"], p["y"]) for p in init_payload["myteamPrison"]),
            opp_targets=tuple((t["x"], t["y"]) for t in init_payload["opponentTarget"]),
            opp_prisons=tuple((p["x"], p["y"]) for p in init_payload["opponentPrison"]),
            blocked=frozenset((w["x"], w["y"]) for w in init_payload["map"]["walls"])
        )

        # 4. 创建奖励系统
        reward_system = AdaptiveRewardSystem()
        reward_system.reset_for_generation(0)

        # 5. 创建TransformerAgent
        agent_l = TransformerAgent(model, "L", temperature=1.0)
        agent_r = TransformerAgent(model, "R", temperature=1.0)

        # 6. 运行对战
        interface = GameInterface(sim, geometry, reward_system, max_steps=50)
        result = interface.run_episode(agent_l, agent_r, record_trajectory=False)

        # 7. 验证结果
        print(f"✓ Transformer对战完成")
        print(f"  胜者: {result.winner}")
        print(f"  比分: L {result.l_score} - {result.r_score} R")
        print(f"  步数: {result.steps}")
        print(f"  L队奖励: {result.l_total_reward:.2f}")
        print(f"  R队奖励: {result.r_total_reward:.2f}")

        assert result is not None
        assert result.steps > 0

        print("✓ 所有断言通过")
        return True

    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("开始测试 game_interface.py")
    print("="*60)

    results = []
    results.append(("GameInterface基础功能", test_game_interface()))
    results.append(("TransformerAgent集成", test_transformer_agent()))

    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")

    all_passed = all(result[1] for result in results)
    print("\n" + "="*60)
    if all_passed:
        print("所有测试通过！")
    else:
        print("部分测试失败，请检查错误信息")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
