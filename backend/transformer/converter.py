from __future__ import annotations

from typing import Any, Dict, List, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ActionConverter:
    """动作输出转换器 - 将模型输出转换为WebSocket响应"""

    ACTION_VOCAB = ["", "up", "down", "left", "right"]

    @staticmethod
    def model_output_to_moves(
        player_names: List[str],
        action_logits: Any,  # torch.Tensor or numpy array
        action_vocab: List[str] = None
    ) -> List[Dict[str, str]]:
        """
        将模型输出的动作logits转换为moves列表

        Args:
            player_names: 玩家名称列表 ["L0", "L1", "L2"]
            action_logits: 形状为 (num_players, num_actions) 的张量
            action_vocab: 动作词汇表，默认为 ["", "up", "down", "left", "right"]

        Returns:
            [{"name": "L0", "move": "up"}, {"name": "L1", "move": "right"}, ...]
        """
        if action_vocab is None:
            action_vocab = ActionConverter.ACTION_VOCAB

        if TORCH_AVAILABLE and isinstance(action_logits, torch.Tensor):
            action_indices = action_logits.argmax(dim=-1).cpu().numpy()
        else:
            import numpy as np
            action_indices = np.argmax(action_logits, axis=-1)

        moves = []
        for name, action_idx in zip(player_names, action_indices):
            move = action_vocab[int(action_idx)]
            moves.append({"name": name, "move": move})

        return moves

    @staticmethod
    def create_websocket_response(moves: List[Dict[str, str]]) -> Dict[str, Any]:
        """创建WebSocket响应格式"""
        return {"players": moves}
