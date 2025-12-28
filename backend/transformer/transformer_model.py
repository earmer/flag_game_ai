from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass(frozen=True, slots=True)
class CTFTransformerConfig:
    """Transformer模型配置"""
    # Token编码参数
    num_entity_types: int = 6          # 实体类型数量
    feature_dim: int = 8               # 特征维度（必须匹配encoding.py）
    max_tokens: int = 32               # 最大token序列长度

    # Transformer架构参数
    d_model: int = 128                 # 模型维度
    nhead: int = 4                     # 注意力头数
    num_layers: int = 2                # Transformer层数
    dim_feedforward: int = 256         # FFN维度
    dropout: float = 0.1               # Dropout率

    # 输出参数
    num_players: int = 3               # 玩家数量
    num_actions: int = 5               # 动作数量 (stay/up/down/left/right)

    # 训练参数
    use_layer_norm: bool = True        # 是否使用LayerNorm
    activation: str = "gelu"           # 激活函数


class CTFTransformer(nn.Module):
    """
    CTF游戏Transformer策略网络

    输入: EncodedBatch (type_ids, features, padding_mask, my_player_indices)
    输出: (B, num_players, num_actions) 动作logits
    """

    def __init__(self, config: CTFTransformerConfig):
        super().__init__()
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not installed; Transformer model unavailable")

        self.config = config

        # 输入嵌入
        self.type_embedding = nn.Embedding(config.num_entity_types, config.d_model)
        self.feature_projection = nn.Linear(config.feature_dim, config.d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, config.max_tokens, config.d_model))

        # 输入LayerNorm
        if config.use_layer_norm:
            self.input_norm = nn.LayerNorm(config.d_model)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=False
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )

        # 多智能体输出头（每个玩家独立）
        self.action_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model // 2, config.num_actions)
            ) for _ in range(config.num_players)
        ])

    def forward(
        self,
        type_ids: torch.Tensor,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
        my_player_token_indices: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            type_ids: (B, T) - token类型ID
            features: (B, T, F) - token特征
            padding_mask: (B, T) - padding掩码 (True表示padding)
            my_player_token_indices: 我方玩家token的索引位置

        Returns:
            action_logits: (B, num_players, num_actions) - 动作logits
        """
        # 1. 输入嵌入
        type_emb = self.type_embedding(type_ids)  # (B, T, D)
        feat_proj = self.feature_projection(features)  # (B, T, D)
        x = type_emb + feat_proj  # (B, T, D)

        # 2. 添加位置编码
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]  # (B, T, D)

        # 3. 输入归一化
        if hasattr(self, 'input_norm'):
            x = self.input_norm(x)

        # 4. Transformer编码
        context = self.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask
        )  # (B, T, D)

        # 5. 提取玩家token的表示
        player_features = []
        for idx in my_player_token_indices:
            player_features.append(context[:, idx, :])  # (B, D)
        player_features = torch.stack(player_features, dim=1)  # (B, num_players, D)

        # 6. 生成动作logits
        action_logits = []
        for i, head in enumerate(self.action_heads):
            logits = head(player_features[:, i, :])  # (B, num_actions)
            action_logits.append(logits)
        action_logits = torch.stack(action_logits, dim=1)  # (B, num_players, num_actions)

        return action_logits

    def _init_weights(self):
        """Xavier初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def count_parameters(self) -> int:
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str):
        """保存模型检查点"""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict(),
        }, path)

    @classmethod
    def load_checkpoint(cls, path: str) -> 'CTFTransformer':
        """加载模型检查点"""
        checkpoint = torch.load(path)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model


def build_ctf_transformer(
    config: Optional[CTFTransformerConfig] = None
) -> CTFTransformer:
    """构建CTF Transformer模型"""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch not installed")

    if config is None:
        config = CTFTransformerConfig()

    model = CTFTransformer(config)
    model._init_weights()

    print(f"✓ Model created with {model.count_parameters():,} parameters")
    return model
