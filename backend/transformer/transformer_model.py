from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import warnings

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True

    # Suppress PyTorch nested tensor prototype warning
    # This warning is informational - nested tensors are an internal optimization
    # that PyTorch uses automatically for better performance with padding masks.
    # The API is stable enough for our use case.
    warnings.filterwarnings(
        "ignore",
        message=".*nested tensors.*",
        category=UserWarning,
        module="torch.nn.modules.transformer"
    )
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
    d_model: int = 96                  # 模型维度 (从128降至96以支持8层)
    nhead: int = 8                     # 注意力头数 (从4增至8)
    num_layers: int = 8                # Transformer层数 (从2增至8)
    dim_feedforward: int = 192         # FFN维度 (默认值，用于向后兼容)
    dropout: float = 0.1               # Dropout率

    # 可变FFN宽度（漏斗架构：宽→窄）
    dim_feedforward_per_layer: Optional[Tuple[int, ...]] = None

    # 输出参数
    num_players: int = 3               # 玩家数量
    num_actions: int = 5               # 动作数量 (stay/up/down/left/right)

    # 训练参数
    use_layer_norm: bool = True        # 是否使用LayerNorm
    activation: str = "gelu"           # 激活函数

    def __post_init__(self):
        """验证配置并设置默认的可变FFN宽度"""
        # 如果未指定dim_feedforward_per_layer，使用漏斗模式
        if self.dim_feedforward_per_layer is None:
            # 8层漏斗架构: 192, 192, 160, 160, 128, 128, 96, 96
            if self.num_layers == 8:
                default_widths = (192, 192, 160, 160, 128, 128, 96, 96)
            else:
                # 其他层数：使用统一的dim_feedforward
                default_widths = tuple([self.dim_feedforward] * self.num_layers)

            object.__setattr__(self, 'dim_feedforward_per_layer', default_widths)

        # 验证：长度必须匹配num_layers
        if len(self.dim_feedforward_per_layer) != self.num_layers:
            raise ValueError(
                f"dim_feedforward_per_layer length ({len(self.dim_feedforward_per_layer)}) "
                f"must match num_layers ({self.num_layers})"
            )


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

        # Transformer编码器层（可变FFN宽度）
        # 每层独立创建，支持不同的dim_feedforward
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward_per_layer[i],  # 可变宽度
                dropout=config.dropout,
                activation=config.activation,
                batch_first=True,
                norm_first=False
            )
            for i in range(config.num_layers)
        ])

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
        # 0. 验证特征维度
        if features.size(-1) != self.config.feature_dim:
            raise ValueError(
                f"Feature dim mismatch: expected {self.config.feature_dim}, "
                f"got {features.size(-1)}. Check encoding.py compatibility."
            )

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

        # 4. Transformer编码（手动迭代各层以支持可变FFN宽度）
        context = x
        for layer in self.encoder_layers:
            context = layer(
                context,
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
