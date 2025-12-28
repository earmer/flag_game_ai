 📋 Game Engine状态管理与转换架构报告

  一、现有架构分析

  1. 数据流向

  WebSocket请求 → game_engine.GameMap → encoding.encode_status_for_team → Transformer模型 → 动作输出
                        ↓
                  sim_env.CTFSim (训练环境)

  2. 现有组件职责

  game_engine.py (GameMap类)
  - 职责: 可视化和基础查询工具
  - 输入: WebSocket的JSON请求 (init/status)
  - 功能:
    - init(): 初始化地图、墙壁、监狱、目标区
    - update(): 更新玩家和旗帜状态
    - list_players(), list_flags(): 查询接口
    - route_to(): BFS路径规划
    - show(): 可视化显示

  sim_env.py (CTFSim类)
  - 职责: 完整的游戏模拟器
  - 功能:
    - reset(): 生成随机地图
    - step(): 执行动作并更新状态
    - status(): 生成标准格式的状态字典
    - init_payload(): 生成初始化数据

  encoding.py
  - 职责: 状态→Token转换
  - 核心函数: encode_status_for_team()
  - 输出: (type_ids, features, padding_mask, my_player_indices)

  ---
  二、状态管理层设计 (state_managing.py)

  核心问题

  1. GameMap只是展示工具,缺少完整的状态管理
  2. CTFSim是训练用模拟器,不适合实际对战
  3. 需要一个中间层来桥接WebSocket输入和AI决策

  state_managing.py 架构设计

  # state_managing.py - 第一部分: 核心数据结构

  from __future__ import annotations
  from dataclasses import dataclass, field
  from typing import Dict, List, Optional, Tuple, Any
  from collections import deque
  import time

  @dataclass
  class GameState:
      """完整的游戏状态快照"""
      timestamp: float  # 游戏时间(ms)

      # 地图静态信息
      width: int
      height: int
      walls: set[Tuple[int, int]]
      my_team: str  # "L" or "R"
      my_prisons: List[Tuple[int, int]]
      my_targets: List[Tuple[int, int]]
      opp_prisons: List[Tuple[int, int]]
      opp_targets: List[Tuple[int, int]]

      # 动态实体
      my_players: List[Dict[str, Any]]
      opp_players: List[Dict[str, Any]]
      my_flags: List[Dict[str, Any]]
      opp_flags: List[Dict[str, Any]]

      # 分数
      my_score: int
      opp_score: int

      # 衍生状态(缓存)
      _my_player_positions: Optional[Dict[str, Tuple[int, int]]] = None
      _opp_player_positions: Optional[Dict[str, Tuple[int, int]]] = None
      def get_my_player_pos(self, name: str) -> Optional[Tuple[int, int]]:
          """获取我方玩家位置"""
          if self._my_player_positions is None:
              self._my_player_positions = {
                  p["name"]: (p["posX"], p["posY"])
                  for p in self.my_players
              }
          return self._my_player_positions.get(name)

      def get_pickable_opp_flags(self) -> List[Dict[str, Any]]:
          """获取可拾取的敌方旗帜"""
          return [f for f in self.opp_flags if f.get("canPickup", False)]

      def get_free_my_players(self) -> List[Dict[str, Any]]:
          """获取未被囚禁的我方玩家"""
          return [p for p in self.my_players if not p.get("inPrison", False)]

  让我继续第二部分:

  # state_managing.py - 第二部分: 状态管理器

  class StateManager:
      """游戏状态管理器 - 连接WebSocket和AI决策"""

      def __init__(self, history_size: int = 10):
          self.current_state: Optional[GameState] = None
          self.history: deque[GameState] = deque(maxlen=history_size)
          self.initialized: bool = False

          # 静态地图信息(init时设置)
          self.width: int = 0
          self.height: int = 0
          self.walls: set[Tuple[int, int]] = set()
          self.my_team: str = ""
          self.my_prisons: List[Tuple[int, int]] = []
          self.my_targets: List[Tuple[int, int]] = []
          self.opp_prisons: List[Tuple[int, int]] = []
          self.opp_targets: List[Tuple[int, int]] = []
      def handle_init(self, init_req: Dict[str, Any]) -> None:
          """处理初始化请求"""
          map_data = init_req["map"]
          self.width = map_data["width"]
          self.height = map_data["height"]

          # 合并walls和obstacles
          self.walls = {
              (w["x"], w["y"])
              for w in (map_data.get("walls", []) + map_data.get("obstacles", []))
          }

          self.my_team = init_req.get("myteamName", "")
          self.my_prisons = [(p["x"], p["y"]) for p in init_req.get("myteamPrison", [])]
          self.my_targets = [(t["x"], t["y"]) for t in init_req.get("myteamTarget", [])]
          self.opp_prisons = [(p["x"], p["y"]) for p in init_req.get("opponentPrison", [])]
          self.opp_targets = [(t["x"], t["y"]) for t in init_req.get("opponentTarget", [])]

          self.initialized = True

      def handle_status(self, status_req: Dict[str, Any]) -> GameState:
          """处理状态更新请求,返回新的GameState"""
          if not self.initialized:
              raise RuntimeError("Must call handle_init() first")

          state = GameState(
              timestamp=status_req.get("time", 0.0),
              width=self.width,
              height=self.height,
              walls=self.walls,
              my_team=self.my_team,
              my_prisons=self.my_prisons,
              my_targets=self.my_targets,
              opp_prisons=self.opp_prisons,
              opp_targets=self.opp_targets,
              my_players=status_req.get("myteamPlayer", []),
              opp_players=status_req.get("opponentPlayer", []),
              my_flags=status_req.get("myteamFlag", []),
              opp_flags=status_req.get("opponentFlag", []),
              my_score=status_req.get("myteamScore", 0),
              opp_score=status_req.get("opponentScore", 0),
          )

          # 保存到历史
          if self.current_state is not None:
              self.history.append(self.current_state)
          self.current_state = state

          return state

      def get_state_for_encoding(self) -> Dict[str, Any]:
          """
          将当前GameState转换为encoding.py需要的格式
          这是关键的转换接口!
          """
          if self.current_state is None:
              raise RuntimeError("No current state available")

          state = self.current_state

          # 构造encoding.encode_status_for_team()需要的字典格式
          return {
              "time": state.timestamp,
              "myteamPlayer": state.my_players,
              "opponentPlayer": state.opp_players,
              "myteamFlag": state.my_flags,
              "opponentFlag": state.opp_flags,
              "myteamScore": state.my_score,
              "opponentScore": state.opp_score,
              # 注意: encoding.py需要这些额外字段用于计算距离特征
              "_myteamTarget": [{"x": x, "y": y} for x, y in state.my_targets],
              "_myteamPrison": [{"x": x, "y": y} for x, y in state.my_prisons],
          }

  继续第三部分:

  # state_managing.py - 第三部分: 输出转换器

  class ActionConverter:
      """动作输出转换器 - 将模型输出转换为WebSocket响应"""

      @staticmethod
      def model_output_to_moves(
          player_names: List[str],
          action_logits: Any,  # torch.Tensor or numpy array
          action_vocab: List[str] = ["", "up", "down", "left", "right"]
      ) -> List[Dict[str, str]]:
          """
          将模型输出的动作logits转换为moves列表

          Args:
              player_names: 玩家名称列表 ["L0", "L1", "L2"]
              action_logits: 形状为 (num_players, num_actions) 的张量
              action_vocab: 动作词汇表

          Returns:
              [{"name": "L0", "move": "up"}, {"name": "L1", "move": "right"}, ...]
          """
          try:
              import torch
              if isinstance(action_logits, torch.Tensor):
                  action_indices = action_logits.argmax(dim=-1).cpu().numpy()
              else:
                  import numpy as np
                  action_indices = np.argmax(action_logits, axis=-1)
          except ImportError:
              # 如果没有torch,假设是numpy
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


  class TokenConverter:
      """Token转换器 - 封装encoding.py的调用"""

      def __init__(self, max_tokens: int = 32):
          self.max_tokens = max_tokens

      def state_to_tokens(
          self,
          state_dict: Dict[str, Any],
          geometry: Any  # Geometry对象
      ) -> Tuple[List[int], List[List[float]], List[bool], Tuple[int, ...]]:
          """
          将状态字典转换为tokens

          这是对encoding.encode_status_for_team()的封装
          """
          from ctf_ai.encoding import encode_status_for_team

          return encode_status_for_team(
              status_req=state_dict,
              geometry=geometry,
              max_tokens=self.max_tokens
          )

      def tokens_to_batch(
          self,
          encoded_list: List[Tuple[List[int], List[List[float]], List[bool], Tuple[int, ...]]]
      ):
          """将多个编码结果转换为batch"""
          from ctf_ai.encoding import to_torch_batch
          return to_torch_batch(encoded_list)

  最后一部分 - 完整的使用示例:

  # state_managing.py - 第四部分: 集成示例

  class GameStateProcessor:
      """完整的状态处理流程 - 从WebSocket到AI决策"""

      def __init__(self, geometry: Any, model: Any, max_tokens: int = 32):
          """
          Args:
              geometry: Geometry对象(来自tree_features.py)
              model: 训练好的Transformer模型
              max_tokens: token序列最大长度
          """
          self.state_manager = StateManager(history_size=10)
          self.token_converter = TokenConverter(max_tokens=max_tokens)
          self.action_converter = ActionConverter()
          self.geometry = geometry
          self.model = model

      def process_init(self, init_req: Dict[str, Any]) -> None:
          """处理初始化"""
          self.state_manager.handle_init(init_req)
      def process_status_and_decide(
          self,
          status_req: Dict[str, Any]
      ) -> Dict[str, Any]:
          """
          完整流程: 状态更新 → Token化 → 模型推理 → 动作转换

          Returns:
              WebSocket响应格式: {"players": [{"name": "L0", "move": "up"}, ...]}
          """
          # 1. 更新状态
          game_state = self.state_manager.handle_status(status_req)

          # 2. 转换为encoding格式
          state_dict = self.state_manager.get_state_for_encoding()

          # 3. Token化
          type_ids, features, padding_mask, my_player_indices = \
              self.token_converter.state_to_tokens(state_dict, self.geometry)

          # 4. 转换为batch(如果需要)
          batch = self.token_converter.tokens_to_batch([
              (type_ids, features, padding_mask, my_player_indices)
          ])

          # 5. 模型推理
          with torch.no_grad():
              action_logits = self.model(
                  type_ids=batch.type_ids,
                  features=batch.features,
                  padding_mask=batch.padding_mask,
                  my_player_token_indices=batch.my_player_token_indices
              )

          # 6. 提取我方玩家名称
          player_names = [p["name"] for p in game_state.my_players]

          # 7. 转换为moves
          moves = self.action_converter.model_output_to_moves(
              player_names,
              action_logits
          )

          # 8. 构造响应
          return self.action_converter.create_websocket_response(moves)


  # ============ 使用示例 ============

  def example_usage():
      """完整使用示例"""
      from ctf_ai.transformer_model import YourTransformerModel
      from lib.tree_features import Geometry

      # 初始化
      geometry = Geometry(width=20, height=20, ...)
      model = YourTransformerModel.load_from_checkpoint("model.pth")
      processor = GameStateProcessor(geometry, model)

      # WebSocket handler中使用
      async def websocket_handler(websocket):
          async for msg in websocket:
              req = json.loads(msg)

              if req["action"] == "init":
                  processor.process_init(req)
              elif req["action"] == "status":
                  response = processor.process_status_and_decide(req)
                  await websocket.send(json.dumps(response))

  ---
  三、关键设计要点

  1. 状态管理层的职责

  - ✅ 维护完整的游戏状态快照
  - ✅ 提供历史状态查询(用于时序决策)
  - ✅ 缓存常用查询结果(性能优化)
  - ✅ 桥接WebSocket格式和encoding格式

  2. 转换层的职责

  - ✅ TokenConverter: 封装encoding.py的调用
  - ✅ ActionConverter: 模型输出→WebSocket响应
  - ✅ 解耦模型实现和通信协议

  3. 与现有代码的对接

  WebSocket JSON↓
  StateManager.handle_status()
      ↓
  GameState对象
      ↓
  StateManager.get_state_for_encoding()
      ↓
  encoding.encode_status_for_team()
      ↓
  (type_ids, features, mask, indices)
      ↓
  Transformer模型
      ↓
  action_logits
      ↓
  ActionConverter.model_output_to_moves()
      ↓
  WebSocket响应

  这个设计完整地解决了状态管理和输出转换的问题,同时保持了与现有encoding.py和sim_env.py的兼容性。

---
四、Token转换详细实现

1. encoding.py核心机制解析

Token类型定义 (encoding.py:17-24)

ENTITY_TYPES = {
    "global": 0,        # 全局状态token
    "my_player": 1,     # 我方玩家token
    "opp_player": 2,    # 敌方玩家token
    "opp_flag": 3,      # 可拾取的敌方旗帜token
    "my_target": 4,     # 我方目标区token
    "my_prison": 5,     # 我方监狱token
}

Token结构

每个token是一个元组: (type_id, features)
- type_id: 实体类型ID (0-5)
- features: 特征向量 (长度根据实体类型不同而不同)

2. encode_status_for_team() 函数详解

函数签名 (encoding.py:49-54)

def encode_status_for_team(
    status_req: Mapping[str, Any],      # WebSocket状态请求
    geometry: Geometry,                  # 地图几何信息
    *,
    max_tokens: int = 32,               # 最大token数量
) -> Tuple[List[int], List[List[float]], List[bool], Tuple[int, ...]]:

返回值说明

- type_ids: List[int] - 每个token的类型ID
- features: List[List[float]] - 每个token的特征向量
- padding_mask: List[bool] - padding掩码 (True表示padding)
- my_player_indices: Tuple[int, ...] - 我方玩家token的索引位置

Token生成顺序 (encoding.py:60-146)

# 1. Global Token (索引0)
全局特征 (7维):
- my_score: 我方得分
- opp_score: 敌方得分
- num_my_prisoners: 我方被囚禁人数
- num_opp_prisoners: 敌方被囚禁人数
- num_opp_flags: 可拾取的敌方旗帜数量
- map_width: 地图宽度
- map_height: 地图高度

# 2. My Player Tokens (索引1-3, 假设3个玩家)
我方玩家特征 (8维):
- pos_x_norm: 归一化X坐标 (0-1)
- pos_y_norm: 归一化Y坐标 (0-1)
- has_flag: 是否持旗 (0/1)
- in_prison: 是否被囚禁 (0/1)
- dist_to_opp_flags: 到最近敌方旗帜的距离 (归一化)
- dist_to_my_target: 到我方目标区的距离 (归一化)
- dist_to_my_prison: 到我方监狱的距离 (归一化)
- dist_to_opp_players: 到最近敌方玩家的距离 (归一化)

# 3. Opp Player Tokens (索引4-6, 假设3个玩家)
敌方玩家特征 (4维):
- pos_x_norm: 归一化X坐标 (0-1)
- pos_y_norm: 归一化Y坐标 (0-1)
- has_flag: 是否持旗 (0/1)
- in_prison: 是否被囚禁 (0/1)

# 4. Opp Flag Tokens (可变数量)
敌方旗帜特征 (2维):
- pos_x_norm: 归一化X坐标 (0-1)
- pos_y_norm: 归一化Y坐标 (0-1)

# 5. My Target Token (1个)
我方目标区特征 (2维):
- pos_x_norm: 归一化X坐标 (0-1)
- pos_y_norm: 归一化Y坐标 (0-1)

# 6. My Prison Token (1个)
我方监狱特征 (2维):
- pos_x_norm: 归一化X坐标 (0-1)
- pos_y_norm: 归一化Y坐标 (0-1)

3. Padding机制 (encoding.py:148-156)

# 截断超长序列
tokens = tokens[:max_tokens]

# Padding到max_tokens长度
while len(type_ids) < max_tokens:
    type_ids.append(0)                      # padding type_id = 0
    feats.append([0.0] * len(feats[0]))    # 全0特征向量
    padding_mask.append(True)               # 标记为padding

关键点:
- padding_mask[i] == True 表示该位置是padding，会被Transformer的attention机制忽略
- padding的type_id设为0 (global类型)，但由于mask的存在不会影响计算

4. 坐标归一化机制 (tree_features.py:51-61)

Geometry类的normalize_pos()方法:

def normalize_pos(self, pos: Tuple[int, int]) -> Tuple[int, int]:
    x, y = pos
    if self.my_side_is_left:
        return x, y
    # 如果我方在右侧，将坐标镜像翻转到左侧视角
    return (self.width - 1 - x), y

作用:
- 统一视角：无论我方在左侧还是右侧，都转换为"我方在左侧"的视角
- 简化模型：模型只需学习一种视角的策略
- 对称性：左右两队使用同一个模型

5. to_torch_batch() 批处理转换 (encoding.py:162-191)

def to_torch_batch(
    encoded: Sequence[Tuple[List[int], List[List[float]], List[bool], Tuple[int, ...]]]
) -> EncodedBatch:
    """将多个编码结果转换为PyTorch batch"""

    # 转换为Tensor
    type_tensor = torch.tensor([e[0] for e in encoded], dtype=torch.long)      # (B, T)
    feat_tensor = torch.tensor([e[1] for e in encoded], dtype=torch.float32)   # (B, T, F)
    pad_mask = torch.tensor([e[2] for e in encoded], dtype=torch.bool)         # (B, T)
    my_player_idx = encoded[0][3]  # 假设batch中所有样本的玩家索引相同

    return EncodedBatch(
        type_ids=type_tensor,
        features=feat_tensor,
        padding_mask=pad_mask,
        my_player_token_indices=my_player_idx,
    )

返回的EncodedBatch数据类:
- type_ids: (B, T) - batch中每个样本的type_id序列
- features: (B, T, F) - batch中每个样本的特征矩阵
- padding_mask: (B, T) - batch中每个样本的padding掩码
- my_player_token_indices: Tuple[int, ...] - 我方玩家token的索引