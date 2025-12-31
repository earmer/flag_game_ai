"""
hall_of_fame.py

Hall of Fame系统 - 保存历史冠军模型
- 保存模型state_dict和元数据
- 提供对手采样接口
- 支持持久化到磁盘
"""

import json
import random
import torch
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List


class HallOfFame:
    """
    Hall of Fame - 保存历史冠军模型

    用途：
    1. 保存每个阶段的冠军模型
    2. 作为训练对手采样源（防止策略漂移）
    3. 持久化到磁盘
    """

    def __init__(self, max_size: int = 10):
        """
        Args:
            max_size: 最大成员数（超过后删除最弱的）
        """
        self.max_size = max_size
        self.members: List[Dict[str, Any]] = []

    def add_champion(
        self,
        model_state_dict: Dict[str, torch.Tensor],
        stage: int,
        generation: int,
        win_rate: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        添加冠军到HoF

        Args:
            model_state_dict: 模型的state_dict
            stage: 训练阶段
            generation: 世代数
            win_rate: 胜率
            metadata: 额外的元数据
        """
        # 创建成员条目
        member = {
            'model_state_dict': model_state_dict,
            'stage': stage,
            'generation': generation,
            'win_rate': win_rate,
            'added_time': datetime.now().isoformat(),
            'metadata': metadata or {}
        }

        self.members.append(member)

        # 按胜率排序（降序）
        self.members.sort(key=lambda x: x['win_rate'], reverse=True)

        # 限制大小
        if len(self.members) > self.max_size:
            removed = self.members.pop()
            print(f"HoF已满，移除最弱成员: Stage {removed['stage']}, Gen {removed['generation']}, WR {removed['win_rate']:.2%}")

    def sample_opponent(self) -> Optional[Dict[str, Any]]:
        """
        随机采样一个HoF成员

        Returns:
            成员字典，如果HoF为空则返回None
        """
        if not self.members:
            return None

        return random.choice(self.members)

    def get_top_k(self, k: int) -> List[Dict[str, Any]]:
        """
        获取前K个最强成员

        Args:
            k: 数量

        Returns:
            前K个成员列表
        """
        return self.members[:min(k, len(self.members))]

    def save(self, save_path: str) -> None:
        """
        保存HoF到磁盘

        Args:
            save_path: 保存路径（目录）
        """
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存每个成员的模型
        for i, member in enumerate(self.members):
            model_path = save_dir / f"member_{i}.pth"
            torch.save(member['model_state_dict'], model_path)

        # 保存元数据
        metadata = []
        for i, member in enumerate(self.members):
            metadata.append({
                'index': i,
                'stage': member['stage'],
                'generation': member['generation'],
                'win_rate': member['win_rate'],
                'added_time': member['added_time'],
                'metadata': member['metadata']
            })

        metadata_path = save_dir / "hof_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"HoF已保存到: {save_path} ({len(self.members)}个成员)")

    def load(self, load_path: str) -> None:
        """
        从磁盘加载HoF

        Args:
            load_path: 加载路径（目录）
        """
        load_dir = Path(load_path)

        if not load_dir.exists():
            print(f"警告: HoF路径不存在: {load_path}")
            return

        # 加载元数据
        metadata_path = load_dir / "hof_metadata.json"
        if not metadata_path.exists():
            print(f"警告: HoF元数据文件不存在: {metadata_path}")
            return

        with open(metadata_path, 'r') as f:
            metadata_list = json.load(f)

        # 加载每个成员
        self.members = []
        for meta in metadata_list:
            model_path = load_dir / f"member_{meta['index']}.pth"
            if not model_path.exists():
                print(f"警告: 模型文件不存在: {model_path}")
                continue

            model_state_dict = torch.load(model_path, map_location='cpu')

            member = {
                'model_state_dict': model_state_dict,
                'stage': meta['stage'],
                'generation': meta['generation'],
                'win_rate': meta['win_rate'],
                'added_time': meta['added_time'],
                'metadata': meta.get('metadata', {})
            }
            self.members.append(member)

        print(f"HoF已加载: {len(self.members)}个成员")

    def __len__(self) -> int:
        """返回成员数量"""
        return len(self.members)

    def is_empty(self) -> bool:
        """检查HoF是否为空"""
        return len(self.members) == 0

    def get_summary(self) -> str:
        """获取HoF摘要信息"""
        if self.is_empty():
            return "HoF: 空"

        summary = f"HoF: {len(self.members)}个成员\n"
        for i, member in enumerate(self.members[:5]):  # 只显示前5个
            summary += f"  {i+1}. Stage {member['stage']}, Gen {member['generation']}, WR {member['win_rate']:.2%}\n"

        if len(self.members) > 5:
            summary += f"  ... 还有{len(self.members) - 5}个成员\n"

        return summary

