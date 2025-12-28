from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Roles:
    STEAL: int = 0
    RETURN: int = 1
    RESCUE: int = 2
    CHASE: int = 3
    DEFEND: int = 4

    @staticmethod
    def names() -> list[str]:
        return ["steal", "return", "rescue", "chase", "defend"]

