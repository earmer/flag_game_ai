"""
sim_env_wrapper.py

Python wrapper for CTF Simulator C library.
Provides unified interface with automatic fallback to Python implementation.
"""

import ctypes
import json
import platform
import struct
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

# Type aliases
Action = str
Pos = Tuple[int, int]

# Action encoding for binary protocol
ACTION_TO_CODE = {"": 0, "up": 1, "down": 2, "left": 3, "right": 4}


class CTFSimC:
    """C implementation wrapper using ctypes."""

    _lib = None
    _lib_path = None

    @classmethod
    def _load_library(cls):
        """Load the C library (singleton pattern)."""
        if cls._lib is not None:
            return cls._lib

        # Determine library path
        base_dir = Path(__file__).parent / "sim_env_c" / "build"
        system = platform.system()

        if system == "Darwin":
            lib_name = "libctfsim.dylib"
        else:
            lib_name = "libctfsim.so"

        lib_path = base_dir / lib_name
        if not lib_path.exists():
            raise FileNotFoundError(f"C library not found: {lib_path}")

        # Load library
        cls._lib = ctypes.CDLL(str(lib_path))
        cls._lib_path = lib_path

        # Define function signatures
        cls._lib.ctf_sim_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        cls._lib.ctf_sim_create.restype = ctypes.c_void_p

        cls._lib.ctf_sim_destroy.argtypes = [ctypes.c_void_p]
        cls._lib.ctf_sim_destroy.restype = None

        cls._lib.ctf_sim_reset.argtypes = [ctypes.c_void_p]
        cls._lib.ctf_sim_reset.restype = ctypes.c_char_p

        cls._lib.ctf_sim_step.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        cls._lib.ctf_sim_step.restype = ctypes.c_char_p

        cls._lib.ctf_sim_step_binary.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte)]
        cls._lib.ctf_sim_step_binary.restype = ctypes.c_char_p

        cls._lib.ctf_sim_status.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        cls._lib.ctf_sim_status.restype = ctypes.c_char_p

        cls._lib.ctf_sim_init_payload.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        cls._lib.ctf_sim_init_payload.restype = ctypes.c_char_p

        cls._lib.ctf_sim_done.argtypes = [ctypes.c_void_p]
        cls._lib.ctf_sim_done.restype = ctypes.c_int

        cls._lib.ctf_sim_l_score.argtypes = [ctypes.c_void_p]
        cls._lib.ctf_sim_l_score.restype = ctypes.c_int

        cls._lib.ctf_sim_r_score.argtypes = [ctypes.c_void_p]
        cls._lib.ctf_sim_r_score.restype = ctypes.c_int

        cls._lib.ctf_sim_step_count.argtypes = [ctypes.c_void_p]
        cls._lib.ctf_sim_step_count.restype = ctypes.c_int

        return cls._lib

    def __init__(
        self,
        *,
        width: int = 20,
        height: int = 20,
        num_players: int = 3,
        num_flags: int = 9,
        seed: Optional[int] = None,
        **kwargs  # Ignore extra args for compatibility
    ):
        self._lib = self._load_library()
        seed_val = seed if seed is not None else -1
        self._handle = self._lib.ctf_sim_create(width, height, num_players, num_flags, seed_val)
        if not self._handle:
            raise RuntimeError("Failed to create CTF simulator")

        self.width = width
        self.height = height
        self.num_players = num_players
        self.num_flags = num_flags

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            self._lib.ctf_sim_destroy(self._handle)
            self._handle = None

    def reset(self) -> None:
        """Reset game to initial state."""
        self._lib.ctf_sim_reset(self._handle)

    def step(self, l_actions: Mapping[str, Action], r_actions: Mapping[str, Action]) -> None:
        """Execute one game step."""
        actions = dict(l_actions)
        actions.update(r_actions)
        actions_json = json.dumps(actions).encode('utf-8')
        self._lib.ctf_sim_step(self._handle, actions_json)

    def step_fast(self, l_actions: Mapping[str, Action], r_actions: Mapping[str, Action]) -> None:
        """Execute one game step using binary protocol (faster)."""
        # Convert actions to binary: [L0, L1, L2, R0, R1, R2]
        action_codes = (ctypes.c_ubyte * 6)(
            ACTION_TO_CODE.get(l_actions.get("L0", ""), 0),
            ACTION_TO_CODE.get(l_actions.get("L1", ""), 0),
            ACTION_TO_CODE.get(l_actions.get("L2", ""), 0),
            ACTION_TO_CODE.get(r_actions.get("R0", ""), 0),
            ACTION_TO_CODE.get(r_actions.get("R1", ""), 0),
            ACTION_TO_CODE.get(r_actions.get("R2", ""), 0),
        )
        self._lib.ctf_sim_step_binary(self._handle, action_codes)

    def status(self, my_team: str) -> Dict[str, object]:
        """Get current game status for a team."""
        result = self._lib.ctf_sim_status(self._handle, my_team.encode('utf-8'))
        if result:
            return json.loads(result.decode('utf-8'))
        return {}

    def init_payload(self, my_team: str) -> Dict[str, object]:
        """Get init payload for a team."""
        result = self._lib.ctf_sim_init_payload(self._handle, my_team.encode('utf-8'))
        if result:
            return json.loads(result.decode('utf-8'))
        return {}

    @property
    def done(self) -> bool:
        return bool(self._lib.ctf_sim_done(self._handle))

    @property
    def l_score(self) -> int:
        return self._lib.ctf_sim_l_score(self._handle)

    @property
    def r_score(self) -> int:
        return self._lib.ctf_sim_r_score(self._handle)

    @property
    def step_count(self) -> int:
        return self._lib.ctf_sim_step_count(self._handle)


# ============================================================
# Unified Interface with Fallback
# ============================================================

_USE_C_IMPL = None  # None = auto-detect, True = force C, False = force Python


def _check_c_available() -> bool:
    """Check if C implementation is available."""
    try:
        CTFSimC._load_library()
        return True
    except (FileNotFoundError, OSError):
        return False


def set_implementation(use_c: Optional[bool] = None):
    """
    Set which implementation to use.

    Args:
        use_c: True = force C, False = force Python, None = auto-detect
    """
    global _USE_C_IMPL
    _USE_C_IMPL = use_c


def get_implementation() -> str:
    """Get current implementation name."""
    global _USE_C_IMPL
    if _USE_C_IMPL is True:
        return "C"
    elif _USE_C_IMPL is False:
        return "Python"
    else:
        return "C" if _check_c_available() else "Python"


def create_simulator(**kwargs) -> "CTFSim":
    """
    Create a CTF simulator instance.

    Automatically selects C or Python implementation based on availability.
    """
    global _USE_C_IMPL

    use_c = _USE_C_IMPL
    if use_c is None:
        use_c = _check_c_available()

    if use_c:
        return CTFSimC(**kwargs)
    else:
        from sim_env import CTFSim as CTFSimPy
        return CTFSimPy(**kwargs)
