"""
_import_bootstrap.py - Centralized import configuration for transformer scripts

This module handles:
1. sys.path setup to find lib/ module
2. Optional dependency detection (torch, numpy)
3. Lazy-loaded helper functions for common imports
"""
import sys
import os

# ============================================================
# Path Setup (runs once on first import)
# ============================================================

_current_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.dirname(_current_dir)

if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# ============================================================
# Optional Dependencies
# ============================================================

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

# ============================================================
# Lazy Import Helpers (cached on first call)
# ============================================================

_Geometry = None

def get_geometry():
    """Get Geometry class from lib.tree_features (cached)"""
    global _Geometry
    if _Geometry is None:
        try:
            from lib.tree_features import Geometry
            _Geometry = Geometry
        except ImportError as e:
            raise ImportError(
                f"Cannot import Geometry from lib.tree_features. "
                f"Make sure you're running from backend/transformer/ directory.\n"
                f"Original error: {e}"
            )
    return _Geometry
