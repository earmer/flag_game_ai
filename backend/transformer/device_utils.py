"""
Device management utilities for automatic GPU/CPU detection and handling.

Provides elegant, centralized device management for training and inference.
"""
import torch
from typing import Optional
import logging

# Configure logging to ensure INFO messages are displayed
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'  # Simple format for device messages
)

logger = logging.getLogger(__name__)


def get_device(verbose: bool = True) -> torch.device:
    """
    Auto-detect best available device with simplified logic.

    Priority: CUDA > MPS > CPU

    Args:
        verbose: Print device info to logger

    Returns:
        torch.device: Best available device
    """
    # Try CUDA first (works on all platforms)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            try:
                name = torch.cuda.get_device_name(0)
                memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                logger.info(f"✓ Using CUDA: {name} ({memory:.2f} GB)")
            except Exception as e:
                # Fallback if device info retrieval fails
                logger.info(f"✓ Using CUDA (device info unavailable: {e})")
        return device

    # Try MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            logger.info("✓ Using Metal Performance Shaders (MPS)")
        return device

    # Fallback to CPU
    if verbose:
        logger.info("✓ Using CPU (no GPU detected)")
    return torch.device("cpu")


def move_to_device(model: torch.nn.Module, device: Optional[torch.device] = None) -> torch.nn.Module:
    """
    Move model to device, auto-detecting if not specified.

    Handles multiprocessing case where models unpickle to CPU.

    Args:
        model: PyTorch model
        device: Target device (auto-detect if None)

    Returns:
        Model on target device
    """
    if device is None:
        device = get_device(verbose=False)
    return model.to(device)


class DeviceManager:
    """
    Centralized device management for training.

    Singleton pattern ensures consistent device usage across the system.
    """

    _instance = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def device(self) -> torch.device:
        """Get current device (lazy initialization)."""
        if self._device is None:
            self._device = get_device(verbose=True)
        return self._device

    def set_device(self, device: torch.device):
        """Manually set device."""
        self._device = device
        logger.info(f"Device manually set to: {device}")

    def move_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Move model to managed device."""
        return model.to(self.device)
