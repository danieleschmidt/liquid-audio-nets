"""Liquid Audio Neural Networks - Edge-efficient audio processing.

This package provides Liquid Neural Network implementations optimized for
ultra-low-power audio processing on edge devices.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.dev"

from typing import Optional

try:
    from ._core import *  # Rust/C++ core bindings
except ImportError:
    # Fallback for development without compiled bindings
    pass

from .lnn import LNN, AdaptiveConfig

# Optional imports - only load if dependencies are available
try:
    from .training import LNNTrainer
    _HAS_TRAINING = True
except ImportError:
    _HAS_TRAINING = False
    LNNTrainer = None

try:
    from .tools import profiler, compression
    _HAS_TOOLS = True
except ImportError:
    _HAS_TOOLS = False
    profiler = None
    compression = None

__all__ = ["LNN", "AdaptiveConfig"]

if _HAS_TRAINING and LNNTrainer:
    __all__.append("LNNTrainer")

if _HAS_TOOLS:
    if profiler:
        __all__.append("profiler")
    if compression:
        __all__.append("compression")