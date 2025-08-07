"""
Pytest configuration and fixtures for liquid-audio-nets tests.
"""
import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import torch


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_audio() -> np.ndarray:
    """Generate sample audio data for testing."""
    sample_rate = 16000
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate a simple sine wave with some noise
    frequency = 440.0  # A4 note
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)
    audio += 0.1 * np.random.normal(0, 1, len(t))
    
    return audio.astype(np.float32)


@pytest.fixture
def sample_features() -> np.ndarray:
    """Generate sample MFCC features for testing."""
    n_frames = 100
    n_features = 40
    features = np.random.randn(n_frames, n_features).astype(np.float32)
    return features


@pytest.fixture
def mock_model_config() -> dict:
    """Mock configuration for LNN model."""
    return {
        'input_dim': 40,
        'hidden_dim': 64,
        'output_dim': 10,
        'ode_solver': 'adaptive_heun',
        'timestep_range': (0.001, 0.050),
        'complexity_penalty': 0.01
    }


@pytest.fixture
def device() -> torch.device:
    """Get the appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducible tests."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def embedded_test_data():
    """Sample data sized for embedded testing."""
    return {
        'audio_frame': np.random.randn(256).astype(np.float32),
        'features': np.random.randn(16, 13).astype(np.float32),
        'expected_output': np.array([0.8, 0.2], dtype=np.float32)
    }


# Markers are now configured in pytest.ini


def pytest_configure(config):
    """Configure pytest with custom options."""
    # Skip embedded tests if hardware not available
    if not os.environ.get('EMBEDDED_HARDWARE_AVAILABLE'):
        config.addinivalue_line("markers", "embedded: requires embedded hardware")
    
    # Skip GPU tests if CUDA not available
    if not torch.cuda.is_available():
        config.addinivalue_line("markers", "gpu: requires CUDA GPU")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle markers."""
    # Skip embedded tests if hardware not available
    if not os.environ.get('EMBEDDED_HARDWARE_AVAILABLE'):
        skip_embedded = pytest.mark.skip(reason="Embedded hardware not available")
        for item in items:
            if "embedded" in item.keywords:
                item.add_marker(skip_embedded)
    
    # Skip GPU tests if CUDA not available
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA GPU not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)