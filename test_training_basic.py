#!/usr/bin/env python3
"""Basic test for Python training framework without heavy dependencies."""

import sys
import os
sys.path.insert(0, 'python')

def test_imports():
    """Test that we can import the training modules."""
    try:
        from liquid_audio_nets.training import TrainingConfig, LiquidNeuralNetworkPyTorch, TimestepController
        print("✅ Training framework imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_creation():
    """Test training configuration creation."""
    try:
        from liquid_audio_nets.training import TrainingConfig
        
        config = TrainingConfig(
            input_dim=40,
            hidden_dim=64,
            output_dim=10,
            learning_rate=1e-3,
            batch_size=32
        )
        
        assert config.input_dim == 40
        assert config.hidden_dim == 64
        assert config.output_dim == 10
        assert config.learning_rate == 1e-3
        assert config.batch_size == 32
        
        print("✅ TrainingConfig creation successful")
        return True
    except Exception as e:
        print(f"❌ TrainingConfig error: {e}")
        return False

def test_synthetic_data():
    """Test synthetic data generation."""
    try:
        from liquid_audio_nets.training import create_synthetic_dataset
        
        X, y = create_synthetic_dataset(num_samples=100, input_dim=40, output_dim=10)
        
        assert X.shape == (100, 40), f"Expected (100, 40), got {X.shape}"
        assert y.shape == (100,), f"Expected (100,), got {y.shape}"
        assert y.max() < 10, f"Labels should be < 10, got max {y.max()}"
        
        print("✅ Synthetic data generation successful")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency for synthetic data: {e}")
        return False
    except Exception as e:
        print(f"❌ Synthetic data error: {e}")
        return False

def test_tools_import():
    """Test that tools modules can be imported."""
    try:
        from liquid_audio_nets.tools import profiler, compression
        print("✅ Tools import successful")
        return True
    except ImportError as e:
        print(f"❌ Tools import error: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Python liquid-audio-nets package...\n")
    
    tests = [
        test_imports,
        test_config_creation, 
        test_synthetic_data,
        test_tools_import,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())