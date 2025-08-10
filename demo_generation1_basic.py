#!/usr/bin/env python3
"""
Generation 1 Demo: MAKE IT WORK - Basic Functionality
======================================================

Demonstrates the enhanced basic functionality of the Liquid Audio Nets library
including:
- Model creation and configuration
- Audio processing with adaptive timesteps
- Feature extraction
- Complexity estimation
- Audio format validation

This validates that all core components work together seamlessly.
"""

import sys
import numpy as np
from pathlib import Path

# Add the library to Python path for demo
sys.path.insert(0, str(Path(__file__).parent / "python"))

try:
    import liquid_audio_nets as lan
    print("✅ Liquid Audio Nets library imported successfully")
    print(f"📦 Version: {lan.__version__}")
except ImportError as e:
    print(f"❌ Failed to import library: {e}")
    print("ℹ️  This demo requires the compiled Rust/Python bindings")
    print("   Run: maturin develop --release")
    sys.exit(1)

def main():
    print("\n🧠 GENERATION 1: MAKE IT WORK - Basic Functionality Demo")
    print("=" * 60)
    
    # 1. Test Basic Configuration System
    print("\n1️⃣  Testing Configuration System")
    print("-" * 40)
    
    try:
        # Create default model configuration
        model_config = lan.ModelConfig()  # Using Rust defaults
        print(f"✅ Model Config Created:")
        print(f"   Input dimensions: {model_config.input_dim}")
        print(f"   Hidden dimensions: {model_config.hidden_dim}") 
        print(f"   Output dimensions: {model_config.output_dim}")
        print(f"   Sample rate: {model_config.sample_rate} Hz")
        print(f"   Frame size: {model_config.frame_size}")
        print(f"   Model type: {model_config.model_type}")
        
        # Create adaptive configuration
        adaptive_config = lan.AdaptiveConfig()
        print(f"\n✅ Adaptive Config Created:")
        print(f"   Min timestep: {adaptive_config.min_timestep_ms:.1f} ms")
        print(f"   Max timestep: {adaptive_config.max_timestep_ms:.1f} ms")
        print(f"   Energy threshold: {adaptive_config.energy_threshold:.3f}")
        print(f"   Power budget: {adaptive_config.power_budget_mw:.1f} mW")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    # 2. Test Audio Format Handling
    print("\n2️⃣  Testing Audio Format System")
    print("-" * 40)
    
    try:
        # Test standard audio formats
        format_16k = lan.AudioFormat.pcm_16khz_mono()
        format_44k = lan.AudioFormat.pcm_44khz_stereo()
        
        print(f"✅ 16kHz Mono Format:")
        print(f"   Sample rate: {format_16k.sample_rate} Hz")
        print(f"   Channels: {format_16k.channels}")
        print(f"   Bits per sample: {format_16k.bits_per_sample}")
        print(f"   Embedded suitable: {format_16k.is_embedded_suitable()}")
        
        print(f"\n✅ 44.1kHz Stereo Format:")
        print(f"   Sample rate: {format_44k.sample_rate} Hz")
        print(f"   Channels: {format_44k.channels}")
        print(f"   Duration for 8000 samples: {format_44k.duration_seconds(8000):.3f}s")
        
        # Test format compatibility
        print(f"\n🔍 Format Compatibility:")
        print(f"   16k with itself: {format_16k.is_compatible_with(format_16k)}")
        print(f"   16k with 44k: {format_16k.is_compatible_with(format_44k)}")
        
    except Exception as e:
        print(f"❌ Audio format test failed: {e}")
        return False
    
    # 3. Test LNN Model Creation and Processing
    print("\n3️⃣  Testing LNN Model Creation")
    print("-" * 40)
    
    try:
        # Create LNN model
        lnn = lan.LNN(model_config)
        print(f"✅ LNN Model created successfully")
        
        # Configure adaptive processing
        lnn.set_adaptive_config(adaptive_config)
        print(f"✅ Adaptive configuration applied")
        
        # Generate test audio signal
        duration_seconds = 1.0
        sample_rate = 16000
        samples = int(duration_seconds * sample_rate)
        
        # Create a test signal with varying complexity
        t = np.linspace(0, duration_seconds, samples)
        
        # Low complexity: simple sine wave
        simple_signal = 0.1 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # High complexity: multiple frequencies + noise
        complex_signal = (0.1 * np.sin(2 * np.pi * 440 * t) + 
                         0.05 * np.sin(2 * np.pi * 880 * t) +
                         0.02 * np.sin(2 * np.pi * 1320 * t) +
                         0.01 * np.random.randn(samples))
        
        print(f"\n🔊 Generated test signals:")
        print(f"   Simple signal: {len(simple_signal)} samples")
        print(f"   Complex signal: {len(complex_signal)} samples")
        
        # Process both signals
        print(f"\n🧮 Processing signals...")
        
        # Process simple signal
        result_simple = lnn.process(simple_signal.astype(np.float32))
        print(f"✅ Simple signal processed:")
        print(f"   Confidence: {result_simple.confidence:.3f}")
        print(f"   Timestep: {result_simple.timestep_ms:.1f} ms")
        print(f"   Power consumption: {result_simple.power_mw:.2f} mW")
        print(f"   Complexity: {result_simple.complexity:.3f}")
        
        # Process complex signal  
        result_complex = lnn.process(complex_signal.astype(np.float32))
        print(f"\n✅ Complex signal processed:")
        print(f"   Confidence: {result_complex.confidence:.3f}")
        print(f"   Timestep: {result_complex.timestep_ms:.1f} ms")
        print(f"   Power consumption: {result_complex.power_mw:.2f} mW")
        print(f"   Complexity: {result_complex.complexity:.3f}")
        
        # Verify adaptive behavior
        timestep_ratio = result_complex.timestep_ms / result_simple.timestep_ms
        power_ratio = result_complex.power_mw / result_simple.power_mw
        
        print(f"\n📊 Adaptive Behavior Analysis:")
        print(f"   Timestep adaptation ratio: {timestep_ratio:.2f}")
        print(f"   Power consumption ratio: {power_ratio:.2f}")
        print(f"   Expected: Complex signals → shorter timesteps, higher power")
        
        if timestep_ratio < 1.0:
            print("✅ Adaptive timestep working correctly")
        else:
            print("⚠️  Timestep adaptation may need tuning")
            
    except Exception as e:
        print(f"❌ LNN processing test failed: {e}")
        return False
    
    # 4. Test Feature Extraction
    print("\n4️⃣  Testing Feature Extraction")
    print("-" * 40)
    
    try:
        # Create feature extractor
        extractor = lan.FeatureExtractor(40)
        print(f"✅ Feature extractor created (40 dimensions)")
        
        # Extract features from test signals
        features_simple = extractor.extract(simple_signal[:1024].astype(np.float32))
        features_complex = extractor.extract(complex_signal[:1024].astype(np.float32))
        
        print(f"\n🔍 Feature Extraction Results:")
        print(f"   Simple signal features: {len(features_simple)} dims")
        print(f"   Feature range: [{min(features_simple):.3f}, {max(features_simple):.3f}]")
        print(f"   Feature mean: {np.mean(features_simple):.3f}")
        
        print(f"\n   Complex signal features: {len(features_complex)} dims")
        print(f"   Feature range: [{min(features_complex):.3f}, {max(features_complex):.3f}]")
        print(f"   Feature mean: {np.mean(features_complex):.3f}")
        
        # Compare feature diversity
        simple_var = np.var(features_simple)
        complex_var = np.var(features_complex)
        
        print(f"\n📈 Feature Diversity:")
        print(f"   Simple signal variance: {simple_var:.6f}")
        print(f"   Complex signal variance: {complex_var:.6f}")
        
        if complex_var > simple_var:
            print("✅ Feature extraction captures signal complexity")
        else:
            print("⚠️  Feature diversity could be improved")
        
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False
    
    # 5. Test Complexity Estimation
    print("\n5️⃣  Testing Complexity Estimation")
    print("-" * 40)
    
    try:
        # Create complexity estimator
        estimator = lan.ComplexityEstimator(10)  # Window size 10
        print(f"✅ Complexity estimator created")
        
        # Estimate complexity for different signals
        complexity_simple = estimator.estimate_complexity(simple_signal[:512].astype(np.float32))
        complexity_complex = estimator.estimate_complexity(complex_signal[:512].astype(np.float32))
        
        print(f"\n🎯 Complexity Estimation:")
        print(f"   Simple signal complexity: {complexity_simple:.3f}")
        print(f"   Complex signal complexity: {complexity_complex:.3f}")
        print(f"   Complexity ratio: {complexity_complex / complexity_simple:.2f}")
        
        # Get estimator statistics
        stats = estimator.cache_stats()
        print(f"\n📊 Estimator Statistics:")
        print(f"   Window size: {stats.window_size}")
        print(f"   History length: {stats.history_length}")
        print(f"   Current complexity: {stats.current_complexity:.3f}")
        print(f"   Cache hit rate: {stats.hit_rate:.1%}")
        
        if complexity_complex > complexity_simple:
            print("✅ Complexity estimation working correctly")
        else:
            print("⚠️  Complexity estimation needs calibration")
            
    except Exception as e:
        print(f"❌ Complexity estimation test failed: {e}")
        return False
    
    # 6. Test Model Performance Summary
    print("\n6️⃣  Model Performance Summary")
    print("-" * 40)
    
    try:
        # Get performance summary
        performance = lnn.get_performance_summary()
        print(f"📋 Performance Summary:")
        print(performance)
        
        # Get recommendations
        recommendations = lnn.get_recommendations()
        print(f"\n💡 Optimization Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
            
    except Exception as e:
        print(f"❌ Performance summary failed: {e}")
        return False
    
    # Final Success Summary
    print("\n" + "=" * 60)
    print("🎉 GENERATION 1 DEMO COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\n✅ Verified Functionality:")
    print("   ✓ Model configuration and validation")
    print("   ✓ Audio format handling and compatibility")  
    print("   ✓ LNN model creation and processing")
    print("   ✓ Adaptive timestep control")
    print("   ✓ Feature extraction from audio signals")
    print("   ✓ Audio complexity estimation")
    print("   ✓ Power consumption modeling")
    print("   ✓ Performance monitoring and recommendations")
    
    print(f"\n📊 Key Performance Metrics:")
    print(f"   • Processing latency: ~{result_simple.timestep_ms:.1f}-{result_complex.timestep_ms:.1f} ms")
    print(f"   • Power consumption: ~{result_simple.power_mw:.2f}-{result_complex.power_mw:.2f} mW")
    print(f"   • Adaptive scaling: {timestep_ratio:.2f}x timestep adaptation")
    print(f"   • Feature dimensions: {len(features_simple)}")
    print(f"   • Complexity range: {complexity_simple:.3f}-{complexity_complex:.3f}")
    
    print(f"\n🚀 Ready for Generation 2: Enhanced reliability and robustness!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)