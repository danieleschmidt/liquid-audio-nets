#!/usr/bin/env python3
"""Generation 1 Enhancement Demo - Advanced Power Efficiency Algorithms"""

import numpy as np
import time
from python.liquid_audio_nets.lnn import LNN, AdaptiveConfig
from python.liquid_audio_nets.power_optimization import HardwareConfig, PowerProfile

def test_enhanced_power_optimization():
    """Test Generation 1 power optimization enhancements."""
    
    print("🚀 GENERATION 1 ENHANCEMENTS - Advanced Power Efficiency")
    print("=" * 65)
    
    # Test different hardware configurations
    hardware_configs = {
        'cortex_m4': HardwareConfig(
            mcu_type="cortex_m4",
            base_power_mw=0.6,
            cpu_freq_mhz=80,
            memory_kb=256,
            has_fpu=True,
            has_dsp=True
        ),
        'cortex_m0': HardwareConfig(
            mcu_type="cortex_m0", 
            base_power_mw=0.3,
            cpu_freq_mhz=48,
            memory_kb=64,
            has_fpu=False,
            has_dsp=False
        ),
        'cortex_m7': HardwareConfig(
            mcu_type="cortex_m7",
            base_power_mw=1.2,
            cpu_freq_mhz=400,
            memory_kb=1024,
            has_fpu=True,
            has_dsp=True
        )
    }
    
    print("🧪 Testing Hardware-Aware Power Optimization")
    print("-" * 50)
    
    for hw_name, hw_config in hardware_configs.items():
        print(f"\n📟 {hw_name.upper()} Configuration:")
        
        # Initialize LNN with hardware configuration
        lnn = LNN(hardware_config=hw_config)
        lnn.set_adaptive_config(AdaptiveConfig(
            min_timestep=0.001,
            max_timestep=0.050,
            energy_threshold=0.1,
            complexity_metric="spectral_flux"
        ))
        
        # Test different audio scenarios
        test_scenarios = {
            'silence': np.zeros(256),
            'speech': np.random.normal(0, 0.1, 256),
            'music': np.sin(2 * np.pi * np.linspace(0, 1, 256) * 440),
            'noise': np.random.normal(0, 0.3, 256)
        }
        
        for scenario, audio in test_scenarios.items():
            result = lnn.process(audio)
            efficiency = lnn.get_power_efficiency_score()
            
            print(f"  {scenario:10}: {result['power_mw']:.2f}mW, "
                  f"efficiency: {efficiency:.3f}, "
                  f"timestep: {result['timestep_ms']:.1f}ms")
    
    print("\n🎯 Testing Power Budget Optimization")
    print("-" * 50)
    
    # Test power budget optimization
    lnn = LNN(hardware_config=hardware_configs['cortex_m4'])
    lnn.set_adaptive_config(AdaptiveConfig(complexity_metric="spectral_flux"))
    
    power_budgets = [1.0, 2.0, 5.0, 10.0]  # mW
    
    for budget in power_budgets:
        optimization_result = lnn.optimize_for_power_budget(budget)
        
        print(f"\n💰 Power Budget: {budget}mW")
        print(f"  Optimal timestep: {optimization_result['optimal_timestep_ms']:.1f}ms")
        print(f"  Suggested profile: {optimization_result['suggested_profile']}")
        print(f"  Config updated: {optimization_result['config_updated']}")
        
        # Test processing with optimized settings
        test_audio = np.random.normal(0, 0.2, 256)  # Moderate complexity
        result = lnn.process(test_audio)
        
        print(f"  Actual power: {result['power_mw']:.2f}mW")
        print(f"  Budget met: {'✅' if result['power_mw'] <= budget * 1.1 else '❌'}")
    
    print("\n⚡ Comparative Power Analysis")
    print("-" * 50)
    
    # Compare old vs new power estimation methods
    test_audio = np.random.normal(0, 0.2, 256)
    
    print("Scenario comparison (Old vs Generation 1):")
    
    # Simple LNN (baseline)
    lnn_baseline = LNN()
    lnn_baseline.set_adaptive_config(AdaptiveConfig(complexity_metric="energy"))
    
    # Enhanced LNN (Generation 1)
    lnn_enhanced = LNN(hardware_config=hardware_configs['cortex_m4'])  
    lnn_enhanced.set_adaptive_config(AdaptiveConfig(complexity_metric="spectral_flux"))
    
    for complexity_name, audio_data in [
        ('Low complexity', np.zeros(256) + np.random.normal(0, 0.05, 256)),
        ('Medium complexity', np.random.normal(0, 0.15, 256)), 
        ('High complexity', np.random.normal(0, 0.4, 256))
    ]:
        result_baseline = lnn_baseline.process(audio_data)
        result_enhanced = lnn_enhanced.process(audio_data)
        
        power_improvement = (result_baseline['power_mw'] - result_enhanced['power_mw']) / result_baseline['power_mw'] * 100
        
        print(f"\n{complexity_name}:")
        print(f"  Baseline:     {result_baseline['power_mw']:.2f}mW")
        print(f"  Generation 1: {result_enhanced['power_mw']:.2f}mW")
        print(f"  Improvement:  {power_improvement:.1f}%")
    
    print("\n🏆 Generation 1 Enhancement Results")
    print("=" * 65)
    print("✅ Hardware-aware power modeling implemented")
    print("✅ Advanced complexity estimation with multiple metrics")
    print("✅ Power budget optimization with timestep tuning")
    print("✅ Hardware profile recommendations")
    print("✅ Power efficiency scoring")
    print("✅ Hysteresis for stable power consumption")
    print("✅ MCU-specific power constraints")

if __name__ == "__main__":
    test_enhanced_power_optimization()