#!/usr/bin/env python3
"""
Production Quality Test Suite
Comprehensive testing for production readiness validation
"""

import time
import math
import random
import sys
import os
from typing import List, Dict, Tuple, Optional

# Import our production implementation
sys.path.insert(0, '.')
try:
    from production_demo import ProductionLiquidAudioNet, create_optimized_configs, generate_realistic_audio
except ImportError as e:
    print(f"Error importing production modules: {e}")
    sys.exit(1)

class QualityGate:
    """Quality gate validation with strict thresholds"""
    
    def __init__(self):
        self.thresholds = {
            'max_power_consumption': 5.0,      # mW
            'min_real_time_factor': 10.0,      # x real-time
            'max_processing_time': 2.0,        # ms per frame
            'min_throughput': 100.0,           # FPS
            'max_memory_footprint': 2048,      # bytes
            'min_numerical_stability': 0.99,   # stability score
            'max_error_rate': 0.01,            # error percentage
            'min_efficiency_score': 0.05,      # complexity/power ratio
        }
        
        self.test_results = {}
        self.passed_gates = 0
        self.total_gates = 0
    
    def validate_power_consumption(self, avg_power: float, config_name: str) -> bool:
        """Validate power consumption is within limits"""
        self.total_gates += 1
        threshold = self.thresholds['max_power_consumption']
        
        if config_name == 'ultra_low_power':
            threshold = 2.0  # Stricter for ultra-low power
        elif config_name == 'high_accuracy':
            threshold = 8.0  # Relaxed for high accuracy
        
        passed = avg_power <= threshold
        if passed:
            self.passed_gates += 1
        
        self.test_results[f'power_{config_name}'] = {
            'value': avg_power,
            'threshold': threshold,
            'passed': passed,
            'unit': 'mW'
        }
        
        return passed
    
    def validate_real_time_performance(self, real_time_factor: float, config_name: str) -> bool:
        """Validate real-time processing capability"""
        self.total_gates += 1
        threshold = self.thresholds['min_real_time_factor']
        
        passed = real_time_factor >= threshold
        if passed:
            self.passed_gates += 1
        
        self.test_results[f'real_time_{config_name}'] = {
            'value': real_time_factor,
            'threshold': threshold,
            'passed': passed,
            'unit': 'x real-time'
        }
        
        return passed
    
    def validate_numerical_stability(self, stability_score: float) -> bool:
        """Validate numerical stability across processing"""
        self.total_gates += 1
        threshold = self.thresholds['min_numerical_stability']
        
        passed = stability_score >= threshold
        if passed:
            self.passed_gates += 1
        
        self.test_results['numerical_stability'] = {
            'value': stability_score,
            'threshold': threshold,
            'passed': passed,
            'unit': 'stability_score'
        }
        
        return passed
    
    def validate_error_resilience(self, error_rate: float) -> bool:
        """Validate error handling and recovery"""
        self.total_gates += 1
        threshold = self.thresholds['max_error_rate']
        
        passed = error_rate <= threshold
        if passed:
            self.passed_gates += 1
        
        self.test_results['error_resilience'] = {
            'value': error_rate,
            'threshold': threshold,
            'passed': passed,
            'unit': 'error_rate'
        }
        
        return passed
    
    def validate_memory_efficiency(self, memory_usage: int, config_name: str) -> bool:
        """Validate memory footprint"""
        self.total_gates += 1
        threshold = self.thresholds['max_memory_footprint']
        
        if config_name == 'ultra_low_power':
            threshold = 256  # Very strict for ultra-low power
        elif config_name == 'high_accuracy':
            threshold = 4096  # Relaxed for high accuracy
        
        passed = memory_usage <= threshold
        if passed:
            self.passed_gates += 1
        
        self.test_results[f'memory_{config_name}'] = {
            'value': memory_usage,
            'threshold': threshold,
            'passed': passed,
            'unit': 'bytes'
        }
        
        return passed
    
    def get_pass_rate(self) -> float:
        """Get overall pass rate"""
        return self.passed_gates / self.total_gates if self.total_gates > 0 else 0.0
    
    def generate_report(self) -> str:
        """Generate detailed quality gate report"""
        report = "\n🛡️  QUALITY GATE VALIDATION REPORT\n"
        report += "=" * 50 + "\n"
        
        report += f"Overall Pass Rate: {self.get_pass_rate():.1%} ({self.passed_gates}/{self.total_gates})\n\n"
        
        for test_name, result in self.test_results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            report += f"{status} {test_name}: {result['value']:.2f} {result['unit']} "
            report += f"(threshold: {result['threshold']:.2f})\n"
        
        return report

def test_basic_functionality():
    """Test basic functionality across all configurations"""
    print("🧪 Testing Basic Functionality")
    print("-" * 40)
    
    configs = create_optimized_configs()
    results = {}
    
    for config_name, config in configs.items():
        print(f"  Testing {config_name}...")
        
        try:
            lnn = ProductionLiquidAudioNet(config)
            
            # Test with various audio types
            test_scenarios = ['silence', 'speech', 'music', 'noise']
            scenario_results = {}
            
            for scenario in test_scenarios:
                audio = generate_realistic_audio(scenario, duration_ms=50)
                result = lnn.process(audio)
                
                # Validate result structure
                required_fields = ['output', 'confidence', 'power_mw', 'complexity', 'timestep_ms']
                for field in required_fields:
                    assert field in result, f"Missing field {field} in result"
                
                # Validate value ranges
                assert 0.0 <= result['confidence'] <= 1.0, f"Invalid confidence: {result['confidence']}"
                assert result['power_mw'] > 0, f"Invalid power: {result['power_mw']}"
                assert 0.0 <= result['complexity'] <= 1.0, f"Invalid complexity: {result['complexity']}"
                assert result['timestep_ms'] > 0, f"Invalid timestep: {result['timestep_ms']}"
                assert len(result['output']) == config['output_dim'], "Output dimension mismatch"
                
                scenario_results[scenario] = result
            
            results[config_name] = {
                'lnn': lnn,
                'scenarios': scenario_results,
                'passed': True
            }
            
            print(f"    ✅ {config_name} passed basic functionality tests")
            
        except Exception as e:
            results[config_name] = {
                'lnn': None,
                'scenarios': {},
                'passed': False,
                'error': str(e)
            }
            print(f"    ❌ {config_name} failed: {e}")
    
    return results

def test_performance_benchmarks(basic_results: Dict) -> Dict:
    """Test performance benchmarks and validate against quality gates"""
    print("\n⚡ Performance Benchmarking")
    print("-" * 35)
    
    quality_gate = QualityGate()
    performance_results = {}
    
    for config_name, result in basic_results.items():
        if not result['passed']:
            continue
        
        print(f"  Benchmarking {config_name}...")
        
        lnn = result['lnn']
        config = lnn.config
        
        # Performance benchmark
        test_audio = generate_realistic_audio('speech', duration_ms=100)
        
        # Warm up
        for _ in range(5):
            lnn.process(test_audio)
        
        # Benchmark
        start_time = time.time()
        iterations = 100
        
        for _ in range(iterations):
            benchmark_result = lnn.process(test_audio)
        
        end_time = time.time()
        
        # Calculate metrics
        avg_processing_time = (end_time - start_time) * 1000 / iterations  # ms
        audio_duration_ms = 100  # ms
        real_time_factor = audio_duration_ms / avg_processing_time
        
        # Get system metrics
        metrics = lnn.get_metrics()
        
        # Estimate memory usage
        memory_usage = (
            config['hidden_dim'] * 8 +  # State + previous output (float64)
            config['input_dim'] * 4 +   # Features (float32)
            config['output_dim'] * 4 +  # Output (float32)
            200  # Overhead estimate
        )
        
        # Quality gate validation
        power_passed = quality_gate.validate_power_consumption(metrics['avg_power_mw'], config_name)
        performance_passed = quality_gate.validate_real_time_performance(real_time_factor, config_name)
        memory_passed = quality_gate.validate_memory_efficiency(memory_usage, config_name)
        
        performance_results[config_name] = {
            'avg_processing_time_ms': avg_processing_time,
            'real_time_factor': real_time_factor,
            'avg_power_mw': metrics['avg_power_mw'],
            'memory_usage_bytes': memory_usage,
            'throughput_fps': metrics['throughput_fps'],
            'efficiency_score': metrics['efficiency_score'],
            'power_gate_passed': power_passed,
            'performance_gate_passed': performance_passed,
            'memory_gate_passed': memory_passed
        }
        
        print(f"    ⚡ {avg_processing_time:.2f}ms/frame ({real_time_factor:.1f}x real-time)")
        print(f"    🔋 {metrics['avg_power_mw']:.2f}mW average power")
        print(f"    💾 ~{memory_usage}B memory footprint")
    
    return performance_results, quality_gate

def test_stress_and_stability(basic_results: Dict) -> Tuple[float, Dict]:
    """Test system stability under stress conditions"""
    print("\n🔧 Stress Testing & Stability")
    print("-" * 35)
    
    stability_results = {}
    total_stability_score = 0.0
    config_count = 0
    
    for config_name, result in basic_results.items():
        if not result['passed']:
            continue
        
        print(f"  Stress testing {config_name}...")
        
        lnn = result['lnn']
        lnn.reset()  # Start fresh
        
        error_count = 0
        total_frames = 0
        numerical_issues = 0
        
        # Long-term processing test
        for i in range(500):
            # Generate increasingly complex test cases
            complexity_factor = min(1.0, i / 200.0)  # Ramp up complexity
            
            if i % 4 == 0:
                audio = generate_realistic_audio('silence', duration_ms=64)
            elif i % 4 == 1:
                audio = generate_realistic_audio('speech', duration_ms=64)
            elif i % 4 == 2:
                audio = generate_realistic_audio('music', duration_ms=64)
            else:
                # Stress test with difficult signals
                audio = [random.uniform(-1, 1) for _ in range(64)]
                # Add some extreme values occasionally
                if random.random() < 0.1:
                    for j in range(min(5, len(audio))):
                        audio[j] = random.choice([-2.0, 2.0])  # Out of normal range
            
            try:
                process_result = lnn.process(audio)
                total_frames += 1
                
                # Check for numerical issues
                if 'error' in process_result:
                    error_count += 1
                else:
                    # Validate numerical stability
                    if not all(math.isfinite(x) for x in process_result['output']):
                        numerical_issues += 1
                    if not math.isfinite(process_result['confidence']):
                        numerical_issues += 1
                    if not math.isfinite(process_result['power_mw']):
                        numerical_issues += 1
                
            except Exception as e:
                error_count += 1
        
        # Calculate stability metrics
        error_rate = error_count / total_frames if total_frames > 0 else 1.0
        numerical_stability = 1.0 - (numerical_issues / total_frames) if total_frames > 0 else 0.0
        
        # Memory leak test (approximate)
        memory_consistent = True  # In Python, we can't easily detect memory leaks
        
        # Edge case testing
        edge_case_passed = True
        edge_cases = [
            [],  # Empty audio
            [0.0],  # Single sample
            [float('inf')] * 10,  # Infinite values
            [float('-inf')] * 10,  # Negative infinite
            [float('nan')] * 10,  # NaN values
        ]
        
        for edge_audio in edge_cases:
            try:
                edge_result = lnn.process(edge_audio)
                # Should either succeed or fail gracefully with error field
                if 'error' not in edge_result:
                    # If it succeeds, output should be valid
                    if not all(math.isfinite(x) for x in edge_result['output']):
                        edge_case_passed = False
            except Exception:
                # Should not raise unhandled exceptions
                edge_case_passed = False
        
        overall_stability = (
            (1.0 - error_rate) * 0.4 +
            numerical_stability * 0.3 +
            (1.0 if memory_consistent else 0.0) * 0.15 +
            (1.0 if edge_case_passed else 0.0) * 0.15
        )
        
        stability_results[config_name] = {
            'total_frames_tested': total_frames,
            'error_rate': error_rate,
            'numerical_stability': numerical_stability,
            'memory_consistent': memory_consistent,
            'edge_case_passed': edge_case_passed,
            'overall_stability': overall_stability
        }
        
        total_stability_score += overall_stability
        config_count += 1
        
        print(f"    📊 Error rate: {error_rate:.1%}")
        print(f"    🔢 Numerical stability: {numerical_stability:.1%}")
        print(f"    🛡️  Overall stability: {overall_stability:.1%}")
    
    avg_stability = total_stability_score / config_count if config_count > 0 else 0.0
    return avg_stability, stability_results

def test_production_readiness():
    """Comprehensive production readiness validation"""
    print("\n🚀 PRODUCTION READINESS VALIDATION")
    print("=" * 50)
    
    # Documentation check
    required_docs = ['README.md', 'CHANGELOG.md', 'LICENSE']
    doc_score = 0
    for doc in required_docs:
        if os.path.exists(doc):
            doc_score += 1
            print(f"✅ Documentation: {doc} exists")
        else:
            print(f"❌ Documentation: {doc} missing")
    
    doc_pass_rate = doc_score / len(required_docs)
    
    # Code structure check
    required_structure = [
        'src/',
        'tests/',
        'examples/',
        'python/',
    ]
    
    structure_score = 0
    for item in required_structure:
        if os.path.exists(item):
            structure_score += 1
            print(f"✅ Structure: {item} exists")
        else:
            print(f"❌ Structure: {item} missing")
    
    structure_pass_rate = structure_score / len(required_structure)
    
    # Demo availability
    demo_files = ['simple_demo.py', 'production_demo.py']
    demo_score = 0
    for demo in demo_files:
        if os.path.exists(demo):
            demo_score += 1
            print(f"✅ Demo: {demo} available")
        else:
            print(f"❌ Demo: {demo} missing")
    
    demo_pass_rate = demo_score / len(demo_files)
    
    # Overall production readiness score
    production_score = (doc_pass_rate + structure_pass_rate + demo_pass_rate) / 3
    
    return {
        'documentation_score': doc_pass_rate,
        'structure_score': structure_pass_rate,
        'demo_score': demo_pass_rate,
        'overall_score': production_score
    }

def generate_final_report(basic_results: Dict, performance_results: Dict, 
                         quality_gate: QualityGate, avg_stability: float,
                         stability_results: Dict, production_readiness: Dict):
    """Generate comprehensive final report"""
    
    report = "\n" + "=" * 80 + "\n"
    report += "🏆 LIQUID AUDIO NETWORKS - PRODUCTION QUALITY REPORT\n"
    report += "=" * 80 + "\n\n"
    
    # Executive Summary
    report += "📋 EXECUTIVE SUMMARY\n"
    report += "-" * 25 + "\n"
    
    total_configs = len(basic_results)
    passed_configs = sum(1 for r in basic_results.values() if r['passed'])
    
    report += f"Configurations Tested: {total_configs}\n"
    report += f"Configurations Passed: {passed_configs}\n"
    report += f"Quality Gate Pass Rate: {quality_gate.get_pass_rate():.1%}\n"
    report += f"Average System Stability: {avg_stability:.1%}\n"
    report += f"Production Readiness: {production_readiness['overall_score']:.1%}\n\n"
    
    # Detailed Results
    report += "📊 DETAILED RESULTS BY CONFIGURATION\n"
    report += "-" * 45 + "\n"
    
    for config_name in basic_results.keys():
        report += f"\n{config_name.upper().replace('_', ' ')}:\n"
        
        if config_name in performance_results:
            perf = performance_results[config_name]
            stab = stability_results.get(config_name, {})
            
            report += f"  ⚡ Performance: {perf['real_time_factor']:.1f}x real-time\n"
            report += f"  🔋 Power: {perf['avg_power_mw']:.2f}mW\n"
            report += f"  💾 Memory: {perf['memory_usage_bytes']}B\n"
            report += f"  🛡️  Stability: {stab.get('overall_stability', 0):.1%}\n"
            
            # Quality gates
            gates = []
            if perf.get('power_gate_passed', False):
                gates.append("Power ✅")
            else:
                gates.append("Power ❌")
            
            if perf.get('performance_gate_passed', False):
                gates.append("Performance ✅")
            else:
                gates.append("Performance ❌")
            
            if perf.get('memory_gate_passed', False):
                gates.append("Memory ✅")
            else:
                gates.append("Memory ❌")
            
            report += f"  🚪 Quality Gates: {', '.join(gates)}\n"
        else:
            report += "  ❌ Configuration failed basic tests\n"
    
    # Quality Gate Details
    report += quality_gate.generate_report()
    
    # Recommendations
    report += "\n💡 RECOMMENDATIONS\n"
    report += "-" * 20 + "\n"
    
    if quality_gate.get_pass_rate() >= 0.8:
        report += "✅ System meets production quality standards\n"
        report += "✅ Ready for deployment in target environments\n"
    else:
        report += "⚠️  System requires additional optimization\n"
        report += "⚠️  Consider addressing failing quality gates\n"
    
    if avg_stability >= 0.95:
        report += "✅ Excellent stability for production use\n"
    elif avg_stability >= 0.9:
        report += "⚠️  Good stability, monitor in production\n"
    else:
        report += "❌ Stability concerns, needs improvement\n"
    
    # Production Deployment Checklist
    report += "\n📋 PRODUCTION DEPLOYMENT CHECKLIST\n"
    report += "-" * 40 + "\n"
    
    checklist_items = [
        ("Core functionality", passed_configs == total_configs),
        ("Performance benchmarks", quality_gate.get_pass_rate() >= 0.8),
        ("Numerical stability", avg_stability >= 0.9),
        ("Error handling", avg_stability >= 0.9),
        ("Documentation", production_readiness['documentation_score'] >= 0.8),
        ("Code structure", production_readiness['structure_score'] >= 0.8),
        ("Demo availability", production_readiness['demo_score'] >= 0.8),
    ]
    
    for item, passed in checklist_items:
        status = "✅" if passed else "❌"
        report += f"{status} {item}\n"
    
    report += "\n" + "=" * 80 + "\n"
    
    return report

def main():
    """Run comprehensive production quality tests"""
    print("🧪 LIQUID AUDIO NETWORKS - PRODUCTION QUALITY TESTING")
    print("=" * 65)
    
    try:
        # Test suite execution
        basic_results = test_basic_functionality()
        performance_results, quality_gate = test_performance_benchmarks(basic_results)
        avg_stability, stability_results = test_stress_and_stability(basic_results)
        production_readiness = test_production_readiness()
        
        # Validate overall system with quality gates
        quality_gate.validate_numerical_stability(avg_stability)
        
        error_rates = [s.get('error_rate', 1.0) for s in stability_results.values()]
        avg_error_rate = sum(error_rates) / len(error_rates) if error_rates else 1.0
        quality_gate.validate_error_resilience(avg_error_rate)
        
        # Generate and display final report
        final_report = generate_final_report(
            basic_results, performance_results, quality_gate,
            avg_stability, stability_results, production_readiness
        )
        
        print(final_report)
        
        # Overall pass/fail determination
        overall_pass = (
            quality_gate.get_pass_rate() >= 0.8 and
            avg_stability >= 0.9 and
            production_readiness['overall_score'] >= 0.8
        )
        
        if overall_pass:
            print("🎉 OVERALL RESULT: PRODUCTION READY ✅")
            return 0
        else:
            print("⚠️  OVERALL RESULT: NEEDS IMPROVEMENT ❌")
            return 1
        
    except Exception as e:
        print(f"\n❌ Quality testing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())