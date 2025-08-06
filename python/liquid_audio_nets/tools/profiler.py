"""Performance profiling tools for liquid audio nets.

Hardware-in-the-loop testing and power measurement utilities.
"""

import time
import statistics
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import numpy as np

from ..lnn import LNN, AdaptiveConfig


@dataclass
class ProfilingResult:
    """Results from performance profiling."""
    
    model_path: str
    device: str
    test_samples: int
    
    # Timing metrics
    avg_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    std_latency_ms: float
    percentile_95_ms: float
    
    # Power metrics  
    avg_power_mw: float
    min_power_mw: float
    max_power_mw: float
    peak_power_mw: float
    
    # Accuracy metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Resource usage
    memory_usage_kb: int
    cpu_utilization_percent: float
    
    # Detailed metrics
    per_sample_metrics: List[Dict[str, float]] = field(default_factory=list)
    power_distribution: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_path': self.model_path,
            'device': self.device,
            'test_samples': self.test_samples,
            'timing': {
                'avg_latency_ms': self.avg_latency_ms,
                'min_latency_ms': self.min_latency_ms,
                'max_latency_ms': self.max_latency_ms,
                'std_latency_ms': self.std_latency_ms,
                'percentile_95_ms': self.percentile_95_ms,
            },
            'power': {
                'avg_power_mw': self.avg_power_mw,
                'min_power_mw': self.min_power_mw,
                'max_power_mw': self.max_power_mw,
                'peak_power_mw': self.peak_power_mw,
                'distribution': self.power_distribution,
            },
            'accuracy': {
                'accuracy': self.accuracy,
                'precision': self.precision,
                'recall': self.recall,
                'f1_score': self.f1_score,
            },
            'resources': {
                'memory_usage_kb': self.memory_usage_kb,
                'cpu_utilization_percent': self.cpu_utilization_percent,
            },
            'detailed_metrics': self.per_sample_metrics,
        }


class ModelProfiler:
    """Profiler for LNN model performance analysis."""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """Initialize profiler.
        
        Args:
            model_path: Path to LNN model file
            device: Target device/platform
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model: Optional[LNN] = None
        
        # Supported devices and their characteristics
        self.device_specs = {
            "stm32f407": {
                "cpu_mhz": 168,
                "ram_kb": 192,
                "flash_kb": 1024,
                "power_base_mw": 0.1,
            },
            "nrf52840": {
                "cpu_mhz": 64,
                "ram_kb": 256,
                "flash_kb": 1024,
                "power_base_mw": 0.05,
            },
            "esp32": {
                "cpu_mhz": 240,
                "ram_kb": 512,
                "flash_kb": 4096,
                "power_base_mw": 0.2,
            },
            "cpu": {
                "cpu_mhz": 2000,  # Typical desktop CPU
                "ram_kb": 8*1024*1024,  # 8GB
                "flash_kb": 256*1024*1024,  # 256GB SSD
                "power_base_mw": 1000,  # 1W base power
            }
        }
        
    def load_model(self) -> None:
        """Load and initialize model for profiling."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
            
        self.model = LNN.from_file(self.model_path)
        print(f"✅ Loaded model: {self.model_path.name}")
        
        # Display model info
        model_info = self.model.get_model_info()
        if model_info:
            print(f"   Architecture: {model_info['input_dim']}→{model_info['hidden_dim']}→{model_info['output_dim']}")
            print(f"   Version: {model_info['version']}")
            
    def profile_latency(self, 
                       audio_samples: List[np.ndarray], 
                       warmup_runs: int = 10) -> Dict[str, float]:
        """Profile inference latency.
        
        Args:
            audio_samples: List of audio samples to test
            warmup_runs: Number of warmup runs before measurement
            
        Returns:
            Latency statistics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        print(f"🔬 Profiling latency with {len(audio_samples)} samples...")
        
        # Warmup runs
        print(f"   Running {warmup_runs} warmup iterations...")
        for i in range(warmup_runs):
            if audio_samples:
                _ = self.model.process(audio_samples[i % len(audio_samples)])
                
        # Measurement runs
        latencies = []
        for i, audio in enumerate(audio_samples):
            start_time = time.perf_counter()
            result = self.model.process(audio)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(audio_samples)} samples")
        
        # Compute statistics
        stats = {
            'avg_latency_ms': statistics.mean(latencies),
            'min_latency_ms': min(latencies),
            'max_latency_ms': max(latencies),
            'std_latency_ms': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            'percentile_95_ms': np.percentile(latencies, 95),
            'percentile_99_ms': np.percentile(latencies, 99),
        }
        
        print(f"   Average latency: {stats['avg_latency_ms']:.2f}ms")
        print(f"   95th percentile: {stats['percentile_95_ms']:.2f}ms")
        
        return stats
        
    def profile_power(self, 
                     audio_samples: List[np.ndarray],
                     adaptive_config: Optional[AdaptiveConfig] = None) -> Dict[str, float]:
        """Profile power consumption.
        
        Args:
            audio_samples: Audio samples for testing
            adaptive_config: Optional adaptive configuration
            
        Returns:
            Power consumption statistics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        print(f"⚡ Profiling power consumption...")
        
        if adaptive_config:
            self.model.set_adaptive_config(adaptive_config)
            print(f"   Using adaptive timestep: {adaptive_config.min_timestep}-{adaptive_config.max_timestep}s")
            
        power_measurements = []
        processing_modes = {"sleep": 0, "minimal": 0, "reduced": 0, "active": 0}
        
        for i, audio in enumerate(audio_samples):
            result = self.model.process(audio)
            power_mw = result.get("power_mw", 0.0)
            power_measurements.append(power_mw)
            
            # Track processing mode distribution (for VAD)
            if hasattr(self.model, 'detect_activity'):
                vad_result = self.model.detect_activity(audio)
                mode = vad_result.get("recommended_power_mode", "unknown")
                if mode in processing_modes:
                    processing_modes[mode] += 1
            
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(audio_samples)} samples")
        
        # Compute power statistics
        stats = {
            'avg_power_mw': statistics.mean(power_measurements),
            'min_power_mw': min(power_measurements),
            'max_power_mw': max(power_measurements),
            'peak_power_mw': max(power_measurements),  # Same as max for now
            'std_power_mw': statistics.stdev(power_measurements) if len(power_measurements) > 1 else 0.0,
        }
        
        # Power distribution
        total_samples = len(audio_samples)
        power_distribution = {
            mode: (count / total_samples * 100) if total_samples > 0 else 0.0
            for mode, count in processing_modes.items()
        }
        
        stats['distribution'] = power_distribution
        
        print(f"   Average power: {stats['avg_power_mw']:.2f}mW")
        print(f"   Peak power: {stats['peak_power_mw']:.2f}mW")
        
        return stats
        
    def profile_accuracy(self, 
                        audio_samples: List[np.ndarray], 
                        ground_truth: List[Any],
                        task_type: str = "classification") -> Dict[str, float]:
        """Profile model accuracy.
        
        Args:
            audio_samples: Test audio samples
            ground_truth: Ground truth labels/results
            task_type: Type of task ("classification", "detection", "vad")
            
        Returns:
            Accuracy metrics
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
            
        if len(audio_samples) != len(ground_truth):
            raise ValueError("Sample count must match ground truth count")
            
        print(f"🎯 Profiling accuracy for {task_type} task...")
        
        predictions = []
        confidences = []
        
        for i, audio in enumerate(audio_samples):
            if task_type == "vad":
                result = self.model.detect_activity(audio)
                pred = result["is_speech"]
                conf = result["confidence"]
            else:
                result = self.model.process(audio)
                pred = result.get("keyword_detected", False)
                conf = result.get("confidence", 0.0)
                
            predictions.append(pred)
            confidences.append(conf)
            
            if (i + 1) % 100 == 0:
                print(f"   Processed {i + 1}/{len(audio_samples)} samples")
        
        # Compute accuracy metrics
        if task_type in ["classification", "detection", "vad"]:
            # Binary or multi-class classification metrics
            correct = sum(1 for p, gt in zip(predictions, ground_truth) if p == gt)
            accuracy = correct / len(predictions)
            
            # For binary classification, compute precision/recall
            if task_type == "vad" or all(isinstance(p, bool) for p in predictions):
                tp = sum(1 for p, gt in zip(predictions, ground_truth) if p and gt)
                fp = sum(1 for p, gt in zip(predictions, ground_truth) if p and not gt)
                fn = sum(1 for p, gt in zip(predictions, ground_truth) if not p and gt)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                # Multi-class metrics (simplified)
                precision = accuracy  # Macro average approximation
                recall = accuracy
                f1 = accuracy
        else:
            # Default metrics
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
            
        stats = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_confidence': statistics.mean(confidences),
        }
        
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   F1 Score: {f1:.1%}")
        
        return stats
        
    def estimate_battery_life(self, 
                             avg_power_mw: float, 
                             battery_mah: int = 220,  # CR2032 battery
                             voltage: float = 3.0) -> Dict[str, float]:
        """Estimate battery life based on power consumption.
        
        Args:
            avg_power_mw: Average power consumption in mW
            battery_mah: Battery capacity in mAh
            voltage: Battery voltage in V
            
        Returns:
            Battery life estimates
        """
        print(f"🔋 Estimating battery life...")
        
        # Convert power to current
        current_ma = avg_power_mw / voltage
        
        # Battery life calculations
        continuous_hours = battery_mah / current_ma if current_ma > 0 else float('inf')
        continuous_days = continuous_hours / 24
        
        # Duty cycle scenarios
        duty_cycles = {
            "always_on": 1.0,
            "high_duty": 0.5,    # 50% duty cycle
            "medium_duty": 0.1,  # 10% duty cycle  
            "low_duty": 0.01,    # 1% duty cycle
        }
        
        estimates = {
            'continuous_hours': continuous_hours,
            'continuous_days': continuous_days,
            'avg_current_ma': current_ma,
        }
        
        for scenario, duty in duty_cycles.items():
            effective_current = current_ma * duty + 0.001  # Base leakage current
            hours = battery_mah / effective_current if effective_current > 0 else float('inf')
            estimates[f'{scenario}_hours'] = hours
            estimates[f'{scenario}_days'] = hours / 24
            
        print(f"   Continuous operation: {continuous_hours:.1f} hours ({continuous_days:.1f} days)")
        print(f"   With 10% duty cycle: {estimates['medium_duty_hours']:.1f} hours ({estimates['medium_duty_days']:.1f} days)")
        
        return estimates
        
    def run_full_profile(self, 
                        audio_samples: List[np.ndarray],
                        ground_truth: Optional[List[Any]] = None,
                        adaptive_config: Optional[AdaptiveConfig] = None,
                        task_type: str = "classification") -> ProfilingResult:
        """Run comprehensive profiling.
        
        Args:
            audio_samples: Test audio samples
            ground_truth: Optional ground truth for accuracy testing
            adaptive_config: Optional adaptive configuration
            task_type: Task type for accuracy evaluation
            
        Returns:
            Complete profiling results
        """
        if self.model is None:
            self.load_model()
            
        print(f"📊 Running full profile on {self.device}...")
        print(f"   Test samples: {len(audio_samples)}")
        
        # Profile latency
        latency_stats = self.profile_latency(audio_samples)
        
        # Profile power
        power_stats = self.profile_power(audio_samples, adaptive_config)
        
        # Profile accuracy (if ground truth provided)
        if ground_truth:
            accuracy_stats = self.profile_accuracy(audio_samples, ground_truth, task_type)
        else:
            accuracy_stats = {
                'accuracy': 0.0,
                'precision': 0.0, 
                'recall': 0.0,
                'f1_score': 0.0,
            }
        
        # Estimate resource usage
        memory_usage = self._estimate_memory_usage()
        cpu_utilization = self._estimate_cpu_utilization(latency_stats['avg_latency_ms'])
        
        # Collect per-sample detailed metrics
        per_sample_metrics = []
        for i, audio in enumerate(audio_samples[:100]):  # Limit to first 100 for detail
            start_time = time.perf_counter()
            result = self.model.process(audio)
            end_time = time.perf_counter()
            
            metrics = {
                'sample_id': i,
                'latency_ms': (end_time - start_time) * 1000,
                'power_mw': result.get('power_mw', 0.0),
                'confidence': result.get('confidence', 0.0),
                'timestep_ms': result.get('timestep_ms', 10.0),
                'liquid_energy': result.get('liquid_state_energy', 0.0),
            }
            per_sample_metrics.append(metrics)
        
        # Create comprehensive result
        result = ProfilingResult(
            model_path=str(self.model_path),
            device=self.device,
            test_samples=len(audio_samples),
            avg_latency_ms=latency_stats['avg_latency_ms'],
            min_latency_ms=latency_stats['min_latency_ms'],
            max_latency_ms=latency_stats['max_latency_ms'],
            std_latency_ms=latency_stats['std_latency_ms'],
            percentile_95_ms=latency_stats['percentile_95_ms'],
            avg_power_mw=power_stats['avg_power_mw'],
            min_power_mw=power_stats['min_power_mw'],
            max_power_mw=power_stats['max_power_mw'],
            peak_power_mw=power_stats['peak_power_mw'],
            accuracy=accuracy_stats['accuracy'],
            precision=accuracy_stats['precision'],
            recall=accuracy_stats['recall'],
            f1_score=accuracy_stats['f1_score'],
            memory_usage_kb=memory_usage,
            cpu_utilization_percent=cpu_utilization,
            per_sample_metrics=per_sample_metrics,
            power_distribution=power_stats.get('distribution', {}),
        )
        
        # Print summary
        print(f"\n📈 Profile Summary:")
        print(f"   Average Latency: {result.avg_latency_ms:.2f}ms")
        print(f"   Average Power: {result.avg_power_mw:.2f}mW")
        print(f"   Accuracy: {result.accuracy:.1%}")
        print(f"   Memory Usage: {result.memory_usage_kb}KB")
        
        # Battery life estimate
        battery_life = self.estimate_battery_life(result.avg_power_mw)
        print(f"   Battery Life (CR2032): {battery_life['continuous_hours']:.1f}h continuous")
        
        return result
        
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in KB."""
        if self.model is None:
            return 0
            
        model_info = self.model.get_model_info()
        if not model_info:
            return 128  # Default estimate
            
        # Model weights
        input_dim = model_info.get('input_dim', 40)
        hidden_dim = model_info.get('hidden_dim', 64)
        output_dim = model_info.get('output_dim', 10)
        
        weights_size = (input_dim * hidden_dim + 
                       hidden_dim * hidden_dim + 
                       hidden_dim * output_dim) * 4  # float32
        
        # State and buffers
        state_size = hidden_dim * 4
        buffer_size = input_dim * 4 * 2  # Double buffer
        
        # Overhead (stack, heap, etc.)
        overhead = 8 * 1024  # 8KB overhead
        
        total_bytes = weights_size + state_size + buffer_size + overhead
        return max(total_bytes // 1024, 16)  # At least 16KB
        
    def _estimate_cpu_utilization(self, avg_latency_ms: float) -> float:
        """Estimate CPU utilization percentage."""
        device_spec = self.device_specs.get(self.device, self.device_specs["cpu"])
        
        # Assume 16kHz sampling, 10ms frames
        frame_period_ms = 10.0
        utilization = (avg_latency_ms / frame_period_ms) * 100
        
        # Scale for device performance
        if self.device in ["stm32f407", "nrf52840"]:
            utilization *= 2.0  # MCUs are slower
        elif self.device == "esp32":
            utilization *= 1.5  # Moderate performance
            
        return min(100.0, max(1.0, utilization))
        
    def save_results(self, result: ProfilingResult, output_path: str) -> None:
        """Save profiling results to file.
        
        Args:
            result: Profiling results to save
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
            
        print(f"💾 Results saved to: {output_file}")


def generate_test_audio(num_samples: int = 100, 
                       sample_rate: int = 16000,
                       frame_size: int = 512) -> List[np.ndarray]:
    """Generate synthetic audio samples for testing.
    
    Args:
        num_samples: Number of audio samples to generate
        sample_rate: Audio sample rate
        frame_size: Frame size in samples
        
    Returns:
        List of audio frame arrays
    """
    print(f"🎵 Generating {num_samples} test audio samples...")
    
    audio_samples = []
    
    for i in range(num_samples):
        # Generate different types of audio content
        if i % 4 == 0:
            # Sine wave (speech-like fundamental)
            freq = 200 + (i % 10) * 50  # 200-700 Hz
            t = np.linspace(0, frame_size/sample_rate, frame_size)
            audio = 0.3 * np.sin(2 * np.pi * freq * t)
            
        elif i % 4 == 1:
            # Noise (background)
            audio = 0.1 * np.random.randn(frame_size)
            
        elif i % 4 == 2:
            # Complex signal (multiple harmonics)
            t = np.linspace(0, frame_size/sample_rate, frame_size)
            f0 = 150 + (i % 8) * 25
            audio = (0.2 * np.sin(2 * np.pi * f0 * t) + 
                    0.1 * np.sin(2 * np.pi * f0 * 2 * t) +
                    0.05 * np.sin(2 * np.pi * f0 * 3 * t))
            
        else:
            # Silence
            audio = np.zeros(frame_size) + 0.01 * np.random.randn(frame_size)
            
        audio_samples.append(audio.astype(np.float32))
        
    return audio_samples


def main():
    """Command line interface for profiling."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Profile LNN model performance')
    parser.add_argument('model', help='Path to .lnn model file')
    parser.add_argument('--device', default='cpu', 
                       choices=['cpu', 'stm32f407', 'nrf52840', 'esp32'],
                       help='Target device')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of test samples')
    parser.add_argument('--output', default='profile_results.json',
                       help='Output file for results')
    parser.add_argument('--adaptive', action='store_true',
                       help='Enable adaptive timestep')
    
    args = parser.parse_args()
    
    # Create profiler
    profiler = ModelProfiler(args.model, args.device)
    
    # Generate test data
    audio_samples = generate_test_audio(args.samples)
    
    # Configure adaptive timestep if requested
    adaptive_config = None
    if args.adaptive:
        adaptive_config = AdaptiveConfig(
            min_timestep=0.001,
            max_timestep=0.05,
            energy_threshold=0.1
        )
    
    # Run profiling
    results = profiler.run_full_profile(
        audio_samples=audio_samples,
        adaptive_config=adaptive_config,
    )
    
    # Save results
    profiler.save_results(results, args.output)
    
    print(f"\n✅ Profiling complete! Results saved to {args.output}")


if __name__ == "__main__":
    main()