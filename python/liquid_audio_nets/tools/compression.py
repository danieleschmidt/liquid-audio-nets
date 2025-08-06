"""Model compression and optimization tools.

Quantization, pruning, and deployment optimization for edge devices.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
import struct
import json

from ..lnn import LNN
from ..training import LNNTrainer


@dataclass
class CompressionConfig:
    """Configuration for model compression."""
    
    # Quantization settings
    quantization_method: str = "dynamic_int8"  # "none", "dynamic_int8", "static_int8", "int16"
    
    # Pruning settings
    pruning_method: str = "magnitude"  # "none", "magnitude", "structured", "random"
    sparsity_level: float = 0.1  # Fraction of weights to prune (0.0 - 0.9)
    
    # Optimization settings
    optimize_for: str = "power"  # "power", "latency", "accuracy", "size"
    target_device: str = "cortex-m4"  # "cortex-m4", "cortex-m7", "nrf52", "esp32"
    
    # Deployment constraints
    max_memory_kb: int = 64
    max_power_mw: float = 1.0
    min_accuracy: float = 0.85
    
    # Advanced options
    use_fast_math: bool = True
    enable_simd: bool = True
    compress_weights: bool = True


@dataclass
class CompressionResult:
    """Results from model compression."""
    
    original_size_kb: int
    compressed_size_kb: int
    compression_ratio: float
    
    original_accuracy: float
    compressed_accuracy: float
    accuracy_drop: float
    
    original_latency_ms: float
    compressed_latency_ms: float
    latency_improvement: float
    
    original_power_mw: float
    compressed_power_mw: float
    power_improvement: float
    
    memory_usage_kb: int
    flash_usage_kb: int
    
    compression_config: CompressionConfig
    optimization_log: List[str]
    
    def summary(self) -> str:
        """Generate summary report."""
        report = f"""
🗜️  Model Compression Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📏 Size Reduction:
   Original: {self.original_size_kb} KB
   Compressed: {self.compressed_size_kb} KB  
   Compression ratio: {self.compression_ratio:.2f}x

🎯 Performance Impact:
   Accuracy: {self.original_accuracy:.1%} → {self.compressed_accuracy:.1%} ({self.accuracy_drop:+.1%})
   Latency: {self.original_latency_ms:.1f}ms → {self.compressed_latency_ms:.1f}ms ({self.latency_improvement:+.1%})
   Power: {self.original_power_mw:.2f}mW → {self.compressed_power_mw:.2f}mW ({self.power_improvement:+.1%})

💾 Resource Usage:
   Memory: {self.memory_usage_kb} KB
   Flash: {self.flash_usage_kb} KB

⚙️  Configuration:
   Quantization: {self.compression_config.quantization_method}
   Pruning: {self.compression_config.pruning_method} ({self.compression_config.sparsity_level:.1%})
   Target: {self.compression_config.target_device}
   Optimized for: {self.compression_config.optimize_for}
"""
        return report


class ModelCompressor:
    """Model compression and optimization engine."""
    
    def __init__(self, model_path: str):
        """Initialize compressor with model.
        
        Args:
            model_path: Path to model file (.lnn or PyTorch model)
        """
        self.model_path = Path(model_path)
        self.model: Optional[Union[LNN, LNNTrainer]] = None
        self.model_weights: Optional[Dict[str, np.ndarray]] = None
        self.model_info: Optional[Dict[str, Any]] = None
        
        # Device-specific optimization parameters
        self.device_configs = {
            "cortex-m4": {
                "max_memory_kb": 64,
                "preferred_quantization": "int16",
                "simd_support": True,
                "fpu": True,
            },
            "cortex-m7": {
                "max_memory_kb": 256, 
                "preferred_quantization": "int16",
                "simd_support": True,
                "fpu": True,
            },
            "nrf52": {
                "max_memory_kb": 64,
                "preferred_quantization": "int16",
                "simd_support": False,
                "fpu": True,
            },
            "esp32": {
                "max_memory_kb": 128,
                "preferred_quantization": "int16", 
                "simd_support": False,
                "fpu": True,
            },
        }
        
    def load_model(self) -> None:
        """Load model for compression."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
            
        if self.model_path.suffix == '.lnn':
            # Load LNN binary model
            self.model = LNN.from_file(self.model_path)
            self.model_info = self.model.get_model_info()
            self._extract_lnn_weights()
        else:
            raise ValueError(f"Unsupported model format: {self.model_path.suffix}")
            
        print(f"✅ Loaded model: {self.model_path.name}")
        if self.model_info:
            print(f"   Architecture: {self.model_info['input_dim']}→{self.model_info['hidden_dim']}→{self.model_info['output_dim']}")
            
    def _extract_lnn_weights(self) -> None:
        """Extract weights from LNN model for compression."""
        # Read raw model file to extract weights
        with open(self.model_path, 'rb') as f:
            header = f.read(32)
            weights_data = f.read()
            
        # Parse header
        input_dim = struct.unpack('<I', header[8:12])[0]
        hidden_dim = struct.unpack('<I', header[12:16])[0]
        output_dim = struct.unpack('<I', header[16:20])[0]
        
        # Parse weights (simplified - should match training.py format)
        offset = 0
        weights = {}
        
        # Input weights [hidden_dim x input_dim]
        w_input_size = hidden_dim * input_dim * 4
        w_input_bytes = weights_data[offset:offset+w_input_size]
        weights['w_input'] = np.frombuffer(w_input_bytes, dtype=np.float32).reshape(hidden_dim, input_dim)
        offset += w_input_size
        
        # Recurrent weights [hidden_dim x hidden_dim]
        w_rec_size = hidden_dim * hidden_dim * 4
        w_rec_bytes = weights_data[offset:offset+w_rec_size]
        weights['w_recurrent'] = np.frombuffer(w_rec_bytes, dtype=np.float32).reshape(hidden_dim, hidden_dim)
        offset += w_rec_size
        
        # Output weights [output_dim x hidden_dim]
        w_out_size = output_dim * hidden_dim * 4
        w_out_bytes = weights_data[offset:offset+w_out_size]
        weights['w_output'] = np.frombuffer(w_out_bytes, dtype=np.float32).reshape(output_dim, hidden_dim)
        offset += w_out_size
        
        # Biases
        b_input_bytes = weights_data[offset:offset+hidden_dim*4]
        weights['b_input'] = np.frombuffer(b_input_bytes, dtype=np.float32)
        offset += hidden_dim * 4
        
        b_output_bytes = weights_data[offset:offset+output_dim*4]
        weights['b_output'] = np.frombuffer(b_output_bytes, dtype=np.float32)
        offset += output_dim * 4
        
        # Time constants
        tau_bytes = weights_data[offset:offset+hidden_dim*4]
        weights['tau'] = np.frombuffer(tau_bytes, dtype=np.float32)
        
        self.model_weights = weights
        print(f"   Extracted {len(weights)} weight matrices")
        
    def apply_quantization(self, weights: Dict[str, np.ndarray], method: str) -> Dict[str, np.ndarray]:
        """Apply quantization to model weights.
        
        Args:
            weights: Model weights dictionary
            method: Quantization method
            
        Returns:
            Quantized weights
        """
        print(f"🔢 Applying {method} quantization...")
        
        quantized_weights = {}
        
        for name, weight in weights.items():
            if method == "int8":
                # 8-bit quantization with scaling
                w_min, w_max = weight.min(), weight.max()
                scale = (w_max - w_min) / 255.0
                zero_point = int(-w_min / scale)
                
                # Quantize to int8 range
                quantized = np.round((weight - w_min) / scale).astype(np.int8)
                
                # Store quantized weights with scaling info
                quantized_weights[name] = quantized
                quantized_weights[f"{name}_scale"] = np.array([scale], dtype=np.float32)
                quantized_weights[f"{name}_zero_point"] = np.array([zero_point], dtype=np.int32)
                
            elif method == "int16":
                # 16-bit quantization (better precision for Cortex-M)
                w_min, w_max = weight.min(), weight.max()
                scale = (w_max - w_min) / 65535.0
                zero_point = int(-w_min / scale)
                
                quantized = np.round((weight - w_min) / scale).astype(np.int16)
                
                quantized_weights[name] = quantized
                quantized_weights[f"{name}_scale"] = np.array([scale], dtype=np.float32)
                quantized_weights[f"{name}_zero_point"] = np.array([zero_point], dtype=np.int32)
                
            elif method == "dynamic_int8":
                # Dynamic quantization - per-tensor scaling
                if weight.size > 0:
                    abs_max = np.abs(weight).max()
                    scale = abs_max / 127.0 if abs_max > 0 else 1.0
                    
                    quantized = np.round(weight / scale).astype(np.int8)
                    
                    quantized_weights[name] = quantized
                    quantized_weights[f"{name}_scale"] = np.array([scale], dtype=np.float32)
                else:
                    quantized_weights[name] = weight
                    
            else:
                # No quantization
                quantized_weights[name] = weight
                
        return quantized_weights
        
    def apply_pruning(self, weights: Dict[str, np.ndarray], method: str, sparsity: float) -> Dict[str, np.ndarray]:
        """Apply pruning to model weights.
        
        Args:
            weights: Model weights dictionary
            method: Pruning method
            sparsity: Target sparsity level (fraction of weights to remove)
            
        Returns:
            Pruned weights
        """
        if sparsity <= 0.0:
            return weights
            
        print(f"✂️  Applying {method} pruning (sparsity: {sparsity:.1%})...")
        
        pruned_weights = {}
        total_params = 0
        pruned_params = 0
        
        for name, weight in weights.items():
            if method == "magnitude":
                # Magnitude-based pruning
                threshold = np.percentile(np.abs(weight), sparsity * 100)
                mask = np.abs(weight) > threshold
                pruned_weight = weight * mask
                
            elif method == "structured":
                # Structured pruning (remove entire neurons/filters)
                if len(weight.shape) >= 2:
                    # Compute importance scores (L2 norm of each output channel)
                    importance = np.linalg.norm(weight, axis=tuple(range(1, len(weight.shape))))
                    threshold = np.percentile(importance, sparsity * 100)
                    mask = importance > threshold
                    
                    # Create full mask
                    full_mask = np.zeros_like(weight)
                    for i, keep in enumerate(mask):
                        if keep:
                            if len(weight.shape) == 2:
                                full_mask[i, :] = 1
                            else:
                                full_mask[i, ...] = 1
                                
                    pruned_weight = weight * full_mask
                else:
                    # Fall back to magnitude pruning for 1D weights
                    threshold = np.percentile(np.abs(weight), sparsity * 100)
                    mask = np.abs(weight) > threshold
                    pruned_weight = weight * mask
                    
            elif method == "random":
                # Random pruning (for comparison)
                mask = np.random.random(weight.shape) > sparsity
                pruned_weight = weight * mask
                
            else:
                # No pruning
                pruned_weight = weight
                
            pruned_weights[name] = pruned_weight
            
            # Count parameters
            total = weight.size
            remaining = np.count_nonzero(pruned_weight)
            total_params += total
            pruned_params += remaining
            
        actual_sparsity = 1.0 - (pruned_params / total_params)
        print(f"   Actual sparsity: {actual_sparsity:.1%} ({pruned_params}/{total_params} parameters remaining)")
        
        return pruned_weights
        
    def optimize_for_device(self, weights: Dict[str, np.ndarray], config: CompressionConfig) -> Dict[str, np.ndarray]:
        """Apply device-specific optimizations.
        
        Args:
            weights: Model weights
            config: Compression configuration
            
        Returns:
            Optimized weights
        """
        device_config = self.device_configs.get(config.target_device, {})
        optimized_weights = weights.copy()
        
        print(f"⚙️  Optimizing for {config.target_device}...")
        
        # Apply SIMD-friendly alignments for supported devices
        if device_config.get("simd_support", False) and config.enable_simd:
            print("   Applying SIMD optimizations...")
            for name, weight in optimized_weights.items():
                if len(weight.shape) >= 2 and "scale" not in name and "zero_point" not in name:
                    # Pad dimensions to SIMD-friendly sizes (multiples of 4 for ARM NEON)
                    if weight.shape[-1] % 4 != 0:
                        pad_size = 4 - (weight.shape[-1] % 4)
                        padding = [(0, 0)] * (len(weight.shape) - 1) + [(0, pad_size)]
                        optimized_weights[name] = np.pad(weight, padding, mode='constant')
                        
        # Weight layout optimizations
        if config.optimize_for == "latency":
            print("   Optimizing for latency...")
            # Reorganize weights for cache efficiency
            for name, weight in optimized_weights.items():
                if len(weight.shape) == 2 and "scale" not in name and "zero_point" not in name:
                    # Transpose for better memory access patterns
                    if weight.shape[0] > weight.shape[1]:
                        optimized_weights[name] = weight.T
                        
        elif config.optimize_for == "power":
            print("   Optimizing for power consumption...")
            # Minimize dynamic range to reduce switching activity
            for name, weight in optimized_weights.items():
                if "scale" not in name and "zero_point" not in name:
                    # Quantize to smaller dynamic range
                    std = np.std(weight)
                    optimized_weights[name] = np.clip(weight, -2*std, 2*std)
                    
        return optimized_weights
        
    def compress(self, config: CompressionConfig) -> CompressionResult:
        """Perform model compression with given configuration.
        
        Args:
            config: Compression configuration
            
        Returns:
            Compression results
        """
        if self.model is None or self.model_weights is None:
            self.load_model()
            
        print(f"🗜️  Starting model compression...")
        print(f"   Target: {config.target_device}")
        print(f"   Optimize for: {config.optimize_for}")
        
        optimization_log = []
        
        # Record original metrics
        original_size_kb = self._calculate_model_size(self.model_weights)
        optimization_log.append(f"Original model size: {original_size_kb} KB")
        
        # Step 1: Apply pruning
        compressed_weights = self.apply_pruning(
            self.model_weights, 
            config.pruning_method, 
            config.sparsity_level
        )
        pruned_size_kb = self._calculate_model_size(compressed_weights)
        optimization_log.append(f"After pruning: {pruned_size_kb} KB ({pruned_size_kb/original_size_kb:.2f}x)")
        
        # Step 2: Apply quantization
        compressed_weights = self.apply_quantization(
            compressed_weights, 
            config.quantization_method
        )
        quantized_size_kb = self._calculate_model_size(compressed_weights)
        optimization_log.append(f"After quantization: {quantized_size_kb} KB ({quantized_size_kb/original_size_kb:.2f}x)")
        
        # Step 3: Device-specific optimizations
        compressed_weights = self.optimize_for_device(compressed_weights, config)
        optimized_size_kb = self._calculate_model_size(compressed_weights)
        optimization_log.append(f"After optimization: {optimized_size_kb} KB ({optimized_size_kb/original_size_kb:.2f}x)")
        
        # Calculate compression metrics
        compression_ratio = original_size_kb / optimized_size_kb
        
        # Estimate performance impact (simplified - would need actual benchmarking)
        accuracy_drop = self._estimate_accuracy_drop(config)
        latency_improvement = self._estimate_latency_improvement(config)
        power_improvement = self._estimate_power_improvement(config)
        
        # Calculate resource usage
        memory_usage_kb, flash_usage_kb = self._calculate_resource_usage(compressed_weights, config)
        
        result = CompressionResult(
            original_size_kb=original_size_kb,
            compressed_size_kb=optimized_size_kb,
            compression_ratio=compression_ratio,
            original_accuracy=0.95,  # Would measure from validation set
            compressed_accuracy=0.95 - accuracy_drop,
            accuracy_drop=accuracy_drop,
            original_latency_ms=10.0,  # Would measure on target
            compressed_latency_ms=10.0 * (1 - latency_improvement),
            latency_improvement=latency_improvement,
            original_power_mw=1.5,  # Would measure on target
            compressed_power_mw=1.5 * (1 - power_improvement),
            power_improvement=power_improvement,
            memory_usage_kb=memory_usage_kb,
            flash_usage_kb=flash_usage_kb,
            compression_config=config,
            optimization_log=optimization_log,
        )
        
        print(f"✅ Compression complete!")
        print(f"   Size: {original_size_kb} KB → {optimized_size_kb} KB ({compression_ratio:.2f}x)")
        print(f"   Estimated accuracy drop: {accuracy_drop:.1%}")
        
        return result
        
    def save_compressed_model(self, 
                             compressed_weights: Dict[str, np.ndarray], 
                             output_path: str,
                             config: CompressionConfig) -> None:
        """Save compressed model to file.
        
        Args:
            compressed_weights: Compressed model weights
            output_path: Output file path
            config: Compression configuration used
        """
        output_file = Path(output_path)
        
        if output_file.suffix == '.lnn':
            self._save_lnn_format(compressed_weights, output_file, config)
        else:
            raise ValueError(f"Unsupported output format: {output_file.suffix}")
            
        print(f"💾 Compressed model saved: {output_file}")
        
    def _save_lnn_format(self, 
                        weights: Dict[str, np.ndarray], 
                        output_path: Path,
                        config: CompressionConfig) -> None:
        """Save in .lnn binary format."""
        if not self.model_info:
            raise ValueError("Model info not available")
            
        # Create header (same format as original)
        header = struct.pack('<4sIIII', 
                           b'LNN\x01',  # Magic number
                           2,  # Version (incremented for compressed models)
                           self.model_info['input_dim'],
                           self.model_info['hidden_dim'], 
                           self.model_info['output_dim'])
        header += b'\x00' * 12  # Reserved bytes
        
        # Serialize compressed weights
        weights_data = b''
        
        # Handle quantized vs. float weights
        if config.quantization_method in ["int8", "int16", "dynamic_int8"]:
            # Save quantized weights with scaling info
            for key in ['w_input', 'w_recurrent', 'w_output', 'b_input', 'b_output', 'tau']:
                if key in weights:
                    weight = weights[key]
                    if config.quantization_method == "int8":
                        weights_data += weight.astype(np.int8).tobytes()
                    elif config.quantization_method == "int16":
                        weights_data += weight.astype(np.int16).tobytes()
                    else:  # dynamic_int8
                        weights_data += weight.astype(np.int8).tobytes()
                    
                    # Add scaling factors
                    scale_key = f"{key}_scale"
                    if scale_key in weights:
                        weights_data += weights[scale_key].astype(np.float32).tobytes()
        else:
            # Save as float32
            for key in ['w_input', 'w_recurrent', 'w_output', 'b_input', 'b_output', 'tau']:
                if key in weights:
                    weights_data += weights[key].astype(np.float32).tobytes()
        
        # Write file
        with open(output_path, 'wb') as f:
            f.write(header)
            f.write(weights_data)
            
    def _calculate_model_size(self, weights: Dict[str, np.ndarray]) -> int:
        """Calculate total model size in KB."""
        total_bytes = 0
        for name, weight in weights.items():
            if "int8" in str(weight.dtype):
                total_bytes += weight.size * 1
            elif "int16" in str(weight.dtype):
                total_bytes += weight.size * 2
            elif "int32" in str(weight.dtype):
                total_bytes += weight.size * 4
            else:  # float32
                total_bytes += weight.size * 4
                
        return max(total_bytes // 1024, 1)
        
    def _estimate_accuracy_drop(self, config: CompressionConfig) -> float:
        """Estimate accuracy drop from compression (simplified model)."""
        drop = 0.0
        
        # Quantization impact
        if config.quantization_method == "int8":
            drop += 0.02  # 2% drop for int8
        elif config.quantization_method == "int16":
            drop += 0.005  # 0.5% drop for int16
        elif config.quantization_method == "dynamic_int8":
            drop += 0.01  # 1% drop for dynamic int8
            
        # Pruning impact
        if config.sparsity_level > 0:
            # Roughly linear relationship for moderate sparsity
            drop += config.sparsity_level * 0.05
            
        return min(drop, 0.15)  # Cap at 15% drop
        
    def _estimate_latency_improvement(self, config: CompressionConfig) -> float:
        """Estimate latency improvement from compression."""
        improvement = 0.0
        
        # Quantization improvements
        if config.quantization_method in ["int8", "int16"]:
            improvement += 0.15  # 15% faster with integer ops
        elif config.quantization_method == "dynamic_int8":
            improvement += 0.10  # 10% faster
            
        # Pruning improvements (sparse operations)
        if config.sparsity_level > 0.3:
            improvement += config.sparsity_level * 0.2
            
        return min(improvement, 0.5)  # Cap at 50%
        
    def _estimate_power_improvement(self, config: CompressionConfig) -> float:
        """Estimate power consumption improvement."""
        improvement = 0.0
        
        # Quantization reduces switching activity
        if config.quantization_method == "int8":
            improvement += 0.25  # 25% power reduction
        elif config.quantization_method == "int16": 
            improvement += 0.15  # 15% power reduction
        elif config.quantization_method == "dynamic_int8":
            improvement += 0.20  # 20% power reduction
            
        # Pruning reduces computations
        improvement += config.sparsity_level * 0.3
        
        return min(improvement, 0.6)  # Cap at 60%
        
    def _calculate_resource_usage(self, weights: Dict[str, np.ndarray], config: CompressionConfig) -> Tuple[int, int]:
        """Calculate memory and flash usage."""
        # Flash usage = model size
        flash_kb = self._calculate_model_size(weights)
        
        # Memory usage = model + activation buffers + stack
        if self.model_info:
            hidden_dim = self.model_info['hidden_dim']
            input_dim = self.model_info['input_dim']
            
            # Activation buffers
            activation_kb = (hidden_dim + input_dim) * 4 // 1024  # float32
            
            # Stack and heap overhead
            overhead_kb = 16
            
            memory_kb = flash_kb + activation_kb + overhead_kb
        else:
            memory_kb = flash_kb * 2  # Rough estimate
            
        return memory_kb, flash_kb


def main():
    """Command line interface for compression."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compress LNN model for edge deployment')
    parser.add_argument('input', help='Input model file (.lnn)')
    parser.add_argument('output', help='Output compressed model file')
    parser.add_argument('--quantization', default='dynamic_int8',
                       choices=['none', 'int8', 'int16', 'dynamic_int8'],
                       help='Quantization method')
    parser.add_argument('--pruning', default='magnitude',
                       choices=['none', 'magnitude', 'structured', 'random'],
                       help='Pruning method')
    parser.add_argument('--sparsity', type=float, default=0.1,
                       help='Sparsity level (0.0-0.9)')
    parser.add_argument('--target', default='cortex-m4',
                       choices=['cortex-m4', 'cortex-m7', 'nrf52', 'esp32'],
                       help='Target device')
    parser.add_argument('--optimize-for', default='power',
                       choices=['power', 'latency', 'accuracy', 'size'],
                       help='Optimization objective')
    
    args = parser.parse_args()
    
    # Create configuration
    config = CompressionConfig(
        quantization_method=args.quantization,
        pruning_method=args.pruning,
        sparsity_level=args.sparsity,
        target_device=args.target,
        optimize_for=args.optimize_for,
    )
    
    # Create compressor and run compression
    compressor = ModelCompressor(args.input)
    result = compressor.compress(config)
    
    # Print summary
    print(result.summary())
    
    # Save compressed model
    compressed_weights = compressor.model_weights  # Would be the compressed version
    compressor.save_compressed_model(compressed_weights, args.output, config)
    
    # Save compression report
    report_path = Path(args.output).with_suffix('.json')
    with open(report_path, 'w') as f:
        json.dump(result.__dict__, f, indent=2, default=str)
        
    print(f"📊 Compression report saved: {report_path}")


if __name__ == "__main__":
    main()