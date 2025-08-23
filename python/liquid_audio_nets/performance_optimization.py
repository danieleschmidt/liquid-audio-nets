"""Generation 3: Advanced Performance Optimization and Scaling for Liquid Neural Networks.

Real-time processing optimization, concurrent processing, memory optimization,
and advanced caching strategies.
"""

from typing import Dict, List, Optional, Any, Callable, Tuple
import time
import threading
from dataclasses import dataclass
from enum import Enum
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class OptimizationLevel(Enum):
    """Optimization levels for different deployment scenarios."""
    DEVELOPMENT = "development"    # No optimization, full debugging
    TESTING = "testing"           # Basic optimization
    PRODUCTION = "production"     # Full optimization
    ULTRA_PERFORMANCE = "ultra"   # Maximum performance, minimal safety


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    optimization_level: OptimizationLevel = OptimizationLevel.PRODUCTION
    enable_caching: bool = True
    enable_concurrent_processing: bool = True
    max_concurrent_streams: int = 4
    cache_size_mb: float = 10.0
    prefetch_enabled: bool = True
    memory_pool_size_kb: int = 1024


class AdvancedMemoryPool:
    """Memory pool for zero-allocation processing."""
    
    def __init__(self, pool_size_kb: int = 1024):
        self.pool_size = pool_size_kb * 1024  # Convert to bytes
        self.allocated_blocks = {}
        self.free_blocks = []
        self.total_allocated = 0
        self.lock = threading.Lock()
        
        # Pre-allocate common buffer sizes
        common_sizes = [256, 512, 1024, 2048, 4096]  # samples
        for size in common_sizes:
            self._preallocate_buffers(size, 4)  # 4 buffers of each size
    
    def _preallocate_buffers(self, size: int, count: int):
        """Pre-allocate buffers of specific size."""
        for _ in range(count):
            if HAS_NUMPY:
                buffer = np.zeros(size, dtype=np.float32)
            else:
                buffer = [0.0] * size
            
            self.free_blocks.append((size, buffer))
    
    def allocate(self, size: int) -> Any:
        """Allocate buffer from pool."""
        with self.lock:
            # Find existing buffer of suitable size
            for i, (buf_size, buffer) in enumerate(self.free_blocks):
                if buf_size >= size:
                    self.free_blocks.pop(i)
                    self.allocated_blocks[id(buffer)] = buffer
                    return buffer
            
            # Allocate new buffer if pool not exhausted
            if self.total_allocated < self.pool_size:
                if HAS_NUMPY:
                    buffer = np.zeros(size, dtype=np.float32)
                else:
                    buffer = [0.0] * size
                
                self.allocated_blocks[id(buffer)] = buffer
                self.total_allocated += size * 4  # Assume 4 bytes per float
                return buffer
            
            # Pool exhausted, return temporary buffer
            if HAS_NUMPY:
                return np.zeros(size, dtype=np.float32)
            else:
                return [0.0] * size
    
    def deallocate(self, buffer: Any):
        """Return buffer to pool."""
        with self.lock:
            buffer_id = id(buffer)
            if buffer_id in self.allocated_blocks:
                del self.allocated_blocks[buffer_id]
                self.free_blocks.append((len(buffer), buffer))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            return {
                'total_allocated_bytes': self.total_allocated,
                'free_blocks': len(self.free_blocks),
                'allocated_blocks': len(self.allocated_blocks),
                'utilization': self.total_allocated / self.pool_size
            }


class StreamingProcessor:
    """High-performance streaming audio processor."""
    
    def __init__(self, 
                 lnn_processor: Any,
                 config: PerformanceConfig,
                 memory_pool: Optional[AdvancedMemoryPool] = None):
        self.lnn = lnn_processor
        self.config = config
        self.memory_pool = memory_pool or AdvancedMemoryPool(config.memory_pool_size_kb)
        
        # Performance monitoring
        self.processing_times = []
        self.throughput_history = []
        self.last_process_time = 0
        
        # Streaming state
        self.overlap_buffer = None
        self.overlap_size = 128  # samples
        
        # Threading for concurrent processing
        self.executor = None
        if config.enable_concurrent_processing:
            import concurrent.futures
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=config.max_concurrent_streams
            )
    
    def process_stream(self, 
                      audio_chunks: List[Any],
                      overlap_processing: bool = True) -> List[Dict[str, Any]]:
        """Process multiple audio chunks with optimized streaming."""
        if not audio_chunks:
            return []
        
        start_time = time.time()
        results = []
        
        if self.config.enable_concurrent_processing and len(audio_chunks) > 1:
            # Concurrent processing for multiple chunks
            results = self._process_concurrent(audio_chunks)
        else:
            # Sequential processing with optimizations
            for chunk in audio_chunks:
                try:
                    if overlap_processing:
                        processed_chunk = self._apply_overlap_processing(chunk)
                    else:
                        processed_chunk = chunk
                    
                    result = self.lnn.process(processed_chunk)
                    results.append(result)
                    
                except Exception as e:
                    # Graceful degradation
                    results.append({
                        'error': True,
                        'message': str(e),
                        'power_mw': 0.0,
                        'confidence': 0.0
                    })
        
        # Update performance metrics
        total_time = time.time() - start_time
        self.processing_times.append(total_time)
        
        throughput = len(audio_chunks) / total_time if total_time > 0 else 0
        self.throughput_history.append(throughput)
        
        # Keep only recent history
        if len(self.processing_times) > 100:
            self.processing_times.pop(0)
            self.throughput_history.pop(0)
        
        return results
    
    def _process_concurrent(self, audio_chunks: List[Any]) -> List[Dict[str, Any]]:
        """Process chunks concurrently."""
        if not self.executor:
            # Fallback to sequential
            return [self.lnn.process(chunk) for chunk in audio_chunks]
        
        # Submit all tasks
        futures = []
        for chunk in audio_chunks:
            future = self.executor.submit(self._safe_process, chunk)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            try:
                result = future.result(timeout=1.0)  # 1 second timeout
                results.append(result)
            except Exception as e:
                results.append({
                    'error': True,
                    'message': str(e),
                    'power_mw': 0.0,
                    'confidence': 0.0
                })
        
        return results
    
    def _safe_process(self, chunk: Any) -> Dict[str, Any]:
        """Thread-safe processing wrapper."""
        try:
            return self.lnn.process(chunk)
        except Exception as e:
            return {
                'error': True,
                'message': str(e),
                'power_mw': 0.0,
                'confidence': 0.0
            }
    
    def _apply_overlap_processing(self, chunk: Any) -> Any:
        """Apply overlap-add processing for streaming."""
        if self.overlap_buffer is None:
            self.overlap_buffer = self.memory_pool.allocate(self.overlap_size)
            # Initialize with zeros
            for i in range(len(self.overlap_buffer)):
                self.overlap_buffer[i] = 0.0
        
        # Combine with previous overlap
        chunk_list = list(chunk) if not isinstance(chunk, list) else chunk
        
        # Apply overlap at the beginning
        overlapped_chunk = list(self.overlap_buffer) + chunk_list
        
        # Update overlap buffer with end of current chunk
        if len(chunk_list) >= self.overlap_size:
            for i in range(self.overlap_size):
                self.overlap_buffer[i] = chunk_list[-(self.overlap_size - i)]
        
        return overlapped_chunk
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        if not self.processing_times:
            return {'no_data': True}
        
        avg_time = sum(self.processing_times) / len(self.processing_times)
        avg_throughput = sum(self.throughput_history) / len(self.throughput_history)
        
        return {
            'average_processing_time_ms': avg_time * 1000,
            'peak_processing_time_ms': max(self.processing_times) * 1000,
            'min_processing_time_ms': min(self.processing_times) * 1000,
            'average_throughput_chunks_per_sec': avg_throughput,
            'memory_pool_stats': self.memory_pool.get_stats(),
            'concurrent_processing_enabled': self.config.enable_concurrent_processing
        }


class AdaptiveQualityController:
    """Adaptive quality control for real-time constraints."""
    
    def __init__(self, target_latency_ms: float = 50.0):
        self.target_latency_ms = target_latency_ms
        self.latency_history = []
        self.quality_level = 1.0  # 0.0 to 1.0
        self.adaptation_rate = 0.1
        
    def adapt_quality(self, measured_latency_ms: float) -> float:
        """Adapt quality based on measured latency."""
        self.latency_history.append(measured_latency_ms)
        if len(self.latency_history) > 10:
            self.latency_history.pop(0)
        
        # Calculate average recent latency
        avg_latency = sum(self.latency_history) / len(self.latency_history)
        
        # Adapt quality level
        if avg_latency > self.target_latency_ms * 1.2:
            # Reduce quality to improve speed
            self.quality_level = max(0.1, self.quality_level - self.adaptation_rate)
        elif avg_latency < self.target_latency_ms * 0.8:
            # Increase quality if we have headroom
            self.quality_level = min(1.0, self.quality_level + self.adaptation_rate * 0.5)
        
        return self.quality_level
    
    def get_recommended_config(self, quality_level: float) -> Dict[str, Any]:
        """Get recommended processing configuration for quality level."""
        if quality_level >= 0.9:
            return {
                'feature_resolution': 'high',
                'timestep_factor': 1.0,
                'complexity_metric': 'full_spectral'
            }
        elif quality_level >= 0.6:
            return {
                'feature_resolution': 'medium',
                'timestep_factor': 1.5,
                'complexity_metric': 'spectral_flux'
            }
        else:
            return {
                'feature_resolution': 'low',
                'timestep_factor': 2.0,
                'complexity_metric': 'energy'
            }


class PerformanceProfiler:
    """Comprehensive performance profiler for LNN processing."""
    
    def __init__(self):
        self.profiles = {}
        self.active_profile = None
        
    def start_profile(self, name: str):
        """Start profiling a processing session."""
        self.active_profile = name
        self.profiles[name] = {
            'start_time': time.time(),
            'stages': [],
            'memory_snapshots': [],
            'errors': []
        }
    
    def record_stage(self, stage_name: str, duration_ms: float, memory_kb: Optional[float] = None):
        """Record a processing stage."""
        if self.active_profile:
            stage_data = {
                'name': stage_name,
                'duration_ms': duration_ms,
                'timestamp': time.time()
            }
            if memory_kb is not None:
                stage_data['memory_kb'] = memory_kb
            
            self.profiles[self.active_profile]['stages'].append(stage_data)
    
    def end_profile(self) -> Dict[str, Any]:
        """End current profile and return results."""
        if not self.active_profile or self.active_profile not in self.profiles:
            return {}
        
        profile = self.profiles[self.active_profile]
        total_time = time.time() - profile['start_time']
        
        # Analyze stages
        stage_analysis = {}
        for stage in profile['stages']:
            name = stage['name']
            if name not in stage_analysis:
                stage_analysis[name] = []
            stage_analysis[name].append(stage['duration_ms'])
        
        # Calculate statistics
        stage_stats = {}
        for name, durations in stage_analysis.items():
            stage_stats[name] = {
                'avg_duration_ms': sum(durations) / len(durations),
                'max_duration_ms': max(durations),
                'min_duration_ms': min(durations),
                'total_duration_ms': sum(durations),
                'percentage': sum(durations) / (total_time * 1000) * 100
            }
        
        result = {
            'profile_name': self.active_profile,
            'total_time_ms': total_time * 1000,
            'stage_stats': stage_stats,
            'num_stages': len(profile['stages']),
            'errors': profile['errors']
        }
        
        self.active_profile = None
        return result


# Singleton instances for global optimization
_memory_pool = None
_performance_profiler = None

def get_global_memory_pool() -> AdvancedMemoryPool:
    """Get global memory pool instance."""
    global _memory_pool
    if _memory_pool is None:
        _memory_pool = AdvancedMemoryPool()
    return _memory_pool

def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance."""
    global _performance_profiler
    if _performance_profiler is None:
        _performance_profiler = PerformanceProfiler()
    return _performance_profiler