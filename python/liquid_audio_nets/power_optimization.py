"""Advanced power optimization algorithms for Liquid Neural Networks.

Generation 1 Enhancement: Hardware-aware power modeling and optimization.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class PowerProfile(Enum):
    """Hardware power profiles for different MCU targets."""
    ULTRA_LOW_POWER = "ultra_low_power"  # < 1mW target
    LOW_POWER = "low_power"             # 1-5mW target  
    BALANCED = "balanced"               # 5-10mW target
    HIGH_PERFORMANCE = "high_performance"  # > 10mW acceptable


@dataclass
class HardwareConfig:
    """Hardware-specific configuration for power modeling."""
    mcu_type: str = "cortex_m4"
    base_power_mw: float = 0.6
    cpu_freq_mhz: float = 80
    memory_kb: int = 256
    has_fpu: bool = True
    has_dsp: bool = True
    cache_kb: int = 8


class AdvancedPowerOptimizer:
    """Advanced power optimization with hardware-aware algorithms."""
    
    def __init__(self, hardware_config: Optional[HardwareConfig] = None):
        self.hw_config = hardware_config or HardwareConfig()
        self._power_history = []
        self._complexity_history = []
        self._optimal_timestep_cache = {}
        
    def estimate_power_consumption(self, 
                                 timestep_ms: float, 
                                 complexity: float,
                                 feature_size: int) -> float:
        """Hardware-aware power estimation with advanced modeling.
        
        Args:
            timestep_ms: Processing timestep in milliseconds
            complexity: Signal complexity (0.0-1.0)
            feature_size: Number of features processed
            
        Returns:
            Estimated power consumption in milliwatts
        """
        # Base power (MCU core, peripherals, always-on)
        base_power = self.hw_config.base_power_mw
        
        # CPU scaling based on frequency and utilization
        cpu_utilization = self._estimate_cpu_utilization(timestep_ms, complexity, feature_size)
        cpu_power = self._calculate_cpu_power(cpu_utilization)
        
        # Memory subsystem power
        memory_power = self._calculate_memory_power(feature_size, complexity)
        
        # DSP/FPU power when active
        dsp_power = self._calculate_dsp_power(complexity) if self.hw_config.has_dsp else 0.0
        
        # Clock management overhead
        clock_power = self._calculate_clock_overhead(timestep_ms)
        
        total_power = base_power + cpu_power + memory_power + dsp_power + clock_power
        
        # Hardware-specific constraints
        return self._apply_hardware_limits(total_power)
    
    def _estimate_cpu_utilization(self, timestep_ms: float, complexity: float, feature_size: int) -> float:
        """Estimate CPU utilization based on workload characteristics."""
        # Base computation cycles for LNN processing
        base_cycles = feature_size * 100  # Cycles per feature
        
        # Complexity scaling (non-linear for realistic MCU behavior)
        complexity_cycles = base_cycles * (1 + complexity * complexity * 2)
        
        # Timestep determines available cycles per processing window
        available_cycles = (timestep_ms / 1000.0) * self.hw_config.cpu_freq_mhz * 1e6
        
        utilization = min(1.0, complexity_cycles / available_cycles)
        return utilization
    
    def _calculate_cpu_power(self, utilization: float) -> float:
        """Calculate CPU power based on utilization and frequency."""
        # Modern MCU power scaling (sub-linear due to leakage)
        freq_factor = self.hw_config.cpu_freq_mhz / 80.0  # Normalize to 80MHz
        base_cpu_power = 2.0 * freq_factor  # mW at full utilization
        
        # Power scales roughly as utilization^0.8 + leakage
        dynamic_power = base_cpu_power * (utilization ** 0.8)
        leakage_power = 0.2 * freq_factor
        
        return dynamic_power + leakage_power
    
    def _calculate_memory_power(self, feature_size: int, complexity: float) -> float:
        """Calculate memory subsystem power consumption."""
        # Memory access power
        accesses_per_feature = 3 + complexity * 2  # Read weights, state, write result
        total_accesses = feature_size * accesses_per_feature
        
        # Cache hit rate depends on data locality
        cache_hit_rate = max(0.6, 0.9 - complexity * 0.3)
        
        # Power per access (cache vs. main memory)
        cache_power_per_access = 0.01  # nJ
        memory_power_per_access = 0.1   # nJ
        
        cache_power = total_accesses * cache_hit_rate * cache_power_per_access
        memory_power = total_accesses * (1 - cache_hit_rate) * memory_power_per_access
        
        # Convert to mW (assuming 1kHz processing rate)
        return (cache_power + memory_power) / 1000.0
    
    def _calculate_dsp_power(self, complexity: float) -> float:
        """Calculate DSP/FPU power when active."""
        if not self.hw_config.has_dsp:
            return 0.0
        
        # DSP usage scales with signal complexity
        dsp_utilization = complexity
        max_dsp_power = 1.5  # mW when fully active
        
        return max_dsp_power * dsp_utilization
    
    def _calculate_clock_overhead(self, timestep_ms: float) -> float:
        """Calculate clock management overhead for adaptive timestep."""
        # Frequent timestep changes incur PLL/clock switching overhead
        if timestep_ms < 5.0:
            return 0.3  # High overhead for very short timesteps
        elif timestep_ms < 20.0:
            return 0.1  # Moderate overhead
        else:
            return 0.05  # Low overhead for longer timesteps
    
    def _apply_hardware_limits(self, power: float) -> float:
        """Apply hardware-specific power limits and constraints."""
        if self.hw_config.mcu_type == "cortex_m0":
            return min(power, 5.0)   # Low-power MCUs
        elif self.hw_config.mcu_type == "cortex_m4":
            return min(power, 15.0)  # Mid-range MCUs
        elif self.hw_config.mcu_type == "cortex_m7":
            return min(power, 25.0)  # High-performance MCUs
        else:
            return min(power, 12.0)  # Conservative default
    
    def optimize_timestep_for_power_budget(self, 
                                         complexity: float,
                                         feature_size: int,
                                         power_budget_mw: float) -> float:
        """Find optimal timestep that meets power budget.
        
        Args:
            complexity: Current signal complexity
            feature_size: Number of features
            power_budget_mw: Maximum allowed power consumption
            
        Returns:
            Optimal timestep in milliseconds
        """
        # Check cache first
        cache_key = (round(complexity, 2), feature_size, power_budget_mw)
        if cache_key in self._optimal_timestep_cache:
            return self._optimal_timestep_cache[cache_key]
        
        # Binary search for optimal timestep
        min_timestep = 1.0   # 1ms minimum
        max_timestep = 100.0 # 100ms maximum
        tolerance = 0.1      # 0.1ms tolerance
        
        best_timestep = max_timestep
        
        while max_timestep - min_timestep > tolerance:
            mid_timestep = (min_timestep + max_timestep) / 2
            estimated_power = self.estimate_power_consumption(mid_timestep, complexity, feature_size)
            
            if estimated_power <= power_budget_mw:
                best_timestep = mid_timestep
                max_timestep = mid_timestep
            else:
                min_timestep = mid_timestep
        
        # Cache the result
        self._optimal_timestep_cache[cache_key] = best_timestep
        
        return best_timestep
    
    def get_power_efficiency_score(self, 
                                  power_mw: float, 
                                  performance_score: float) -> float:
        """Calculate overall power efficiency score.
        
        Args:
            power_mw: Current power consumption
            performance_score: Performance metric (higher is better)
            
        Returns:
            Efficiency score (higher is better)
        """
        if power_mw <= 0:
            return 0.0
        
        # Efficiency = Performance per milliwatt
        base_efficiency = performance_score / power_mw
        
        # Bonus for ultra-low power operation
        if power_mw < 2.0:
            ultra_low_power_bonus = 1.5
        elif power_mw < 5.0:
            ultra_low_power_bonus = 1.2
        else:
            ultra_low_power_bonus = 1.0
        
        return base_efficiency * ultra_low_power_bonus
    
    def suggest_hardware_profile(self, 
                                target_power_mw: float,
                                required_performance: float) -> PowerProfile:
        """Suggest optimal power profile based on requirements.
        
        Args:
            target_power_mw: Target power consumption
            required_performance: Required performance level
            
        Returns:
            Recommended power profile
        """
        if target_power_mw < 1.0:
            return PowerProfile.ULTRA_LOW_POWER
        elif target_power_mw < 5.0:
            return PowerProfile.LOW_POWER
        elif target_power_mw < 10.0 or required_performance < 0.8:
            return PowerProfile.BALANCED
        else:
            return PowerProfile.HIGH_PERFORMANCE
    
    def update_power_history(self, power_mw: float, complexity: float):
        """Update power consumption history for adaptive optimization."""
        self._power_history.append(power_mw)
        self._complexity_history.append(complexity)
        
        # Keep only recent history (last 100 samples)
        if len(self._power_history) > 100:
            self._power_history.pop(0)
            self._complexity_history.pop(0)
    
    def get_power_statistics(self) -> Dict[str, float]:
        """Get power consumption statistics."""
        if not self._power_history:
            return {
                'mean_power_mw': 0.0,
                'peak_power_mw': 0.0,
                'min_power_mw': 0.0,
                'std_power_mw': 0.0
            }
        
        if HAS_NUMPY:
            return {
                'mean_power_mw': np.mean(self._power_history),
                'peak_power_mw': np.max(self._power_history),
                'min_power_mw': np.min(self._power_history),
                'std_power_mw': np.std(self._power_history)
            }
        else:
            # Fallback implementation without numpy
            mean_power = sum(self._power_history) / len(self._power_history)
            return {
                'mean_power_mw': mean_power,
                'peak_power_mw': max(self._power_history),
                'min_power_mw': min(self._power_history),
                'std_power_mw': math.sqrt(sum((x - mean_power) ** 2 for x in self._power_history) / len(self._power_history))
            }