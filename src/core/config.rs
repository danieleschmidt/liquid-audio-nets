//! Configuration types for Liquid Neural Networks

use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
use alloc::string::String;

/// Main model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Input feature dimension
    pub input_dim: usize,
    /// Hidden liquid layer dimension  
    pub hidden_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Audio sample rate (Hz)
    pub sample_rate: u32,
    /// Audio frame size for processing
    pub frame_size: usize,
    /// Model type identifier
    pub model_type: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            input_dim: 40,         // MFCC features
            hidden_dim: 64,        // Liquid neurons
            output_dim: 8,         // Keywords
            sample_rate: 16000,    // 16kHz audio
            frame_size: 512,       // ~32ms frames
            model_type: "keyword_spotting".to_string(),
        }
    }
}

/// Adaptive timestep control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Minimum timestep (seconds)
    pub min_timestep: f32,
    /// Maximum timestep (seconds)  
    pub max_timestep: f32,
    /// Energy threshold for adaptation
    pub energy_threshold: f32,
    /// Complexity estimation method
    pub complexity_metric: ComplexityMetric,
    /// Adaptation aggressiveness (0.0 to 1.0)
    pub adaptation_rate: f32,
    /// Enable timestep smoothing
    pub smooth_transitions: bool,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            min_timestep: 0.001,   // 1ms
            max_timestep: 0.050,   // 50ms  
            energy_threshold: 0.1,
            complexity_metric: ComplexityMetric::SpectralFlux,
            adaptation_rate: 0.5,
            smooth_transitions: true,
        }
    }
}

/// Signal complexity estimation methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ComplexityMetric {
    /// Energy-based complexity
    Energy,
    /// Spectral flux (rate of spectral change)
    SpectralFlux,
    /// Zero crossing rate
    ZeroCrossingRate,
    /// Spectral centroid
    SpectralCentroid,
    /// Combined multi-metric approach
    Combined,
}

/// Power management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerConfig {
    /// Target power budget (mW)
    pub power_budget_mw: f32,
    /// Enable aggressive power saving
    pub aggressive_mode: bool,
    /// Power estimation accuracy vs speed tradeoff
    pub estimation_quality: PowerEstimationQuality,
    /// Minimum processing quality threshold
    pub min_quality_threshold: f32,
}

impl Default for PowerConfig {
    fn default() -> Self {
        Self {
            power_budget_mw: 1.0,  // 1mW target
            aggressive_mode: false,
            estimation_quality: PowerEstimationQuality::Balanced,
            min_quality_threshold: 0.8,
        }
    }
}

/// Power estimation quality levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PowerEstimationQuality {
    /// Fast but less accurate estimation
    Fast,
    /// Balanced accuracy and speed
    Balanced,
    /// High accuracy power modeling
    Accurate,
}

/// Runtime configuration for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Enable debug logging
    pub debug_mode: bool,
    /// Performance profiling level
    pub profiling_level: ProfilingLevel,
    /// Memory allocation strategy
    pub memory_strategy: MemoryStrategy,
    /// Numerical precision mode
    pub precision_mode: PrecisionMode,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            debug_mode: false,
            profiling_level: ProfilingLevel::None,
            memory_strategy: MemoryStrategy::Conservative,
            precision_mode: PrecisionMode::Float32,
        }
    }
}

/// Performance profiling levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ProfilingLevel {
    /// No profiling overhead
    None,
    /// Basic timing information
    Basic,
    /// Detailed performance metrics
    Detailed,
    /// Full profiling with memory tracking
    Full,
}

/// Memory allocation strategies for embedded systems
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryStrategy {
    /// Minimize memory usage
    Conservative,
    /// Balance memory and performance
    Balanced,
    /// Optimize for speed over memory
    Performance,
}

/// Numerical precision modes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PrecisionMode {
    /// 16-bit floating point (experimental)
    Float16,
    /// 32-bit floating point (standard)
    Float32,
    /// 64-bit floating point (high precision)
    Float64,
    /// Fixed-point arithmetic
    FixedPoint,
}

/// Validation for configurations
impl ModelConfig {
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.input_dim == 0 {
            return Err("Input dimension must be > 0".to_string());
        }
        if self.hidden_dim == 0 {
            return Err("Hidden dimension must be > 0".to_string());
        }
        if self.output_dim == 0 {
            return Err("Output dimension must be > 0".to_string());
        }
        if self.sample_rate < 8000 || self.sample_rate > 48000 {
            return Err("Sample rate must be between 8kHz and 48kHz".to_string());
        }
        if self.frame_size == 0 || self.frame_size > 4096 {
            return Err("Frame size must be between 1 and 4096 samples".to_string());
        }
        Ok(())
    }
    
    /// Estimate memory usage for this configuration
    pub fn estimate_memory_usage(&self) -> usize {
        // Rough estimation in bytes
        let weights_memory = 
            self.input_dim * self.hidden_dim * 4 +      // W_input
            self.hidden_dim * self.hidden_dim * 4 +     // W_recurrent  
            self.output_dim * self.hidden_dim * 4 +     // W_output
            self.hidden_dim * 4 +                       // b_input
            self.output_dim * 4 +                       // b_output
            self.hidden_dim * 4;                        // tau
            
        let state_memory = self.hidden_dim * 4;         // Liquid state
        let buffer_memory = self.frame_size * 4 * 3;    // Audio + features + temp buffers
        
        weights_memory + state_memory + buffer_memory
    }
}

impl AdaptiveConfig {
    /// Validate adaptive configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.min_timestep <= 0.0 {
            return Err("Minimum timestep must be > 0".to_string());
        }
        if self.max_timestep <= self.min_timestep {
            return Err("Maximum timestep must be > minimum timestep".to_string());
        }
        if self.energy_threshold < 0.0 {
            return Err("Energy threshold must be >= 0".to_string());
        }
        if self.adaptation_rate < 0.0 || self.adaptation_rate > 1.0 {
            return Err("Adaptation rate must be between 0.0 and 1.0".to_string());
        }
        Ok(())
    }
    
    /// Get timestep range ratio
    pub fn timestep_ratio(&self) -> f32 {
        self.max_timestep / self.min_timestep
    }
}

impl PowerConfig {
    /// Validate power configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.power_budget_mw <= 0.0 {
            return Err("Power budget must be > 0".to_string());
        }
        if self.min_quality_threshold < 0.0 || self.min_quality_threshold > 1.0 {
            return Err("Quality threshold must be between 0.0 and 1.0".to_string());
        }
        Ok(())
    }
}