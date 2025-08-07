//! # Liquid Audio Nets
//! 
//! Edge-efficient Liquid Neural Network models for always-on audio sensing.
//! 
//! This crate provides ultra-low-power neural network implementations optimized
//! for ARM Cortex-M microcontrollers and edge devices with advanced scaling,
//! optimization, and deployment capabilities.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use core::alloc::{vec::Vec, string::String, vec};

#[cfg(not(feature = "std"))]
use alloc::string::ToString;

/// Minimal AdaptiveConfig for compilation
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveConfig {
    pub min_timestep_ms: f32,
    pub max_timestep_ms: f32,
    pub energy_threshold: f32,
    pub complexity_penalty: f32,
    pub power_budget_mw: f32,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            min_timestep_ms: 5.0,
            max_timestep_ms: 50.0,
            energy_threshold: 0.1,
            complexity_penalty: 0.02,
            power_budget_mw: 1.0,
        }
    }
}

/// Minimal ModelConfig
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub sample_rate: u32,
    pub frame_size: usize,
    pub model_type: String,
}

/// Enhanced ProcessingResult
#[derive(Debug, Clone)]
pub struct ProcessingResult {
    pub output: Vec<f32>,
    pub confidence: f32,
    pub timestep_ms: f32,
    pub power_mw: f32,
    pub complexity: f32,
    pub liquid_energy: f32,
    pub metadata: Option<String>,
}

/// Main error type
#[derive(Debug, Clone)]
pub enum LiquidAudioError {
    ModelError(String),
    AudioError(String),
    ConfigError(String),
    ComputationError(String),
    IoError(String),
    InvalidInput(String),
    SecurityError(String),
    ResourceExhausted(String),
    InvalidState(String),
    ThreadError(String),
}

impl core::fmt::Display for LiquidAudioError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LiquidAudioError::ModelError(msg) => write!(f, "Model error: {}", msg),
            LiquidAudioError::AudioError(msg) => write!(f, "Audio error: {}", msg),
            LiquidAudioError::ConfigError(msg) => write!(f, "Config error: {}", msg),
            LiquidAudioError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            LiquidAudioError::IoError(msg) => write!(f, "I/O error: {}", msg),
            LiquidAudioError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            LiquidAudioError::SecurityError(msg) => write!(f, "Security error: {}", msg),
            LiquidAudioError::ResourceExhausted(msg) => write!(f, "Resource exhausted: {}", msg),
            LiquidAudioError::InvalidState(msg) => write!(f, "Invalid state: {}", msg),
            LiquidAudioError::ThreadError(msg) => write!(f, "Thread error: {}", msg),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LiquidAudioError {}

pub type Result<T> = core::result::Result<T, LiquidAudioError>;

/// Enhanced LNN implementation with diagnostics and validation
#[derive(Debug)]
pub struct LNN {
    config: ModelConfig,
    power_mw: f32,
    diagnostics: DiagnosticsCollector,
    adaptive_config: Option<AdaptiveConfig>,
    validation_enabled: bool,
    max_input_magnitude: f32,
}

impl LNN {
    pub fn new(config: ModelConfig) -> Result<Self> {
        // Validate configuration
        Self::validate_config(&config)?;
        
        Logger::info(&format!("Creating LNN: {}→{}→{}", config.input_dim, config.hidden_dim, config.output_dim));
        
        Ok(LNN {
            config,
            power_mw: 1.0,
            diagnostics: DiagnosticsCollector::new(),
            adaptive_config: None,
            validation_enabled: true,
            max_input_magnitude: 10.0, // Default maximum input value
        })
    }
    
    /// Validate model configuration
    fn validate_config(config: &ModelConfig) -> Result<()> {
        if config.input_dim == 0 {
            return Err(LiquidAudioError::ConfigError("Input dimension must be > 0".to_string()));
        }
        if config.hidden_dim == 0 {
            return Err(LiquidAudioError::ConfigError("Hidden dimension must be > 0".to_string()));
        }
        if config.output_dim == 0 {
            return Err(LiquidAudioError::ConfigError("Output dimension must be > 0".to_string()));
        }
        if config.sample_rate < 1000 || config.sample_rate > 192000 {
            return Err(LiquidAudioError::ConfigError(
                format!("Sample rate {} is outside valid range 1000-192000 Hz", config.sample_rate)
            ));
        }
        if config.frame_size == 0 || config.frame_size > 8192 {
            return Err(LiquidAudioError::ConfigError(
                format!("Frame size {} is outside valid range 1-8192", config.frame_size)
            ));
        }
        
        // Check for reasonable model sizes
        if config.hidden_dim > 1024 {
            Logger::warn(&format!("Large hidden dimension {} may impact performance", config.hidden_dim));
        }
        
        let estimated_memory = (config.input_dim * config.hidden_dim + 
                               config.hidden_dim * config.hidden_dim +
                               config.hidden_dim * config.output_dim) * 4;
        if estimated_memory > 1024 * 1024 {
            Logger::warn(&format!("Model size {}KB may exceed embedded memory limits", estimated_memory / 1024));
        }
        
        Ok(())
    }

    #[cfg(feature = "std")]
    pub fn load_from_file(_path: impl AsRef<std::path::Path>) -> Result<Self> {
        let config = ModelConfig {
            input_dim: 40,
            hidden_dim: 64,
            output_dim: 10,
            sample_rate: 16000,
            frame_size: 512,
            model_type: "demo".to_string(),
        };
        Ok(LNN::new(config)?)
    }

    pub fn process(&mut self, audio_buffer: &[f32]) -> Result<ProcessingResult> {
        let start_time = Self::get_timestamp();
        
        // Input validation
        if let Err(e) = self.validate_input(audio_buffer) {
            self.diagnostics.record_error(&e);
            return Err(e);
        }

        // Process with error recovery
        let result = match self.process_internal(audio_buffer) {
            Ok(result) => result,
            Err(e) => {
                Logger::error(&format!("Processing failed: {}", e));
                self.diagnostics.record_error(&e);
                
                // Attempt error recovery
                if let Ok(recovered_result) = self.attempt_error_recovery(audio_buffer, &e) {
                    Logger::warn("Error recovery successful");
                    recovered_result
                } else {
                    return Err(e);
                }
            }
        };

        // Record performance metrics
        let processing_time_ms = (Self::get_timestamp() - start_time) as f32 / 1000.0;
        self.diagnostics.record_processing(&result, processing_time_ms);

        Ok(result)
    }

    /// Validate input audio buffer
    fn validate_input(&self, audio_buffer: &[f32]) -> Result<()> {
        if !self.validation_enabled {
            return Ok(());
        }

        if audio_buffer.is_empty() {
            return Err(LiquidAudioError::InvalidInput("Empty audio buffer".to_string()));
        }

        if audio_buffer.len() > self.config.frame_size * 4 {
            return Err(LiquidAudioError::InvalidInput(
                format!("Buffer size {} exceeds maximum {}", audio_buffer.len(), self.config.frame_size * 4)
            ));
        }

        // Check for invalid values
        for (i, &sample) in audio_buffer.iter().enumerate() {
            if !sample.is_finite() {
                return Err(LiquidAudioError::InvalidInput(
                    format!("Non-finite value at index {}: {}", i, sample)
                ));
            }
            
            if sample.abs() > self.max_input_magnitude {
                Logger::warn(&format!("Large input value at index {}: {}", i, sample));
            }
        }

        // Check for silence (all zeros)
        let max_abs = audio_buffer.iter().map(|x| x.abs()).fold(0.0f32, |a, b| a.max(b));
        if max_abs == 0.0 {
            Logger::warn("Input buffer contains only zeros");
        }

        // Check for clipping
        let clipped_count = audio_buffer.iter().filter(|&&x| x.abs() >= 1.0).count();
        if clipped_count > audio_buffer.len() / 10 {
            Logger::warn(&format!("Possible clipping detected: {}/{} samples at maximum", clipped_count, audio_buffer.len()));
        }

        Ok(())
    }

    /// Internal processing with enhanced error handling
    fn process_internal(&mut self, audio_buffer: &[f32]) -> Result<ProcessingResult> {
        // Enhanced dummy processing with validation
        let energy = audio_buffer.iter().map(|x| x * x).sum::<f32>() / audio_buffer.len() as f32;
        
        if !energy.is_finite() {
            return Err(LiquidAudioError::ComputationError("Energy calculation produced non-finite result".to_string()));
        }
        
        let confidence = energy.sqrt().min(1.0);
        
        // Adaptive timestep calculation
        let timestep_ms = if let Some(ref adaptive_config) = self.adaptive_config {
            let complexity = energy.min(1.0);
            adaptive_config.min_timestep_ms + 
            (adaptive_config.max_timestep_ms - adaptive_config.min_timestep_ms) * (1.0 - complexity)
        } else {
            10.0
        };

        // Dynamic power calculation
        let base_power = 0.5;
        let signal_power = energy.sqrt() * 0.8;
        let complexity_power = if let Some(ref adaptive_config) = self.adaptive_config {
            (10.0 / timestep_ms) * adaptive_config.complexity_penalty
        } else {
            0.2
        };
        
        self.power_mw = base_power + signal_power + complexity_power;
        
        let output = vec![confidence, 1.0 - confidence];
        
        // Validate output
        if output.iter().any(|x| !x.is_finite()) {
            return Err(LiquidAudioError::ComputationError("Output contains non-finite values".to_string()));
        }

        Ok(ProcessingResult {
            output,
            confidence,
            timestep_ms,
            power_mw: self.power_mw,
            complexity: energy.sqrt(),
            liquid_energy: energy,
            metadata: Some("generation3_optimized".to_string()),
        })
    }

    /// Attempt to recover from processing errors
    fn attempt_error_recovery(&mut self, audio_buffer: &[f32], error: &LiquidAudioError) -> Result<ProcessingResult> {
        match error {
            LiquidAudioError::ComputationError(_) => {
                // Reset state and try with fallback processing
                Logger::warn("Attempting computational error recovery");
                self.reset_state();
                
                // Simplified fallback processing
                let safe_energy = audio_buffer.iter()
                    .map(|x| if x.is_finite() { x * x } else { 0.0 })
                    .sum::<f32>() / audio_buffer.len() as f32;
                
                let safe_confidence = if safe_energy.is_finite() { safe_energy.sqrt().min(1.0) } else { 0.0 };
                
                Ok(ProcessingResult {
                    output: vec![safe_confidence, 1.0 - safe_confidence],
                    confidence: safe_confidence,
                    timestep_ms: 20.0, // Conservative timestep
                    power_mw: 0.8,     // Conservative power estimate
                    complexity: safe_energy.sqrt(),
                    liquid_energy: safe_energy,
                    metadata: Some("error_recovery".to_string()),
                })
            }
            LiquidAudioError::InvalidInput(_) => {
                // Try processing with cleaned input
                Logger::warn("Attempting input cleaning for error recovery");
                let cleaned_buffer: Vec<f32> = audio_buffer.iter()
                    .map(|&x| if x.is_finite() && x.abs() <= self.max_input_magnitude { 
                        x 
                    } else { 
                        0.0 
                    })
                    .collect();
                
                self.process_internal(&cleaned_buffer)
            }
            _ => {
                // Cannot recover from other error types
                Err(LiquidAudioError::ComputationError("Error recovery failed".to_string()))
            }
        }
    }

    /// Get current timestamp (microseconds)
    fn get_timestamp() -> u64 {
        // Simple counter for demo - real implementation would use proper timer
        static mut COUNTER: u64 = 0;
        unsafe {
            COUNTER += 100; // Simulate 100μs increments
            COUNTER
        }
    }

    pub fn current_power_mw(&self) -> f32 {
        self.power_mw
    }

    pub fn reset_state(&mut self) {
        Logger::info("Resetting LNN state");
        self.diagnostics.reset_stats();
        self.power_mw = 1.0;
        // Reset internal neural state (placeholder)
    }

    pub fn set_adaptive_config(&mut self, config: AdaptiveConfig) {
        Logger::info(&format!("Setting adaptive config: {:.1}-{:.1}ms timestep", 
                             config.min_timestep_ms, config.max_timestep_ms));
        
        // Validate adaptive configuration
        if config.min_timestep_ms >= config.max_timestep_ms {
            Logger::error("Invalid adaptive config: min_timestep >= max_timestep");
            return;
        }
        
        if config.min_timestep_ms <= 0.0 || config.max_timestep_ms > 1000.0 {
            Logger::error("Adaptive timestep values outside reasonable range");
            return;
        }
        
        self.adaptive_config = Some(config);
    }

    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Perform health check and return diagnostics
    pub fn health_check(&mut self) -> Result<HealthReport> {
        self.diagnostics.health_check(&self.config)
    }

    /// Enable or disable input validation
    pub fn set_validation_enabled(&mut self, enabled: bool) {
        Logger::info(&format!("Input validation {}", if enabled { "enabled" } else { "disabled" }));
        self.validation_enabled = enabled;
    }

    /// Set maximum allowed input magnitude
    pub fn set_max_input_magnitude(&mut self, max_magnitude: f32) {
        if max_magnitude <= 0.0 {
            Logger::error("Invalid max input magnitude - must be positive");
            return;
        }
        
        Logger::info(&format!("Max input magnitude set to {:.2}", max_magnitude));
        self.max_input_magnitude = max_magnitude;
    }

    /// Get performance statistics
    pub fn get_performance_summary(&mut self) -> String {
        match self.health_check() {
            Ok(report) => self.diagnostics.format_health_summary(&report),
            Err(_) => "Health check failed".to_string(),
        }
    }

    /// Get diagnostic recommendations
    pub fn get_recommendations(&mut self) -> Vec<String> {
        match self.health_check() {
            Ok(report) => self.diagnostics.get_recommendations(&report),
            Err(_) => vec!["Unable to generate recommendations - health check failed".to_string()],
        }
    }
}

// Implement AudioModel trait for LNN
impl AudioModel for LNN {
    fn process_audio(&mut self, audio_buffer: &[f32]) -> Result<ProcessingResult> {
        self.process(audio_buffer)
    }
    
    fn current_power_mw(&self) -> f32 {
        self.current_power_mw()
    }
    
    fn reset(&mut self) {
        self.reset_state();
    }
    
    fn model_type(&self) -> &str {
        &self.config.model_type
    }
    
    fn is_ready(&self) -> bool {
        true // LNN is always ready after creation
    }
}

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Core modules
pub mod diagnostics;
pub mod models;

// Generation 3: Advanced scaling and optimization modules
pub mod cache;
pub mod optimization;
pub mod concurrent;
pub mod scaling;
pub mod pretrained;
pub mod deployment;
pub mod benchmark;

// Global-first modules for international deployment
pub mod i18n;
pub mod compliance;
pub mod regions;

pub use diagnostics::{DiagnosticsCollector, HealthReport, HealthStatus, Logger};
pub use models::{AudioModel, ModelFactory};
pub use i18n::{Language, MessageKey, I18nManager, set_global_language, t, t_error};
pub use compliance::{PrivacyFramework, ComplianceConfig, PrivacyManager};
pub use regions::{Region, RegionalConfig, RegionalManager, PerformanceProfile};

// Panic handler for no_std mode
#[cfg(not(feature = "std"))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}