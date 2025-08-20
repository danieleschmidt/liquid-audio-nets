//! Model trait definitions and implementations for Liquid Audio Networks

use crate::{Result, LiquidAudioError, ProcessingResult, ModelConfig, AdaptiveConfig, LNN};

#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, vec::Vec, string::String};

/// Core audio processing model trait
pub trait AudioModel {
    /// Process audio buffer and return results
    fn process_audio(&mut self, audio_buffer: &[f32]) -> Result<ProcessingResult>;
    
    /// Get current power consumption in milliwatts
    fn current_power_mw(&self) -> f32;
    
    /// Reset model state
    fn reset(&mut self);
    
    /// Get model type identifier
    fn model_type(&self) -> &str;
    
    /// Check if model is ready for processing
    fn is_ready(&self) -> bool;
}

/// Factory for creating audio models
pub struct ModelFactory;

impl ModelFactory {
    /// Create a new LNN model instance
    pub fn create_lnn(config: ModelConfig) -> Result<LNN> {
        LNN::new(config)
    }
    
    /// Create LNN with adaptive timestep control
    pub fn create_adaptive_lnn(config: ModelConfig, adaptive_config: AdaptiveConfig) -> Result<LNN> {
        let mut lnn = LNN::new(config)?;
        lnn.set_adaptive_config(adaptive_config);
        Ok(lnn)
    }
    
    /// Load model from file path
    #[cfg(feature = "std")]
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<LNN> {
        LNN::load_from_file(path.as_ref().as_os_str())
    }
    
    /// Create model from model type string
    pub fn create_by_type(model_type: &str, config: ModelConfig) -> Result<Box<dyn AudioModel>> {
        match model_type.to_lowercase().as_str() {
            "lnn" | "liquid" => Ok(Box::new(Self::create_lnn(config)?)),
            "adaptive_lnn" | "adaptive" => {
                let adaptive_config = AdaptiveConfig::default();
                Ok(Box::new(Self::create_adaptive_lnn(config, adaptive_config)?))
            }
            _ => Err(LiquidAudioError::ModelError(
                format!("Unknown model type: {}", model_type)
            ))
        }
    }
    
    /// Create model for specific use case
    pub fn create_for_use_case(use_case: &str, config: Option<ModelConfig>) -> Result<Box<dyn AudioModel>> {
        let model_config = config.unwrap_or_default();
        
        match use_case.to_lowercase().as_str() {
            "keyword_spotting" | "wake_word" => {
                let config = ModelConfig {
                    output_dim: 10,  // Common keywords
                    model_type: "keyword_spotting".to_string(),
                    ..model_config
                };
                Self::create_by_type("adaptive_lnn", config)
            }
            "voice_activity" | "vad" => {
                let config = ModelConfig {
                    output_dim: 2,  // Speech/no-speech
                    model_type: "voice_activity".to_string(),
                    ..model_config
                };
                Self::create_by_type("lnn", config)
            }
            "speech_enhancement" => {
                let config = ModelConfig {
                    output_dim: model_config.input_dim,  // Same as input
                    model_type: "speech_enhancement".to_string(),
                    ..model_config
                };
                Self::create_by_type("adaptive_lnn", config)
            }
            _ => Err(LiquidAudioError::ConfigError(
                format!("Unknown use case: {}", use_case)
            ))
        }
    }
    
    /// Get available model types
    pub fn available_models() -> &'static [&'static str] {
        &["lnn", "liquid", "adaptive_lnn", "adaptive"]
    }
    
    /// Get available use cases
    pub fn available_use_cases() -> &'static [&'static str] {
        &["keyword_spotting", "wake_word", "voice_activity", "vad", "speech_enhancement"]
    }
}

/// AudioModel implementation for LNN
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
        "lnn"
    }
    
    fn is_ready(&self) -> bool {
        // LNN is ready if it has valid configuration
        let config = self.config();
        config.hidden_dim > 0 && config.input_dim > 0
    }
}