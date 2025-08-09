//! Model trait definitions and implementations for Liquid Audio Networks

use crate::{Result, LiquidAudioError, ProcessingResult};

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
    pub fn create_lnn(config: crate::ModelConfig) -> Result<crate::LNN> {
        crate::LNN::new(config)
    }
    
    /// Load model from file path
    #[cfg(feature = "std")]
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<crate::LNN> {
        crate::LNN::load_from_file(path.as_ref().as_os_str())
    }
    
    /// Create model from model type string
    pub fn create_by_type(model_type: &str, config: crate::ModelConfig) -> Result<Box<dyn AudioModel>> {
        match model_type {
            "lnn" | "liquid" => Ok(Box::new(Self::create_lnn(config)?)),
            _ => Err(LiquidAudioError::ModelError(
                format!("Unknown model type: {}", model_type)
            ))
        }
    }
}