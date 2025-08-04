//! # Liquid Audio Nets
//! 
//! Edge-efficient Liquid Neural Network models for always-on audio sensing.
//! 
//! This crate provides ultra-low-power neural network implementations optimized
//! for ARM Cortex-M microcontrollers and edge devices.

#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String};

pub mod core;
pub mod audio;
pub mod adaptive;
pub mod models;

#[cfg(feature = "python")]
pub mod python;

#[cfg(feature = "embedded")]
pub mod embedded;

// Re-export main types
pub use core::{LNN, AdaptiveConfig, ProcessingResult, ModelConfig};
pub use audio::{AudioProcessor, FeatureExtractor, AudioFormat};
pub use adaptive::{TimestepController, ComplexityEstimator};
// pub use models::{KeywordSpotter, VoiceActivityDetector};

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Result type used throughout the library  
pub type Result<T> = ::core::result::Result<T, LiquidAudioError>;

/// Main error type for the library
#[derive(Debug, Clone)]
pub enum LiquidAudioError {
    /// Model loading/parsing errors
    ModelError(String),
    /// Audio processing errors
    AudioError(String),
    /// Configuration errors
    ConfigError(String),
    /// Runtime computation errors
    ComputationError(String),
    /// I/O errors (file operations, etc.)
    IoError(String),
    /// Invalid input parameters
    InvalidInput(String),
    /// Security errors
    SecurityError(String),
}

impl ::core::fmt::Display for LiquidAudioError {
    fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
        match self {
            LiquidAudioError::ModelError(msg) => write!(f, "Model error: {}", msg),
            LiquidAudioError::AudioError(msg) => write!(f, "Audio error: {}", msg),
            LiquidAudioError::ConfigError(msg) => write!(f, "Config error: {}", msg),
            LiquidAudioError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
            LiquidAudioError::IoError(msg) => write!(f, "I/O error: {}", msg),
            LiquidAudioError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            LiquidAudioError::SecurityError(msg) => write!(f, "Security error: {}", msg),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LiquidAudioError {}