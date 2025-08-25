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
use alloc::{vec::Vec, string::String, vec, collections::BTreeMap};


#[cfg(not(feature = "std"))]
use alloc::string::ToString;

use ::core::fmt;

/// Core modules for LNN implementation
pub mod core;
pub mod audio;
pub mod adaptive;
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

// Next-Generation Features (Beyond Generation 3)
pub mod self_optimization;
pub mod multimodal;
pub mod adaptive_learning;
pub mod hardware_acceleration;
pub mod quantum_classical;

// Novel Research Contributions
pub mod adaptive_meta_learning;
pub mod quantum_optimizer;

// Global-first modules for international deployment
pub mod i18n;
pub mod compliance;
pub mod regions;

// Re-export core types
pub use core::{LNN, ModelConfig, AdaptiveConfig, PowerConfig, LiquidState, ProcessingResult, ODESolver, EulerSolver, HeunSolver};
pub use audio::{AudioProcessor, FeatureExtractor, AudioFormat, SampleFormat};
pub use adaptive::{TimestepController, ComplexityEstimator};
pub use diagnostics::{DiagnosticsCollector, HealthReport, HealthStatus, Logger};
pub use models::{AudioModel, ModelFactory};
pub use scaling::{
    LoadBalancer, AutoScaler, ScalingSystem, AdvancedAutoScaler,
    LoadBalancingStrategy, ScalingConfig, AdvancedScalingConfig,
    ProcessingNode, ScalingMetrics, ScalingEvent, ScalingAction
};
pub use i18n::{Language, MessageKey, I18nManager, set_global_language, t, t_error};
pub use compliance::{PrivacyFramework, ComplianceConfig, PrivacyManager};
pub use regions::{Region, RegionalConfig, RegionalManager, PerformanceProfile};

/// Enhanced error type with security and validation context
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
    ValidationError { field: String, reason: String, severity: ErrorSeverity },
    SecurityViolation { action: String, context: String, risk_level: RiskLevel },
    RateLimitExceeded { resource: String, limit: u64, current: u64 },
    AuthenticationFailed { reason: String },
    PermissionDenied { action: String, required_level: String },
    DataCorruption { location: String, checksum_expected: String, checksum_actual: String },
    SystemOverload { component: String, load_percent: f32 },
    DependencyFailure { dependency: String, error: String },
}

/// Error severity levels for validation and security
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Security risk levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for LiquidAudioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
            LiquidAudioError::ValidationError { field, reason, severity } => {
                write!(f, "Validation error [{:?}] in '{}': {}", severity, field, reason)
            },
            LiquidAudioError::SecurityViolation { action, context, risk_level } => {
                write!(f, "Security violation [{:?}] - Action '{}' in context '{}'", risk_level, action, context)
            },
            LiquidAudioError::RateLimitExceeded { resource, limit, current } => {
                write!(f, "Rate limit exceeded for '{}': {}/{}", resource, current, limit)
            },
            LiquidAudioError::AuthenticationFailed { reason } => {
                write!(f, "Authentication failed: {}", reason)
            },
            LiquidAudioError::PermissionDenied { action, required_level } => {
                write!(f, "Permission denied for '{}' (requires '{}')", action, required_level)
            },
            LiquidAudioError::DataCorruption { location, checksum_expected, checksum_actual } => {
                write!(f, "Data corruption at '{}': expected checksum {}, got {}", location, checksum_expected, checksum_actual)
            },
            LiquidAudioError::SystemOverload { component, load_percent } => {
                write!(f, "System overload in '{}': {:.1}% load", component, load_percent)
            },
            LiquidAudioError::DependencyFailure { dependency, error } => {
                write!(f, "Dependency '{}' failed: {}", dependency, error)
            },
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for LiquidAudioError {}

#[cfg(feature = "std")]
pub type Result<T> = std::result::Result<T, LiquidAudioError>;

#[cfg(not(feature = "std"))]
pub type Result<T> = core::result::Result<T, LiquidAudioError>;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Security context for authentication and authorization
#[derive(Debug, Clone)]
pub struct SecurityContext {
    pub session_id: String,
    pub permissions: Vec<String>,
    pub rate_limits: Vec<RateLimit>,
    pub security_level: SecurityLevel,
    pub last_auth_time: u64,
    pub failed_attempts: u32,
}

/// Security levels
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum SecurityLevel {
    Public,
    Authenticated,
    Privileged,
    Admin,
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimit {
    pub resource: String,
    pub max_requests: u64,
    pub window_ms: u64,
    pub current_count: u64,
    pub window_start: u64,
}

impl Default for SecurityContext {
    fn default() -> Self {
        Self {
            session_id: "default_session".to_string(),
            permissions: vec!["basic_processing".to_string()],
            rate_limits: vec![],
            security_level: SecurityLevel::Public,
            last_auth_time: 0,
            failed_attempts: 0,
        }
    }
}

// Additional common exports

// Panic handler for no_std mode
#[cfg(not(feature = "std"))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}