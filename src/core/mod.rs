//! Core Liquid Neural Network implementation
//! 
//! This module contains the main LNN implementation with continuous-time dynamics,
//! adaptive timestep control, and ultra-low-power optimizations.

pub mod lnn;
pub mod config;
pub mod ode_solver;
pub mod state;
pub mod result;

pub use self::lnn::LNN;
pub use self::config::{AdaptiveConfig, ModelConfig, PowerConfig, ComplexityMetric};
pub use self::ode_solver::{ODESolver, EulerSolver, HeunSolver};
pub use self::state::{LiquidState, NetworkState};
pub use self::result::ProcessingResult;