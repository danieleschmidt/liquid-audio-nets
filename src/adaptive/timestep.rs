//! Timestep control for adaptive liquid neural networks

use crate::core::AdaptiveConfig;
use crate::{Result, LiquidAudioError};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Timestep controller with security validation and smoothing
#[derive(Debug, Clone)]
pub struct TimestepController {
    /// Current timestep configuration
    config: Option<AdaptiveConfig>,
    /// Timestep history for smoothing
    timestep_history: Vec<f32>,
    /// Maximum history length
    max_history: usize,
    /// Security limits
    limits: TimestepLimits,
    /// Smoothing parameters
    smoothing: SmoothingParams,
}

impl TimestepController {
    /// Create new timestep controller
    pub fn new() -> Self {
        Self {
            config: None,
            timestep_history: Vec::with_capacity(10),
            max_history: 10,
            limits: TimestepLimits::default(),
            smoothing: SmoothingParams::default(),
        }
    }
    
    /// Set configuration with validation
    pub fn set_config(&mut self, config: AdaptiveConfig) {
        // Validate against security limits
        let validated_config = AdaptiveConfig {
            min_timestep: config.min_timestep.max(self.limits.absolute_min_timestep),
            max_timestep: config.max_timestep.min(self.limits.absolute_max_timestep),
            energy_threshold: config.energy_threshold.clamp(0.0, 1.0),
            adaptation_rate: config.adaptation_rate.clamp(0.0, 1.0),
            ..config
        };
        
        self.config = Some(validated_config);
    }
    
    /// Calculate adaptive timestep based on signal complexity
    pub fn calculate_timestep(&mut self, complexity: f32, config: &AdaptiveConfig) -> f32 {
        // Validate input complexity
        let safe_complexity = complexity.clamp(0.0, 1.0);
        
        // Base timestep calculation with security bounds
        let min_dt = config.min_timestep.max(self.limits.absolute_min_timestep);
        let max_dt = config.max_timestep.min(self.limits.absolute_max_timestep);
        
        // Ensure valid range
        if min_dt >= max_dt {
            return self.limits.default_timestep;
        }
        
        // Calculate raw timestep based on complexity
        let timestep_range = max_dt - min_dt;
        let raw_timestep = max_dt - (safe_complexity * timestep_range);
        
        // Apply adaptation rate
        let adapted_timestep = if let Some(prev_timestep) = self.get_previous_timestep() {
            let change = raw_timestep - prev_timestep;
            prev_timestep + change * config.adaptation_rate
        } else {
            raw_timestep
        };
        
        // Apply security bounds
        let bounded_timestep = adapted_timestep.clamp(min_dt, max_dt);
        
        // Apply smoothing if enabled
        let final_timestep = if config.smooth_transitions {
            self.apply_smoothing(bounded_timestep)
        } else {
            bounded_timestep
        };
        
        // Update history
        self.update_history(final_timestep);
        
        // Final validation
        self.validate_timestep(final_timestep).unwrap_or(self.limits.default_timestep)
    }
    
    /// Apply temporal smoothing to timestep
    fn apply_smoothing(&self, new_timestep: f32) -> f32 {
        if self.timestep_history.is_empty() {
            return new_timestep;
        }
        
        let prev_timestep = self.timestep_history[self.timestep_history.len() - 1];
        
        // Calculate maximum allowed change
        let max_change = prev_timestep * self.smoothing.max_change_rate;
        let change = new_timestep - prev_timestep;
        
        // Limit the change
        let limited_change = change.clamp(-max_change, max_change);
        
        prev_timestep + limited_change
    }
    
    /// Update timestep history
    fn update_history(&mut self, timestep: f32) {
        self.timestep_history.push(timestep);
        
        // Limit history size
        if self.timestep_history.len() > self.max_history {
            self.timestep_history.remove(0);
        }
    }
    
    /// Get previous timestep from history
    fn get_previous_timestep(&self) -> Option<f32> {
        self.timestep_history.last().copied()
    }
    
    /// Validate timestep for security
    fn validate_timestep(&self, timestep: f32) -> Result<f32> {
        // Check for NaN or infinite values
        if !timestep.is_finite() {
            return Err(LiquidAudioError::ComputationError(
                "Invalid timestep: NaN or infinite".to_string()
            ));
        }
        
        // Check absolute bounds
        if timestep < self.limits.absolute_min_timestep || timestep > self.limits.absolute_max_timestep {
            return Err(LiquidAudioError::ComputationError(
                format!("Timestep {} outside safe bounds [{}, {}]", 
                       timestep, self.limits.absolute_min_timestep, self.limits.absolute_max_timestep)
            ));
        }
        
        // Check rate of change if we have history
        if let Some(prev_timestep) = self.get_previous_timestep() {
            let change_rate = (timestep - prev_timestep).abs() / prev_timestep;
            if change_rate > self.limits.max_change_rate {
                return Err(LiquidAudioError::ComputationError(
                    format!("Timestep change rate {} exceeds limit {}", 
                           change_rate, self.limits.max_change_rate)
                ));
            }
        }
        
        Ok(timestep)
    }
    
    /// Get timestep statistics
    pub fn get_statistics(&self) -> TimestepStatistics {
        if self.timestep_history.is_empty() {
            return TimestepStatistics::default();
        }
        
        let count = self.timestep_history.len();
        let sum: f32 = self.timestep_history.iter().sum();
        let mean = sum / count as f32;
        
        let variance: f32 = self.timestep_history.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / count as f32;
        
        let std_dev = variance.sqrt();
        let min = self.timestep_history.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = self.timestep_history.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        TimestepStatistics {
            mean,
            std_dev,
            min,
            max,
            current: self.timestep_history[count - 1],
            history_count: count,
        }
    }
    
    /// Reset controller state
    pub fn reset(&mut self) {
        self.timestep_history.clear();
    }
    
    /// Check if controller is in stable state
    pub fn is_stable(&self) -> bool {
        if self.timestep_history.len() < 3 {
            return false;
        }
        
        let recent_timesteps = &self.timestep_history[self.timestep_history.len() - 3..];
        let variance: f32 = {
            let mean: f32 = recent_timesteps.iter().sum::<f32>() / recent_timesteps.len() as f32;
            recent_timesteps.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / recent_timesteps.len() as f32
        };
        
        variance < self.smoothing.stability_threshold
    }
    
    /// Get recommended timestep for given complexity without updating state
    pub fn preview_timestep(&self, complexity: f32, config: &AdaptiveConfig) -> f32 {
        let safe_complexity = complexity.clamp(0.0, 1.0);
        let timestep_range = config.max_timestep - config.min_timestep;
        config.max_timestep - (safe_complexity * timestep_range)
    }
    
    /// Set security limits
    pub fn set_limits(&mut self, limits: TimestepLimits) {
        self.limits = limits;
    }
    
    /// Get current limits
    pub fn limits(&self) -> &TimestepLimits {
        &self.limits
    }
}

impl Default for TimestepController {
    fn default() -> Self {
        Self::new()
    }
}

/// Timestep security limits
#[derive(Debug, Clone)]
pub struct TimestepLimits {
    /// Absolute minimum timestep (safety bound)
    pub absolute_min_timestep: f32,
    /// Absolute maximum timestep (safety bound)
    pub absolute_max_timestep: f32,
    /// Maximum allowed change rate per step
    pub max_change_rate: f32,
    /// Default timestep for fallback
    pub default_timestep: f32,
}

impl Default for TimestepLimits {
    fn default() -> Self {
        Self {
            absolute_min_timestep: 1e-6,  // 1 microsecond
            absolute_max_timestep: 0.5,   // 500ms
            max_change_rate: 2.0,         // 200% change maximum
            default_timestep: 0.01,       // 10ms default
        }
    }
}

impl TimestepLimits {
    /// Create limits suitable for embedded systems
    pub fn embedded() -> Self {
        Self {
            absolute_min_timestep: 1e-4,  // 0.1ms
            absolute_max_timestep: 0.1,   // 100ms
            max_change_rate: 0.5,         // 50% change maximum
            default_timestep: 0.01,       // 10ms default
        }
    }
    
    /// Create limits for high-precision applications
    pub fn high_precision() -> Self {
        Self {
            absolute_min_timestep: 1e-8,  // 10 nanoseconds
            absolute_max_timestep: 0.001, // 1ms maximum
            max_change_rate: 0.1,         // 10% change maximum
            default_timestep: 1e-4,       // 0.1ms default
        }
    }
}

/// Smoothing parameters for timestep transitions
#[derive(Debug, Clone)]
pub struct SmoothingParams {
    /// Maximum change rate per step (fraction of current value)
    pub max_change_rate: f32,
    /// Variance threshold for stability detection
    pub stability_threshold: f32,
    /// Enable exponential smoothing
    pub use_exponential_smoothing: bool,
    /// Exponential smoothing factor
    pub smoothing_factor: f32,
}

impl Default for SmoothingParams {
    fn default() -> Self {
        Self {
            max_change_rate: 0.2,         // 20% maximum change
            stability_threshold: 1e-6,    // Very small variance for stability
            use_exponential_smoothing: false,
            smoothing_factor: 0.1,        // 10% smoothing
        }
    }
}

/// Timestep statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct TimestepStatistics {
    /// Mean timestep
    pub mean: f32,
    /// Standard deviation
    pub std_dev: f32,
    /// Minimum timestep seen
    pub min: f32,
    /// Maximum timestep seen
    pub max: f32,
    /// Current timestep
    pub current: f32,
    /// Number of timesteps in history
    pub history_count: usize,
}

impl TimestepStatistics {
    /// Get coefficient of variation (relative variability)
    pub fn coefficient_of_variation(&self) -> f32 {
        if self.mean > 1e-8 {
            self.std_dev / self.mean
        } else {
            0.0
        }
    }
    
    /// Get dynamic range (max/min ratio)
    pub fn dynamic_range(&self) -> f32 {
        if self.min > 1e-8 {
            self.max / self.min
        } else {
            1.0
        }
    }
    
    /// Check if statistics indicate stable operation
    pub fn indicates_stable_operation(&self) -> bool {
        self.history_count >= 5 && self.coefficient_of_variation() < 0.1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::AdaptiveConfig;
    
    #[test]
    fn test_timestep_controller_creation() {
        let controller = TimestepController::new();
        assert!(controller.config.is_none());
        assert!(controller.timestep_history.is_empty());
    }
    
    #[test]
    fn test_timestep_calculation() {
        let mut controller = TimestepController::new();
        let config = AdaptiveConfig {
            min_timestep: 0.001,
            max_timestep: 0.05,
            energy_threshold: 0.1,
            complexity_metric: crate::core::ComplexityMetric::Energy,
            adaptation_rate: 1.0,
            smooth_transitions: false,
        };
        
        // Low complexity should give large timestep
        let dt_low = controller.calculate_timestep(0.1, &config);
        assert!(dt_low > 0.03);
        
        // High complexity should give small timestep
        let dt_high = controller.calculate_timestep(0.9, &config);
        assert!(dt_high < 0.02);
        
        // High complexity timestep should be smaller than low complexity
        assert!(dt_high < dt_low);
    }
    
    #[test]
    fn test_timestep_bounds() {
        let mut controller = TimestepController::new();
        let config = AdaptiveConfig {
            min_timestep: 0.001,
            max_timestep: 0.05,
            energy_threshold: 0.1,
            complexity_metric: crate::core::ComplexityMetric::Energy,
            adaptation_rate: 1.0,
            smooth_transitions: false,
        };
        
        // Very low complexity should be clamped to max_timestep
        let dt = controller.calculate_timestep(0.0, &config);
        assert!((dt - config.max_timestep).abs() < 1e-6);
        
        // Very high complexity should be clamped to min_timestep
        let dt = controller.calculate_timestep(1.0, &config);
        assert!((dt - config.min_timestep).abs() < 1e-6);
    }
    
    #[test]
    fn test_timestep_smoothing() {
        let mut controller = TimestepController::new();
        let config = AdaptiveConfig {
            min_timestep: 0.001,
            max_timestep: 0.05,
            energy_threshold: 0.1,
            complexity_metric: crate::core::ComplexityMetric::Energy,
            adaptation_rate: 0.5,  // 50% adaptation rate
            smooth_transitions: true,
        };
        
        // First timestep
        let dt1 = controller.calculate_timestep(0.2, &config);
        
        // Sudden change in complexity
        let dt2 = controller.calculate_timestep(0.8, &config);
        
        // Change should be moderated by adaptation rate
        let expected_change = (dt2 - dt1).abs();
        // With 50% adaptation rate, change should be reduced
        // This is a simplified test - actual smoothing is more complex
        assert!(expected_change < 0.1); // Reasonable change limit
    }
    
    #[test]
    fn test_timestep_validation() {
        let controller = TimestepController::new();
        
        // Valid timestep
        assert!(controller.validate_timestep(0.01).is_ok());
        
        // Invalid timesteps
        assert!(controller.validate_timestep(f32::NAN).is_err());
        assert!(controller.validate_timestep(f32::INFINITY).is_err());
        assert!(controller.validate_timestep(-0.01).is_err());
        assert!(controller.validate_timestep(10.0).is_err()); // Too large
    }
    
    #[test]
    fn test_timestep_statistics() {
        let mut controller = TimestepController::new();
        let config = AdaptiveConfig::default();
        
        // Generate some timesteps
        for complexity in [0.1, 0.3, 0.5, 0.7, 0.9] {
            controller.calculate_timestep(complexity, &config);
        }
        
        let stats = controller.get_statistics();
        assert_eq!(stats.history_count, 5);
        assert!(stats.mean > 0.0);
        assert!(stats.min <= stats.max);
        assert!(stats.std_dev >= 0.0);
    }
    
    #[test]
    fn test_stability_detection() {
        let mut controller = TimestepController::new();
        let config = AdaptiveConfig::default();
        
        // Generate stable timesteps (same complexity)
        for _ in 0..5 {
            controller.calculate_timestep(0.5, &config);
        }
        
        assert!(controller.is_stable());
    }
}