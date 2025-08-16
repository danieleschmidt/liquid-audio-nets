//! Adaptive quality control system for real-time audio processing
//!
//! Provides intelligent quality adaptation to maintain real-time performance
//! while preserving audio quality as much as possible.

use crate::{Result, LiquidAudioError};
use super::{PerformanceProfile, get_current_time_us};

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String, collections::BTreeMap};

#[cfg(feature = "std")]
use std::collections::BTreeMap;

/// Adaptive quality controller for real-time processing
#[derive(Debug)]
pub struct AdaptiveQualityController {
    /// Maximum allowable quality degradation
    max_degradation: f32,
    /// Current quality factor (0.0-1.0)
    current_quality: f32,
    /// Target quality factor
    target_quality: f32,
    /// Quality adaptation rate
    adaptation_rate: f32,
    /// Quality history for smoothing
    quality_history: Vec<QualityMeasurement>,
    /// Maximum history length
    max_history: usize,
    /// Quality degradation strategies
    degradation_strategies: Vec<QualityStrategy>,
    /// Current active strategy
    active_strategy: Option<usize>,
    /// Quality recovery controller
    recovery_controller: QualityRecoveryController,
    /// Quality metrics
    metrics: QualityMetrics,
    /// Emergency mode state
    emergency_mode: bool,
}

/// Quality measurement point
#[derive(Debug, Clone)]
struct QualityMeasurement {
    timestamp_us: u64,
    quality_factor: f32,
    performance_health: f32,
    deadline_pressure: f32,
    strategy_used: Option<String>,
    recovery_rate: f32,
}

/// Quality degradation strategy
#[derive(Debug, Clone)]
pub struct QualityStrategy {
    /// Strategy name
    pub name: String,
    /// Quality reduction amount (0.0-1.0)
    pub quality_reduction: f32,
    /// Performance improvement estimate
    pub performance_gain: f32,
    /// Recovery time estimate (microseconds)
    pub recovery_time_us: u64,
    /// Applicability conditions
    pub conditions: QualityConditions,
    /// Strategy priority (higher = more preferred)
    pub priority: u32,
}

/// Conditions for applying quality strategy
#[derive(Debug, Clone)]
pub struct QualityConditions {
    /// Minimum deadline pressure to activate
    pub min_deadline_pressure: f32,
    /// Maximum acceptable quality after application
    pub min_acceptable_quality: f32,
    /// System load threshold
    pub load_threshold: f32,
    /// Consecutive frames before activation
    pub activation_delay: u32,
}

/// Quality recovery controller
#[derive(Debug)]
struct QualityRecoveryController {
    /// Recovery rate (quality increase per frame)
    recovery_rate: f32,
    /// Recovery delay after degradation
    recovery_delay_frames: u32,
    /// Current recovery delay countdown
    current_delay: u32,
    /// Gradual recovery enabled
    gradual_recovery: bool,
    /// Recovery smoothing window
    smoothing_window: usize,
}

/// Quality control metrics
#[derive(Debug, Clone)]
struct QualityMetrics {
    /// Total frames processed
    total_frames: u64,
    /// Frames with quality degradation
    degraded_frames: u64,
    /// Average quality maintained
    avg_quality: f32,
    /// Quality preservation rate
    preservation_rate: f32,
    /// Recovery success rate
    recovery_success_rate: f32,
    /// Strategy effectiveness scores
    strategy_effectiveness: BTreeMap<String, f32>,
}

impl AdaptiveQualityController {
    /// Create new adaptive quality controller
    pub fn new(max_degradation: f32, adaptation_rate: f32) -> Self {
        let mut controller = Self {
            max_degradation: max_degradation.clamp(0.0, 1.0),
            current_quality: 1.0,
            target_quality: 1.0,
            adaptation_rate: adaptation_rate.clamp(0.01, 1.0),
            quality_history: Vec::with_capacity(100),
            max_history: 100,
            degradation_strategies: Vec::new(),
            active_strategy: None,
            recovery_controller: QualityRecoveryController::new(),
            metrics: QualityMetrics::new(),
            emergency_mode: false,
        };

        // Initialize default strategies
        controller.initialize_default_strategies();
        controller
    }

    /// Update quality control based on current performance
    pub fn update(&mut self, deadline_pressure: f32, performance_health: f32) {
        let measurement = QualityMeasurement {
            timestamp_us: get_current_time_us(),
            quality_factor: self.current_quality,
            performance_health,
            deadline_pressure,
            strategy_used: self.active_strategy.map(|i| self.degradation_strategies[i].name.clone()),
            recovery_rate: self.recovery_controller.recovery_rate,
        };

        // Record measurement
        self.quality_history.push(measurement);
        if self.quality_history.len() > self.max_history {
            self.quality_history.remove(0);
        }

        // Determine target quality based on system state
        self.determine_target_quality(deadline_pressure, performance_health);

        // Apply quality adaptation
        self.apply_quality_adaptation();

        // Update recovery controller
        self.recovery_controller.update(self.current_quality, self.target_quality);

        // Update metrics
        self.update_metrics(deadline_pressure, performance_health);
    }

    /// Get current quality factor
    pub fn get_quality_factor(&self) -> f32 {
        self.current_quality
    }

    /// Get smoothed quality factor
    pub fn get_smoothed_quality_factor(&self) -> f32 {
        if self.quality_history.len() < 3 {
            return self.current_quality;
        }

        let recent_qualities: Vec<f32> = self.quality_history
            .iter()
            .rev()
            .take(5)
            .map(|m| m.quality_factor)
            .collect();

        recent_qualities.iter().sum::<f32>() / recent_qualities.len() as f32
    }

    /// Check if quality is currently degraded
    pub fn is_degraded(&self) -> bool {
        self.current_quality < 0.98
    }

    /// Force emergency mode (maximum degradation)
    pub fn force_emergency_mode(&mut self) {
        self.emergency_mode = true;
        self.target_quality = 1.0 - self.max_degradation;
        self.current_quality = self.target_quality;
    }

    /// Exit emergency mode
    pub fn exit_emergency_mode(&mut self) {
        self.emergency_mode = false;
        self.recovery_controller.start_recovery();
    }

    /// Get quality recommendations for different processing components
    pub fn get_component_recommendations(&self) -> QualityRecommendations {
        QualityRecommendations {
            input_decimation: self.calculate_input_decimation(),
            feature_reduction: self.calculate_feature_reduction(),
            model_complexity: self.calculate_model_complexity(),
            output_precision: self.calculate_output_precision(),
            buffer_optimization: self.calculate_buffer_optimization(),
        }
    }

    /// Get quality control statistics
    pub fn get_statistics(&self) -> QualityControlStatistics {
        QualityControlStatistics {
            current_quality: self.current_quality,
            target_quality: self.target_quality,
            avg_quality: self.metrics.avg_quality,
            preservation_rate: self.metrics.preservation_rate,
            degraded_frames: self.metrics.degraded_frames,
            total_frames: self.metrics.total_frames,
            active_strategy: self.active_strategy.map(|i| self.degradation_strategies[i].name.clone()),
            emergency_mode: self.emergency_mode,
            recovery_progress: self.recovery_controller.get_progress(),
        }
    }

    /// Add custom quality strategy
    pub fn add_strategy(&mut self, strategy: QualityStrategy) -> Result<()> {
        // Validate strategy
        if strategy.quality_reduction < 0.0 || strategy.quality_reduction > 1.0 {
            return Err(LiquidAudioError::ConfigError(
                "Quality reduction must be between 0.0 and 1.0".to_string()
            ));
        }

        if strategy.performance_gain <= 0.0 {
            return Err(LiquidAudioError::ConfigError(
                "Performance gain must be positive".to_string()
            ));
        }

        // Insert strategy in priority order
        let insert_pos = self.degradation_strategies
            .iter()
            .position(|s| s.priority < strategy.priority)
            .unwrap_or(self.degradation_strategies.len());

        self.degradation_strategies.insert(insert_pos, strategy);
        Ok(())
    }

    /// Reset quality controller
    pub fn reset(&mut self) {
        self.current_quality = 1.0;
        self.target_quality = 1.0;
        self.quality_history.clear();
        self.active_strategy = None;
        self.recovery_controller.reset();
        self.metrics = QualityMetrics::new();
        self.emergency_mode = false;
    }

    // Private implementation methods

    fn initialize_default_strategies(&mut self) {
        // Minimal quality reduction - reduce precision
        let minimal_strategy = QualityStrategy {
            name: "minimal_precision_reduction".to_string(),
            quality_reduction: 0.05,
            performance_gain: 0.15,
            recovery_time_us: 50000, // 50ms
            conditions: QualityConditions {
                min_deadline_pressure: 0.7,
                min_acceptable_quality: 0.9,
                load_threshold: 0.6,
                activation_delay: 2,
            },
            priority: 100,
        };

        // Moderate quality reduction - reduce features
        let moderate_strategy = QualityStrategy {
            name: "feature_reduction".to_string(),
            quality_reduction: 0.15,
            performance_gain: 0.3,
            recovery_time_us: 100000, // 100ms
            conditions: QualityConditions {
                min_deadline_pressure: 0.8,
                min_acceptable_quality: 0.8,
                load_threshold: 0.75,
                activation_delay: 3,
            },
            priority: 80,
        };

        // Aggressive quality reduction - reduce model complexity
        let aggressive_strategy = QualityStrategy {
            name: "model_complexity_reduction".to_string(),
            quality_reduction: 0.3,
            performance_gain: 0.5,
            recovery_time_us: 200000, // 200ms
            conditions: QualityConditions {
                min_deadline_pressure: 0.9,
                min_acceptable_quality: 0.6,
                load_threshold: 0.85,
                activation_delay: 1,
            },
            priority: 60,
        };

        // Emergency strategy - maximum reduction
        let emergency_strategy = QualityStrategy {
            name: "emergency_degradation".to_string(),
            quality_reduction: self.max_degradation,
            performance_gain: 0.8,
            recovery_time_us: 500000, // 500ms
            conditions: QualityConditions {
                min_deadline_pressure: 0.95,
                min_acceptable_quality: 0.3,
                load_threshold: 0.95,
                activation_delay: 0,
            },
            priority: 10,
        };

        self.degradation_strategies.push(minimal_strategy);
        self.degradation_strategies.push(moderate_strategy);
        self.degradation_strategies.push(aggressive_strategy);
        self.degradation_strategies.push(emergency_strategy);
    }

    fn determine_target_quality(&mut self, deadline_pressure: f32, performance_health: f32) {
        if self.emergency_mode {
            return; // Target already set in emergency mode
        }

        // Find appropriate strategy
        let mut selected_strategy = None;
        for (i, strategy) in self.degradation_strategies.iter().enumerate() {
            if self.should_apply_strategy(strategy, deadline_pressure, performance_health) {
                selected_strategy = Some(i);
                break;
            }
        }

        // Update target quality based on selected strategy
        if let Some(strategy_idx) = selected_strategy {
            let strategy = &self.degradation_strategies[strategy_idx];
            let new_target = 1.0 - strategy.quality_reduction;
            
            // Only degrade if necessary, allow gradual recovery otherwise
            if new_target < self.target_quality || deadline_pressure > 0.8 {
                self.target_quality = new_target;
                self.active_strategy = Some(strategy_idx);
            }
        } else {
            // No degradation needed, allow recovery
            if performance_health > 0.8 && deadline_pressure < 0.5 {
                self.target_quality = (self.target_quality + 0.02).min(1.0);
                self.active_strategy = None;
            }
        }

        // Ensure target doesn't exceed maximum degradation
        let min_quality = 1.0 - self.max_degradation;
        self.target_quality = self.target_quality.max(min_quality);
    }

    fn should_apply_strategy(&self, strategy: &QualityStrategy, deadline_pressure: f32, performance_health: f32) -> bool {
        let conditions = &strategy.conditions;
        
        // Check basic conditions
        if deadline_pressure < conditions.min_deadline_pressure {
            return false;
        }

        if (1.0 - strategy.quality_reduction) < conditions.min_acceptable_quality {
            return false;
        }

        // Check if current performance is below threshold
        if performance_health > (1.0 - conditions.load_threshold) {
            return false;
        }

        // Check activation delay (simplified - would need frame counting)
        true
    }

    fn apply_quality_adaptation(&mut self) {
        let quality_diff = self.target_quality - self.current_quality;
        let adaptation_step = quality_diff * self.adaptation_rate;

        // Apply rate limiting for smooth transitions
        let max_step = match self.emergency_mode {
            true => 0.5,   // Fast emergency adaptation
            false => 0.1,  // Normal adaptation speed
        };

        let limited_step = adaptation_step.clamp(-max_step, max_step);
        self.current_quality = (self.current_quality + limited_step).clamp(0.0, 1.0);
    }

    fn calculate_input_decimation(&self) -> f32 {
        if self.current_quality > 0.9 {
            1.0 // No decimation
        } else if self.current_quality > 0.7 {
            0.8 // Light decimation
        } else if self.current_quality > 0.5 {
            0.6 // Moderate decimation
        } else {
            0.4 // Heavy decimation
        }
    }

    fn calculate_feature_reduction(&self) -> f32 {
        if self.current_quality > 0.8 {
            1.0 // Full features
        } else if self.current_quality > 0.6 {
            0.75 // Reduced features
        } else {
            0.5 // Minimal features
        }
    }

    fn calculate_model_complexity(&self) -> f32 {
        self.current_quality // Direct mapping
    }

    fn calculate_output_precision(&self) -> f32 {
        if self.current_quality > 0.9 {
            1.0 // Full precision
        } else if self.current_quality > 0.7 {
            0.8 // Reduced precision
        } else {
            0.6 // Low precision
        }
    }

    fn calculate_buffer_optimization(&self) -> f32 {
        // Buffer size adjustment based on quality
        if self.current_quality < 0.8 {
            1.2 // Increase buffer for stability
        } else {
            1.0 // Normal buffer size
        }
    }

    fn update_metrics(&mut self, deadline_pressure: f32, performance_health: f32) {
        self.metrics.total_frames += 1;

        if self.current_quality < 0.98 {
            self.metrics.degraded_frames += 1;
        }

        // Update average quality (exponential moving average)
        let alpha = 0.05;
        self.metrics.avg_quality = self.metrics.avg_quality * (1.0 - alpha) + 
                                  self.current_quality * alpha;

        // Update preservation rate
        self.metrics.preservation_rate = 1.0 - (self.metrics.degraded_frames as f32 / self.metrics.total_frames as f32);

        // Update strategy effectiveness
        if let Some(strategy_idx) = self.active_strategy {
            let strategy_name = &self.degradation_strategies[strategy_idx].name;
            let effectiveness = performance_health; // Simplified effectiveness measure
            
            let current_effectiveness = self.metrics.strategy_effectiveness
                .get(strategy_name)
                .copied()
                .unwrap_or(0.5);
            
            let new_effectiveness = current_effectiveness * 0.9 + effectiveness * 0.1;
            self.metrics.strategy_effectiveness.insert(strategy_name.clone(), new_effectiveness);
        }
    }
}

impl QualityRecoveryController {
    fn new() -> Self {
        Self {
            recovery_rate: 0.02,
            recovery_delay_frames: 10,
            current_delay: 0,
            gradual_recovery: true,
            smoothing_window: 5,
        }
    }

    fn update(&mut self, current_quality: f32, target_quality: f32) {
        if current_quality < target_quality && self.current_delay == 0 {
            // Start gradual recovery
            if self.gradual_recovery {
                // Recovery logic is handled in main controller
            }
        } else if self.current_delay > 0 {
            self.current_delay -= 1;
        }
    }

    fn start_recovery(&mut self) {
        self.current_delay = self.recovery_delay_frames;
    }

    fn get_progress(&self) -> f32 {
        if self.current_delay == 0 {
            1.0
        } else {
            1.0 - (self.current_delay as f32 / self.recovery_delay_frames as f32)
        }
    }

    fn reset(&mut self) {
        self.current_delay = 0;
    }
}

impl QualityMetrics {
    fn new() -> Self {
        Self {
            total_frames: 0,
            degraded_frames: 0,
            avg_quality: 1.0,
            preservation_rate: 1.0,
            recovery_success_rate: 1.0,
            strategy_effectiveness: BTreeMap::new(),
        }
    }
}

/// Quality recommendations for different processing components
#[derive(Debug, Clone)]
pub struct QualityRecommendations {
    /// Input decimation factor (0.0-1.0)
    pub input_decimation: f32,
    /// Feature reduction factor (0.0-1.0)
    pub feature_reduction: f32,
    /// Model complexity factor (0.0-1.0)
    pub model_complexity: f32,
    /// Output precision factor (0.0-1.0)
    pub output_precision: f32,
    /// Buffer optimization factor (> 1.0 = larger buffer)
    pub buffer_optimization: f32,
}

/// Quality control statistics
#[derive(Debug, Clone)]
pub struct QualityControlStatistics {
    /// Current quality factor
    pub current_quality: f32,
    /// Target quality factor
    pub target_quality: f32,
    /// Average quality maintained
    pub avg_quality: f32,
    /// Quality preservation rate (0.0-1.0)
    pub preservation_rate: f32,
    /// Number of degraded frames
    pub degraded_frames: u64,
    /// Total frames processed
    pub total_frames: u64,
    /// Currently active strategy
    pub active_strategy: Option<String>,
    /// Emergency mode active
    pub emergency_mode: bool,
    /// Recovery progress (0.0-1.0)
    pub recovery_progress: f32,
}

impl QualityControlStatistics {
    /// Get formatted summary
    pub fn summary(&self) -> String {
        format!(
            "Quality: {:.1}% (target: {:.1}%), preservation: {:.1}%, strategy: {}",
            self.current_quality * 100.0,
            self.target_quality * 100.0,
            self.preservation_rate * 100.0,
            self.active_strategy.as_deref().unwrap_or("none")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_controller_creation() {
        let controller = AdaptiveQualityController::new(0.5, 0.1);
        assert_eq!(controller.current_quality, 1.0);
        assert_eq!(controller.max_degradation, 0.5);
    }

    #[test]
    fn test_quality_degradation() {
        let mut controller = AdaptiveQualityController::new(0.5, 0.2);
        
        // Simulate high deadline pressure
        controller.update(0.9, 0.3);
        
        // Quality should degrade
        assert!(controller.current_quality < 1.0);
        assert!(controller.is_degraded());
    }

    #[test]
    fn test_emergency_mode() {
        let mut controller = AdaptiveQualityController::new(0.5, 0.1);
        
        controller.force_emergency_mode();
        assert!(controller.emergency_mode);
        assert_eq!(controller.current_quality, 0.5);
        
        controller.exit_emergency_mode();
        assert!(!controller.emergency_mode);
    }

    #[test]
    fn test_quality_recovery() {
        let mut controller = AdaptiveQualityController::new(0.5, 0.1);
        
        // Degrade quality
        controller.update(0.9, 0.3);
        let degraded_quality = controller.current_quality;
        
        // Improve conditions
        controller.update(0.3, 0.9);
        
        // Quality should start recovering
        assert!(controller.current_quality >= degraded_quality);
    }

    #[test]
    fn test_component_recommendations() {
        let mut controller = AdaptiveQualityController::new(0.5, 0.1);
        
        // Degrade quality
        controller.update(0.8, 0.4);
        
        let recommendations = controller.get_component_recommendations();
        assert!(recommendations.input_decimation <= 1.0);
        assert!(recommendations.feature_reduction <= 1.0);
        assert!(recommendations.model_complexity <= 1.0);
    }

    #[test]
    fn test_custom_strategy() {
        let mut controller = AdaptiveQualityController::new(0.5, 0.1);
        
        let custom_strategy = QualityStrategy {
            name: "test_strategy".to_string(),
            quality_reduction: 0.2,
            performance_gain: 0.4,
            recovery_time_us: 150000,
            conditions: QualityConditions {
                min_deadline_pressure: 0.75,
                min_acceptable_quality: 0.7,
                load_threshold: 0.8,
                activation_delay: 2,
            },
            priority: 90,
        };
        
        assert!(controller.add_strategy(custom_strategy).is_ok());
        assert_eq!(controller.degradation_strategies.len(), 5); // 4 default + 1 custom
    }

    #[test]
    fn test_statistics() {
        let mut controller = AdaptiveQualityController::new(0.5, 0.1);
        
        // Process some frames
        for i in 0..10 {
            let pressure = (i as f32) / 10.0;
            controller.update(pressure, 1.0 - pressure);
        }
        
        let stats = controller.get_statistics();
        assert_eq!(stats.total_frames, 10);
        assert!(stats.preservation_rate <= 1.0);
    }
}