//! High-level adaptive controller combining timestep and complexity estimation

use crate::{Result, LiquidAudioError};
use crate::core::AdaptiveConfig;
use crate::adaptive::{TimestepController, ComplexityEstimator};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// High-level adaptive controller with state management and optimization
#[derive(Debug)]
pub struct AdaptiveController {
    /// Timestep controller
    timestep_controller: TimestepController,
    /// Complexity estimator
    complexity_estimator: ComplexityEstimator,
    /// Current configuration
    config: Option<AdaptiveConfig>,
    /// Controller state
    state: ControllerState,
    /// Performance optimization settings
    optimization: ControllerOptimization,
    /// Statistics tracking
    stats: ControllerStatistics,
}

impl AdaptiveController {
    /// Create new adaptive controller
    pub fn new() -> Self {
        Self {
            timestep_controller: TimestepController::new(),
            complexity_estimator: ComplexityEstimator::default(),
            config: None,
            state: ControllerState::new(),
            optimization: ControllerOptimization::default(),
            stats: ControllerStatistics::new(),
        }
    }
    
    /// Set adaptive configuration
    pub fn set_config(&mut self, config: AdaptiveConfig) -> Result<()> {
        // Validate configuration
        config.validate().map_err(|e| LiquidAudioError::ConfigError(e))?;
        
        // Apply to sub-components
        self.timestep_controller.set_config(config.clone());
        self.complexity_estimator.set_primary_metric(config.complexity_metric);
        
        // Update state
        self.config = Some(config);
        self.state.reset();
        
        Ok(())
    }
    
    /// Process audio and determine optimal timestep
    pub fn process_audio(&mut self, audio: &[f32]) -> Result<AdaptiveResult> {
        let start_time = self.get_time();
        
        // Validate input
        if audio.is_empty() {
            return Err(LiquidAudioError::InvalidInput("Empty audio buffer".to_string()));
        }
        
        if audio.len() > self.optimization.max_buffer_size {
            return Err(LiquidAudioError::SecurityError(
                format!("Buffer size {} exceeds limit {}", 
                       audio.len(), self.optimization.max_buffer_size)
            ));
        }
        
        // Get configuration
        let config = self.config.as_ref().ok_or_else(|| {
            LiquidAudioError::ConfigError("No adaptive configuration set".to_string())
        })?;
        
        // Estimate signal complexity
        let complexity = if self.optimization.enable_complexity_caching {
            self.complexity_estimator.estimate_complexity(audio)?
        } else {
            // Bypass cache
            let mut temp_estimator = ComplexityEstimator::new(config.complexity_metric);
            temp_estimator.set_caching_enabled(false);
            temp_estimator.estimate_complexity(audio)?
        };
        
        // Apply complexity smoothing if enabled
        let smoothed_complexity = if self.optimization.enable_complexity_smoothing {
            self.apply_complexity_smoothing(complexity)
        } else {
            complexity
        };
        
        // Calculate optimal timestep
        let timestep = self.timestep_controller.calculate_timestep(smoothed_complexity, config);
        
        // Update controller state
        self.state.update(smoothed_complexity, timestep, audio);
        
        // Check for adaptation stability
        let is_stable = self.check_adaptation_stability();
        
        // Compute processing time
        let processing_time = self.get_time() - start_time;
        
        // Create result
        let result = AdaptiveResult {
            complexity: smoothed_complexity,
            raw_complexity: complexity,
            timestep_seconds: timestep,
            timestep_ms: timestep * 1000.0,
            is_stable,
            processing_time_ms: processing_time as f32,
            adaptation_confidence: self.compute_adaptation_confidence(),
            recommended_action: self.recommend_action(smoothed_complexity, timestep),
        };
        
        // Update statistics
        self.stats.update(&result);
        
        Ok(result)
    }
    
    /// Apply temporal smoothing to complexity estimates
    fn apply_complexity_smoothing(&mut self, raw_complexity: f32) -> f32 {
        let smoothing_factor = self.optimization.complexity_smoothing_factor;
        
        if let Some(prev_complexity) = self.state.get_previous_complexity() {
            // Exponential smoothing
            prev_complexity * (1.0 - smoothing_factor) + raw_complexity * smoothing_factor
        } else {
            raw_complexity
        }
    }
    
    /// Check if adaptation is stable
    fn check_adaptation_stability(&self) -> bool {
        // Check timestep stability
        let timestep_stable = self.timestep_controller.is_stable();
        
        // Check complexity stability
        let complexity_stable = self.state.is_complexity_stable(0.1); // 10% variation threshold
        
        // Check minimum stable frames
        let min_frames_stable = self.state.frame_count >= self.optimization.min_stable_frames;
        
        timestep_stable && complexity_stable && min_frames_stable
    }
    
    /// Compute confidence in current adaptation
    fn compute_adaptation_confidence(&self) -> f32 {
        let mut confidence = 1.0;
        
        // Reduce confidence for high variability
        if let Some(complexity_variance) = self.state.get_complexity_variance() {
            confidence *= (1.0 - complexity_variance).clamp(0.0, 1.0);
        }
        
        // Reduce confidence for frequent timestep changes
        if let Some(timestep_variance) = self.state.get_timestep_variance() {
            confidence *= (1.0 - timestep_variance * 2.0).clamp(0.0, 1.0);
        }
        
        // Increase confidence with more data
        let data_confidence = (self.state.frame_count as f32 / 100.0).clamp(0.0, 1.0);
        confidence = confidence * 0.7 + data_confidence * 0.3;
        
        confidence.clamp(0.0, 1.0)
    }
    
    /// Recommend action based on current state
    fn recommend_action(&self, complexity: f32, timestep: f32) -> RecommendedAction {
        let config = self.config.as_ref().unwrap(); // Safe due to earlier check
        
        // Check for extreme conditions
        if complexity > 0.9 && timestep > config.min_timestep * 2.0 {
            return RecommendedAction::ReduceTimestep;
        }
        
        if complexity < 0.1 && timestep < config.max_timestep * 0.5 {
            return RecommendedAction::IncreaseTimestep;
        }
        
        // Check for instability
        if !self.check_adaptation_stability() {
            return RecommendedAction::StabilizeParameters;
        }
        
        // Check for potential power optimization
        if self.stats.avg_power_efficiency < 0.5 {
            return RecommendedAction::OptimizePower;
        }
        
        RecommendedAction::Continue
    }
    
    /// Get current time
    fn get_time(&self) -> f64 {
        #[cfg(feature = "std")]
        {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64()
        }
        
        #[cfg(not(feature = "std"))]
        {
            static mut COUNTER: u64 = 0;
            unsafe {
                COUNTER += 1;
                COUNTER as f64 * 0.001 // Simulate millisecond precision
            }
        }
    }
    
    /// Reset controller state
    pub fn reset(&mut self) {
        self.timestep_controller.reset();
        self.complexity_estimator.clear_cache();
        self.state.reset();
        self.stats.reset();
    }
    
    /// Get controller statistics
    pub fn stats(&self) -> &ControllerStatistics {
        &self.stats
    }
    
    /// Get detailed diagnostics
    pub fn get_diagnostics(&self) -> ControllerDiagnostics {
        ControllerDiagnostics {
            timestep_stats: self.timestep_controller.get_statistics(),
            complexity_cache_stats: self.complexity_estimator.cache_stats(),
            controller_stats: self.stats.clone(),
            state_info: self.state.get_info(),
            current_config: self.config.clone(),
        }
    }
    
    /// Enable/disable specific optimizations
    pub fn set_optimization(&mut self, optimization: ControllerOptimizationType, enabled: bool) {
        match optimization {
            ControllerOptimizationType::ComplexityCaching => {
                self.optimization.enable_complexity_caching = enabled;
                self.complexity_estimator.set_caching_enabled(enabled);
            },
            ControllerOptimizationType::ComplexitySmoothing => {
                self.optimization.enable_complexity_smoothing = enabled;
            },
        }
    }
    
    /// Set optimization parameters
    pub fn set_optimization_params(&mut self, params: ControllerOptimization) {
        self.optimization = params;
        
        // Apply to sub-components
        self.complexity_estimator.set_caching_enabled(params.enable_complexity_caching);
    }
    
    /// Predict timestep for given complexity without updating state
    pub fn predict_timestep(&self, complexity: f32) -> Result<f32> {
        let config = self.config.as_ref().ok_or_else(|| {
            LiquidAudioError::ConfigError("No adaptive configuration set".to_string())
        })?;
        
        Ok(self.timestep_controller.preview_timestep(complexity, config))
    }
    
    /// Get current controller state
    pub fn state(&self) -> &ControllerState {
        &self.state
    }
}

impl Default for AdaptiveController {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of adaptive processing
#[derive(Debug, Clone)]
pub struct AdaptiveResult {
    /// Estimated signal complexity (0.0-1.0)
    pub complexity: f32,
    /// Raw complexity before smoothing
    pub raw_complexity: f32,
    /// Optimal timestep in seconds
    pub timestep_seconds: f32,
    /// Optimal timestep in milliseconds
    pub timestep_ms: f32,
    /// Whether adaptation is stable
    pub is_stable: bool,
    /// Processing time for this adaptation
    pub processing_time_ms: f32,
    /// Confidence in adaptation (0.0-1.0)
    pub adaptation_confidence: f32,
    /// Recommended action
    pub recommended_action: RecommendedAction,
}

/// Recommended actions based on adaptive state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendedAction {
    /// Continue with current parameters
    Continue,
    /// Reduce timestep for better accuracy
    ReduceTimestep,
    /// Increase timestep for better efficiency
    IncreaseTimestep,
    /// Stabilize parameters (reduce adaptation rate)
    StabilizeParameters,
    /// Optimize for power efficiency
    OptimizePower,
}

/// Controller internal state
#[derive(Debug, Clone)]
pub struct ControllerState {
    /// Frame processing count
    pub frame_count: u64,
    /// Complexity history for smoothing
    complexity_history: Vec<f32>,
    /// Timestep history
    timestep_history: Vec<f32>,
    /// Audio energy history
    energy_history: Vec<f32>,
    /// Maximum history length
    max_history: usize,
}

impl ControllerState {
    /// Create new controller state
    pub fn new() -> Self {
        Self {
            frame_count: 0,
            complexity_history: Vec::with_capacity(50),
            timestep_history: Vec::with_capacity(50),
            energy_history: Vec::with_capacity(50),
            max_history: 50,
        }
    }
    
    /// Update state with new processing frame
    pub fn update(&mut self, complexity: f32, timestep: f32, audio: &[f32]) {
        self.frame_count += 1;
        
        // Update complexity history
        self.complexity_history.push(complexity);
        if self.complexity_history.len() > self.max_history {
            self.complexity_history.remove(0);
        }
        
        // Update timestep history
        self.timestep_history.push(timestep);
        if self.timestep_history.len() > self.max_history {
            self.timestep_history.remove(0);
        }
        
        // Update energy history
        let energy: f32 = audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32;
        self.energy_history.push(energy);
        if self.energy_history.len() > self.max_history {
            self.energy_history.remove(0);
        }
    }
    
    /// Reset state
    pub fn reset(&mut self) {
        self.frame_count = 0;
        self.complexity_history.clear();
        self.timestep_history.clear();
        self.energy_history.clear();
    }
    
    /// Get previous complexity
    pub fn get_previous_complexity(&self) -> Option<f32> {
        self.complexity_history.last().copied()
    }
    
    /// Get complexity variance over recent history
    pub fn get_complexity_variance(&self) -> Option<f32> {
        if self.complexity_history.len() < 2 {
            return None;
        }
        
        let mean = self.complexity_history.iter().sum::<f32>() / self.complexity_history.len() as f32;
        let variance = self.complexity_history.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / self.complexity_history.len() as f32;
        
        Some(variance)
    }
    
    /// Get timestep variance
    pub fn get_timestep_variance(&self) -> Option<f32> {
        if self.timestep_history.len() < 2 {
            return None;
        }
        
        let mean = self.timestep_history.iter().sum::<f32>() / self.timestep_history.len() as f32;
        let variance = self.timestep_history.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / self.timestep_history.len() as f32;
        
        Some(variance / mean.powi(2)) // Coefficient of variation
    }
    
    /// Check if complexity is stable
    pub fn is_complexity_stable(&self, threshold: f32) -> bool {
        if let Some(variance) = self.get_complexity_variance() {
            variance < threshold
        } else {
            false
        }
    }
    
    /// Get state information summary
    pub fn get_info(&self) -> StateInfo {
        StateInfo {
            frame_count: self.frame_count,
            complexity_history_len: self.complexity_history.len(),
            current_complexity: self.complexity_history.last().copied(),
            complexity_variance: self.get_complexity_variance(),
            current_timestep: self.timestep_history.last().copied(),
            timestep_variance: self.get_timestep_variance(),
            avg_energy: if !self.energy_history.is_empty() {
                Some(self.energy_history.iter().sum::<f32>() / self.energy_history.len() as f32)
            } else {
                None
            },
        }
    }
}

impl Default for ControllerState {
    fn default() -> Self {
        Self::new()
    }
}

/// State information summary
#[derive(Debug, Clone)]
pub struct StateInfo {
    pub frame_count: u64,
    pub complexity_history_len: usize,
    pub current_complexity: Option<f32>,
    pub complexity_variance: Option<f32>,
    pub current_timestep: Option<f32>,
    pub timestep_variance: Option<f32>,
    pub avg_energy: Option<f32>,
}

/// Controller optimization settings
#[derive(Debug, Clone)]
pub struct ControllerOptimization {
    /// Enable complexity result caching
    pub enable_complexity_caching: bool,
    /// Enable complexity temporal smoothing
    pub enable_complexity_smoothing: bool,
    /// Complexity smoothing factor (0.0-1.0)
    pub complexity_smoothing_factor: f32,
    /// Maximum buffer size for security
    pub max_buffer_size: usize,
    /// Minimum frames for stability detection
    pub min_stable_frames: u64,
}

impl Default for ControllerOptimization {
    fn default() -> Self {
        Self {
            enable_complexity_caching: true,
            enable_complexity_smoothing: true,
            complexity_smoothing_factor: 0.2,
            max_buffer_size: 65536,
            min_stable_frames: 10,
        }
    }
}

/// Controller optimization types
#[derive(Debug, Clone, Copy)]
pub enum ControllerOptimizationType {
    ComplexityCaching,
    ComplexitySmoothing,
}

/// Controller performance statistics
#[derive(Debug, Clone, Default)]
pub struct ControllerStatistics {
    pub total_frames: u64,
    pub avg_complexity: f32,
    pub avg_timestep_ms: f32,
    pub avg_processing_time_ms: f32,
    pub stability_ratio: f32,
    pub avg_adaptation_confidence: f32,
    pub avg_power_efficiency: f32,
    pub complexity_range: (f32, f32),
    pub timestep_range_ms: (f32, f32),
}

impl ControllerStatistics {
    fn new() -> Self {
        Self {
            complexity_range: (f32::INFINITY, f32::NEG_INFINITY),
            timestep_range_ms: (f32::INFINITY, f32::NEG_INFINITY),
            ..Default::default()
        }
    }
    
    fn update(&mut self, result: &AdaptiveResult) {
        self.total_frames += 1;
        
        // Update running averages with exponential smoothing
        let alpha = 1.0 / self.total_frames.min(100) as f32; // Adapt smoothing over time
        
        self.avg_complexity = self.avg_complexity * (1.0 - alpha) + result.complexity * alpha;
        self.avg_timestep_ms = self.avg_timestep_ms * (1.0 - alpha) + result.timestep_ms * alpha;
        self.avg_processing_time_ms = self.avg_processing_time_ms * (1.0 - alpha) + 
                                     result.processing_time_ms * alpha;
        self.avg_adaptation_confidence = self.avg_adaptation_confidence * (1.0 - alpha) + 
                                        result.adaptation_confidence * alpha;
        
        // Update stability ratio
        let stable_increment = if result.is_stable { 1.0 } else { 0.0 };
        self.stability_ratio = self.stability_ratio * (1.0 - alpha) + stable_increment * alpha;
        
        // Update power efficiency (inverse of timestep for simplicity)
        let power_efficiency = 1.0 / (result.timestep_ms + 1.0);
        self.avg_power_efficiency = self.avg_power_efficiency * (1.0 - alpha) + 
                                   power_efficiency * alpha;
        
        // Update ranges
        self.complexity_range.0 = self.complexity_range.0.min(result.complexity);
        self.complexity_range.1 = self.complexity_range.1.max(result.complexity);
        self.timestep_range_ms.0 = self.timestep_range_ms.0.min(result.timestep_ms);
        self.timestep_range_ms.1 = self.timestep_range_ms.1.max(result.timestep_ms);
    }
    
    fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Complete controller diagnostics
#[derive(Debug, Clone)]
pub struct ControllerDiagnostics {
    pub timestep_stats: crate::adaptive::timestep::TimestepStatistics,
    pub complexity_cache_stats: crate::adaptive::complexity::ComplexityCacheStats,
    pub controller_stats: ControllerStatistics,
    pub state_info: StateInfo,
    pub current_config: Option<AdaptiveConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{AdaptiveConfig, ComplexityMetric};
    
    #[test]
    fn test_adaptive_controller_creation() {
        let controller = AdaptiveController::new();
        assert!(controller.config.is_none());
        assert_eq!(controller.state.frame_count, 0);
    }
    
    #[test]
    fn test_config_setting() {
        let mut controller = AdaptiveController::new();
        let config = AdaptiveConfig {
            min_timestep: 0.001,
            max_timestep: 0.05,
            energy_threshold: 0.1,
            complexity_metric: ComplexityMetric::Energy,
            adaptation_rate: 0.5,
            smooth_transitions: true,
        };
        
        assert!(controller.set_config(config.clone()).is_ok());
        assert!(controller.config.is_some());
    }
    
    #[test]
    fn test_audio_processing() {
        let mut controller = AdaptiveController::new();
        let config = AdaptiveConfig::default();
        controller.set_config(config).unwrap();
        
        let audio = vec![0.1, 0.2, -0.1, 0.3, -0.2, 0.1];
        let result = controller.process_audio(&audio).unwrap();
        
        assert!(result.complexity >= 0.0 && result.complexity <= 1.0);
        assert!(result.timestep_seconds > 0.0);
        assert!(result.timestep_ms > 0.0);
        assert!(result.adaptation_confidence >= 0.0 && result.adaptation_confidence <= 1.0);
    }
    
    #[test]
    fn test_empty_audio_error() {
        let mut controller = AdaptiveController::new();
        let config = AdaptiveConfig::default();
        controller.set_config(config).unwrap();
        
        let empty_audio = vec![];
        assert!(controller.process_audio(&empty_audio).is_err());
    }
    
    #[test]
    fn test_complexity_smoothing() {
        let mut controller = AdaptiveController::new();
        controller.optimization.enable_complexity_smoothing = true;
        controller.optimization.complexity_smoothing_factor = 0.5;
        
        // First call - no previous complexity
        let smoothed1 = controller.apply_complexity_smoothing(0.8);
        assert_eq!(smoothed1, 0.8);
        
        // Simulate previous complexity
        controller.state.complexity_history.push(0.2);
        
        // Second call - should be smoothed
        let smoothed2 = controller.apply_complexity_smoothing(0.8);
        let expected = 0.2 * 0.5 + 0.8 * 0.5; // 0.5
        assert!((smoothed2 - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_state_management() {
        let mut state = ControllerState::new();
        let audio = vec![0.1, 0.2, 0.3];
        
        // Initial state
        assert_eq!(state.frame_count, 0);
        assert!(state.get_previous_complexity().is_none());
        
        // Update state
        state.update(0.5, 0.01, &audio);
        assert_eq!(state.frame_count, 1);
        assert_eq!(state.get_previous_complexity(), Some(0.5));
        
        // Add more data for variance calculation
        state.update(0.6, 0.012, &audio);
        state.update(0.4, 0.008, &audio);
        
        let variance = state.get_complexity_variance();
        assert!(variance.is_some());
        assert!(variance.unwrap() >= 0.0);
    }
    
    #[test]
    fn test_recommended_actions() {
        let mut controller = AdaptiveController::new();
        let config = AdaptiveConfig::default();
        controller.set_config(config).unwrap();
        
        // High complexity should recommend reducing timestep
        let action = controller.recommend_action(0.95, 0.02);
        assert_eq!(action, RecommendedAction::ReduceTimestep);
        
        // Low complexity should recommend increasing timestep
        let action = controller.recommend_action(0.05, 0.01);
        assert_eq!(action, RecommendedAction::IncreaseTimestep);
    }
    
    #[test]
    fn test_statistics_update() {
        let mut stats = ControllerStatistics::new();
        let result = AdaptiveResult {
            complexity: 0.5,
            raw_complexity: 0.5,
            timestep_seconds: 0.01,
            timestep_ms: 10.0,
            is_stable: true,
            processing_time_ms: 2.0,
            adaptation_confidence: 0.8,
            recommended_action: RecommendedAction::Continue,
        };
        
        stats.update(&result);
        assert_eq!(stats.total_frames, 1);
        assert_eq!(stats.avg_complexity, 0.5);
        assert_eq!(stats.avg_timestep_ms, 10.0);
        assert_eq!(stats.stability_ratio, 1.0);
    }
    
    #[test]
    fn test_timestep_prediction() {
        let mut controller = AdaptiveController::new();
        let config = AdaptiveConfig::default();
        controller.set_config(config).unwrap();
        
        let predicted_timestep = controller.predict_timestep(0.5).unwrap();
        assert!(predicted_timestep > 0.0);
        assert!(predicted_timestep <= config.max_timestep);
        assert!(predicted_timestep >= config.min_timestep);
    }
}