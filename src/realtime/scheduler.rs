//! Advanced predictive scheduler for real-time audio processing
//!
//! Provides intelligent scheduling and resource allocation for optimal
//! real-time performance with machine learning-based predictions.

use crate::{Result, LiquidAudioError};
use super::{RealTimeConstraints, PerformanceProfile, get_current_time_us};

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String, collections::BTreeMap};

#[cfg(feature = "std")]
use std::collections::BTreeMap;

/// Predictive scheduler for real-time processing optimization
#[derive(Debug)]
pub struct PredictiveScheduler {
    /// Performance profile
    profile: PerformanceProfile,
    /// Processing time history for predictions
    processing_history: Vec<ProcessingMeasurement>,
    /// Maximum history length
    max_history: usize,
    /// Prediction models
    prediction_models: PredictionModels,
    /// Scheduling decisions history
    decision_history: Vec<SchedulingDecision>,
    /// Current system load estimate
    system_load: SystemLoad,
    /// Scheduling metrics
    metrics: SchedulingMetrics,
}

/// Processing measurement for prediction
#[derive(Debug, Clone)]
struct ProcessingMeasurement {
    timestamp_us: u64,
    processing_time_us: u64,
    input_complexity: f32,
    system_load: f32,
    met_deadline: bool,
    quality_factor: f32,
}

/// Prediction models for different aspects
#[derive(Debug)]
struct PredictionModels {
    /// Processing time predictor
    time_predictor: TimePredictor,
    /// System load predictor
    load_predictor: LoadPredictor,
    /// Quality impact predictor
    quality_predictor: QualityPredictor,
}

/// Time-based prediction model
#[derive(Debug)]
struct TimePredictor {
    /// Exponential weighted moving average
    ewma_alpha: f32,
    /// Current prediction
    current_prediction_us: f32,
    /// Confidence in prediction (0.0-1.0)
    confidence: f32,
    /// Prediction error history
    error_history: Vec<f32>,
}

/// System load prediction model
#[derive(Debug)]
struct LoadPredictor {
    /// Recent load measurements
    load_samples: Vec<f32>,
    /// Trend analysis
    trend_direction: TrendDirection,
    /// Load prediction for next N frames
    future_load: Vec<f32>,
}

/// Quality impact prediction model
#[derive(Debug)]
struct QualityPredictor {
    /// Quality degradation patterns
    degradation_patterns: BTreeMap<String, QualityPattern>,
    /// Current quality trend
    quality_trend: f32,
}

/// Trend direction for load prediction
#[derive(Debug, Clone, Copy)]
enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

/// Quality degradation pattern
#[derive(Debug, Clone)]
struct QualityPattern {
    /// Pattern identifier
    id: String,
    /// Load threshold for activation
    load_threshold: f32,
    /// Quality reduction factor
    quality_reduction: f32,
    /// Recovery time estimate
    recovery_time_us: u64,
}

/// System load estimation
#[derive(Debug, Clone)]
struct SystemLoad {
    /// CPU utilization estimate (0.0-1.0)
    cpu_utilization: f32,
    /// Memory pressure estimate (0.0-1.0)
    memory_pressure: f32,
    /// I/O pressure estimate (0.0-1.0)
    io_pressure: f32,
    /// Overall system pressure (0.0-1.0)
    overall_pressure: f32,
    /// Load stability indicator
    stability: f32,
}

/// Scheduling decision record
#[derive(Debug, Clone)]
struct SchedulingDecision {
    timestamp_us: u64,
    decision_type: DecisionType,
    predicted_time_us: u64,
    actual_time_us: Option<u64>,
    success: Option<bool>,
    quality_impact: f32,
}

/// Types of scheduling decisions
#[derive(Debug, Clone, Copy)]
enum DecisionType {
    /// Continue with full quality
    FullQuality,
    /// Reduce quality preemptively
    PreemptiveQualityReduction,
    /// Emergency quality reduction
    EmergencyQualityReduction,
    /// Skip frame for timing recovery
    SkipFrame,
    /// Adjust buffer size
    BufferAdjustment,
}

/// Scheduling recommendation from predictor
#[derive(Debug, Clone)]
pub struct SchedulingRecommendation {
    /// Recommended decision type
    pub decision: DecisionType,
    /// Confidence in recommendation (0.0-1.0)
    pub confidence: f32,
    /// Predicted processing time
    pub predicted_time_us: u64,
    /// Recommended quality factor
    pub quality_factor: f32,
    /// Urgency level (0.0-1.0)
    pub urgency: f32,
    /// Additional metadata
    pub metadata: BTreeMap<String, String>,
}

/// Scheduling performance metrics
#[derive(Debug, Clone)]
struct SchedulingMetrics {
    /// Total decisions made
    total_decisions: u64,
    /// Successful predictions
    successful_predictions: u64,
    /// Prediction accuracy (0.0-1.0)
    prediction_accuracy: f32,
    /// Average prediction error (microseconds)
    avg_prediction_error_us: f32,
    /// Quality preservation rate
    quality_preservation_rate: f32,
    /// Emergency activations
    emergency_activations: u64,
}

impl PredictiveScheduler {
    /// Create new predictive scheduler
    pub fn new(profile: PerformanceProfile) -> Self {
        let max_history = match profile {
            PerformanceProfile::UltraLowLatency => 50,  // Minimal history for speed
            PerformanceProfile::HighPerformance => 100,
            PerformanceProfile::Balanced => 200,
            PerformanceProfile::PowerOptimized => 500,
            PerformanceProfile::BestEffort => 1000,
        };

        Self {
            profile,
            processing_history: Vec::with_capacity(max_history),
            max_history,
            prediction_models: PredictionModels::new(profile),
            decision_history: Vec::with_capacity(max_history),
            system_load: SystemLoad::new(),
            metrics: SchedulingMetrics::new(),
        }
    }

    /// Start processing a new frame
    pub fn start_frame(&mut self) {
        // Update system load estimation
        self.update_system_load();
        
        // Update prediction models
        self.update_prediction_models();
    }

    /// Record completed frame timing
    pub fn record_frame_time(&mut self, processing_time_us: u64, met_deadline: bool) {
        let measurement = ProcessingMeasurement {
            timestamp_us: get_current_time_us(),
            processing_time_us,
            input_complexity: self.estimate_input_complexity(),
            system_load: self.system_load.overall_pressure,
            met_deadline,
            quality_factor: 1.0, // Would be provided by quality controller
        };

        // Add to history
        self.processing_history.push(measurement);
        if self.processing_history.len() > self.max_history {
            self.processing_history.remove(0);
        }

        // Update prediction models with new data
        self.prediction_models.update(&measurement);
        
        // Update metrics
        self.update_metrics(processing_time_us, met_deadline);
    }

    /// Get scheduling recommendation for current frame
    pub fn get_recommendation(&self) -> SchedulingRecommendation {
        let predicted_time = self.prediction_models.time_predictor.predict();
        let system_load = self.system_load.overall_pressure;
        let quality_impact = self.prediction_models.quality_predictor.predict_impact(system_load);

        // Determine decision type based on predictions
        let decision = self.determine_decision_type(predicted_time, system_load);
        
        // Calculate confidence based on prediction accuracy and stability
        let confidence = self.calculate_confidence();
        
        // Calculate urgency based on system pressure and deadline proximity
        let urgency = self.calculate_urgency(predicted_time, system_load);
        
        // Determine quality factor
        let quality_factor = self.calculate_quality_factor(decision, system_load);

        // Build metadata
        let mut metadata = BTreeMap::new();
        metadata.insert("system_load".to_string(), format!("{:.3}", system_load));
        metadata.insert("prediction_confidence".to_string(), format!("{:.3}", confidence));
        metadata.insert("trend".to_string(), format!("{:?}", self.prediction_models.load_predictor.trend_direction));

        SchedulingRecommendation {
            decision,
            confidence,
            predicted_time_us: predicted_time as u64,
            quality_factor,
            urgency,
            metadata,
        }
    }

    /// Check if frame should be preempted
    pub fn should_preempt(&self, remaining_time_us: u64) -> bool {
        let predicted_time = self.prediction_models.time_predictor.predict() as u64;
        let safety_margin = match self.profile {
            PerformanceProfile::UltraLowLatency => 1.1, // 10% margin
            PerformanceProfile::HighPerformance => 1.2, // 20% margin
            PerformanceProfile::Balanced => 1.3,        // 30% margin
            PerformanceProfile::PowerOptimized => 1.4,  // 40% margin
            PerformanceProfile::BestEffort => 1.5,      // 50% margin
        };

        (predicted_time as f32 * safety_margin) > remaining_time_us as f32
    }

    /// Get prediction accuracy
    pub fn get_prediction_accuracy(&self) -> f32 {
        self.metrics.prediction_accuracy
    }

    /// Get scheduling metrics
    pub fn get_metrics(&self) -> SchedulingMetrics {
        self.metrics.clone()
    }

    /// Reset scheduler state
    pub fn reset(&mut self) {
        self.processing_history.clear();
        self.decision_history.clear();
        self.prediction_models.reset();
        self.system_load = SystemLoad::new();
        self.metrics = SchedulingMetrics::new();
    }

    // Private implementation methods

    fn update_system_load(&mut self) {
        // Update CPU utilization estimate
        let recent_processing_times: Vec<u64> = self.processing_history
            .iter()
            .rev()
            .take(10)
            .map(|m| m.processing_time_us)
            .collect();

        if !recent_processing_times.is_empty() {
            let avg_time = recent_processing_times.iter().sum::<u64>() as f32 / recent_processing_times.len() as f32;
            let target_time = match self.profile {
                PerformanceProfile::UltraLowLatency => 800.0,
                PerformanceProfile::HighPerformance => 4000.0,
                PerformanceProfile::Balanced => 8000.0,
                PerformanceProfile::PowerOptimized => 20000.0,
                PerformanceProfile::BestEffort => 50000.0,
            };

            self.system_load.cpu_utilization = (avg_time / target_time).min(1.0);
        }

        // Estimate memory pressure (simplified)
        self.system_load.memory_pressure = (self.processing_history.len() as f32 / self.max_history as f32) * 0.5;

        // I/O pressure is minimal for audio processing
        self.system_load.io_pressure = 0.1;

        // Calculate overall pressure
        self.system_load.overall_pressure = (
            self.system_load.cpu_utilization * 0.6 +
            self.system_load.memory_pressure * 0.3 +
            self.system_load.io_pressure * 0.1
        ).min(1.0);

        // Update stability based on variance in processing times
        if self.processing_history.len() >= 5 {
            let recent_times: Vec<f32> = self.processing_history
                .iter()
                .rev()
                .take(5)
                .map(|m| m.processing_time_us as f32)
                .collect();

            let mean = recent_times.iter().sum::<f32>() / recent_times.len() as f32;
            let variance = recent_times.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / recent_times.len() as f32;
            let coefficient_of_variation = variance.sqrt() / mean;

            self.system_load.stability = (1.0 - coefficient_of_variation.min(1.0)).max(0.0);
        }
    }

    fn update_prediction_models(&mut self) {
        if let Some(latest) = self.processing_history.last() {
            self.prediction_models.time_predictor.update(latest.processing_time_us as f32);
            self.prediction_models.load_predictor.update(self.system_load.overall_pressure);
            self.prediction_models.quality_predictor.update(latest);
        }
    }

    fn estimate_input_complexity(&self) -> f32 {
        // Simplified complexity estimation based on recent processing times
        if let Some(latest) = self.processing_history.last() {
            let base_time = match self.profile {
                PerformanceProfile::UltraLowLatency => 500.0,
                PerformanceProfile::HighPerformance => 2000.0,
                PerformanceProfile::Balanced => 5000.0,
                PerformanceProfile::PowerOptimized => 15000.0,
                PerformanceProfile::BestEffort => 40000.0,
            };

            (latest.processing_time_us as f32 / base_time).min(2.0)
        } else {
            0.5 // Default moderate complexity
        }
    }

    fn determine_decision_type(&self, predicted_time_us: f32, system_load: f32) -> DecisionType {
        let deadline_us = match self.profile {
            PerformanceProfile::UltraLowLatency => 800.0,
            PerformanceProfile::HighPerformance => 4000.0,
            PerformanceProfile::Balanced => 8000.0,
            PerformanceProfile::PowerOptimized => 20000.0,
            PerformanceProfile::BestEffort => 50000.0,
        };

        let deadline_pressure = predicted_time_us / deadline_us;
        let emergency_threshold = 0.95;
        let preemptive_threshold = 0.8;

        if deadline_pressure > emergency_threshold || system_load > 0.9 {
            DecisionType::EmergencyQualityReduction
        } else if deadline_pressure > preemptive_threshold || system_load > 0.7 {
            DecisionType::PreemptiveQualityReduction
        } else if system_load > 0.5 {
            DecisionType::BufferAdjustment
        } else {
            DecisionType::FullQuality
        }
    }

    fn calculate_confidence(&self) -> f32 {
        let prediction_confidence = self.prediction_models.time_predictor.confidence;
        let stability_confidence = self.system_load.stability;
        let history_confidence = (self.processing_history.len() as f32 / self.max_history as f32).min(1.0);

        (prediction_confidence + stability_confidence + history_confidence) / 3.0
    }

    fn calculate_urgency(&self, predicted_time_us: f32, system_load: f32) -> f32 {
        let deadline_us = match self.profile {
            PerformanceProfile::UltraLowLatency => 800.0,
            PerformanceProfile::HighPerformance => 4000.0,
            PerformanceProfile::Balanced => 8000.0,
            PerformanceProfile::PowerOptimized => 20000.0,
            PerformanceProfile::BestEffort => 50000.0,
        };

        let time_urgency = (predicted_time_us / deadline_us).min(1.0);
        let load_urgency = system_load;

        (time_urgency * 0.7 + load_urgency * 0.3).min(1.0)
    }

    fn calculate_quality_factor(&self, decision: DecisionType, system_load: f32) -> f32 {
        match decision {
            DecisionType::FullQuality => 1.0,
            DecisionType::PreemptiveQualityReduction => 0.8 - (system_load * 0.2),
            DecisionType::EmergencyQualityReduction => 0.5 - (system_load * 0.3),
            DecisionType::SkipFrame => 0.0,
            DecisionType::BufferAdjustment => 0.9,
        }
    }

    fn update_metrics(&mut self, processing_time_us: u64, met_deadline: bool) {
        self.metrics.total_decisions += 1;

        if met_deadline {
            self.metrics.successful_predictions += 1;
        }

        // Update prediction accuracy
        self.metrics.prediction_accuracy = self.metrics.successful_predictions as f32 / self.metrics.total_decisions as f32;

        // Update prediction error (simplified)
        if let Some(prediction) = self.prediction_models.time_predictor.get_last_prediction() {
            let error = (prediction - processing_time_us as f32).abs();
            let alpha = 0.1;
            self.metrics.avg_prediction_error_us = self.metrics.avg_prediction_error_us * (1.0 - alpha) + error * alpha;
        }
    }
}

impl PredictionModels {
    fn new(profile: PerformanceProfile) -> Self {
        Self {
            time_predictor: TimePredictor::new(profile),
            load_predictor: LoadPredictor::new(),
            quality_predictor: QualityPredictor::new(),
        }
    }

    fn update(&mut self, measurement: &ProcessingMeasurement) {
        self.time_predictor.update(measurement.processing_time_us as f32);
        self.load_predictor.update(measurement.system_load);
        self.quality_predictor.update(measurement);
    }

    fn reset(&mut self) {
        self.time_predictor.reset();
        self.load_predictor.reset();
        self.quality_predictor.reset();
    }
}

impl TimePredictor {
    fn new(profile: PerformanceProfile) -> Self {
        let ewma_alpha = match profile {
            PerformanceProfile::UltraLowLatency => 0.3, // Fast adaptation
            PerformanceProfile::HighPerformance => 0.2,
            PerformanceProfile::Balanced => 0.15,
            PerformanceProfile::PowerOptimized => 0.1,
            PerformanceProfile::BestEffort => 0.05, // Slow, stable adaptation
        };

        Self {
            ewma_alpha,
            current_prediction_us: 1000.0, // Default 1ms
            confidence: 0.5,
            error_history: Vec::with_capacity(20),
        }
    }

    fn update(&mut self, actual_time_us: f32) {
        // Update prediction using exponential weighted moving average
        self.current_prediction_us = self.current_prediction_us * (1.0 - self.ewma_alpha) + 
                                   actual_time_us * self.ewma_alpha;

        // Update confidence based on prediction error
        let error = (self.current_prediction_us - actual_time_us).abs();
        self.error_history.push(error);
        if self.error_history.len() > 20 {
            self.error_history.remove(0);
        }

        // Calculate confidence as inverse of normalized error variance
        if self.error_history.len() > 3 {
            let mean_error = self.error_history.iter().sum::<f32>() / self.error_history.len() as f32;
            let error_variance = self.error_history.iter()
                .map(|e| (e - mean_error).powi(2))
                .sum::<f32>() / self.error_history.len() as f32;
            
            let normalized_variance = error_variance / (self.current_prediction_us + 1.0);
            self.confidence = (1.0 - normalized_variance.sqrt()).max(0.1).min(1.0);
        }
    }

    fn predict(&self) -> f32 {
        self.current_prediction_us
    }

    fn get_last_prediction(&self) -> Option<f32> {
        Some(self.current_prediction_us)
    }

    fn reset(&mut self) {
        self.current_prediction_us = 1000.0;
        self.confidence = 0.5;
        self.error_history.clear();
    }
}

impl LoadPredictor {
    fn new() -> Self {
        Self {
            load_samples: Vec::with_capacity(50),
            trend_direction: TrendDirection::Stable,
            future_load: Vec::with_capacity(10),
        }
    }

    fn update(&mut self, load: f32) {
        self.load_samples.push(load);
        if self.load_samples.len() > 50 {
            self.load_samples.remove(0);
        }

        // Update trend analysis
        if self.load_samples.len() >= 5 {
            let recent: Vec<f32> = self.load_samples.iter().rev().take(5).cloned().collect();
            let older: Vec<f32> = self.load_samples.iter().rev().skip(5).take(5).cloned().collect();

            if !older.is_empty() {
                let recent_avg = recent.iter().sum::<f32>() / recent.len() as f32;
                let older_avg = older.iter().sum::<f32>() / older.len() as f32;

                let diff = recent_avg - older_avg;
                let threshold = 0.05;

                self.trend_direction = if diff > threshold {
                    TrendDirection::Increasing
                } else if diff < -threshold {
                    TrendDirection::Decreasing
                } else {
                    TrendDirection::Stable
                };
            }
        }

        // Predict future load (simple linear extrapolation)
        self.update_future_predictions();
    }

    fn update_future_predictions(&mut self) {
        self.future_load.clear();
        
        if self.load_samples.len() >= 3 {
            let current = self.load_samples.last().copied().unwrap_or(0.5);
            let trend_factor = match self.trend_direction {
                TrendDirection::Increasing => 0.02,
                TrendDirection::Decreasing => -0.02,
                TrendDirection::Stable => 0.0,
            };

            for i in 1..=5 {
                let predicted = (current + trend_factor * i as f32).clamp(0.0, 1.0);
                self.future_load.push(predicted);
            }
        }
    }

    fn reset(&mut self) {
        self.load_samples.clear();
        self.trend_direction = TrendDirection::Stable;
        self.future_load.clear();
    }
}

impl QualityPredictor {
    fn new() -> Self {
        let mut degradation_patterns = BTreeMap::new();
        
        // Initialize common patterns
        degradation_patterns.insert("high_load".to_string(), QualityPattern {
            id: "high_load".to_string(),
            load_threshold: 0.8,
            quality_reduction: 0.2,
            recovery_time_us: 100000, // 100ms
        });

        degradation_patterns.insert("extreme_load".to_string(), QualityPattern {
            id: "extreme_load".to_string(),
            load_threshold: 0.95,
            quality_reduction: 0.5,
            recovery_time_us: 500000, // 500ms
        });

        Self {
            degradation_patterns,
            quality_trend: 0.0,
        }
    }

    fn update(&mut self, measurement: &ProcessingMeasurement) {
        // Update quality trend based on recent performance
        let quality_impact = if measurement.met_deadline { 0.0 } else { -0.1 };
        let alpha = 0.2;
        self.quality_trend = self.quality_trend * (1.0 - alpha) + quality_impact * alpha;
    }

    fn predict_impact(&self, system_load: f32) -> f32 {
        let mut max_impact = 0.0;

        for pattern in self.degradation_patterns.values() {
            if system_load > pattern.load_threshold {
                max_impact = max_impact.max(pattern.quality_reduction);
            }
        }

        max_impact + self.quality_trend.abs() * 0.5
    }

    fn reset(&mut self) {
        self.quality_trend = 0.0;
    }
}

impl SystemLoad {
    fn new() -> Self {
        Self {
            cpu_utilization: 0.5,
            memory_pressure: 0.1,
            io_pressure: 0.1,
            overall_pressure: 0.2,
            stability: 1.0,
        }
    }
}

impl SchedulingMetrics {
    fn new() -> Self {
        Self {
            total_decisions: 0,
            successful_predictions: 0,
            prediction_accuracy: 0.0,
            avg_prediction_error_us: 0.0,
            quality_preservation_rate: 1.0,
            emergency_activations: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictive_scheduler_creation() {
        let scheduler = PredictiveScheduler::new(PerformanceProfile::HighPerformance);
        assert_eq!(scheduler.max_history, 100);
    }

    #[test]
    fn test_frame_recording() {
        let mut scheduler = PredictiveScheduler::new(PerformanceProfile::Balanced);
        
        scheduler.start_frame();
        scheduler.record_frame_time(1000, true);
        
        assert_eq!(scheduler.processing_history.len(), 1);
        assert!(scheduler.processing_history[0].met_deadline);
    }

    #[test]
    fn test_recommendation_generation() {
        let mut scheduler = PredictiveScheduler::new(PerformanceProfile::HighPerformance);
        
        // Add some history
        for i in 0..10 {
            scheduler.start_frame();
            scheduler.record_frame_time(1000 + i * 100, true);
        }
        
        let recommendation = scheduler.get_recommendation();
        assert!(recommendation.confidence > 0.0);
        assert!(recommendation.quality_factor > 0.0);
    }

    #[test]
    fn test_preemption_decision() {
        let mut scheduler = PredictiveScheduler::new(PerformanceProfile::UltraLowLatency);
        
        // Record consistently high processing times
        for _ in 0..5 {
            scheduler.record_frame_time(2000, false); // High time, missed deadline
        }
        
        // Should recommend preemption with little remaining time
        assert!(scheduler.should_preempt(500));
    }

    #[test]
    fn test_time_predictor() {
        let mut predictor = TimePredictor::new(PerformanceProfile::Balanced);
        
        // Feed consistent timing data
        for _ in 0..10 {
            predictor.update(1000.0);
        }
        
        let prediction = predictor.predict();
        assert!((prediction - 1000.0).abs() < 100.0); // Should be close to 1000
        assert!(predictor.confidence > 0.5); // Should be confident
    }

    #[test]
    fn test_load_predictor_trend() {
        let mut predictor = LoadPredictor::new();
        
        // Feed increasing load data
        for i in 0..10 {
            predictor.update(0.1 + (i as f32 * 0.05));
        }
        
        assert!(matches!(predictor.trend_direction, TrendDirection::Increasing));
    }
}