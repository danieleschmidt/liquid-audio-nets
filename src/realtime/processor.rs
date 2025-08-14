//! Real-time audio processor with predictable latency and adaptive quality

use crate::{Result, LiquidAudioError, ProcessingResult, LNN, ModelConfig};
use crate::realtime::{RealTimeContext, RealTimeConstraints, get_current_time_us};
use crate::realtime::buffer::CircularBuffer;
use crate::realtime::metrics::LatencyMetrics;

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, boxed::Box, string::String};

/// Processing priority levels for real-time scheduling
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProcessingPriority {
    /// Background processing (lowest priority)
    Background = 0,
    /// Normal processing priority
    Normal = 1,
    /// High priority processing
    High = 2,
    /// Real-time critical processing (highest priority)
    RealTime = 3,
}

/// Configuration for real-time processor
#[derive(Debug, Clone)]
pub struct ProcessorConfig {
    /// Model configuration for LNN
    pub model_config: ModelConfig,
    /// Real-time constraints
    pub rt_constraints: RealTimeConstraints,
    /// Input buffer size (samples)
    pub input_buffer_size: usize,
    /// Output buffer size (samples)
    pub output_buffer_size: usize,
    /// Enable adaptive quality degradation
    pub adaptive_quality: bool,
    /// Enable predictive scheduling
    pub predictive_scheduling: bool,
    /// Maximum quality degradation factor
    pub max_quality_degradation: f32,
    /// Thread affinity (CPU core to bind to)
    pub thread_affinity: Option<usize>,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            model_config: ModelConfig::default(),
            rt_constraints: RealTimeConstraints::default(),
            input_buffer_size: 1024,
            output_buffer_size: 1024,
            adaptive_quality: true,
            predictive_scheduling: true,
            max_quality_degradation: 0.3,
            thread_affinity: None,
        }
    }
}

impl ProcessorConfig {
    /// Create configuration optimized for embedded systems
    pub fn embedded() -> Self {
        Self {
            model_config: ModelConfig {
                hidden_dim: 32,
                input_dim: 20,  // Reduced features
                ..Default::default()
            },
            rt_constraints: RealTimeConstraints::embedded(),
            input_buffer_size: 256,
            output_buffer_size: 256,
            adaptive_quality: true,
            predictive_scheduling: false, // Simpler for embedded
            max_quality_degradation: 0.5,
            thread_affinity: Some(0),
        }
    }
    
    /// Create configuration for high-performance applications
    pub fn high_performance() -> Self {
        Self {
            model_config: ModelConfig {
                hidden_dim: 128,
                input_dim: 80,  // Full features
                ..Default::default()
            },
            rt_constraints: RealTimeConstraints::high_performance(),
            input_buffer_size: 2048,
            output_buffer_size: 2048,
            adaptive_quality: true,
            predictive_scheduling: true,
            max_quality_degradation: 0.1,
            thread_affinity: None,
        }
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        self.rt_constraints.validate()?;
        self.model_config.validate()?;
        
        if self.input_buffer_size == 0 || self.output_buffer_size == 0 {
            return Err(LiquidAudioError::ConfigError(
                "Buffer sizes must be greater than zero".to_string()
            ));
        }
        
        if self.max_quality_degradation < 0.0 || self.max_quality_degradation > 1.0 {
            return Err(LiquidAudioError::ConfigError(
                "Quality degradation must be between 0.0 and 1.0".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Real-time audio processor with adaptive quality and predictable latency
pub struct RealTimeProcessor {
    /// Core LNN model
    lnn: LNN,
    /// Processor configuration
    config: ProcessorConfig,
    /// Real-time context
    rt_context: RealTimeContext,
    /// Input circular buffer
    input_buffer: CircularBuffer<f32>,
    /// Output circular buffer
    output_buffer: CircularBuffer<f32>,
    /// Latency measurement system
    latency_metrics: LatencyMetrics,
    /// Processing state
    state: ProcessorState,
    /// Quality controller
    quality_controller: QualityController,
    /// Predictive scheduler
    scheduler: Option<PredictiveScheduler>,
}

/// Processing state tracking
#[derive(Debug, Clone)]
struct ProcessorState {
    /// Current processing frame
    current_frame: u64,
    /// Total samples processed
    samples_processed: u64,
    /// Current quality level (0.0-1.0)
    current_quality: f32,
    /// Processing mode
    mode: ProcessingMode,
    /// Last processing time
    last_processing_time_us: u64,
    /// Estimated next processing time
    estimated_next_time_us: u64,
}

/// Processing modes for adaptive behavior
#[derive(Debug, Clone, Copy, PartialEq)]
enum ProcessingMode {
    /// Full quality processing
    FullQuality,
    /// Adaptive quality processing
    Adaptive,
    /// Degraded quality for deadline safety
    Degraded,
    /// Emergency mode (minimal processing)
    Emergency,
}

/// Adaptive quality controller
#[derive(Debug)]
struct QualityController {
    /// Current quality factor (0.0-1.0)
    quality_factor: f32,
    /// Quality adaptation rate
    adaptation_rate: f32,
    /// Minimum quality threshold
    min_quality: f32,
    /// Quality history for smoothing
    quality_history: Vec<f32>,
    /// Maximum history length
    max_history: usize,
}

impl QualityController {
    fn new(min_quality: f32) -> Self {
        Self {
            quality_factor: 1.0,
            adaptation_rate: 0.1,
            min_quality,
            quality_history: Vec::with_capacity(10),
            max_history: 10,
        }
    }
    
    fn update(&mut self, deadline_pressure: f32, performance_health: f32) {
        // Calculate target quality based on deadline pressure and performance
        let target_quality = if deadline_pressure > 0.9 {
            // High deadline pressure - reduce quality aggressively
            (1.0 - deadline_pressure).max(self.min_quality)
        } else if performance_health > 0.8 {
            // Good performance - can increase quality
            (self.quality_factor + 0.05).min(1.0)
        } else {
            // Moderate performance - slight adjustment
            self.quality_factor * (1.0 + (performance_health - 0.5) * 0.1)
        };
        
        // Apply adaptation rate
        self.quality_factor = self.quality_factor * (1.0 - self.adaptation_rate) + 
                             target_quality * self.adaptation_rate;
        
        // Clamp to valid range
        self.quality_factor = self.quality_factor.clamp(self.min_quality, 1.0);
        
        // Update history
        self.quality_history.push(self.quality_factor);
        if self.quality_history.len() > self.max_history {
            self.quality_history.remove(0);
        }
    }
    
    fn get_quality(&self) -> f32 {
        self.quality_factor
    }
    
    fn get_smoothed_quality(&self) -> f32 {
        if self.quality_history.is_empty() {
            self.quality_factor
        } else {
            self.quality_history.iter().sum::<f32>() / self.quality_history.len() as f32
        }
    }
}

/// Predictive scheduler for processing optimization
#[derive(Debug)]
struct PredictiveScheduler {
    /// Processing time history
    processing_times: Vec<u64>,
    /// Maximum history size
    max_history: usize,
    /// Prediction horizon (frames)
    prediction_horizon: usize,
}

impl PredictiveScheduler {
    fn new() -> Self {
        Self {
            processing_times: Vec::with_capacity(100),
            max_history: 100,
            prediction_horizon: 5,
        }
    }
    
    fn record_processing_time(&mut self, time_us: u64) {
        self.processing_times.push(time_us);
        if self.processing_times.len() > self.max_history {
            self.processing_times.remove(0);
        }
    }
    
    fn predict_next_processing_time(&self) -> u64 {
        if self.processing_times.len() < 3 {
            return 1000; // Default 1ms estimate
        }
        
        // Use exponential weighted average for prediction
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        
        for (i, &time) in self.processing_times.iter().rev().enumerate() {
            if i >= self.prediction_horizon {
                break;
            }
            let weight = 2.0_f64.powi(-(i as i32));
            weighted_sum += time as f64 * weight;
            weight_sum += weight;
        }
        
        if weight_sum > 0.0 {
            (weighted_sum / weight_sum) as u64
        } else {
            1000
        }
    }
    
    fn should_preempt(&self, remaining_time_us: u64) -> bool {
        let predicted_time = self.predict_next_processing_time();
        // Add 20% safety margin
        predicted_time > remaining_time_us * 80 / 100
    }
}

impl RealTimeProcessor {
    /// Create new real-time processor
    pub fn new(config: ProcessorConfig) -> Result<Self> {
        config.validate()?;
        
        let lnn = LNN::new(config.model_config.clone())?;
        let rt_context = RealTimeContext::new(config.rt_constraints.clone())?;
        
        let input_buffer = CircularBuffer::new(config.input_buffer_size);
        let output_buffer = CircularBuffer::new(config.output_buffer_size);
        
        let latency_metrics = LatencyMetrics::new(1000); // 1000 sample history
        
        let state = ProcessorState {
            current_frame: 0,
            samples_processed: 0,
            current_quality: 1.0,
            mode: ProcessingMode::FullQuality,
            last_processing_time_us: 0,
            estimated_next_time_us: 1000,
        };
        
        let quality_controller = QualityController::new(
            1.0 - config.max_quality_degradation
        );
        
        let scheduler = if config.predictive_scheduling {
            Some(PredictiveScheduler::new())
        } else {
            None
        };
        
        Ok(Self {
            lnn,
            config,
            rt_context,
            input_buffer,
            output_buffer,
            latency_metrics,
            state,
            quality_controller,
            scheduler,
        })
    }
    
    /// Process audio frame with real-time constraints
    pub fn process_frame(&mut self, input: &[f32]) -> Result<ProcessingResult> {
        // Start real-time frame processing
        self.rt_context.start_frame();
        let frame_start_time = get_current_time_us();
        
        // Update frame counter
        self.state.current_frame += 1;
        self.state.samples_processed += input.len() as u64;
        
        // Check if we should preempt based on prediction
        if let Some(ref scheduler) = self.scheduler {
            if scheduler.should_preempt(self.rt_context.remaining_time_us()) {
                return self.emergency_processing(input);
            }
        }
        
        // Determine processing mode based on constraints
        let processing_mode = self.determine_processing_mode();
        self.state.mode = processing_mode;
        
        // Apply adaptive quality control
        if self.config.adaptive_quality {
            self.update_quality_control();
        }
        
        // Process with appropriate quality level
        let result = match processing_mode {
            ProcessingMode::FullQuality => {
                self.full_quality_processing(input)?
            },
            ProcessingMode::Adaptive => {
                self.adaptive_quality_processing(input)?
            },
            ProcessingMode::Degraded => {
                self.degraded_quality_processing(input)?
            },
            ProcessingMode::Emergency => {
                self.emergency_processing(input)?
            },
        };
        
        // Record timing metrics
        let (processing_time_us, jitter_us) = self.rt_context.finish_frame();
        self.state.last_processing_time_us = processing_time_us;
        
        // Update latency metrics
        self.latency_metrics.record_latency(processing_time_us);
        
        // Update predictive scheduler
        if let Some(ref mut scheduler) = self.scheduler {
            scheduler.record_processing_time(processing_time_us);
            self.state.estimated_next_time_us = scheduler.predict_next_processing_time();
        }
        
        // Store output in buffer for smoothing
        self.output_buffer.write_slice(&result.output)?;
        
        Ok(result)
    }
    
    /// Determine appropriate processing mode based on constraints
    fn determine_processing_mode(&self) -> ProcessingMode {
        let deadline_pressure = self.rt_context.elapsed_processing_time_us() as f32 / 
                               self.rt_context.constraints.deadline_us as f32;
        
        let performance_health = self.rt_context.performance_health();
        
        if deadline_pressure > 0.9 {
            ProcessingMode::Emergency
        } else if deadline_pressure > 0.7 || performance_health < 0.3 {
            ProcessingMode::Degraded
        } else if self.config.adaptive_quality && performance_health < 0.8 {
            ProcessingMode::Adaptive
        } else {
            ProcessingMode::FullQuality
        }
    }
    
    /// Update quality control parameters
    fn update_quality_control(&mut self) {
        let deadline_pressure = self.rt_context.elapsed_processing_time_us() as f32 / 
                               self.rt_context.constraints.deadline_us as f32;
        let performance_health = self.rt_context.performance_health();
        
        self.quality_controller.update(deadline_pressure, performance_health);
        self.state.current_quality = self.quality_controller.get_quality();
    }
    
    /// Full quality processing
    fn full_quality_processing(&mut self, input: &[f32]) -> Result<ProcessingResult> {
        self.lnn.process(input)
    }
    
    /// Adaptive quality processing with dynamic adjustments
    fn adaptive_quality_processing(&mut self, input: &[f32]) -> Result<ProcessingResult> {
        let quality_factor = self.quality_controller.get_quality();
        
        // Reduce input dimension based on quality factor
        let reduced_input = if quality_factor < 1.0 {
            let reduction_factor = (1.0 - quality_factor) * 0.5; // Reduce up to 50%
            let new_len = ((input.len() as f32 * (1.0 - reduction_factor)) as usize).max(1);
            &input[..new_len.min(input.len())]
        } else {
            input
        };
        
        // Process with potentially reduced input
        let mut result = self.lnn.process(reduced_input)?;
        
        // Adjust confidence based on quality reduction
        result.confidence *= quality_factor;
        
        // Add metadata about quality
        result.metadata = Some(format!("adaptive_quality_{:.2}", quality_factor));
        
        Ok(result)
    }
    
    /// Degraded quality processing for deadline safety
    fn degraded_quality_processing(&mut self, input: &[f32]) -> Result<ProcessingResult> {
        // Use only first 50% of input for faster processing
        let reduced_input = &input[..input.len() / 2];
        
        let mut result = self.lnn.process(reduced_input)?;
        
        // Mark as degraded quality
        result.confidence *= 0.7; // Reduce confidence due to quality degradation
        result.metadata = Some("degraded_quality".to_string());
        
        Ok(result)
    }
    
    /// Emergency processing mode - minimal computation
    fn emergency_processing(&mut self, input: &[f32]) -> Result<ProcessingResult> {
        // Ultra-simple processing - just energy-based detection
        let energy = input.iter().map(|x| x * x).sum::<f32>() / input.len() as f32;
        let confidence = energy.sqrt().min(1.0);
        
        Ok(ProcessingResult {
            output: vec![confidence, 1.0 - confidence],
            confidence,
            timestep_ms: 50.0, // Conservative timestep
            power_mw: 0.1,     // Minimal power
            complexity: 0.1,   // Very low complexity
            liquid_energy: energy,
            metadata: Some("emergency_mode".to_string()),
        })
    }
    
    /// Get current processor statistics
    pub fn get_statistics(&self) -> ProcessorStatistics {
        ProcessorStatistics {
            frames_processed: self.state.current_frame,
            samples_processed: self.state.samples_processed,
            current_quality: self.state.current_quality,
            processing_mode: self.state.mode,
            avg_processing_time_us: self.rt_context.stats.avg_processing_time_us,
            max_processing_time_us: self.rt_context.stats.max_processing_time_us,
            success_rate: self.rt_context.stats.success_rate(),
            performance_health: self.rt_context.performance_health(),
            latency_p50: self.latency_metrics.percentile(0.5),
            latency_p95: self.latency_metrics.percentile(0.95),
            latency_p99: self.latency_metrics.percentile(0.99),
            buffer_utilization: self.output_buffer.utilization(),
        }
    }
    
    /// Reset processor state and statistics
    pub fn reset(&mut self) {
        self.rt_context.stats.reset();
        self.latency_metrics.reset();
        self.state.current_frame = 0;
        self.state.samples_processed = 0;
        self.state.current_quality = 1.0;
        self.state.mode = ProcessingMode::FullQuality;
        self.quality_controller.quality_factor = 1.0;
        self.quality_controller.quality_history.clear();
        self.input_buffer.reset();
        self.output_buffer.reset();
    }
    
    /// Check if processor is meeting real-time constraints
    pub fn is_meeting_constraints(&self) -> bool {
        self.rt_context.stats.success_rate() > 0.95 && // 95% success rate
        self.rt_context.performance_health() > 0.7      // Good performance health
    }
    
    /// Get recommended configuration adjustments
    pub fn get_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        let success_rate = self.rt_context.stats.success_rate();
        let health = self.rt_context.performance_health();
        
        if success_rate < 0.9 {
            recommendations.push("Consider increasing deadline or reducing model complexity".to_string());
        }
        
        if health < 0.5 {
            recommendations.push("Enable adaptive quality if not already enabled".to_string());
            recommendations.push("Consider using embedded configuration preset".to_string());
        }
        
        if self.rt_context.stats.avg_jitter_us > self.rt_context.constraints.max_jitter_us as f64 {
            recommendations.push("Increase buffer size to reduce jitter".to_string());
        }
        
        if self.state.current_quality < 0.8 {
            recommendations.push("System under stress - consider reducing load".to_string());
        }
        
        recommendations
    }
}

/// Processor performance statistics
#[derive(Debug, Clone)]
pub struct ProcessorStatistics {
    /// Total frames processed
    pub frames_processed: u64,
    /// Total samples processed
    pub samples_processed: u64,
    /// Current quality level (0.0-1.0)
    pub current_quality: f32,
    /// Current processing mode
    pub processing_mode: ProcessingMode,
    /// Average processing time (microseconds)
    pub avg_processing_time_us: f64,
    /// Maximum processing time seen
    pub max_processing_time_us: u64,
    /// Success rate (frames meeting deadline)
    pub success_rate: f64,
    /// Overall performance health (0.0-1.0)
    pub performance_health: f32,
    /// 50th percentile latency
    pub latency_p50: u64,
    /// 95th percentile latency
    pub latency_p95: u64,
    /// 99th percentile latency
    pub latency_p99: u64,
    /// Output buffer utilization (0.0-1.0)
    pub buffer_utilization: f32,
}

impl ProcessorStatistics {
    /// Get performance summary string
    pub fn summary(&self) -> String {
        format!(
            "RT Processor: {:.1}% success, {:.1}% health, quality: {:.1}%, latency: {}µs (p95: {}µs)",
            self.success_rate * 100.0,
            self.performance_health * 100.0,
            self.current_quality * 100.0,
            self.latency_p50,
            self.latency_p95
        )
    }
    
    /// Check if performance is acceptable
    pub fn is_performance_acceptable(&self) -> bool {
        self.success_rate > 0.95 && 
        self.performance_health > 0.7 && 
        self.current_quality > 0.8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_processor_creation() {
        let config = ProcessorConfig::default();
        let processor = RealTimeProcessor::new(config);
        assert!(processor.is_ok());
    }
    
    #[test]
    fn test_embedded_config() {
        let config = ProcessorConfig::embedded();
        assert!(config.validate().is_ok());
        assert_eq!(config.rt_constraints.max_latency_us, 2000);
    }
    
    #[test]
    fn test_quality_controller() {
        let mut controller = QualityController::new(0.5);
        
        // High deadline pressure should reduce quality
        controller.update(0.95, 0.5);
        assert!(controller.get_quality() < 1.0);
        
        // Good performance should increase quality
        controller.update(0.3, 0.9);
        let quality_after = controller.get_quality();
        // Quality should increase or stay same
        assert!(quality_after >= controller.get_quality() * 0.95);
    }
    
    #[test]
    fn test_predictive_scheduler() {
        let mut scheduler = PredictiveScheduler::new();
        
        // Record some processing times
        for i in 1..=10 {
            scheduler.record_processing_time(i * 100);
        }
        
        let prediction = scheduler.predict_next_processing_time();
        assert!(prediction > 0);
        assert!(prediction < 2000); // Should be reasonable
    }
    
    #[test]
    fn test_processing_mode_determination() {
        let config = ProcessorConfig::default();
        let processor = RealTimeProcessor::new(config).unwrap();
        
        let mode = processor.determine_processing_mode();
        assert_eq!(mode, ProcessingMode::FullQuality); // Should start with full quality
    }
}