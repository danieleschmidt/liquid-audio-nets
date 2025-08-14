//! Real-time audio processing with ultra-low latency capabilities
//! 
//! Advanced real-time processing framework for edge devices with predictable
//! latency, jitter minimization, and adaptive resource management.

pub mod processor;
pub mod buffer;
pub mod scheduler;
pub mod metrics;

pub use processor::{RealTimeProcessor, ProcessorConfig, ProcessingPriority};
pub use buffer::{CircularBuffer, BufferConfig, BufferMetrics};
pub use scheduler::{RealTimeScheduler, TaskPriority, SchedulerConfig};
pub use metrics::{LatencyMetrics, ThroughputMetrics, JitterAnalyzer};

use crate::{Result, LiquidAudioError, ProcessingResult};

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, boxed::Box};

/// Real-time processing constraints
#[derive(Debug, Clone)]
pub struct RealTimeConstraints {
    /// Maximum acceptable latency (microseconds)
    pub max_latency_us: u64,
    /// Maximum acceptable jitter (microseconds)
    pub max_jitter_us: u64,
    /// Target sample rate (Hz)
    pub sample_rate: u32,
    /// Processing deadline (microseconds from input)
    pub deadline_us: u64,
    /// Priority level for real-time scheduling
    pub priority: ProcessingPriority,
    /// CPU budget percentage (0.0-1.0)
    pub cpu_budget: f32,
    /// Memory budget (bytes)
    pub memory_budget: usize,
}

impl Default for RealTimeConstraints {
    fn default() -> Self {
        Self {
            max_latency_us: 1000,      // 1ms maximum latency
            max_jitter_us: 100,        // 100µs maximum jitter
            sample_rate: 16000,        // 16kHz sample rate
            deadline_us: 800,          // 800µs deadline
            priority: ProcessingPriority::High,
            cpu_budget: 0.8,           // 80% CPU budget
            memory_budget: 64 * 1024,  // 64KB memory budget
        }
    }
}

impl RealTimeConstraints {
    /// Create constraints optimized for embedded systems
    pub fn embedded() -> Self {
        Self {
            max_latency_us: 2000,      // 2ms for embedded
            max_jitter_us: 200,        // 200µs jitter
            sample_rate: 16000,
            deadline_us: 1500,         // 1.5ms deadline
            priority: ProcessingPriority::RealTime,
            cpu_budget: 0.6,           // Conservative 60%
            memory_budget: 32 * 1024,  // 32KB memory
        }
    }
    
    /// Create constraints for high-performance applications
    pub fn high_performance() -> Self {
        Self {
            max_latency_us: 500,       // 0.5ms ultra-low latency
            max_jitter_us: 50,         // 50µs minimal jitter
            sample_rate: 48000,        // High sample rate
            deadline_us: 400,          // 400µs tight deadline
            priority: ProcessingPriority::RealTime,
            cpu_budget: 0.9,           // 90% CPU budget
            memory_budget: 128 * 1024, // 128KB memory
        }
    }
    
    /// Validate constraints for feasibility
    pub fn validate(&self) -> Result<()> {
        if self.max_latency_us == 0 {
            return Err(LiquidAudioError::ConfigError(
                "Maximum latency must be greater than zero".to_string()
            ));
        }
        
        if self.deadline_us >= self.max_latency_us {
            return Err(LiquidAudioError::ConfigError(
                "Deadline must be less than maximum latency".to_string()
            ));
        }
        
        if self.cpu_budget <= 0.0 || self.cpu_budget > 1.0 {
            return Err(LiquidAudioError::ConfigError(
                "CPU budget must be between 0.0 and 1.0".to_string()
            ));
        }
        
        if self.sample_rate < 1000 || self.sample_rate > 192000 {
            return Err(LiquidAudioError::ConfigError(
                format!("Sample rate {} is outside valid range", self.sample_rate)
            ));
        }
        
        Ok(())
    }
    
    /// Calculate frame size for given latency target
    pub fn calculate_frame_size(&self) -> usize {
        let frame_time_us = self.max_latency_us / 2; // Use half max latency
        let frame_time_s = frame_time_us as f64 / 1_000_000.0;
        (self.sample_rate as f64 * frame_time_s) as usize
    }
    
    /// Check if processing time meets constraints
    pub fn meets_constraints(&self, processing_time_us: u64, jitter_us: u64) -> bool {
        processing_time_us <= self.deadline_us && jitter_us <= self.max_jitter_us
    }
}

/// Real-time processing statistics
#[derive(Debug, Clone, Default)]
pub struct RealTimeStats {
    /// Total frames processed
    pub frames_processed: u64,
    /// Frames that met deadline
    pub frames_on_time: u64,
    /// Frames that missed deadline
    pub frames_late: u64,
    /// Average processing time (microseconds)
    pub avg_processing_time_us: f64,
    /// Maximum processing time seen (microseconds)
    pub max_processing_time_us: u64,
    /// Current jitter (microseconds)
    pub current_jitter_us: u64,
    /// Average jitter (microseconds)
    pub avg_jitter_us: f64,
    /// CPU utilization (0.0-1.0)
    pub cpu_utilization: f32,
    /// Memory utilization (0.0-1.0)
    pub memory_utilization: f32,
    /// Number of buffer underruns
    pub buffer_underruns: u64,
    /// Number of buffer overruns
    pub buffer_overruns: u64,
}

impl RealTimeStats {
    /// Calculate success rate (frames meeting deadline)
    pub fn success_rate(&self) -> f64 {
        if self.frames_processed > 0 {
            self.frames_on_time as f64 / self.frames_processed as f64
        } else {
            0.0
        }
    }
    
    /// Calculate late frame rate
    pub fn late_frame_rate(&self) -> f64 {
        if self.frames_processed > 0 {
            self.frames_late as f64 / self.frames_processed as f64
        } else {
            0.0
        }
    }
    
    /// Check if performance meets target
    pub fn meets_performance_target(&self, target_success_rate: f64) -> bool {
        self.success_rate() >= target_success_rate
    }
    
    /// Update statistics with new processing measurement
    pub fn update(&mut self, processing_time_us: u64, jitter_us: u64, deadline_us: u64) {
        self.frames_processed += 1;
        
        if processing_time_us <= deadline_us {
            self.frames_on_time += 1;
        } else {
            self.frames_late += 1;
        }
        
        // Update average processing time (exponential moving average)
        let alpha = 0.1;
        self.avg_processing_time_us = self.avg_processing_time_us * (1.0 - alpha) + 
                                     processing_time_us as f64 * alpha;
        
        // Update maximum processing time
        if processing_time_us > self.max_processing_time_us {
            self.max_processing_time_us = processing_time_us;
        }
        
        // Update jitter statistics
        self.current_jitter_us = jitter_us;
        self.avg_jitter_us = self.avg_jitter_us * (1.0 - alpha) + jitter_us as f64 * alpha;
    }
    
    /// Reset statistics
    pub fn reset(&mut self) {
        *self = Self::default();
    }
    
    /// Get performance summary string
    pub fn summary(&self) -> String {
        format!(
            "RT Stats: {:.1}% on-time ({}/{}), avg: {:.0}µs, max: {}µs, jitter: {:.0}µs",
            self.success_rate() * 100.0,
            self.frames_on_time,
            self.frames_processed,
            self.avg_processing_time_us,
            self.max_processing_time_us,
            self.avg_jitter_us
        )
    }
}

/// Real-time processing context
#[derive(Debug)]
pub struct RealTimeContext {
    /// Processing constraints
    pub constraints: RealTimeConstraints,
    /// Performance statistics
    pub stats: RealTimeStats,
    /// Current timestamp (microseconds)
    pub current_time_us: u64,
    /// Processing start timestamp
    pub processing_start_us: u64,
    /// Frame sequence number
    pub frame_sequence: u64,
    /// Adaptive parameters
    pub adaptive_params: AdaptiveParams,
}

/// Adaptive parameters for real-time optimization
#[derive(Debug, Clone)]
pub struct AdaptiveParams {
    /// Dynamic CPU frequency scaling factor
    pub cpu_scale_factor: f32,
    /// Dynamic buffer size adjustment
    pub buffer_size_factor: f32,
    /// Quality degradation factor (for graceful degradation)
    pub quality_factor: f32,
    /// Processing complexity adjustment
    pub complexity_factor: f32,
}

impl Default for AdaptiveParams {
    fn default() -> Self {
        Self {
            cpu_scale_factor: 1.0,
            buffer_size_factor: 1.0,
            quality_factor: 1.0,
            complexity_factor: 1.0,
        }
    }
}

impl RealTimeContext {
    /// Create new real-time context
    pub fn new(constraints: RealTimeConstraints) -> Result<Self> {
        constraints.validate()?;
        
        Ok(Self {
            constraints,
            stats: RealTimeStats::default(),
            current_time_us: get_current_time_us(),
            processing_start_us: 0,
            frame_sequence: 0,
            adaptive_params: AdaptiveParams::default(),
        })
    }
    
    /// Start processing frame
    pub fn start_frame(&mut self) {
        self.processing_start_us = get_current_time_us();
        self.current_time_us = self.processing_start_us;
        self.frame_sequence += 1;
    }
    
    /// Finish processing frame and update statistics
    pub fn finish_frame(&mut self) -> (u64, u64) {
        let end_time_us = get_current_time_us();
        let processing_time_us = end_time_us.saturating_sub(self.processing_start_us);
        
        // Calculate jitter (variation from expected timing)
        let expected_interval_us = 1_000_000 / self.constraints.sample_rate as u64 * 
                                   self.constraints.calculate_frame_size() as u64;
        let actual_interval_us = end_time_us.saturating_sub(self.current_time_us);
        let jitter_us = if actual_interval_us > expected_interval_us {
            actual_interval_us - expected_interval_us
        } else {
            expected_interval_us - actual_interval_us
        };
        
        // Update statistics
        self.stats.update(processing_time_us, jitter_us, self.constraints.deadline_us);
        
        // Adaptive adjustments based on performance
        self.update_adaptive_params(processing_time_us, jitter_us);
        
        self.current_time_us = end_time_us;
        
        (processing_time_us, jitter_us)
    }
    
    /// Update adaptive parameters based on performance
    fn update_adaptive_params(&mut self, processing_time_us: u64, jitter_us: u64) {
        let deadline_ratio = processing_time_us as f32 / self.constraints.deadline_us as f32;
        let jitter_ratio = jitter_us as f32 / self.constraints.max_jitter_us as f32;
        
        // If we're consistently missing deadlines, reduce complexity
        if deadline_ratio > 0.9 {
            self.adaptive_params.complexity_factor *= 0.95;
            self.adaptive_params.quality_factor *= 0.98;
        } else if deadline_ratio < 0.5 {
            // If we have headroom, can increase quality
            self.adaptive_params.complexity_factor = (self.adaptive_params.complexity_factor * 1.02).min(1.0);
            self.adaptive_params.quality_factor = (self.adaptive_params.quality_factor * 1.01).min(1.0);
        }
        
        // Jitter adaptation
        if jitter_ratio > 0.8 {
            self.adaptive_params.buffer_size_factor *= 1.1;
        } else if jitter_ratio < 0.2 {
            self.adaptive_params.buffer_size_factor = (self.adaptive_params.buffer_size_factor * 0.95).max(0.5);
        }
        
        // Clamp adaptive parameters to reasonable ranges
        self.adaptive_params.complexity_factor = self.adaptive_params.complexity_factor.clamp(0.1, 1.0);
        self.adaptive_params.quality_factor = self.adaptive_params.quality_factor.clamp(0.5, 1.0);
        self.adaptive_params.buffer_size_factor = self.adaptive_params.buffer_size_factor.clamp(0.5, 2.0);
    }
    
    /// Check if current frame is within deadline
    pub fn is_within_deadline(&self) -> bool {
        let elapsed_us = get_current_time_us().saturating_sub(self.processing_start_us);
        elapsed_us <= self.constraints.deadline_us
    }
    
    /// Get remaining time until deadline
    pub fn remaining_time_us(&self) -> u64 {
        let elapsed_us = get_current_time_us().saturating_sub(self.processing_start_us);
        self.constraints.deadline_us.saturating_sub(elapsed_us)
    }
    
    /// Get current processing time
    pub fn elapsed_processing_time_us(&self) -> u64 {
        get_current_time_us().saturating_sub(self.processing_start_us)
    }
    
    /// Should use degraded processing mode
    pub fn should_degrade_quality(&self) -> bool {
        let elapsed_us = self.elapsed_processing_time_us();
        let deadline_pressure = elapsed_us as f32 / self.constraints.deadline_us as f32;
        deadline_pressure > 0.7 // Start degrading at 70% of deadline
    }
    
    /// Get recommended processing complexity
    pub fn recommended_complexity(&self) -> f32 {
        self.adaptive_params.complexity_factor
    }
    
    /// Get performance health indicator (0.0-1.0)
    pub fn performance_health(&self) -> f32 {
        let success_rate = self.stats.success_rate() as f32;
        let latency_health = 1.0 - (self.stats.avg_processing_time_us as f32 / self.constraints.deadline_us as f32).min(1.0);
        let jitter_health = 1.0 - (self.stats.avg_jitter_us as f32 / self.constraints.max_jitter_us as f32).min(1.0);
        
        (success_rate + latency_health + jitter_health) / 3.0
    }
}

/// Get current high-resolution timestamp in microseconds
pub fn get_current_time_us() -> u64 {
    #[cfg(feature = "std")]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64
    }
    
    #[cfg(not(feature = "std"))]
    {
        // Simple counter for embedded systems - replace with hardware timer
        static mut COUNTER: u64 = 0;
        unsafe {
            COUNTER += 100; // Increment by 100µs
            COUNTER
        }
    }
}

/// High-precision sleep function for real-time scheduling
pub fn precise_sleep_us(duration_us: u64) {
    #[cfg(feature = "std")]
    {
        use std::thread;
        use std::time::Duration;
        thread::sleep(Duration::from_micros(duration_us));
    }
    
    #[cfg(not(feature = "std"))]
    {
        // Busy wait for embedded systems - replace with hardware timer
        let start = get_current_time_us();
        while get_current_time_us().saturating_sub(start) < duration_us {
            // Busy wait - in real implementation, use WFI or similar
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_realtime_constraints_validation() {
        let constraints = RealTimeConstraints::default();
        assert!(constraints.validate().is_ok());
        
        let invalid = RealTimeConstraints {
            max_latency_us: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }
    
    #[test]
    fn test_realtime_constraints_frame_size() {
        let constraints = RealTimeConstraints {
            max_latency_us: 1000,
            sample_rate: 16000,
            ..Default::default()
        };
        
        let frame_size = constraints.calculate_frame_size();
        assert!(frame_size > 0);
        assert!(frame_size <= 16); // Should be reasonable for 1ms at 16kHz
    }
    
    #[test]
    fn test_realtime_stats_update() {
        let mut stats = RealTimeStats::default();
        
        // Simulate successful processing
        stats.update(500, 50, 1000);
        assert_eq!(stats.frames_processed, 1);
        assert_eq!(stats.frames_on_time, 1);
        assert_eq!(stats.frames_late, 0);
        
        // Simulate late processing
        stats.update(1500, 100, 1000);
        assert_eq!(stats.frames_processed, 2);
        assert_eq!(stats.frames_on_time, 1);
        assert_eq!(stats.frames_late, 1);
        
        assert_eq!(stats.success_rate(), 0.5);
    }
    
    #[test]
    fn test_realtime_context_deadline_checking() {
        let constraints = RealTimeConstraints {
            deadline_us: 1000,
            ..Default::default()
        };
        
        let mut context = RealTimeContext::new(constraints).unwrap();
        context.start_frame();
        
        assert!(context.is_within_deadline());
        assert!(context.remaining_time_us() <= 1000);
    }
    
    #[test]
    fn test_adaptive_params_adjustment() {
        let constraints = RealTimeConstraints::default();
        let mut context = RealTimeContext::new(constraints).unwrap();
        
        // Simulate consistently missing deadlines
        for _ in 0..10 {
            context.start_frame();
            context.update_adaptive_params(1500, 200); // Over deadline
        }
        
        // Complexity factor should be reduced
        assert!(context.adaptive_params.complexity_factor < 1.0);
        assert!(context.adaptive_params.quality_factor < 1.0);
    }
    
    #[test]
    fn test_performance_health() {
        let constraints = RealTimeConstraints::default();
        let mut context = RealTimeContext::new(constraints).unwrap();
        
        // Simulate good performance
        for _ in 0..10 {
            context.stats.update(300, 30, 1000); // Well under deadline
        }
        
        let health = context.performance_health();
        assert!(health > 0.8); // Should indicate good health
    }
}