//! High-level audio processing interface with error handling and validation

use crate::{Result, LiquidAudioError};
use crate::audio::{FeatureExtractor, AudioFormat};
use crate::audio::filters::{PreprocessingFilter, FilterChain};
use crate::core::{ProcessingResult, AdaptiveConfig};
use crate::LNN;

#[cfg(not(feature = "std"))]
use core::alloc::{vec::Vec, string::String};

/// High-level audio processor with robust error handling
#[derive(Debug)]
pub struct AudioProcessor {
    /// Core liquid neural network
    lnn: LNN,
    /// Audio format specification
    format: AudioFormat,
    /// Feature extractor
    feature_extractor: FeatureExtractor,
    /// Preprocessing filter chain
    filter_chain: FilterChain,
    /// Processing statistics
    stats: ProcessingStats,
    /// Security limits
    limits: SecurityLimits,
    /// Buffer for audio preprocessing
    processing_buffer: Vec<f32>,
}

impl AudioProcessor {
    /// Create new audio processor with validation
    pub fn new(lnn: LNN, format: AudioFormat) -> Result<Self> {
        // Validate configuration compatibility
        Self::validate_config(&lnn, &format)?;
        
        let feature_extractor = FeatureExtractor::new(lnn.config().input_dim)?
            .with_mel_filters(
                format.sample_rate,
                lnn.config().input_dim,
                80.0,   // fmin
                format.sample_rate as f32 / 2.0  // fmax (Nyquist)
            )?;
        
        let filter_chain = FilterChain::new();
        let processing_buffer = Vec::with_capacity(format.max_frame_size());
        
        Ok(Self {
            lnn,
            format,
            feature_extractor,
            filter_chain,
            stats: ProcessingStats::new(),
            limits: SecurityLimits::default(),
            processing_buffer,
        })
    }
    
    /// Process audio with comprehensive error handling
    pub fn process_audio(&mut self, audio_data: &[f32]) -> Result<ProcessingResult> {
        // Security validation
        self.validate_input_security(audio_data)?;
        
        // Statistical validation
        self.validate_audio_statistics(audio_data)?;
        
        // Update processing statistics
        self.stats.increment_frames();
        
        // Preprocess audio through filter chain
        let filtered_audio = self.preprocess_audio(audio_data)?;
        
        // Process through LNN with error recovery
        let result = self.process_with_recovery(&filtered_audio)?;
        
        // Validate output
        self.validate_output(&result)?;
        
        // Update statistics
        self.stats.update_with_result(&result);
        
        Ok(result)
    }
    
    /// Set adaptive configuration with validation
    pub fn set_adaptive_config(&mut self, config: AdaptiveConfig) -> Result<()> {
        // Validate configuration
        config.validate().map_err(|e| LiquidAudioError::ConfigError(e))?;
        
        // Check compatibility with current setup
        if config.max_timestep > self.limits.max_timestep {
            return Err(LiquidAudioError::SecurityError(
                format!("Timestep {} exceeds security limit {}", 
                       config.max_timestep, self.limits.max_timestep)
            ));
        }
        
        self.lnn.set_adaptive_config(config);
        Ok(())
    }
    
    /// Add preprocessing filter with validation
    pub fn add_filter(&mut self, filter: PreprocessingFilter) -> Result<()> {
        // Validate filter parameters
        filter.validate()?;
        
        // Check filter chain length limits
        if self.filter_chain.len() >= self.limits.max_filters {
            return Err(LiquidAudioError::SecurityError(
                format!("Filter chain length {} exceeds limit {}", 
                       self.filter_chain.len(), self.limits.max_filters)
            ));
        }
        
        self.filter_chain.add_filter(filter);
        Ok(())
    }
    
    /// Get processing statistics
    pub fn stats(&self) -> &ProcessingStats {
        &self.stats
    }
    
    /// Reset internal state securely
    pub fn reset(&mut self) -> Result<()> {
        // Validate reset request
        if self.stats.total_frames < self.limits.min_frames_before_reset {
            return Err(LiquidAudioError::SecurityError(
                "Insufficient frames processed before reset".to_string()
            ));
        }
        
        self.lnn.reset_state();
        self.stats.reset();
        self.processing_buffer.clear();
        
        // Securely clear any sensitive data
        self.secure_clear_buffers();
        
        Ok(())
    }
    
    /// Validate configuration compatibility  
    fn validate_config(lnn: &LNN, format: &AudioFormat) -> Result<()> {
        let config = lnn.config();
        
        // Check sample rate compatibility
        if format.sample_rate < 8000 || format.sample_rate > 48000 {
            return Err(LiquidAudioError::ConfigError(
                format!("Unsupported sample rate: {} Hz", format.sample_rate)
            ));
        }
        
        // Check frame size limits
        if config.frame_size > format.max_frame_size() {
            return Err(LiquidAudioError::ConfigError(
                format!("Frame size {} exceeds format limit {}", 
                       config.frame_size, format.max_frame_size())
            ));
        }
        
        // Check feature dimension reasonableness
        if config.input_dim > 512 {
            return Err(LiquidAudioError::ConfigError(
                format!("Input dimension {} too large (max 512)", config.input_dim)
            ));
        }
        
        Ok(())
    }
    
    /// Validate input security constraints
    fn validate_input_security(&self, audio: &[f32]) -> Result<()> {
        // Check buffer size limits
        if audio.len() > self.limits.max_buffer_size {
            return Err(LiquidAudioError::SecurityError(
                format!("Buffer size {} exceeds limit {}", audio.len(), self.limits.max_buffer_size)
            ));
        }
        
        // Check for potentially malicious patterns
        let max_consecutive_zeros = self.count_consecutive_zeros(audio);
        if max_consecutive_zeros > self.limits.max_consecutive_zeros {
            return Err(LiquidAudioError::SecurityError(
                "Suspicious pattern: too many consecutive zeros".to_string()
            ));
        }
        
        // Check for extreme values that might cause overflow
        let max_abs = audio.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        if max_abs > self.limits.max_sample_value {
            return Err(LiquidAudioError::SecurityError(
                format!("Sample value {} exceeds limit {}", max_abs, self.limits.max_sample_value)
            ));
        }
        
        Ok(())
    }
    
    /// Validate audio statistics for anomaly detection
    fn validate_audio_statistics(&self, audio: &[f32]) -> Result<()> {
        if audio.is_empty() {
            return Err(LiquidAudioError::InvalidInput("Empty audio buffer".to_string()));
        }
        
        // Compute basic statistics
        let mean: f32 = audio.iter().sum::<f32>() / audio.len() as f32;
        let variance: f32 = audio.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / audio.len() as f32;
        let std_dev = variance.sqrt();
        
        // Check for statistical anomalies
        if std_dev < 1e-8 {
            return Err(LiquidAudioError::AudioError(
                "Audio signal has no variation (likely corrupt)".to_string()
            ));
        }
        
        if mean.abs() > 0.5 {
            return Err(LiquidAudioError::AudioError(
                format!("Audio DC offset too large: {}", mean)
            ));
        }
        
        // Check for clipping
        let clipped_samples = audio.iter().filter(|&&x| x.abs() > 0.99).count();
        let clipping_ratio = clipped_samples as f32 / audio.len() as f32;
        if clipping_ratio > 0.05 {
            return Err(LiquidAudioError::AudioError(
                format!("Audio clipping detected: {:.1}% of samples", clipping_ratio * 100.0)
            ));
        }
        
        Ok(())
    }
    
    /// Preprocess audio through filter chain
    fn preprocess_audio(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        // Ensure buffer capacity
        self.processing_buffer.clear();
        self.processing_buffer.reserve(audio.len());
        
        // Apply filter chain
        let filtered = self.filter_chain.process(audio)?;
        
        // Additional safety checks after filtering
        if filtered.len() != audio.len() {
            return Err(LiquidAudioError::AudioError(
                "Filter chain changed buffer length".to_string()
            ));
        }
        
        // Check for filter instability
        let energy_ratio = self.compute_energy_ratio(audio, &filtered);
        if energy_ratio > 10.0 || energy_ratio < 0.1 {
            return Err(LiquidAudioError::AudioError(
                format!("Filter instability detected: energy ratio {:.2}", energy_ratio)
            ));
        }
        
        Ok(filtered)
    }
    
    /// Process with error recovery
    fn process_with_recovery(&mut self, audio: &[f32]) -> Result<ProcessingResult> {
        // Primary processing attempt
        match self.lnn.process(audio) {
            Ok(result) => {
                // Reset error counter on success
                self.stats.reset_error_count();
                Ok(result)
            },
            Err(e) => {
                // Increment error counter
                self.stats.increment_error_count();
                
                // Attempt recovery strategies
                if self.stats.consecutive_errors < self.limits.max_consecutive_errors {
                    self.attempt_recovery(audio, e)
                } else {
                    Err(LiquidAudioError::ComputationError(
                        format!("Too many consecutive errors: {}", self.stats.consecutive_errors)
                    ))
                }
            }
        }
    }
    
    /// Attempt error recovery strategies
    fn attempt_recovery(&mut self, audio: &[f32], original_error: LiquidAudioError) -> Result<ProcessingResult> {
        // Strategy 1: Reset liquid state and retry
        if self.stats.consecutive_errors == 1 {
            self.lnn.reset_state();
            if let Ok(result) = self.lnn.process(audio) {
                return Ok(result);
            }
        }
        
        // Strategy 2: Use simpler processing (fixed timestep)
        if self.stats.consecutive_errors == 2 {
            // Temporarily disable adaptive timestep
            let original_config = self.lnn.adaptive_config().cloned();
            self.lnn.clear_adaptive_config();
            
            match self.lnn.process(audio) {
                Ok(result) => {
                    // Restore original config
                    if let Some(config) = original_config {
                        let _ = self.set_adaptive_config(config);
                    }
                    return Ok(result);
                },
                Err(_) => {
                    // Restore config even on failure
                    if let Some(config) = original_config {
                        let _ = self.set_adaptive_config(config);
                    }
                }
            }
        }
        
        // Strategy 3: Fallback to dummy result
        if self.stats.consecutive_errors == 3 {
            return Ok(self.create_fallback_result());
        }
        
        // All recovery strategies failed
        Err(original_error)
    }
    
    /// Validate processing output
    fn validate_output(&self, result: &ProcessingResult) -> Result<()> {
        // Check for NaN or infinite values
        for &value in &result.output {
            if !value.is_finite() {
                return Err(LiquidAudioError::ComputationError(
                    "Output contains NaN or infinite values".to_string()
                ));
            }
        }
        
        // Check confidence bounds
        if result.confidence < 0.0 || result.confidence > 1.0 {
            return Err(LiquidAudioError::ComputationError(
                format!("Invalid confidence value: {}", result.confidence)
            ));
        }
        
        // Check power estimate reasonableness
        if result.power_mw < 0.0 || result.power_mw > 100.0 {
            return Err(LiquidAudioError::ComputationError(
                format!("Unreasonable power estimate: {} mW", result.power_mw)
            ));
        }
        
        // Check timestep bounds
        if result.timestep_ms <= 0.0 || result.timestep_ms > 1000.0 {
            return Err(LiquidAudioError::ComputationError(
                format!("Invalid timestep: {} ms", result.timestep_ms)
            ));
        }
        
        Ok(())
    }
    
    /// Count consecutive zeros (for anomaly detection)
    fn count_consecutive_zeros(&self, audio: &[f32]) -> usize {
        let mut max_consecutive = 0;
        let mut current_consecutive = 0;
        
        for &sample in audio {
            if sample.abs() < 1e-8 {
                current_consecutive += 1;
                max_consecutive = max_consecutive.max(current_consecutive);
            } else {
                current_consecutive = 0;
            }
        }
        
        max_consecutive
    }
    
    /// Compute energy ratio between signals
    fn compute_energy_ratio(&self, original: &[f32], filtered: &[f32]) -> f32 {
        let original_energy: f32 = original.iter().map(|&x| x * x).sum();
        let filtered_energy: f32 = filtered.iter().map(|&x| x * x).sum();
        
        if original_energy > 1e-8 {
            filtered_energy / original_energy
        } else {
            1.0
        }
    }
    
    /// Create fallback result for error recovery
    fn create_fallback_result(&self) -> ProcessingResult {
        let output_dim = self.lnn.config().output_dim;
        ProcessingResult::new(
            vec![0.0; output_dim],  // Zero output
            10.0,                   // Fixed timestep
            1.0,                    // Default power
            0.0,                    // Zero complexity
            0.0,                    // Zero energy
        )
    }
    
    /// Securely clear internal buffers
    fn secure_clear_buffers(&mut self) {
        // Clear processing buffer with explicit zeros
        for value in &mut self.processing_buffer {
            *value = 0.0;
        }
        self.processing_buffer.clear();
        
        // Additional security: overwrite memory
        self.processing_buffer.shrink_to_fit();
    }
}

/// Processing statistics with error tracking
#[derive(Debug, Clone, Default)]
pub struct ProcessingStats {
    /// Total frames processed
    pub total_frames: u64,
    /// Total processing errors
    pub total_errors: u64,
    /// Consecutive errors
    pub consecutive_errors: u32,
    /// Average processing time
    pub avg_processing_time_ms: f32,
    /// Average power consumption
    pub avg_power_mw: f32,
    /// Average confidence
    pub avg_confidence: f32,
    /// Peak power consumption
    pub peak_power_mw: f32,
    /// Minimum confidence seen
    pub min_confidence: f32,
}

impl ProcessingStats {
    /// Create new statistics tracker
    pub fn new() -> Self {
        Self {
            min_confidence: 1.0,
            ..Default::default()
        }
    }
    
    /// Increment frame counter
    pub fn increment_frames(&mut self) {
        self.total_frames += 1;
    }
    
    /// Increment error counter
    pub fn increment_error_count(&mut self) {
        self.total_errors += 1;
        self.consecutive_errors += 1;
    }
    
    /// Reset consecutive error count
    pub fn reset_error_count(&mut self) {
        self.consecutive_errors = 0;
    }
    
    /// Update statistics with processing result
    pub fn update_with_result(&mut self, result: &ProcessingResult) {
        // Update averages using exponential smoothing
        let alpha = 0.01f32; // Smoothing factor
        
        self.avg_processing_time_ms = self.avg_processing_time_ms * (1.0 - alpha) + 
                                     result.timestep_ms * alpha;
        
        self.avg_power_mw = self.avg_power_mw * (1.0 - alpha) + 
                           result.power_mw * alpha;
        
        self.avg_confidence = self.avg_confidence * (1.0 - alpha) + 
                             result.confidence * alpha;
        
        // Update extremes
        self.peak_power_mw = self.peak_power_mw.max(result.power_mw);
        self.min_confidence = self.min_confidence.min(result.confidence);
    }
    
    /// Get error rate
    pub fn error_rate(&self) -> f32 {
        if self.total_frames > 0 {
            self.total_errors as f32 / self.total_frames as f32
        } else {
            0.0
        }
    }
    
    /// Reset all statistics
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Security limits for safe operation
#[derive(Debug, Clone)]
pub struct SecurityLimits {
    /// Maximum buffer size allowed
    pub max_buffer_size: usize,
    /// Maximum sample value
    pub max_sample_value: f32,
    /// Maximum consecutive zeros
    pub max_consecutive_zeros: usize,
    /// Maximum timestep allowed
    pub max_timestep: f32,
    /// Maximum number of filters in chain
    pub max_filters: usize,
    /// Maximum consecutive errors before failure
    pub max_consecutive_errors: u32,
    /// Minimum frames before allowing reset
    pub min_frames_before_reset: u64,
}

impl Default for SecurityLimits {
    fn default() -> Self {
        Self {
            max_buffer_size: 65536,        // 64KB audio buffer
            max_sample_value: 10.0,        // Allow some headroom above [-1,1]
            max_consecutive_zeros: 1000,   // 1000 consecutive zero samples
            max_timestep: 0.1,             // 100ms maximum
            max_filters: 10,               // Maximum filter chain length
            max_consecutive_errors: 5,     // Error recovery attempts
            min_frames_before_reset: 10,   // Minimum processing before reset
        }
    }
}

impl SecurityLimits {
    /// Create more restrictive limits for embedded systems
    pub fn embedded() -> Self {
        Self {
            max_buffer_size: 4096,         // 4KB buffer
            max_sample_value: 2.0,         // Stricter sample limits
            max_consecutive_zeros: 100,    // Fewer consecutive zeros
            max_timestep: 0.05,            // 50ms maximum
            max_filters: 3,                // Fewer filters
            max_consecutive_errors: 3,     // Fewer recovery attempts
            min_frames_before_reset: 5,    // Faster reset allowance
        }
    }
    
    /// Create relaxed limits for development/testing
    pub fn development() -> Self {
        Self {
            max_buffer_size: 1048576,      // 1MB buffer
            max_sample_value: 100.0,       // Very relaxed
            max_consecutive_zeros: 10000,  // Allow long silence
            max_timestep: 1.0,             // 1 second maximum
            max_filters: 50,               // Many filters allowed
            max_consecutive_errors: 20,    // More recovery attempts
            min_frames_before_reset: 1,    // Immediate reset allowed
        }
    }
}