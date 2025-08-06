//! Audio preprocessing filters with security validation

use crate::{Result, LiquidAudioError};
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
use core::alloc::{vec::Vec, string::String};

/// Preprocessing filter for audio data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PreprocessingFilter {
    /// High-pass filter
    HighPass { cutoff_hz: f32, order: u8 },
    /// Low-pass filter
    LowPass { cutoff_hz: f32, order: u8 },
    /// Band-pass filter
    BandPass { low_hz: f32, high_hz: f32, order: u8 },
    /// Notch filter (band-stop)
    Notch { center_hz: f32, width_hz: f32, order: u8 },
    /// DC removal filter
    DCRemoval { time_constant: f32 },
    /// Automatic gain control
    AGC { target_level: f32, attack_time: f32, release_time: f32 },
    /// Pre-emphasis filter  
    PreEmphasis { coefficient: f32 },
    /// De-emphasis filter
    DeEmphasis { coefficient: f32 },
    /// Normalization
    Normalize { target_peak: f32 },
    /// Gate (silence threshold)
    Gate { threshold: f32, attack_time: f32, release_time: f32 },
}

impl PreprocessingFilter {
    /// Validate filter parameters for security
    pub fn validate(&self) -> Result<()> {
        match self {
            PreprocessingFilter::HighPass { cutoff_hz, order } => {
                Self::validate_frequency(*cutoff_hz)?;
                Self::validate_filter_order(*order)?;
            },
            PreprocessingFilter::LowPass { cutoff_hz, order } => {
                Self::validate_frequency(*cutoff_hz)?;
                Self::validate_filter_order(*order)?;
            },
            PreprocessingFilter::BandPass { low_hz, high_hz, order } => {
                Self::validate_frequency(*low_hz)?;
                Self::validate_frequency(*high_hz)?;
                Self::validate_filter_order(*order)?;
                if *low_hz >= *high_hz {
                    return Err(LiquidAudioError::ConfigError(
                        "Band-pass: low frequency must be < high frequency".to_string()
                    ));
                }
            },
            PreprocessingFilter::Notch { center_hz, width_hz, order } => {
                Self::validate_frequency(*center_hz)?;
                Self::validate_positive(*width_hz, "width")?;
                Self::validate_filter_order(*order)?;
            },
            PreprocessingFilter::DCRemoval { time_constant } => {
                Self::validate_positive(*time_constant, "time constant")?;
                if *time_constant > 10.0 {
                    return Err(LiquidAudioError::ConfigError(
                        "DC removal time constant too large".to_string()
                    ));
                }
            },
            PreprocessingFilter::AGC { target_level, attack_time, release_time } => {
                Self::validate_level(*target_level)?;
                Self::validate_positive(*attack_time, "attack time")?;
                Self::validate_positive(*release_time, "release time")?;
                if *attack_time > 10.0 || *release_time > 10.0 {
                    return Err(LiquidAudioError::ConfigError(
                        "AGC time constants too large".to_string()
                    ));
                }
            },
            PreprocessingFilter::PreEmphasis { coefficient } => {
                if coefficient.abs() >= 1.0 {
                    return Err(LiquidAudioError::ConfigError(
                        "Pre-emphasis coefficient must be < 1.0".to_string()
                    ));
                }
            },
            PreprocessingFilter::DeEmphasis { coefficient } => {
                if coefficient.abs() >= 1.0 {
                    return Err(LiquidAudioError::ConfigError(
                        "De-emphasis coefficient must be < 1.0".to_string()
                    ));
                }
            },
            PreprocessingFilter::Normalize { target_peak } => {
                Self::validate_positive(*target_peak, "target peak")?;
                if *target_peak > 10.0 {
                    return Err(LiquidAudioError::ConfigError(
                        "Normalization target too high".to_string()
                    ));
                }
            },
            PreprocessingFilter::Gate { threshold, attack_time, release_time } => {
                Self::validate_level(*threshold)?;
                Self::validate_positive(*attack_time, "attack time")?;
                Self::validate_positive(*release_time, "release time")?;
            },
        }
        Ok(())
    }
    
    /// Apply filter to audio data
    pub fn apply(&self, input: &[f32], sample_rate: u32, state: &mut FilterState) -> Result<Vec<f32>> {
        if input.is_empty() {
            return Ok(Vec::new());
        }
        
        // Security check: prevent excessive processing
        if input.len() > 1_000_000 {
            return Err(LiquidAudioError::SecurityError(
                "Input buffer too large for filtering".to_string()
            ));
        }
        
        match self {
            PreprocessingFilter::HighPass { cutoff_hz, order } => {
                self.apply_highpass(input, sample_rate, *cutoff_hz, *order, state)
            },
            PreprocessingFilter::LowPass { cutoff_hz, order } => {
                self.apply_lowpass(input, sample_rate, *cutoff_hz, *order, state)
            },
            PreprocessingFilter::BandPass { low_hz, high_hz, order } => {
                self.apply_bandpass(input, sample_rate, *low_hz, *high_hz, *order, state)
            },
            PreprocessingFilter::Notch { center_hz, width_hz, order } => {
                self.apply_notch(input, sample_rate, *center_hz, *width_hz, *order, state)
            },
            PreprocessingFilter::DCRemoval { time_constant } => {
                self.apply_dc_removal(input, sample_rate, *time_constant, state)
            },
            PreprocessingFilter::AGC { target_level, attack_time, release_time } => {
                self.apply_agc(input, sample_rate, *target_level, *attack_time, *release_time, state)
            },
            PreprocessingFilter::PreEmphasis { coefficient } => {
                self.apply_pre_emphasis(input, *coefficient, state)
            },
            PreprocessingFilter::DeEmphasis { coefficient } => {
                self.apply_de_emphasis(input, *coefficient, state)
            },
            PreprocessingFilter::Normalize { target_peak } => {
                self.apply_normalize(input, *target_peak)
            },
            PreprocessingFilter::Gate { threshold, attack_time, release_time } => {
                self.apply_gate(input, sample_rate, *threshold, *attack_time, *release_time, state)
            },
        }
    }
    
    /// Validation helpers
    fn validate_frequency(freq: f32) -> Result<()> {
        if freq <= 0.0 || freq > 100000.0 {
            return Err(LiquidAudioError::ConfigError(
                format!("Invalid frequency: {} Hz", freq)
            ));
        }
        Ok(())
    }
    
    fn validate_filter_order(order: u8) -> Result<()> {
        if order == 0 || order > 10 {
            return Err(LiquidAudioError::ConfigError(
                format!("Invalid filter order: {} (must be 1-10)", order)
            ));
        }
        Ok(())
    }
    
    fn validate_positive(value: f32, name: &str) -> Result<()> {
        if value <= 0.0 {
            return Err(LiquidAudioError::ConfigError(
                format!("Invalid {}: {} (must be > 0)", name, value)
            ));
        }
        Ok(())
    }
    
    fn validate_level(level: f32) -> Result<()> {
        if level < 0.0 || level > 1.0 {
            return Err(LiquidAudioError::ConfigError(
                format!("Invalid level: {} (must be 0-1)", level)
            ));
        }
        Ok(())
    }
    
    /// Filter implementations
    fn apply_highpass(&self, input: &[f32], sample_rate: u32, cutoff: f32, order: u8, state: &mut FilterState) -> Result<Vec<f32>> {
        // Simple first-order high-pass filter implementation
        let rc = 1.0 / (2.0 * core::f32::consts::PI * cutoff);
        let dt = 1.0 / sample_rate as f32;
        let alpha = rc / (rc + dt);
        
        let mut output = Vec::with_capacity(input.len());
        let mut prev_input = state.get_prev_input().unwrap_or(0.0);
        let mut prev_output = state.get_prev_output().unwrap_or(0.0);
        
        for &sample in input {
            let filtered = alpha * (prev_output + sample - prev_input);
            output.push(filtered);
            prev_input = sample;
            prev_output = filtered;
        }
        
        // Update state
        state.set_prev_input(prev_input);
        state.set_prev_output(prev_output);
        
        Ok(output)
    }
    
    fn apply_lowpass(&self, input: &[f32], sample_rate: u32, cutoff: f32, order: u8, state: &mut FilterState) -> Result<Vec<f32>> {
        // Simple first-order low-pass filter
        let rc = 1.0 / (2.0 * core::f32::consts::PI * cutoff);
        let dt = 1.0 / sample_rate as f32;
        let alpha = dt / (rc + dt);
        
        let mut output = Vec::with_capacity(input.len());
        let mut prev_output = state.get_prev_output().unwrap_or(0.0);
        
        for &sample in input {
            let filtered = prev_output + alpha * (sample - prev_output);
            output.push(filtered);
            prev_output = filtered;
        }
        
        state.set_prev_output(prev_output);
        Ok(output)
    }
    
    fn apply_bandpass(&self, input: &[f32], sample_rate: u32, low: f32, high: f32, order: u8, state: &mut FilterState) -> Result<Vec<f32>> {
        // Cascade high-pass and low-pass filters
        let mut temp_state = FilterState::new();
        let highpassed = self.apply_highpass(input, sample_rate, low, order, &mut temp_state)?;
        self.apply_lowpass(&highpassed, sample_rate, high, order, state)
    }
    
    fn apply_notch(&self, input: &[f32], sample_rate: u32, center: f32, width: f32, order: u8, state: &mut FilterState) -> Result<Vec<f32>> {
        // Simplified notch filter - subtract narrow bandpass from original
        let mut temp_state = FilterState::new();
        let low = center - width / 2.0;
        let high = center + width / 2.0;
        let bandpassed = self.apply_bandpass(input, sample_rate, low, high, order, &mut temp_state)?;
        
        let mut output = Vec::with_capacity(input.len());
        for (orig, band) in input.iter().zip(bandpassed.iter()) {
            output.push(orig - band * 0.5); // Attenuate rather than completely remove
        }
        
        Ok(output)
    }
    
    fn apply_dc_removal(&self, input: &[f32], sample_rate: u32, time_constant: f32, state: &mut FilterState) -> Result<Vec<f32>> {
        // DC blocking filter
        let alpha = 1.0 - 1.0 / (time_constant * sample_rate as f32);
        
        let mut output = Vec::with_capacity(input.len());
        let mut prev_input = state.get_prev_input().unwrap_or(0.0);
        let mut prev_output = state.get_prev_output().unwrap_or(0.0);
        
        for &sample in input {
            let filtered = alpha * (prev_output + sample - prev_input);
            output.push(filtered);
            prev_input = sample;
            prev_output = filtered;
        }
        
        state.set_prev_input(prev_input);
        state.set_prev_output(prev_output);
        Ok(output)
    }
    
    fn apply_agc(&self, input: &[f32], sample_rate: u32, target: f32, attack: f32, release: f32, state: &mut FilterState) -> Result<Vec<f32>> {
        let attack_coeff = (-1.0 / (attack * sample_rate as f32)).exp();
        let release_coeff = (-1.0 / (release * sample_rate as f32)).exp();
        
        let mut output = Vec::with_capacity(input.len());
        let mut envelope = state.get_envelope().unwrap_or(0.0);
        
        for &sample in input {
            let sample_level = sample.abs();
            
            // Update envelope
            if sample_level > envelope {
                envelope = envelope * attack_coeff + sample_level * (1.0 - attack_coeff);
            } else {
                envelope = envelope * release_coeff + sample_level * (1.0 - release_coeff);
            }
            
            // Apply gain
            let gain = if envelope > 1e-8 { target / envelope } else { 1.0 };
            let safe_gain = gain.clamp(0.1, 10.0); // Limit gain range for safety
            
            output.push(sample * safe_gain);
        }
        
        state.set_envelope(envelope);
        Ok(output)
    }
    
    fn apply_pre_emphasis(&self, input: &[f32], coefficient: f32, state: &mut FilterState) -> Result<Vec<f32>> {
        let mut output = Vec::with_capacity(input.len());
        let mut prev_sample = state.get_prev_input().unwrap_or(0.0);
        
        for &sample in input {
            let emphasized = sample - coefficient * prev_sample;
            output.push(emphasized);
            prev_sample = sample;
        }
        
        state.set_prev_input(prev_sample);
        Ok(output)
    }
    
    fn apply_de_emphasis(&self, input: &[f32], coefficient: f32, state: &mut FilterState) -> Result<Vec<f32>> {
        let mut output = Vec::with_capacity(input.len());
        let mut prev_output = state.get_prev_output().unwrap_or(0.0);
        
        for &sample in input {
            let deemphasized = sample + coefficient * prev_output;
            output.push(deemphasized);
            prev_output = deemphasized;
        }
        
        state.set_prev_output(prev_output);
        Ok(output)
    }
    
    fn apply_normalize(&self, input: &[f32], target_peak: f32) -> Result<Vec<f32>> {
        // Find peak
        let peak = input.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        
        if peak < 1e-8 {
            // Silent signal, return as-is
            return Ok(input.to_vec());
        }
        
        let scale = target_peak / peak;
        let safe_scale = scale.clamp(0.01, 100.0); // Safety limits
        
        let output = input.iter().map(|&x| x * safe_scale).collect();
        Ok(output)
    }
    
    fn apply_gate(&self, input: &[f32], sample_rate: u32, threshold: f32, attack: f32, release: f32, state: &mut FilterState) -> Result<Vec<f32>> {
        let attack_coeff = (-1.0 / (attack * sample_rate as f32)).exp();
        let release_coeff = (-1.0 / (release * sample_rate as f32)).exp();
        
        let mut output = Vec::with_capacity(input.len());
        let mut gate_gain = state.get_gate_gain().unwrap_or(1.0);
        
        for &sample in input {
            let sample_level = sample.abs();
            let should_gate = sample_level < threshold;
            
            let target_gain = if should_gate { 0.0 } else { 1.0 };
            
            // Smooth gain changes
            let coeff = if target_gain > gate_gain { attack_coeff } else { release_coeff };
            gate_gain = gate_gain * coeff + target_gain * (1.0 - coeff);
            
            output.push(sample * gate_gain);
        }
        
        state.set_gate_gain(gate_gain);
        Ok(output)
    }
}

/// Filter state for maintaining continuity between processing blocks
#[derive(Debug, Clone, Default)]
pub struct FilterState {
    prev_input: Option<f32>,
    prev_output: Option<f32>,
    envelope: Option<f32>,
    gate_gain: Option<f32>,
    delay_line: Vec<f32>,
}

impl FilterState {
    /// Create new filter state
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Getters and setters for state variables
    pub fn get_prev_input(&self) -> Option<f32> { self.prev_input }
    pub fn set_prev_input(&mut self, value: f32) { self.prev_input = Some(value); }
    
    pub fn get_prev_output(&self) -> Option<f32> { self.prev_output }
    pub fn set_prev_output(&mut self, value: f32) { self.prev_output = Some(value); }
    
    pub fn get_envelope(&self) -> Option<f32> { self.envelope }
    pub fn set_envelope(&mut self, value: f32) { self.envelope = Some(value); }
    
    pub fn get_gate_gain(&self) -> Option<f32> { self.gate_gain }
    pub fn set_gate_gain(&mut self, value: f32) { self.gate_gain = Some(value); }
    
    /// Reset all state
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

/// Filter chain for applying multiple filters in sequence
#[derive(Debug, Clone)]
pub struct FilterChain {
    filters: Vec<PreprocessingFilter>,
    states: Vec<FilterState>,
}

impl FilterChain {
    /// Create new empty filter chain
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
            states: Vec::new(),
        }
    }
    
    /// Add filter to chain
    pub fn add_filter(&mut self, filter: PreprocessingFilter) {
        self.filters.push(filter);
        self.states.push(FilterState::new());
    }
    
    /// Get number of filters in chain
    pub fn len(&self) -> usize {
        self.filters.len()
    }
    
    /// Check if chain is empty
    pub fn is_empty(&self) -> bool {
        self.filters.is_empty()
    }
    
    /// Process audio through entire filter chain
    pub fn process(&mut self, input: &[f32]) -> Result<Vec<f32>> {
        if self.filters.is_empty() {
            return Ok(input.to_vec());
        }
        
        // Default sample rate if not specified
        let sample_rate = 16000; // This should be configurable
        
        let mut current = input.to_vec();
        
        for (filter, state) in self.filters.iter().zip(self.states.iter_mut()) {
            current = filter.apply(&current, sample_rate, state)?;
        }
        
        Ok(current)
    }
    
    /// Clear all state
    pub fn reset(&mut self) {
        for state in &mut self.states {
            state.reset();
        }
    }
    
    /// Validate entire filter chain
    pub fn validate(&self) -> Result<()> {
        for filter in &self.filters {
            filter.validate()?;
        }
        Ok(())
    }
}

impl Default for FilterChain {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_filter_validation() {
        let valid_filter = PreprocessingFilter::HighPass { cutoff_hz: 100.0, order: 2 };
        assert!(valid_filter.validate().is_ok());
        
        let invalid_filter = PreprocessingFilter::HighPass { cutoff_hz: -100.0, order: 2 };
        assert!(invalid_filter.validate().is_err());
        
        let invalid_order = PreprocessingFilter::HighPass { cutoff_hz: 100.0, order: 0 };
        assert!(invalid_order.validate().is_err());
    }
    
    #[test]
    fn test_dc_removal() {
        let filter = PreprocessingFilter::DCRemoval { time_constant: 0.1 };
        let mut state = FilterState::new();
        
        // Signal with DC offset
        let input = vec![1.1, 1.2, 1.0, 1.3, 1.1];
        let output = filter.apply(&input, 16000, &mut state).unwrap();
        
        // Output should have reduced DC component
        let output_mean: f32 = output.iter().sum::<f32>() / output.len() as f32;
        let input_mean: f32 = input.iter().sum::<f32>() / input.len() as f32;
        
        assert!(output_mean.abs() < input_mean.abs());
    }
    
    #[test]
    fn test_normalize() {
        let filter = PreprocessingFilter::Normalize { target_peak: 0.5 };
        
        let input = vec![2.0, -1.0, 1.5, -2.0]; // Peak = 2.0
        let output = filter.apply(&input, 16000, &mut FilterState::new()).unwrap();
        
        let output_peak = output.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        assert!((output_peak - 0.5).abs() < 1e-6);
    }
    
    #[test]
    fn test_filter_chain() {
        let mut chain = FilterChain::new();
        chain.add_filter(PreprocessingFilter::DCRemoval { time_constant: 0.1 });
        chain.add_filter(PreprocessingFilter::Normalize { target_peak: 1.0 });
        
        let input = vec![1.1, 1.2, 1.0, 1.3, 1.1];
        let output = chain.process(&input).unwrap();
        
        assert_eq!(output.len(), input.len());
        assert!(chain.validate().is_ok());
    }
    
    #[test]
    fn test_pre_emphasis() {
        let filter = PreprocessingFilter::PreEmphasis { coefficient: 0.97 };
        let mut state = FilterState::new();
        
        let input = vec![1.0, 2.0, 1.5, 0.5];
        let output = filter.apply(&input, 16000, &mut state).unwrap();
        
        // First sample should be unchanged
        assert!((output[0] - 1.0).abs() < 1e-6);
        
        // Second sample should be: 2.0 - 0.97 * 1.0 = 1.03
        assert!((output[1] - 1.03).abs() < 1e-6);
    }
}