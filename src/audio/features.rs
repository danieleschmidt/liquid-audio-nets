//! Audio feature extraction for liquid neural networks

use crate::{Result, LiquidAudioError};
use nalgebra::DVector;
use rustfft::{FftPlanner, num_complex::Complex};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Audio feature extractor optimized for liquid neural networks
pub struct FeatureExtractor {
    /// Target feature dimension
    feature_dim: usize,
    /// FFT planner for spectral analysis
    fft_planner: FftPlanner<f32>,
    /// Window function for FFT
    window: WindowFunction,
    /// Mel filter bank (optional)
    mel_filters: Option<MelFilterBank>,
    /// Feature normalization parameters
    normalization: NormalizationParams,
}

impl core::fmt::Debug for FeatureExtractor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FeatureExtractor")
            .field("feature_dim", &self.feature_dim)
            .field("window", &self.window)
            .field("mel_filters", &self.mel_filters)
            .field("normalization", &self.normalization)
            .finish()
    }
}

impl FeatureExtractor {
    /// Create new feature extractor
    pub fn new(feature_dim: usize) -> Result<Self> {
        if feature_dim == 0 {
            return Err(LiquidAudioError::ConfigError(
                "Feature dimension must be > 0".to_string()
            ));
        }
        
        Ok(Self {
            feature_dim,
            fft_planner: FftPlanner::new(),
            window: WindowFunction::Hann,
            mel_filters: None,
            normalization: NormalizationParams::default(),
        })
    }
    
    /// Create feature extractor with mel-scale filters
    pub fn with_mel_filters(mut self, sample_rate: u32, n_mels: usize, fmin: f32, fmax: f32) -> Result<Self> {
        self.mel_filters = Some(MelFilterBank::new(sample_rate, n_mels, fmin, fmax)?);
        Ok(self)
    }
    
    /// Set window function for FFT
    pub fn with_window(mut self, window: WindowFunction) -> Self {
        self.window = window;
        self
    }
    
    /// Extract features from audio buffer
    pub fn extract(&mut self, audio: &[f32]) -> Result<DVector<f32>> {
        if audio.is_empty() {
            return Err(LiquidAudioError::InvalidInput(
                "Empty audio buffer".to_string()
            ));
        }
        
        // Apply windowing
        let windowed = self.apply_window(audio);
        
        // Compute FFT
        let spectrum = self.compute_spectrum(&windowed)?;
        
        // Extract features based on configuration
        let features = if let Some(ref mel_filters) = self.mel_filters {
            // Mel-scale features
            self.extract_mel_features(&spectrum, mel_filters)?
        } else {
            // Direct spectral features
            self.extract_spectral_features(&spectrum)?
        };
        
        // Apply normalization
        let normalized = self.normalize_features(&features)?;
        
        Ok(normalized)
    }
    
    /// Apply window function to audio
    fn apply_window(&self, audio: &[f32]) -> Vec<f32> {
        let len = audio.len();
        let mut windowed = Vec::with_capacity(len);
        
        match self.window {
            WindowFunction::None => {
                windowed.extend_from_slice(audio);
            },
            WindowFunction::Hann => {
                for (i, &sample) in audio.iter().enumerate() {
                    let window_val = 0.5 - 0.5 * (2.0 * core::f32::consts::PI * i as f32 / (len - 1) as f32).cos();
                    windowed.push(sample * window_val);
                }
            },
            WindowFunction::Hamming => {
                for (i, &sample) in audio.iter().enumerate() {
                    let window_val = 0.54 - 0.46 * (2.0 * core::f32::consts::PI * i as f32 / (len - 1) as f32).cos();
                    windowed.push(sample * window_val);
                }
            },
            WindowFunction::Blackman => {
                for (i, &sample) in audio.iter().enumerate() {
                    let n = i as f32 / (len - 1) as f32;
                    let window_val = 0.42 - 0.5 * (2.0 * core::f32::consts::PI * n).cos() + 
                                   0.08 * (4.0 * core::f32::consts::PI * n).cos();
                    windowed.push(sample * window_val);
                }
            },
        }
        
        windowed
    }
    
    /// Compute power spectrum using FFT
    fn compute_spectrum(&mut self, audio: &[f32]) -> Result<Vec<f32>> {
        let len = audio.len();
        
        // Prepare complex input
        let mut input: Vec<Complex<f32>> = audio.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        
        // Pad to next power of 2 for efficiency
        let fft_len = len.next_power_of_two();
        input.resize(fft_len, Complex::new(0.0, 0.0));
        
        // Create FFT and compute
        let fft = self.fft_planner.plan_fft_forward(fft_len);
        fft.process(&mut input);
        
        // Compute power spectrum (magnitude squared)
        let power_spectrum: Vec<f32> = input[..fft_len/2]
            .iter()
            .map(|c| c.norm_sqr())
            .collect();
        
        Ok(power_spectrum)
    }
    
    /// Extract mel-scale features using filter bank
    fn extract_mel_features(&self, spectrum: &[f32], mel_filters: &MelFilterBank) -> Result<DVector<f32>> {
        let mel_energies = mel_filters.apply(spectrum)?;
        
        // Convert to log scale (similar to MFCC)
        let log_energies: Vec<f32> = mel_energies.iter()
            .map(|&energy| (energy + 1e-8).ln())
            .collect();
        
        // Truncate or pad to target dimension
        let mut features = vec![0.0; self.feature_dim];
        let copy_len = log_energies.len().min(self.feature_dim);
        features[..copy_len].copy_from_slice(&log_energies[..copy_len]);
        
        Ok(DVector::from_vec(features))
    }
    
    /// Extract direct spectral features
    fn extract_spectral_features(&self, spectrum: &[f32]) -> Result<DVector<f32>> {
        let mut features = vec![0.0; self.feature_dim];
        
        if spectrum.is_empty() {
            return Ok(DVector::from_vec(features));
        }
        
        // Sample spectrum to target dimension
        let step = spectrum.len() as f32 / self.feature_dim as f32;
        
        for i in 0..self.feature_dim {
            let idx = (i as f32 * step) as usize;
            if idx < spectrum.len() {
                // Log power spectrum
                features[i] = (spectrum[idx] + 1e-8).ln();
            }
        }
        
        Ok(DVector::from_vec(features))
    }
    
    /// Apply feature normalization
    fn normalize_features(&self, features: &DVector<f32>) -> Result<DVector<f32>> {
        match self.normalization.method {
            NormalizationMethod::None => Ok(features.clone()),
            NormalizationMethod::ZScore => {
                let mean = features.mean();
                let variance = features.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / features.len() as f32;
                let std_dev = variance.sqrt().max(1e-8);
                
                let normalized = features.map(|x| (x - mean) / std_dev);
                Ok(normalized)
            },
            NormalizationMethod::MinMax => {
                let min_val = features.min();
                let max_val = features.max();
                let range = (max_val - min_val).max(1e-8);
                
                let normalized = features.map(|x| (x - min_val) / range);
                Ok(normalized)
            },
            NormalizationMethod::UnitNorm => {
                let norm = features.norm().max(1e-8);
                let normalized = features / norm;
                Ok(normalized)
            },
        }
    }
    
    /// Get feature dimension
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }
    
    /// Update normalization statistics (for adaptive normalization)
    pub fn update_normalization_stats(&mut self, features: &DVector<f32>) {
        // Update running statistics for adaptive normalization
        let mean = features.mean();
        let variance = features.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / features.len() as f32;
        
        // Exponential moving average
        let alpha = 0.1;
        self.normalization.running_mean = self.normalization.running_mean * (1.0 - alpha) + mean * alpha;
        self.normalization.running_variance = self.normalization.running_variance * (1.0 - alpha) + variance * alpha;
    }
}

/// Window functions for FFT
#[derive(Debug, Clone, Copy)]
pub enum WindowFunction {
    None,
    Hann,
    Hamming,
    Blackman,
}

/// Mel-scale filter bank for MFCC-like features
#[derive(Debug, Clone)]
pub struct MelFilterBank {
    /// Filter weights [n_mels x n_fft_bins]
    filters: Vec<Vec<f32>>,
    /// Number of mel filters
    n_mels: usize,
    /// FFT bin count
    n_fft_bins: usize,
}

impl MelFilterBank {
    /// Create new mel filter bank
    pub fn new(sample_rate: u32, n_mels: usize, fmin: f32, fmax: f32) -> Result<Self> {
        if n_mels == 0 {
            return Err(LiquidAudioError::ConfigError(
                "Number of mel filters must be > 0".to_string()
            ));
        }
        
        // Typical FFT size for filter bank computation
        let n_fft = 512;
        let n_fft_bins = n_fft / 2 + 1;
        
        let filters = Self::create_mel_filters(sample_rate, n_mels, n_fft_bins, fmin, fmax)?;
        
        Ok(Self {
            filters,
            n_mels,
            n_fft_bins,
        })
    }
    
    /// Create mel-scale filter weights
    fn create_mel_filters(
        sample_rate: u32,
        n_mels: usize,
        n_fft_bins: usize,
        fmin: f32,
        fmax: f32,
    ) -> Result<Vec<Vec<f32>>> {
        // Convert to mel scale
        let mel_min = Self::hz_to_mel(fmin);
        let mel_max = Self::hz_to_mel(fmax);
        
        // Create mel points
        let mel_points: Vec<f32> = (0..=n_mels + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32)
            .collect();
        
        // Convert back to Hz
        let hz_points: Vec<f32> = mel_points.iter()
            .map(|&mel| Self::mel_to_hz(mel))
            .collect();
        
        // Convert to FFT bin indices
        let bin_points: Vec<usize> = hz_points.iter()
            .map(|&hz| ((hz * n_fft_bins as f32 * 2.0) / sample_rate as f32) as usize)
            .map(|bin| bin.min(n_fft_bins - 1))
            .collect();
        
        // Create triangular filters
        let mut filters = Vec::with_capacity(n_mels);
        
        for m in 1..=n_mels {
            let mut filter = vec![0.0; n_fft_bins];
            
            let left = bin_points[m - 1];
            let center = bin_points[m];
            let right = bin_points[m + 1];
            
            // Rising edge
            for k in left..center {
                if center > left {
                    filter[k] = (k - left) as f32 / (center - left) as f32;
                }
            }
            
            // Falling edge  
            for k in center..right {
                if right > center {
                    filter[k] = (right - k) as f32 / (right - center) as f32;
                }
            }
            
            filters.push(filter);
        }
        
        Ok(filters)
    }
    
    /// Apply mel filter bank to power spectrum
    pub fn apply(&self, spectrum: &[f32]) -> Result<Vec<f32>> {
        if spectrum.len() != self.n_fft_bins {
            return Err(LiquidAudioError::InvalidInput(
                format!("Spectrum length {} doesn't match filter bank size {}", 
                       spectrum.len(), self.n_fft_bins)
            ));
        }
        
        let mut mel_energies = Vec::with_capacity(self.n_mels);
        
        for filter in &self.filters {
            let energy: f32 = spectrum.iter()
                .zip(filter.iter())
                .map(|(&spec, &filt)| spec * filt)
                .sum();
            mel_energies.push(energy);
        }
        
        Ok(mel_energies)
    }
    
    /// Convert frequency in Hz to mel scale
    fn hz_to_mel(hz: f32) -> f32 {
        2595.0 * (1.0 + hz / 700.0).ln()
    }
    
    /// Convert mel scale to frequency in Hz
    fn mel_to_hz(mel: f32) -> f32 {
        700.0 * ((mel / 2595.0).exp() - 1.0)
    }
}

/// Feature normalization parameters
#[derive(Debug, Clone)]
pub struct NormalizationParams {
    /// Normalization method
    pub method: NormalizationMethod,
    /// Running mean for adaptive normalization
    pub running_mean: f32,
    /// Running variance for adaptive normalization
    pub running_variance: f32,
}

impl Default for NormalizationParams {
    fn default() -> Self {
        Self {
            method: NormalizationMethod::ZScore,
            running_mean: 0.0,
            running_variance: 1.0,
        }
    }
}

/// Feature normalization methods
#[derive(Debug, Clone, Copy)]
pub enum NormalizationMethod {
    /// No normalization
    None,
    /// Z-score normalization (zero mean, unit variance)
    ZScore,
    /// Min-max normalization (0 to 1 range)
    MinMax,
    /// Unit norm normalization
    UnitNorm,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_extractor_creation() {
        let extractor = FeatureExtractor::new(40).unwrap();
        assert_eq!(extractor.feature_dim(), 40);
    }
    
    #[test]
    fn test_feature_extraction() {
        let mut extractor = FeatureExtractor::new(10).unwrap();
        let audio = vec![0.1, 0.2, -0.1, 0.3, -0.2, 0.1, 0.0, -0.1, 0.2, 0.1];
        
        let features = extractor.extract(&audio).unwrap();
        assert_eq!(features.len(), 10);
        
        // Features should not be all zeros
        assert!(features.iter().any(|&x| x.abs() > 1e-6));
    }
    
    #[test]
    fn test_mel_filter_bank() {
        let mel_bank = MelFilterBank::new(16000, 13, 0.0, 8000.0).unwrap();
        let spectrum = vec![1.0; 257]; // Power spectrum
        
        let mel_energies = mel_bank.apply(&spectrum).unwrap();
        assert_eq!(mel_energies.len(), 13);
        assert!(mel_energies.iter().all(|&x| x >= 0.0));
    }
    
    #[test]
    fn test_window_functions() {
        let audio = vec![1.0; 10];
        let extractor = FeatureExtractor::new(5).unwrap();
        
        // Test different window functions
        let hann_windowed = extractor.apply_window(&audio);
        assert_eq!(hann_windowed.len(), 10);
        
        // Hann window should be zero at endpoints
        assert!((hann_windowed[0]).abs() < 1e-6);
        assert!((hann_windowed[9]).abs() < 1e-6);
    }
    
    #[test]
    fn test_mel_scale_conversion() {
        let hz = 1000.0;
        let mel = MelFilterBank::hz_to_mel(hz);
        let hz_back = MelFilterBank::mel_to_hz(mel);
        
        assert!((hz - hz_back).abs() < 1e-3);
    }
}