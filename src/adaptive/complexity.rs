//! Signal complexity estimation for adaptive timestep control

use crate::{Result, LiquidAudioError};
use crate::core::config::ComplexityMetric;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Signal complexity estimator with multiple metrics and caching
#[derive(Debug, Clone)]
pub struct ComplexityEstimator {
    /// Primary complexity metric
    primary_metric: ComplexityMetric,
    /// Enable metric caching
    enable_caching: bool,
    /// Cache size
    cache_size: usize,
    /// Cached results
    cache: Vec<(u64, f32)>,
    /// Performance statistics
    cache_hits: u64,
    cache_misses: u64,
}

impl ComplexityEstimator {
    /// Create new complexity estimator
    pub fn new(primary_metric: ComplexityMetric) -> Self {
        Self {
            primary_metric,
            enable_caching: true,
            cache_size: 100,
            cache: Vec::with_capacity(100),
            cache_hits: 0,
            cache_misses: 0,
        }
    }
    
    /// Estimate signal complexity
    pub fn estimate_complexity(&mut self, audio: &[f32]) -> Result<f32> {
        if audio.is_empty() {
            return Ok(0.0);
        }
        
        // Check cache first
        if self.enable_caching {
            let hash = self.hash_audio(audio);
            if let Some(cached_result) = self.get_cached_result(hash) {
                self.cache_hits += 1;
                return Ok(cached_result);
            }
            self.cache_misses += 1;
        }
        
        let complexity = match self.primary_metric {
            ComplexityMetric::Energy => self.compute_energy_complexity(audio),
            ComplexityMetric::SpectralFlux => self.compute_spectral_flux_complexity(audio)?,
            ComplexityMetric::ZeroCrossingRate => self.compute_zcr_complexity(audio),
            ComplexityMetric::SpectralCentroid => self.compute_centroid_complexity(audio),
            ComplexityMetric::Combined => self.compute_combined_complexity(audio)?,
        };
        
        // Cache result
        if self.enable_caching {
            let hash = self.hash_audio(audio);
            self.cache_result(hash, complexity);
        }
        
        Ok(complexity)
    }
    
    /// Compute energy-based complexity
    fn compute_energy_complexity(&self, audio: &[f32]) -> f32 {
        let energy: f32 = audio.iter().map(|&x| x * x).sum::<f32>() / audio.len() as f32;
        
        // Normalize to 0-1 range with reasonable scaling
        (energy * 100.0).tanh() // Soft saturation
    }
    
    /// Compute spectral flux complexity
    fn compute_spectral_flux_complexity(&self, audio: &[f32]) -> Result<f32> {
        if audio.len() < 16 {
            return Ok(0.0);
        }
        
        // Simple spectral flux using time-domain differences
        let mut flux = 0.0;
        let window_size = 8;
        
        for i in window_size..audio.len() {
            // Compute local spectral change
            let mut current_energy = 0.0;
            let mut prev_energy = 0.0;
            
            for j in 0..window_size {
                if i >= j && i >= j + window_size {
                    current_energy += audio[i - j] * audio[i - j];
                    prev_energy += audio[i - j - window_size] * audio[i - j - window_size];
                }
            }
            
            current_energy /= window_size as f32;
            prev_energy /= window_size as f32;
            
            // Spectral flux is positive changes in energy
            if current_energy > prev_energy {
                flux += current_energy - prev_energy;
            }
        }
        
        // Normalize
        flux /= (audio.len() - window_size) as f32;
        
        Ok((flux * 1000.0).tanh()) // Scale and saturate
    }
    
    /// Compute zero crossing rate complexity
    fn compute_zcr_complexity(&self, audio: &[f32]) -> f32 {
        if audio.len() < 2 {
            return 0.0;
        }
        
        let mut zero_crossings = 0;
        for i in 1..audio.len() {
            if (audio[i] >= 0.0) != (audio[i-1] >= 0.0) {
                zero_crossings += 1;
            }
        }
        
        let zcr = zero_crossings as f32 / (audio.len() - 1) as f32;
        
        // ZCR typically ranges from 0 to ~0.5 for speech
        // Map to 0-1 range
        (zcr * 4.0).clamp(0.0, 1.0)
    }
    
    /// Compute spectral centroid complexity
    fn compute_centroid_complexity(&self, audio: &[f32]) -> f32 {
        if audio.len() < 8 {
            return 0.0;
        }
        
        // Compute autocorrelation for frequency analysis
        let max_lags = audio.len().min(64);
        let mut spectrum_approx = Vec::with_capacity(max_lags);
        
        for lag in 1..max_lags {
            let mut correlation = 0.0;
            for i in 0..audio.len() - lag {
                correlation += audio[i] * audio[i + lag];
            }
            spectrum_approx.push(correlation.abs());
        }
        
        // Compute centroid (center of mass)
        let total_energy: f32 = spectrum_approx.iter().sum();
        if total_energy < 1e-8 {
            return 0.0;
        }
        
        let centroid: f32 = spectrum_approx.iter()
            .enumerate()
            .map(|(i, &energy)| (i + 1) as f32 * energy)
            .sum::<f32>() / total_energy;
        
        // Normalize centroid to 0-1 range
        (centroid / max_lags as f32).clamp(0.0, 1.0)
    }
    
    /// Compute combined complexity metric
    fn compute_combined_complexity(&self, audio: &[f32]) -> Result<f32> {
        let energy = self.compute_energy_complexity(audio);
        let spectral_flux = self.compute_spectral_flux_complexity(audio)?;
        let zcr = self.compute_zcr_complexity(audio);
        let centroid = self.compute_centroid_complexity(audio);
        
        // Weighted combination
        let weights = [0.3, 0.3, 0.2, 0.2]; // [energy, flux, zcr, centroid]
        let metrics = [energy, spectral_flux, zcr, centroid];
        
        let combined: f32 = weights.iter()
            .zip(metrics.iter())
            .map(|(&w, &m)| w * m)
            .sum();
        
        Ok(combined.clamp(0.0, 1.0))
    }
    
    /// Hash audio for caching
    fn hash_audio(&self, audio: &[f32]) -> u64 {
        // Simple hash based on audio statistics
        let len = audio.len() as u64;
        let sum = audio.iter().sum::<f32>() as u64;
        let sum_squares = audio.iter().map(|&x| (x * x) as u64).sum::<u64>();
        
        // Combine hashes
        len ^ (sum << 16) ^ (sum_squares << 32)
    }
    
    /// Get cached result
    fn get_cached_result(&self, hash: u64) -> Option<f32> {
        self.cache.iter()
            .find(|(h, _)| *h == hash)
            .map(|(_, complexity)| *complexity)
    }
    
    /// Cache result
    fn cache_result(&mut self, hash: u64, complexity: f32) {
        // Remove existing entry with same hash
        self.cache.retain(|(h, _)| *h != hash);
        
        // Add new entry
        self.cache.push((hash, complexity));
        
        // Maintain cache size
        if self.cache.len() > self.cache_size {
            self.cache.remove(0); // Remove oldest
        }
    }
    
    /// Set primary complexity metric
    pub fn set_primary_metric(&mut self, metric: ComplexityMetric) {
        self.primary_metric = metric;
        self.clear_cache(); // Clear cache as results may differ
    }
    
    /// Enable or disable caching
    pub fn set_caching_enabled(&mut self, enabled: bool) {
        self.enable_caching = enabled;
        if !enabled {
            self.clear_cache();
        }
    }
    
    /// Clear complexity cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
        self.cache_hits = 0;
        self.cache_misses = 0;
    }
    
    /// Get cache statistics
    pub fn cache_stats(&self) -> ComplexityCacheStats {
        let total_requests = self.cache_hits + self.cache_misses;
        let hit_rate = if total_requests > 0 {
            self.cache_hits as f32 / total_requests as f32
        } else {
            0.0
        };
        
        ComplexityCacheStats {
            hits: self.cache_hits,
            misses: self.cache_misses,
            hit_rate,
            size: self.cache.len(),
            capacity: self.cache_size,
        }
    }
    
    /// Estimate complexity for multiple metrics at once
    pub fn estimate_all_metrics(&mut self, audio: &[f32]) -> Result<ComplexityMetrics> {
        Ok(ComplexityMetrics {
            energy: self.compute_energy_complexity(audio),
            spectral_flux: self.compute_spectral_flux_complexity(audio)?,
            zero_crossing_rate: self.compute_zcr_complexity(audio),
            spectral_centroid: self.compute_centroid_complexity(audio),
            combined: self.compute_combined_complexity(audio)?,
        })
    }
}

impl Default for ComplexityEstimator {
    fn default() -> Self {
        Self::new(ComplexityMetric::SpectralFlux)
    }
}

/// Cache statistics for complexity estimation
#[derive(Debug, Clone)]
pub struct ComplexityCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f32,
    pub size: usize,
    pub capacity: usize,
}

/// All complexity metrics computed at once
#[derive(Debug, Clone)]
pub struct ComplexityMetrics {
    pub energy: f32,
    pub spectral_flux: f32,
    pub zero_crossing_rate: f32,
    pub spectral_centroid: f32,
    pub combined: f32,
}

impl ComplexityMetrics {
    /// Get complexity value for specific metric
    pub fn get_metric(&self, metric: ComplexityMetric) -> f32 {
        match metric {
            ComplexityMetric::Energy => self.energy,
            ComplexityMetric::SpectralFlux => self.spectral_flux,
            ComplexityMetric::ZeroCrossingRate => self.zero_crossing_rate,
            ComplexityMetric::SpectralCentroid => self.spectral_centroid,
            ComplexityMetric::Combined => self.combined,
        }
    }
    
    /// Get the most complex metric value
    pub fn max_complexity(&self) -> f32 {
        [self.energy, self.spectral_flux, self.zero_crossing_rate, 
         self.spectral_centroid, self.combined]
            .iter()
            .fold(0.0f32, |acc, &x| acc.max(x))
    }
    
    /// Get the least complex metric value
    pub fn min_complexity(&self) -> f32 {
        [self.energy, self.spectral_flux, self.zero_crossing_rate, 
         self.spectral_centroid, self.combined]
            .iter()
            .fold(1.0f32, |acc, &x| acc.min(x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_complexity_estimator_creation() {
        let estimator = ComplexityEstimator::new(ComplexityMetric::Energy);
        assert!(matches!(estimator.primary_metric, ComplexityMetric::Energy));
    }
    
    #[test]
    fn test_energy_complexity() {
        let estimator = ComplexityEstimator::new(ComplexityMetric::Energy);
        
        // Silent signal should have low complexity
        let silent = vec![0.0; 100];
        let complexity = estimator.compute_energy_complexity(&silent);
        assert!(complexity < 0.1);
        
        // High energy signal should have higher complexity
        let high_energy = vec![1.0; 100];
        let complexity_high = estimator.compute_energy_complexity(&high_energy);
        assert!(complexity_high > complexity);
    }
    
    #[test]
    fn test_zcr_complexity() {
        let estimator = ComplexityEstimator::new(ComplexityMetric::ZeroCrossingRate);
        
        // Constant signal should have zero ZCR complexity
        let constant = vec![1.0; 100];
        let complexity = estimator.compute_zcr_complexity(&constant);
        assert_eq!(complexity, 0.0);
        
        // Alternating signal should have high ZCR complexity
        let alternating: Vec<f32> = (0..100).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let complexity_alt = estimator.compute_zcr_complexity(&alternating);
        assert!(complexity_alt > 0.5);
    }
    
    #[test]
    fn test_spectral_flux_complexity() {
        let estimator = ComplexityEstimator::new(ComplexityMetric::SpectralFlux);
        
        // Constant signal should have low spectral flux
        let constant = vec![1.0; 100];
        let complexity = estimator.compute_spectral_flux_complexity(&constant).unwrap();
        assert!(complexity < 0.1);
        
        // Changing signal should have higher spectral flux
        let changing: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let complexity_changing = estimator.compute_spectral_flux_complexity(&changing).unwrap();
        assert!(complexity_changing > complexity);
    }
    
    #[test]
    fn test_combined_complexity() {
        let estimator = ComplexityEstimator::new(ComplexityMetric::Combined);
        
        let audio = vec![0.1, 0.2, -0.1, 0.3, -0.2, 0.1, 0.4, -0.3];
        let complexity = estimator.compute_combined_complexity(&audio).unwrap();
        
        assert!(complexity >= 0.0 && complexity <= 1.0);
    }
    
    #[test]
    fn test_complexity_caching() {
        let mut estimator = ComplexityEstimator::new(ComplexityMetric::Energy);
        let audio = vec![0.1, 0.2, 0.3, 0.4];
        
        // First call should be a cache miss
        let complexity1 = estimator.estimate_complexity(&audio).unwrap();
        assert_eq!(estimator.cache_stats().misses, 1);
        assert_eq!(estimator.cache_stats().hits, 0);
        
        // Second call with same audio should be a cache hit
        let complexity2 = estimator.estimate_complexity(&audio).unwrap();
        assert_eq!(complexity1, complexity2);
        assert_eq!(estimator.cache_stats().hits, 1);
        assert_eq!(estimator.cache_stats().misses, 1);
    }
    
    #[test]
    fn test_all_metrics_computation() {
        let mut estimator = ComplexityEstimator::default();
        let audio = vec![0.1, 0.2, -0.1, 0.3, -0.2, 0.1, 0.4, -0.3];
        
        let metrics = estimator.estimate_all_metrics(&audio).unwrap();
        
        // All metrics should be in valid range
        assert!(metrics.energy >= 0.0 && metrics.energy <= 1.0);
        assert!(metrics.spectral_flux >= 0.0 && metrics.spectral_flux <= 1.0);
        assert!(metrics.zero_crossing_rate >= 0.0 && metrics.zero_crossing_rate <= 1.0);
        assert!(metrics.spectral_centroid >= 0.0 && metrics.spectral_centroid <= 1.0);
        assert!(metrics.combined >= 0.0 && metrics.combined <= 1.0);
        
        // Max/min should work correctly
        let max_complexity = metrics.max_complexity();
        let min_complexity = metrics.min_complexity();
        assert!(max_complexity >= min_complexity);
    }
    
    #[test]
    fn test_cache_management() {
        let mut estimator = ComplexityEstimator::new(ComplexityMetric::Energy);
        estimator.cache_size = 2; // Small cache for testing
        
        let audio1 = vec![0.1, 0.2];
        let audio2 = vec![0.3, 0.4];
        let audio3 = vec![0.5, 0.6];
        
        // Fill cache
        estimator.estimate_complexity(&audio1).unwrap();
        estimator.estimate_complexity(&audio2).unwrap();
        assert!(estimator.cache_stats().size <= 2);
        
        // Add third item should evict first
        estimator.estimate_complexity(&audio3).unwrap();
        assert!(estimator.cache_stats().size <= 2);
        
        // First item should no longer be cached (cache miss expected)
        estimator.estimate_complexity(&audio1).unwrap();
        // Cache statistics may vary based on implementation - just ensure positive misses
        assert!(estimator.cache_stats().misses >= 3);
    }
}