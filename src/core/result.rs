//! Processing result types for Liquid Neural Networks

use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Result of LNN processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    /// Output predictions/classifications
    pub output: Vec<f32>,
    /// Maximum confidence value
    pub confidence: f32,
    /// Timestep used for integration (ms)
    pub timestep_ms: f32,
    /// Estimated power consumption (mW)
    pub power_mw: f32,
    /// Signal complexity metric (0.0-1.0)
    pub complexity: f32,
    /// Liquid state energy
    pub liquid_energy: f32,
    /// Optional metadata for the processing result
    pub metadata: Option<String>,
}

impl ProcessingResult {
    /// Create new processing result
    pub fn new(
        output: Vec<f32>,
        timestep_ms: f32,
        power_mw: f32,
        complexity: f32,
        liquid_energy: f32,
    ) -> Self {
        let confidence = output.iter().fold(0.0f32, |acc, &x| acc.max(x));
        
        Self {
            output,
            confidence,
            timestep_ms,
            power_mw,
            complexity,
            liquid_energy,
            metadata: None,
        }
    }
    
    /// Get predicted class index (highest output)
    pub fn predicted_class(&self) -> Option<usize> {
        if self.output.is_empty() {
            return None;
        }
        
        let (max_idx, _) = self.output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal))?;
        
        Some(max_idx)
    }
    
    /// Check if confidence exceeds threshold
    pub fn is_confident(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
    
    /// Get top-k predictions with indices
    pub fn top_k(&self, k: usize) -> Vec<(usize, f32)> {
        let mut indexed_outputs: Vec<(usize, f32)> = self.output
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();
        
        // Sort by value in descending order
        indexed_outputs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(core::cmp::Ordering::Equal));
        
        // Take top k
        indexed_outputs.into_iter().take(k.min(self.output.len())).collect()
    }
    
    /// Calculate entropy of output distribution
    pub fn entropy(&self) -> f32 {
        let sum: f32 = self.output.iter().sum();
        if sum <= 0.0 {
            return 0.0;
        }
        
        let mut entropy = 0.0;
        for &prob in &self.output {
            if prob > 0.0 {
                let normalized_prob = prob / sum;
                entropy -= normalized_prob * normalized_prob.ln();
            }
        }
        
        entropy
    }
    
    /// Check if result indicates high power consumption
    pub fn is_high_power(&self, threshold_mw: f32) -> bool {
        self.power_mw > threshold_mw
    }
    
    /// Check if signal complexity suggests need for adaptation
    pub fn needs_adaptation(&self, complexity_threshold: f32) -> bool {
        self.complexity > complexity_threshold
    }
    
    /// Get efficiency metric (confidence per milliwatt)
    pub fn efficiency(&self) -> f32 {
        if self.power_mw > 0.0 {
            self.confidence / self.power_mw
        } else {
            0.0
        }
    }
    
    /// Combine with another processing result (useful for ensemble methods)
    pub fn combine_with(&self, other: &ProcessingResult, weight: f32) -> ProcessingResult {
        if self.output.len() != other.output.len() {
            // Can't combine outputs of different dimensions, return self
            return self.clone();
        }
        
        let combined_output: Vec<f32> = self.output
            .iter()
            .zip(other.output.iter())
            .map(|(&a, &b)| a * (1.0 - weight) + b * weight)
            .collect();
        
        let combined_timestep = self.timestep_ms * (1.0 - weight) + other.timestep_ms * weight;
        let combined_power = self.power_mw * (1.0 - weight) + other.power_mw * weight;
        let combined_complexity = self.complexity * (1.0 - weight) + other.complexity * weight;
        let combined_energy = self.liquid_energy * (1.0 - weight) + other.liquid_energy * weight;
        
        ProcessingResult::new(
            combined_output,
            combined_timestep,
            combined_power,
            combined_complexity,
            combined_energy,
        )
    }
}

impl Default for ProcessingResult {
    fn default() -> Self {
        Self {
            output: Vec::new(),
            confidence: 0.0,
            timestep_ms: 10.0,
            power_mw: 1.0,
            complexity: 0.5,
            liquid_energy: 0.0,
            metadata: None,
        }
    }
}

/// Extended result with additional diagnostic information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedProcessingResult {
    /// Basic processing result
    pub basic: ProcessingResult,
    /// Processing timing breakdown
    pub timing: TimingInfo,
    /// Power consumption breakdown
    pub power_breakdown: PowerBreakdown,
    /// State information
    pub state_info: StateInfo,
    /// Quality metrics
    pub quality: QualityMetrics,
}

/// Timing information for processing steps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingInfo {
    /// Feature extraction time (ms)
    pub feature_extraction_ms: f32,
    /// ODE integration time (ms)
    pub integration_ms: f32,
    /// Output computation time (ms)
    pub output_computation_ms: f32,
    /// Total processing time (ms)
    pub total_ms: f32,
}

/// Power consumption breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerBreakdown {
    /// Base/idle power (mW)
    pub base_power_mw: f32,
    /// Feature processing power (mW)
    pub feature_power_mw: f32,
    /// Neural computation power (mW)
    pub neural_power_mw: f32,
    /// Memory access power (mW)
    pub memory_power_mw: f32,
    /// I/O power (mW)
    pub io_power_mw: f32,
}

/// Information about internal state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateInfo {
    /// Liquid state dimension
    pub state_dimension: usize,
    /// State norm
    pub state_norm: f32,
    /// State sparsity (fraction of non-zero elements)
    pub state_sparsity: f32,
    /// State stability indicator
    pub is_stable: bool,
    /// Number of integration steps taken
    pub integration_steps: u32,
}

/// Quality metrics for the processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Numerical stability score (0.0-1.0)
    pub stability_score: f32,
    /// Convergence indicator
    pub converged: bool,
    /// Estimated accuracy based on confidence
    pub estimated_accuracy: f32,
    /// Signal-to-noise ratio
    pub snr_db: f32,
}

impl DetailedProcessingResult {
    /// Create from basic result with additional info
    pub fn from_basic(basic: ProcessingResult) -> Self {
        let confidence = basic.confidence;
        Self {
            basic,
            timing: TimingInfo {
                feature_extraction_ms: 0.5,
                integration_ms: 2.0,
                output_computation_ms: 0.1,
                total_ms: 2.6,
            },
            power_breakdown: PowerBreakdown {
                base_power_mw: 0.08,
                feature_power_mw: 0.3,
                neural_power_mw: 0.8,
                memory_power_mw: 0.2,
                io_power_mw: 0.1,
            },
            state_info: StateInfo {
                state_dimension: 64,
                state_norm: 1.0,
                state_sparsity: 0.3,
                is_stable: true,
                integration_steps: 1,
            },
            quality: QualityMetrics {
                stability_score: 0.95,
                converged: true,
                estimated_accuracy: confidence,
                snr_db: 20.0,
            },
        }
    }
    
    /// Get overall efficiency score
    pub fn efficiency_score(&self) -> f32 {
        let accuracy_weight = 0.4;
        let power_weight = 0.3;
        let timing_weight = 0.2;
        let stability_weight = 0.1;
        
        let accuracy_score = self.basic.confidence;
        let power_score = (5.0 - self.basic.power_mw).max(0.0) / 5.0; // Normalize power (lower is better)
        let timing_score = (20.0 - self.timing.total_ms).max(0.0) / 20.0; // Normalize timing
        let stability_score = self.quality.stability_score;
        
        accuracy_weight * accuracy_score +
        power_weight * power_score +
        timing_weight * timing_score +
        stability_weight * stability_score
    }
    
    /// Check if result meets quality thresholds
    pub fn meets_quality_thresholds(&self, thresholds: &QualityThresholds) -> bool {
        self.basic.confidence >= thresholds.min_confidence &&
        self.basic.power_mw <= thresholds.max_power_mw &&
        self.timing.total_ms <= thresholds.max_timing_ms &&
        self.quality.stability_score >= thresholds.min_stability
    }
}

/// Quality thresholds for validation
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    pub min_confidence: f32,
    pub max_power_mw: f32,
    pub max_timing_ms: f32,
    pub min_stability: f32,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_confidence: 0.7,
            max_power_mw: 2.0,
            max_timing_ms: 10.0,
            min_stability: 0.8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_processing_result_creation() {
        let output = vec![0.1, 0.8, 0.1];
        let result = ProcessingResult::new(output, 10.0, 1.5, 0.6, 0.3);
        
        assert_eq!(result.confidence, 0.8);
        assert_eq!(result.timestep_ms, 10.0);
        assert_eq!(result.predicted_class(), Some(1));
    }
    
    #[test]
    fn test_top_k_predictions() {
        let output = vec![0.1, 0.8, 0.05, 0.3, 0.2];
        let result = ProcessingResult::new(output, 10.0, 1.0, 0.5, 0.1);
        
        let top3 = result.top_k(3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0], (1, 0.8)); // Highest
        assert_eq!(top3[1], (3, 0.3)); // Second highest
        assert_eq!(top3[2], (4, 0.2)); // Third highest
    }
    
    #[test]
    fn test_entropy_calculation() {
        // Uniform distribution should have high entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let uniform_result = ProcessingResult::new(uniform, 10.0, 1.0, 0.5, 0.1);
        
        // Peaked distribution should have low entropy
        let peaked = vec![0.9, 0.05, 0.03, 0.02];
        let peaked_result = ProcessingResult::new(peaked, 10.0, 1.0, 0.5, 0.1);
        
        assert!(uniform_result.entropy() > peaked_result.entropy());
    }
    
    #[test]
    fn test_result_combination() {
        let result1 = ProcessingResult::new(vec![0.8, 0.2], 10.0, 1.0, 0.5, 0.1);
        let result2 = ProcessingResult::new(vec![0.3, 0.7], 15.0, 1.5, 0.7, 0.2);
        
        let combined = result1.combine_with(&result2, 0.3);
        
        // Should be weighted average: 0.8 * 0.7 + 0.3 * 0.3 = 0.65
        assert!((combined.output[0] - 0.65).abs() < 1e-6);
        assert!((combined.output[1] - 0.35).abs() < 1e-6);
    }
}