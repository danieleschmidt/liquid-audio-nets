//! Enhanced liquid state management for neural networks
//! 
//! Extended with neuromorphic computing, quantum-inspired states, and multi-scale dynamics

use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use crate::{Result, LiquidAudioError};

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, format};

/// Processing modes for different computational paradigms
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StateMode {
    /// Standard continuous dynamics
    Standard,
    /// Neuromorphic spiking dynamics
    Neuromorphic,
    /// Quantum-inspired superposition
    Quantum,
    /// Multi-scale hierarchical
    MultiScale,
    /// Hybrid mode combining multiple paradigms
    Hybrid,
}

impl Default for StateMode {
    fn default() -> Self {
        StateMode::Standard
    }
}

/// Enhanced liquid neural network state with multiple computational modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiquidState {
    /// Primary hidden state vector
    hidden: DVector<f32>,
    /// Neuromorphic spike times (for spiking neural networks)
    #[serde(skip_serializing_if = "Option::is_none")]
    spike_times: Option<Vec<f32>>,
    /// Membrane potentials (for neuromorphic mode)
    #[serde(skip_serializing_if = "Option::is_none")]
    membrane_potentials: Option<DVector<f32>>,
    /// Quantum amplitudes (for quantum-inspired mode)
    #[serde(skip_serializing_if = "Option::is_none")]
    quantum_amplitudes: Option<DVector<f32>>,
    /// Multi-scale state hierarchy
    #[serde(skip_serializing_if = "Option::is_none")]
    scale_states: Option<Vec<DVector<f32>>>,
    /// Timestamp of last update
    last_update: f64,
    /// State energy (for monitoring)
    energy: f32,
    /// Entropy measure
    entropy: f32,
    /// Sparsity level (fraction of active neurons)
    sparsity: f32,
    /// Stability indicator
    stability: f32,
    /// Processing mode
    mode: StateMode,
}

impl LiquidState {
    /// Create new liquid state with given dimension
    pub fn new(dim: usize) -> Self {
        Self {
            hidden: DVector::zeros(dim),
            spike_times: None,
            membrane_potentials: None,
            quantum_amplitudes: None,
            scale_states: None,
            last_update: 0.0,
            energy: 0.0,
            entropy: 0.0,
            sparsity: 1.0,
            stability: 1.0,
            mode: StateMode::Standard,
        }
    }
    
    /// Create state from vector
    pub fn from_vector(hidden: DVector<f32>) -> Self {
        let energy = hidden.norm_squared() / hidden.len() as f32;
        let entropy = Self::calculate_entropy(&hidden);
        let sparsity = Self::calculate_sparsity(&hidden);
        
        Self {
            hidden,
            spike_times: None,
            membrane_potentials: None,
            quantum_amplitudes: None,
            scale_states: None,
            last_update: 0.0,
            energy,
            entropy,
            sparsity,
            stability: 1.0 / (1.0 + energy),
            mode: StateMode::Standard,
        }
    }
    
    /// Create neuromorphic state with spike timing
    pub fn from_neuromorphic(
        hidden: DVector<f32>,
        spike_times: Vec<f32>,
        membrane_potentials: DVector<f32>
    ) -> Result<Self> {
        if spike_times.len() != hidden.len() || membrane_potentials.len() != hidden.len() {
            return Err(LiquidAudioError::ConfigError(
                "Dimension mismatch in neuromorphic state".to_string()
            ));
        }
        
        let energy = hidden.norm_squared() / hidden.len() as f32;
        let entropy = Self::calculate_entropy(&hidden);
        let sparsity = Self::calculate_sparsity(&hidden);
        
        Ok(Self {
            hidden,
            spike_times: Some(spike_times),
            membrane_potentials: Some(membrane_potentials),
            quantum_amplitudes: None,
            scale_states: None,
            last_update: 0.0,
            energy,
            entropy,
            sparsity,
            stability: 1.0 / (1.0 + energy),
            mode: StateMode::Neuromorphic,
        })
    }
    
    /// Create quantum-inspired state with amplitude information
    pub fn from_quantum(
        hidden: DVector<f32>,
        quantum_amplitudes: DVector<f32>
    ) -> Result<Self> {
        if quantum_amplitudes.len() != hidden.len() {
            return Err(LiquidAudioError::ConfigError(
                "Dimension mismatch in quantum state".to_string()
            ));
        }
        
        let energy = Self::calculate_quantum_energy(&hidden, &quantum_amplitudes);
        let entropy = Self::calculate_quantum_entropy(&quantum_amplitudes);
        let sparsity = Self::calculate_sparsity(&hidden);
        
        Ok(Self {
            hidden,
            spike_times: None,
            membrane_potentials: None,
            quantum_amplitudes: Some(quantum_amplitudes),
            scale_states: None,
            last_update: 0.0,
            energy,
            entropy,
            sparsity,
            stability: 1.0 / (1.0 + energy),
            mode: StateMode::Quantum,
        })
    }
    
    /// Create multi-scale hierarchical state
    pub fn from_multiscale(
        hidden: DVector<f32>,
        scale_states: Vec<DVector<f32>>
    ) -> Self {
        let energy = hidden.norm_squared() / hidden.len() as f32;
        let entropy = Self::calculate_entropy(&hidden);
        let sparsity = Self::calculate_sparsity(&hidden);
        
        Self {
            hidden,
            spike_times: None,
            membrane_potentials: None,
            quantum_amplitudes: None,
            scale_states: Some(scale_states),
            last_update: 0.0,
            energy,
            entropy,
            sparsity,
            stability: 1.0 / (1.0 + energy),
            mode: StateMode::MultiScale,
        }
    }
    
    /// Get hidden state vector (immutable reference)
    pub fn hidden_state(&self) -> &DVector<f32> {
        &self.hidden
    }
    
    /// Get mutable reference to hidden state
    pub fn hidden_state_mut(&mut self) -> &mut DVector<f32> {
        &mut self.hidden
    }
    
    /// Get state dimension
    pub fn dim(&self) -> usize {
        self.hidden.len()
    }
    
    /// Get current energy
    pub fn energy(&self) -> f32 {
        self.energy
    }
    
    /// Update energy calculation
    pub fn update_energy(&mut self) {
        self.energy = self.hidden.norm_squared() / self.hidden.len() as f32;
    }
    
    /// Get last update timestamp
    pub fn last_update(&self) -> f64 {
        self.last_update
    }
    
    /// Set update timestamp
    pub fn set_timestamp(&mut self, timestamp: f64) {
        self.last_update = timestamp;
    }
    
    /// Reset state to zeros
    pub fn reset(&mut self) {
        self.hidden.fill(0.0);
        self.energy = 0.0;
        self.last_update = 0.0;
    }
    
    /// Apply decay to state (useful for stability)
    pub fn apply_decay(&mut self, decay_factor: f32) {
        self.hidden *= decay_factor;
        self.update_energy();
    }
    
    /// Check if state is stable (low energy)
    pub fn is_stable(&self, threshold: f32) -> bool {
        self.energy < threshold
    }
    
    /// Get state statistics
    pub fn statistics(&self) -> StateStatistics {
        let mean = self.hidden.mean();
        let variance = self.hidden.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / self.hidden.len() as f32;
        let min_val = self.hidden.min();
        let max_val = self.hidden.max();
        
        StateStatistics {
            mean,
            variance,
            std_dev: variance.sqrt(),
            min: min_val,
            max: max_val,
            energy: self.energy,
            norm: self.hidden.norm(),
        }
    }
    
    /// Normalize state to unit norm
    pub fn normalize(&mut self) {
        let norm = self.hidden.norm();
        if norm > 1e-8 {
            self.hidden /= norm;
            self.update_energy();
        }
    }
    
    /// Clip state values to range
    pub fn clip(&mut self, min_val: f32, max_val: f32) {
        for value in self.hidden.iter_mut() {
            *value = value.clamp(min_val, max_val);
        }
        self.update_energy();
    }
    
    /// Apply activation function element-wise
    pub fn apply_activation<F>(&mut self, activation: F) 
    where
        F: Fn(f32) -> f32,
    {
        for value in self.hidden.iter_mut() {
            *value = activation(*value);
        }
        self.update_energy();
    }
    
    /// Add noise to state (useful for exploration)
    pub fn add_noise(&mut self, noise_scale: f32) {
        for value in self.hidden.iter_mut() {
            *value += noise_scale * (random() - 0.5) * 2.0;
        }
        self.update_energy();
    }
    
    /// Interpolate towards target state
    pub fn interpolate_to(&mut self, target: &LiquidState, alpha: f32) {
        if self.hidden.len() == target.hidden.len() {
            self.hidden = &self.hidden * (1.0 - alpha) + &target.hidden * alpha;
            self.update_energy();
        }
    }
    
    /// Distance to another state
    pub fn distance_to(&self, other: &LiquidState) -> f32 {
        if self.hidden.len() == other.hidden.len() {
            (&self.hidden - &other.hidden).norm()
        } else {
            f32::INFINITY
        }
    }
    
    /// Create sparse version (zero out small values)
    pub fn sparsify(&mut self, threshold: f32) {
        for value in self.hidden.iter_mut() {
            if value.abs() < threshold {
                *value = 0.0;
            }
        }
        self.update_energy();
    }
    
    /// Count non-zero elements
    pub fn sparsity(&self) -> f32 {
        let non_zero = self.hidden.iter().filter(|&&x| x.abs() > 1e-8).count();
        non_zero as f32 / self.hidden.len() as f32
    }
}

/// Statistics about liquid state
#[derive(Debug, Clone)]
pub struct StateStatistics {
    pub mean: f32,
    pub variance: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub energy: f32,
    pub norm: f32,
}

/// Network state that includes multiple components
#[derive(Debug, Clone)]
pub struct NetworkState {
    /// Main liquid state
    pub liquid: LiquidState,
    /// Output state/activations
    pub output: DVector<f32>,
    /// Processing metadata
    pub metadata: StateMetadata,
}

impl NetworkState {
    /// Create new network state
    pub fn new(liquid_dim: usize, output_dim: usize) -> Self {
        Self {
            liquid: LiquidState::new(liquid_dim),
            output: DVector::zeros(output_dim),
            metadata: StateMetadata::default(),
        }
    }
    
    /// Reset all state components
    pub fn reset(&mut self) {
        self.liquid.reset();
        self.output.fill(0.0);
        self.metadata = StateMetadata::default();
    }
    
    /// Get total energy across all state components
    pub fn total_energy(&self) -> f32 {
        let liquid_energy = self.liquid.energy();
        let output_energy = self.output.norm_squared() / self.output.len() as f32;
        liquid_energy + output_energy
    }
}

/// Metadata about state processing
#[derive(Debug, Clone, Default)]
pub struct StateMetadata {
    /// Number of processing steps
    pub step_count: u64,
    /// Total processing time
    pub total_time: f64,
    /// Average timestep size
    pub avg_timestep: f32,
    /// Power consumption estimate
    pub power_estimate: f32,
    /// Complexity metric
    pub complexity: f32,
}

impl StateMetadata {
    /// Update with new processing step
    pub fn update_step(&mut self, timestep: f32, power: f32, complexity: f32) {
        self.step_count += 1;
        self.total_time += timestep as f64;
        
        // Running average of timestep
        let alpha = 0.1; // Exponential smoothing factor
        self.avg_timestep = self.avg_timestep * (1.0 - alpha) + timestep * alpha;
        
        self.power_estimate = power;
        self.complexity = complexity;
    }
    
    /// Get processing rate (steps per second)
    pub fn processing_rate(&self) -> f64 {
        if self.total_time > 0.0 {
            self.step_count as f64 / self.total_time
        } else {
            0.0
        }
    }
}

/// Simple random number generator for noise
fn random() -> f32 {
    // Simple linear congruential generator
    static mut SEED: u32 = 1337;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED as f32) / (u32::MAX as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_liquid_state_creation() {
        let state = LiquidState::new(64);
        assert_eq!(state.dim(), 64);
        assert_eq!(state.energy(), 0.0);
    }
    
    #[test]
    fn test_state_energy_calculation() {
        let vector = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let mut state = LiquidState::from_vector(vector);
        
        // Energy should be mean squared: (1 + 4 + 9 + 16) / 4 = 7.5
        assert!((state.energy() - 7.5).abs() < 1e-6);
        
        state.update_energy();
        assert!((state.energy() - 7.5).abs() < 1e-6);
    }
    
    #[test]
    fn test_state_normalization() {
        let vector = DVector::from_vec(vec![3.0, 4.0]); // Norm = 5.0
        let mut state = LiquidState::from_vector(vector);
        
        state.normalize();
        assert!((state.hidden_state().norm() - 1.0).abs() < 1e-6);
        assert!((state.hidden_state()[0] - 0.6).abs() < 1e-6);
        assert!((state.hidden_state()[1] - 0.8).abs() < 1e-6);
    }
    
    #[test]
    fn test_state_distance() {
        let state1 = LiquidState::from_vector(DVector::from_vec(vec![1.0, 2.0]));
        let state2 = LiquidState::from_vector(DVector::from_vec(vec![4.0, 6.0]));
        
        let distance = state1.distance_to(&state2);
        assert!((distance - 5.0).abs() < 1e-6); // sqrt((4-1)^2 + (6-2)^2) = sqrt(9+16) = 5
    }
    
    #[test]
    fn test_state_sparsity() {
        let vector = DVector::from_vec(vec![1.0, 0.0, 0.001, 2.0, 0.0]);
        let mut state = LiquidState::from_vector(vector);
        
        // Before sparsification: 3/5 = 0.6 non-zero
        assert!((state.sparsity() - 0.6).abs() < 1e-6);
        
        state.sparsify(0.01);
        
        // After sparsification (threshold 0.01): 2/5 = 0.4 non-zero
        assert!((state.sparsity() - 0.4).abs() < 1e-6);
    }
}