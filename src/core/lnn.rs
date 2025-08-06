//! Main Liquid Neural Network implementation

use crate::{Result, LiquidAudioError};
use crate::core::{AdaptiveConfig, ModelConfig, ODESolver, EulerSolver, LiquidState, ProcessingResult};
use crate::audio::{AudioProcessor, FeatureExtractor};
use crate::adaptive::TimestepController;

#[cfg(not(feature = "std"))]
use core::alloc::{vec::Vec, string::String};
use core::fmt;

use nalgebra::{DVector, DMatrix};
use serde::{Deserialize, Serialize};

/// Main Liquid Neural Network struct
#[derive(Debug, Clone)]
pub struct LNN {
    /// Model configuration
    config: ModelConfig,
    /// Adaptive timestep configuration
    adaptive_config: Option<AdaptiveConfig>,
    /// Current liquid state
    liquid_state: LiquidState,
    /// ODE solver for integration
    solver: Box<dyn ODESolver>,
    /// Timestep controller
    timestep_controller: TimestepController,
    /// Audio feature extractor
    feature_extractor: FeatureExtractor,
    /// Model weights and parameters
    weights: NetworkWeights,
    /// Current power consumption estimate
    current_power_mw: f32,
}

/// Network weights and parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkWeights {
    /// Input to liquid layer weights [hidden_dim x input_dim]
    w_input: DMatrix<f32>,
    /// Recurrent liquid layer weights [hidden_dim x hidden_dim]
    w_recurrent: DMatrix<f32>,
    /// Liquid to output weights [output_dim x hidden_dim]
    w_output: DMatrix<f32>,
    /// Input bias
    b_input: DVector<f32>,
    /// Output bias
    b_output: DVector<f32>,
    /// Time constants for each neuron
    tau: DVector<f32>,
}

impl LNN {
    /// Create new LNN with given configuration
    pub fn new(config: ModelConfig) -> Result<Self> {
        let liquid_state = LiquidState::new(config.hidden_dim);
        let solver: Box<dyn ODESolver> = Box::new(EulerSolver::new());
        let timestep_controller = TimestepController::new();
        let feature_extractor = FeatureExtractor::new(config.input_dim)?;
        
        // Initialize random weights (in practice, loaded from file)
        let weights = NetworkWeights::random(&config)?;
        
        Ok(LNN {
            config,
            adaptive_config: None,
            liquid_state,
            solver,
            timestep_controller,
            feature_extractor,
            weights,
            current_power_mw: 0.0,
        })
    }
    
    /// Load LNN from .lnn model file
    #[cfg(feature = "std")]
    pub fn load_from_file(path: impl AsRef<std::ffi::OsStr>) -> Result<Self> {
        #[cfg(feature = "std")]
        {
            use std::fs;
            use std::path::Path;
            
            let path = Path::new(path.as_ref());
            if !path.exists() {
                return Err(LiquidAudioError::IoError(
                    format!("Model file not found: {}", path.display())
                ));
            }
            
            let data = fs::read(path)
                .map_err(|e| LiquidAudioError::IoError(format!("Failed to read model: {}", e)))?;
            
            Self::load_from_bytes(&data)
        }
        
        #[cfg(not(feature = "std"))]
        {
            Err(LiquidAudioError::ConfigError(
                "File loading not supported in no_std mode".to_string()
            ))
        }
    }
    
    /// Load LNN from binary data (.lnn format)
    pub fn load_from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 32 {
            return Err(LiquidAudioError::ModelError(
                "Invalid model file - too short".to_string()
            ));
        }
        
        // Parse header
        let magic = &data[0..4];
        if magic != b"LNN\x01" {
            return Err(LiquidAudioError::ModelError(
                "Invalid model file - wrong magic number".to_string()
            ));
        }
        
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let input_dim = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let hidden_dim = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
        let output_dim = u32::from_le_bytes([data[16], data[17], data[18], data[19]]) as usize;
        
        if version != 1 {
            return Err(LiquidAudioError::ModelError(
                format!("Unsupported model version: {}", version)
            ));
        }
        
        // Parse weights from remaining data
        let weights_data = &data[32..];
        let weights = NetworkWeights::from_bytes(weights_data, input_dim, hidden_dim, output_dim)?;
        
        let config = ModelConfig {
            input_dim,
            hidden_dim,
            output_dim,
            sample_rate: 16000,
            frame_size: 512,
            model_type: "keyword_spotting".to_string(),
        };
        
        let mut lnn = Self::new(config)?;
        lnn.weights = weights;
        
        Ok(lnn)
    }
    
    /// Set adaptive timestep configuration
    pub fn set_adaptive_config(&mut self, config: AdaptiveConfig) {
        self.adaptive_config = Some(config);
        self.timestep_controller.set_config(config.clone());
    }
    
    /// Process audio buffer
    pub fn process(&mut self, audio_buffer: &[f32]) -> Result<ProcessingResult> {
        if audio_buffer.is_empty() {
            return Err(LiquidAudioError::InvalidInput(
                "Empty audio buffer".to_string()
            ));
        }
        
        // Extract features
        let features = self.feature_extractor.extract(audio_buffer)?;
        
        // Determine timestep
        let complexity = self.estimate_complexity(audio_buffer);
        let timestep = if let Some(ref config) = self.adaptive_config {
            self.timestep_controller.calculate_timestep(complexity, config)
        } else {
            0.01 // Default 10ms timestep
        };
        
        // Integrate liquid dynamics
        let input_current = &self.weights.w_input * &features + &self.weights.b_input;
        let recurrent_current = &self.weights.w_recurrent * self.liquid_state.hidden_state();
        let decay_current = -self.liquid_state.hidden_state().component_div(&self.weights.tau);
        
        let total_current = input_current + recurrent_current + decay_current;
        
        // Use ODE solver to integrate
        self.liquid_state = self.solver.step(
            &self.liquid_state,
            &total_current,
            timestep
        )?;
        
        // Compute output
        let output = &self.weights.w_output * self.liquid_state.hidden_state() + &self.weights.b_output;
        let output = softmax(&output);
        
        // Power estimation
        let power_mw = self.estimate_power(audio_buffer.len(), complexity, timestep);
        self.current_power_mw = power_mw;
        
        // Create result
        let result = ProcessingResult {
            output: output.as_slice().to_vec(),
            confidence: output.max(),
            timestep_ms: timestep * 1000.0,
            power_mw,
            complexity,
            liquid_energy: self.liquid_state.energy(),
        };
        
        Ok(result)
    }
    
    /// Get current power consumption
    pub fn current_power_mw(&self) -> f32 {
        self.current_power_mw
    }
    
    /// Estimate signal complexity for adaptive timestep
    fn estimate_complexity(&self, audio_buffer: &[f32]) -> f32 {
        if audio_buffer.is_empty() {
            return 0.0;
        }
        
        // Simple energy-based complexity
        let energy: f32 = audio_buffer.iter().map(|x| x * x).sum::<f32>() / audio_buffer.len() as f32;
        
        // Spectral flux approximation using differences
        let mut spectral_change = 0.0;
        for i in 1..audio_buffer.len() {
            spectral_change += (audio_buffer[i] - audio_buffer[i-1]).abs();
        }
        spectral_change /= audio_buffer.len() as f32;
        
        // Combine metrics
        let complexity = (energy.sqrt() + spectral_change) * 0.5;
        complexity.min(1.0)
    }
    
    /// Estimate power consumption
    fn estimate_power(&self, buffer_len: usize, complexity: f32, timestep: f32) -> f32 {
        const BASE_POWER: f32 = 0.08; // mW baseline
        
        // Signal-dependent power
        let signal_power = complexity * 1.2;
        
        // Computation power (depends on timestep - smaller timestep = more computation)
        let computation_power = (1.0 / timestep) * 0.1;
        
        // Buffer size dependent power
        let buffer_power = (buffer_len as f32 / 1024.0) * 0.3;
        
        // Network size dependent power
        let network_power = (self.config.hidden_dim as f32 / 64.0) * 0.4;
        
        let total_power = BASE_POWER + signal_power + computation_power + buffer_power + network_power;
        
        // Apply efficiency factor for adaptive timestep
        let efficiency = if self.adaptive_config.is_some() {
            1.0 - complexity * 0.3
        } else {
            1.0
        };
        
        (total_power * efficiency).max(0.05).min(5.0)
    }
    
    /// Get model configuration
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }
    
    /// Get liquid state
    pub fn liquid_state(&self) -> &LiquidState {
        &self.liquid_state
    }
    
    /// Reset liquid state to zero
    pub fn reset_state(&mut self) {
        self.liquid_state = LiquidState::new(self.config.hidden_dim);
    }
}

impl NetworkWeights {
    /// Create random weights for given configuration
    pub fn random(config: &ModelConfig) -> Result<Self> {
        use nalgebra::DMatrix;
        
        let w_input = DMatrix::from_fn(config.hidden_dim, config.input_dim, |_, _| {
            (rand() - 0.5) * 0.2
        });
        
        let w_recurrent = DMatrix::from_fn(config.hidden_dim, config.hidden_dim, |_, _| {
            (rand() - 0.5) * 0.1
        });
        
        let w_output = DMatrix::from_fn(config.output_dim, config.hidden_dim, |_, _| {
            (rand() - 0.5) * 0.1
        });
        
        let b_input = DVector::from_fn(config.hidden_dim, |_, _| {
            (rand() - 0.5) * 0.1
        });
        
        let b_output = DVector::from_fn(config.output_dim, |_, _| {
            (rand() - 0.5) * 0.1
        });
        
        let tau = DVector::from_fn(config.hidden_dim, |_, _| {
            0.1 + rand() * 0.05 // Time constants between 0.1 and 0.15
        });
        
        Ok(NetworkWeights {
            w_input,
            w_recurrent,
            w_output,
            b_input,
            b_output,
            tau,
        })
    }
    
    /// Load weights from binary data
    pub fn from_bytes(data: &[u8], input_dim: usize, hidden_dim: usize, output_dim: usize) -> Result<Self> {
        // Calculate expected sizes
        let w_input_size = hidden_dim * input_dim;
        let w_recurrent_size = hidden_dim * hidden_dim;
        let w_output_size = output_dim * hidden_dim;
        let b_input_size = hidden_dim;
        let b_output_size = output_dim;
        let tau_size = hidden_dim;
        
        let total_floats = w_input_size + w_recurrent_size + w_output_size + 
                          b_input_size + b_output_size + tau_size;
        let expected_bytes = total_floats * core::mem::size_of::<f32>();
        
        if data.len() < expected_bytes {
            return Err(LiquidAudioError::ModelError(
                format!("Insufficient weight data: got {} bytes, need {}", data.len(), expected_bytes)
            ));
        }
        
        // Parse weights (simplified - real implementation would handle endianness)
        let mut offset = 0;
        
        // Helper to read matrix from bytes
        let read_matrix = |data: &[u8], offset: &mut usize, rows: usize, cols: usize| -> Result<DMatrix<f32>> {
            let size = rows * cols * 4; // 4 bytes per f32
            if *offset + size > data.len() {
                return Err(LiquidAudioError::ModelError("Truncated weight data".to_string()));
            }
            
            let mut values = Vec::with_capacity(rows * cols);
            for i in 0..rows * cols {
                let start = *offset + i * 4;
                let bytes = [data[start], data[start+1], data[start+2], data[start+3]];
                values.push(f32::from_le_bytes(bytes));
            }
            *offset += size;
            
            Ok(DMatrix::from_row_slice(rows, cols, &values))
        };
        
        let w_input = read_matrix(data, &mut offset, hidden_dim, input_dim)?;
        let w_recurrent = read_matrix(data, &mut offset, hidden_dim, hidden_dim)?;
        let w_output = read_matrix(data, &mut offset, output_dim, hidden_dim)?;
        
        // Read bias vectors (simplified)
        let b_input = DVector::from_fn(hidden_dim, |i, _| {
            let start = offset + i * 4;
            f32::from_le_bytes([data[start], data[start+1], data[start+2], data[start+3]])
        });
        offset += b_input_size * 4;
        
        let b_output = DVector::from_fn(output_dim, |i, _| {
            let start = offset + i * 4;
            f32::from_le_bytes([data[start], data[start+1], data[start+2], data[start+3]])
        });
        offset += b_output_size * 4;
        
        let tau = DVector::from_fn(hidden_dim, |i, _| {
            let start = offset + i * 4;
            f32::from_le_bytes([data[start], data[start+1], data[start+2], data[start+3]])
        });
        
        Ok(NetworkWeights {
            w_input,
            w_recurrent,
            w_output,
            b_input,
            b_output,
            tau,
        })
    }
}

/// Simple softmax activation
fn softmax(x: &DVector<f32>) -> DVector<f32> {
    let max_val = x.max();
    let exp_vals: Vec<f32> = x.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    
    DVector::from_vec(exp_vals.into_iter().map(|v| v / sum).collect())
}

/// Simple random number generator (for no_std compatibility)
fn rand() -> f32 {
    // Simple linear congruential generator
    static mut SEED: u32 = 1;
    unsafe {
        SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
        (SEED as f32) / (u32::MAX as f32)
    }
}

impl fmt::Display for LNN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LNN({}→{}→{}, power: {:.2}mW)", 
               self.config.input_dim,
               self.config.hidden_dim, 
               self.config.output_dim,
               self.current_power_mw)
    }
}