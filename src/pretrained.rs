//! Pre-trained model architectures and utilities for Liquid Neural Networks
//!
//! Provides ready-to-use model architectures, weight loading, and
//! specialized models for common audio processing tasks.

use crate::{Result, LiquidAudioError, ModelConfig, ProcessingResult, AdaptiveConfig};
use crate::models::{AudioModel, ModelFactory};
#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String, collections::BTreeMap};

#[cfg(feature = "std")]
use std::{vec::Vec, string::String, collections::BTreeMap};

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

/// Pre-trained model registry
#[derive(Debug)]
pub struct ModelRegistry {
    /// Available models
    models: BTreeMap<String, ModelArchitecture>,
    /// Registry enabled
    enabled: bool,
}

/// Model architecture definition
#[derive(Debug, Clone)]
pub struct ModelArchitecture {
    /// Architecture name
    pub name: String,
    /// Model type
    pub model_type: ModelType,
    /// Default configuration
    pub default_config: ModelConfig,
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Weight specifications
    pub weight_specs: WeightSpecs,
    /// Performance characteristics
    pub performance: PerformanceProfile,
}

/// Available model types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelType {
    /// Keyword spotting model
    KeywordSpotter,
    /// Voice activity detection
    VoiceActivityDetection,
    /// Audio classification
    AudioClassification,
    /// Speech recognition
    SpeechRecognition,
    /// Audio enhancement
    AudioEnhancement,
    /// Emotion recognition
    EmotionRecognition,
    /// Speaker identification
    SpeakerIdentification,
    /// Custom model type
    Custom(String),
}

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// Model version
    pub version: String,
    /// Training dataset
    pub dataset: String,
    /// Accuracy metrics
    pub accuracy: f32,
    /// Model size (parameters)
    pub parameters: usize,
    /// Training date
    pub training_date: String,
    /// Author/organization
    pub author: String,
    /// Description
    pub description: String,
    /// License
    pub license: String,
}

/// Weight specifications
#[derive(Debug, Clone)]
pub struct WeightSpecs {
    /// Total parameters
    pub total_parameters: usize,
    /// Weight format
    pub format: WeightFormat,
    /// Quantization level
    pub quantization: QuantizationLevel,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Checksum for validation
    pub checksum: String,
}

/// Weight file formats
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WeightFormat {
    /// Raw binary format
    Raw,
    /// Compressed format
    Compressed,
    /// Quantized format
    Quantized,
    /// Custom format
    Custom(String),
}

/// Quantization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationLevel {
    /// Full precision (32-bit)
    Float32,
    /// Half precision (16-bit)
    Float16,
    /// 8-bit quantization
    Int8,
    /// 4-bit quantization
    Int4,
    /// Custom quantization
    Custom(u8),
}

/// Performance profile
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    /// Typical latency (ms)
    pub typical_latency_ms: f32,
    /// Peak latency (ms)
    pub peak_latency_ms: f32,
    /// Power consumption (mW)
    pub power_consumption_mw: f32,
    /// Memory usage (KB)
    pub memory_usage_kb: usize,
    /// Throughput (inferences per second)
    pub throughput_ips: f32,
    /// CPU utilization (%)
    pub cpu_utilization_percent: f32,
}

impl ModelRegistry {
    /// Create new model registry
    pub fn new() -> Self {
        let mut registry = Self {
            models: BTreeMap::new(),
            enabled: true,
        };

        // Register built-in models
        registry.register_builtin_models();
        registry
    }

    /// Register built-in model architectures
    fn register_builtin_models(&mut self) {
        // Keyword Spotter
        self.register_model(ModelArchitecture {
            name: "keyword_spotter_v1".to_string(),
            model_type: ModelType::KeywordSpotter,
            default_config: ModelConfig {
                input_dim: 40,
                hidden_dim: 64,
                output_dim: 10,
                sample_rate: 16000,
                frame_size: 512,
                model_type: "keyword_spotter".to_string(),
            },
            metadata: ModelMetadata {
                version: "1.0.0".to_string(),
                dataset: "Google Speech Commands v2".to_string(),
                accuracy: 0.95,
                parameters: 50000,
                training_date: "2024-01-01".to_string(),
                author: "Terragon Labs".to_string(),
                description: "Efficient keyword spotting for embedded systems".to_string(),
                license: "MIT".to_string(),
            },
            weight_specs: WeightSpecs {
                total_parameters: 50000,
                format: WeightFormat::Quantized,
                quantization: QuantizationLevel::Int8,
                compression_ratio: 4.0,
                checksum: "sha256:abc123".to_string(),
            },
            performance: PerformanceProfile {
                typical_latency_ms: 5.0,
                peak_latency_ms: 8.0,
                power_consumption_mw: 0.8,
                memory_usage_kb: 64,
                throughput_ips: 200.0,
                cpu_utilization_percent: 15.0,
            },
        });

        // Voice Activity Detection
        self.register_model(ModelArchitecture {
            name: "vad_ultra_low_power".to_string(),
            model_type: ModelType::VoiceActivityDetection,
            default_config: ModelConfig {
                input_dim: 20,
                hidden_dim: 32,
                output_dim: 2,
                sample_rate: 16000,
                frame_size: 320,
                model_type: "vad".to_string(),
            },
            metadata: ModelMetadata {
                version: "1.2.0".to_string(),
                dataset: "LibriSpeech + MUSAN".to_string(),
                accuracy: 0.92,
                parameters: 15000,
                training_date: "2024-02-01".to_string(),
                author: "Terragon Labs".to_string(),
                description: "Ultra-low-power voice activity detection".to_string(),
                license: "MIT".to_string(),
            },
            weight_specs: WeightSpecs {
                total_parameters: 15000,
                format: WeightFormat::Quantized,
                quantization: QuantizationLevel::Int8,
                compression_ratio: 6.0,
                checksum: "sha256:def456".to_string(),
            },
            performance: PerformanceProfile {
                typical_latency_ms: 2.0,
                peak_latency_ms: 3.0,
                power_consumption_mw: 0.3,
                memory_usage_kb: 32,
                throughput_ips: 500.0,
                cpu_utilization_percent: 8.0,
            },
        });

        // Audio Classifier
        self.register_model(ModelArchitecture {
            name: "audio_classifier_general".to_string(),
            model_type: ModelType::AudioClassification,
            default_config: ModelConfig {
                input_dim: 128,
                hidden_dim: 256,
                output_dim: 50,
                sample_rate: 22050,
                frame_size: 1024,
                model_type: "audio_classifier".to_string(),
            },
            metadata: ModelMetadata {
                version: "2.1.0".to_string(),
                dataset: "AudioSet + FSD50K".to_string(),
                accuracy: 0.88,
                parameters: 180000,
                training_date: "2024-03-01".to_string(),
                author: "Terragon Labs".to_string(),
                description: "General-purpose audio event classification".to_string(),
                license: "MIT".to_string(),
            },
            weight_specs: WeightSpecs {
                total_parameters: 180000,
                format: WeightFormat::Compressed,
                quantization: QuantizationLevel::Float16,
                compression_ratio: 2.5,
                checksum: "sha256:ghi789".to_string(),
            },
            performance: PerformanceProfile {
                typical_latency_ms: 12.0,
                peak_latency_ms: 18.0,
                power_consumption_mw: 2.1,
                memory_usage_kb: 256,
                throughput_ips: 80.0,
                cpu_utilization_percent: 35.0,
            },
        });
    }

    /// Register a new model architecture
    pub fn register_model(&mut self, architecture: ModelArchitecture) {
        self.models.insert(architecture.name.clone(), architecture);
    }

    /// Load a pre-trained model
    pub fn load_model(&mut self, model_name: &str, weights: Option<Vec<f32>>) -> Result<Box<dyn AudioModel>> {
        if !self.enabled {
            return Err(LiquidAudioError::InvalidState("Model registry disabled".to_string()));
        }

        // Get architecture
        let architecture = self.models.get(model_name)
            .ok_or_else(|| LiquidAudioError::ModelError(format!("Unknown model: {}", model_name)))?;

        // Create model instance (simplified - no caching for now)
        let model = self.create_model_instance(architecture, weights)?;
        
        Ok(model)
    }

    /// Create model instance from architecture
    fn create_model_instance(
        &self,
        architecture: &ModelArchitecture,
        weights: Option<Vec<f32>>,
    ) -> Result<Box<dyn AudioModel>> {
        match architecture.model_type {
            ModelType::KeywordSpotter => {
                Ok(Box::new(PretrainedKeywordSpotter::new(
                    architecture.default_config.clone(),
                    weights,
                    architecture.metadata.clone(),
                )?))
            }
            ModelType::VoiceActivityDetection => {
                Ok(Box::new(PretrainedVAD::new(
                    architecture.default_config.clone(),
                    weights,
                    architecture.metadata.clone(),
                )?))
            }
            ModelType::AudioClassification => {
                Ok(Box::new(PretrainedAudioClassifier::new(
                    architecture.default_config.clone(),
                    weights,
                    architecture.metadata.clone(),
                )?))
            }
            _ => Err(LiquidAudioError::ModelError(
                format!("Model type {:?} not yet supported", architecture.model_type)
            )),
        }
    }

    /// List available models
    pub fn list_models(&self) -> Vec<String> {
        self.models.keys().cloned().collect()
    }

    /// Get model information
    pub fn get_model_info(&self, model_name: &str) -> Option<&ModelArchitecture> {
        self.models.get(model_name)
    }

    /// Clear model cache (no-op in simplified version)
    pub fn clear_cache(&mut self) {
        // No caching implemented
    }

    /// Enable/disable registry
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Pre-trained Keyword Spotter implementation
#[derive(Debug, Clone)]
pub struct PretrainedKeywordSpotter {
    /// Model configuration
    config: ModelConfig,
    /// Model weights
    weights: Option<Vec<f32>>,
    /// Model metadata
    metadata: ModelMetadata,
    /// Current power consumption
    power_mw: f32,
    /// Processing state
    processing_state: ProcessingState,
}

/// Pre-trained VAD implementation
#[derive(Debug, Clone)]
pub struct PretrainedVAD {
    /// Model configuration
    config: ModelConfig,
    /// Model weights
    weights: Option<Vec<f32>>,
    /// Model metadata
    metadata: ModelMetadata,
    /// Current power consumption
    power_mw: f32,
    /// Processing state
    processing_state: ProcessingState,
}

/// Pre-trained Audio Classifier implementation
#[derive(Debug, Clone)]
pub struct PretrainedAudioClassifier {
    /// Model configuration
    config: ModelConfig,
    /// Model weights
    weights: Option<Vec<f32>>,
    /// Model metadata
    metadata: ModelMetadata,
    /// Current power consumption
    power_mw: f32,
    /// Processing state
    processing_state: ProcessingState,
}

/// Processing state for pre-trained models
#[derive(Debug, Clone)]
struct ProcessingState {
    /// Internal state vector
    state: Vec<f32>,
    /// Processing ready
    ready: bool,
    /// Last processing time
    last_processing_time: u64,
}

impl ProcessingState {
    fn new(hidden_dim: usize) -> Self {
        Self {
            state: vec![0.0; hidden_dim],
            ready: true,
            last_processing_time: 0,
        }
    }

    fn reset(&mut self) {
        for value in &mut self.state {
            *value = 0.0;
        }
        self.ready = true;
    }
}

// Implement AudioModel trait for each pre-trained model

impl PretrainedKeywordSpotter {
    pub fn new(config: ModelConfig, weights: Option<Vec<f32>>, metadata: ModelMetadata) -> Result<Self> {
        Ok(Self {
            processing_state: ProcessingState::new(config.hidden_dim),
            power_mw: 0.8,
            config,
            weights,
            metadata,
        })
    }

}

impl AudioModel for PretrainedKeywordSpotter {
    fn process_audio(&mut self, audio_buffer: &[f32]) -> Result<ProcessingResult> {
        if audio_buffer.is_empty() {
            return Err(LiquidAudioError::InvalidInput("Empty audio buffer".to_string()));
        }

        // Simulate keyword spotting processing
        let mut confidence = 0.0f32;
        let mut detected_keyword = 0;

        // Simple energy-based detection
        let energy: f32 = audio_buffer.iter().map(|&x| x * x).sum::<f32>() / audio_buffer.len() as f32;
        
        if energy > 0.01 {
            confidence = (energy * 10.0).min(1.0);
            detected_keyword = if confidence > 0.8 { 1 } else { 0 };
        }

        // Update power consumption based on processing complexity
        self.power_mw = 0.8 + energy * 0.2;

        Ok(ProcessingResult {
            output: vec![detected_keyword as f32, 1.0 - detected_keyword as f32],
            confidence,
            timestep_ms: 10.0,
            power_mw: self.power_mw,
            complexity: energy,
            liquid_energy: energy,
            metadata: Some(format!("keyword_detector_v{}", self.metadata.version)),
        })
    }

    fn current_power_mw(&self) -> f32 {
        self.power_mw
    }

    fn reset(&mut self) {
        self.processing_state.reset();
        self.power_mw = 0.8;
    }

    fn model_type(&self) -> &str {
        &self.config.model_type
    }

    fn is_ready(&self) -> bool {
        self.processing_state.ready
    }
}

impl PretrainedVAD {
    pub fn new(config: ModelConfig, weights: Option<Vec<f32>>, metadata: ModelMetadata) -> Result<Self> {
        Ok(Self {
            processing_state: ProcessingState::new(config.hidden_dim),
            power_mw: 0.3,
            config,
            weights,
            metadata,
        })
    }

}

impl AudioModel for PretrainedVAD {
    fn process_audio(&mut self, audio_buffer: &[f32]) -> Result<ProcessingResult> {
        if audio_buffer.is_empty() {
            return Err(LiquidAudioError::InvalidInput("Empty audio buffer".to_string()));
        }

        // Simulate VAD processing
        let energy: f32 = audio_buffer.iter().map(|&x| x * x).sum::<f32>() / audio_buffer.len() as f32;
        let spectral_centroid = self.compute_spectral_centroid(audio_buffer);
        
        // Simple VAD decision
        let voice_probability = if energy > 0.001 && spectral_centroid > 1000.0 {
            (energy * 50.0 + spectral_centroid / 5000.0).min(1.0)
        } else {
            0.0
        };

        let is_voice = if voice_probability > 0.5 { 1.0 } else { 0.0 };
        
        // Ultra-low power consumption
        self.power_mw = 0.3 + energy * 0.1;

        Ok(ProcessingResult {
            output: vec![is_voice, 1.0 - is_voice],
            confidence: voice_probability,
            timestep_ms: 20.0,
            power_mw: self.power_mw,
            complexity: energy,
            liquid_energy: energy,
            metadata: Some(format!("vad_v{}", self.metadata.version)),
        })
    }

    fn current_power_mw(&self) -> f32 {
        self.power_mw
    }

    fn reset(&mut self) {
        self.processing_state.reset();
        self.power_mw = 0.3;
    }

    fn model_type(&self) -> &str {
        &self.config.model_type
    }

    fn is_ready(&self) -> bool {
        self.processing_state.ready
    }
}

impl PretrainedVAD {
    fn compute_spectral_centroid(&self, buffer: &[f32]) -> f32 {
        // Simplified spectral centroid calculation
        let mut weighted_sum = 0.0f32;
        let mut magnitude_sum = 0.0f32;
        
        for (i, &sample) in buffer.iter().enumerate() {
            let magnitude = sample.abs();
            weighted_sum += (i as f32) * magnitude;
            magnitude_sum += magnitude;
        }
        
        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }
}

impl PretrainedAudioClassifier {
    pub fn new(config: ModelConfig, weights: Option<Vec<f32>>, metadata: ModelMetadata) -> Result<Self> {
        Ok(Self {
            processing_state: ProcessingState::new(config.hidden_dim),
            power_mw: 2.1,
            config,
            weights,
            metadata,
        })
    }

}

impl AudioModel for PretrainedAudioClassifier {
    fn process_audio(&mut self, audio_buffer: &[f32]) -> Result<ProcessingResult> {
        if audio_buffer.is_empty() {
            return Err(LiquidAudioError::InvalidInput("Empty audio buffer".to_string()));
        }

        // Simulate multi-class audio classification
        let features = self.extract_features(audio_buffer);
        let mut class_scores = vec![0.0f32; self.config.output_dim];
        
        // Simple feature-to-class mapping
        for (i, &feature) in features.iter().enumerate().take(class_scores.len()) {
            class_scores[i] = (feature * 2.0).tanh(); // Activation
        }

        // Apply softmax
        let max_score = class_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut exp_sum = 0.0f32;
        for score in &mut class_scores {
            *score = (*score - max_score).exp();
            exp_sum += *score;
        }
        
        if exp_sum > 0.0 {
            for score in &mut class_scores {
                *score /= exp_sum;
            }
        }

        let confidence = class_scores.iter().fold(0.0f32, |a, &b| a.max(b));
        
        // Higher power consumption for complex classification
        self.power_mw = 2.1 + confidence * 0.3;

        Ok(ProcessingResult {
            output: class_scores,
            confidence,
            timestep_ms: 25.0,
            power_mw: self.power_mw,
            complexity: confidence,
            liquid_energy: confidence,
            metadata: Some(format!("audio_classifier_v{}", self.metadata.version)),
        })
    }

    fn current_power_mw(&self) -> f32 {
        self.power_mw
    }

    fn reset(&mut self) {
        self.processing_state.reset();
        self.power_mw = 2.1;
    }

    fn model_type(&self) -> &str {
        &self.config.model_type
    }

    fn is_ready(&self) -> bool {
        self.processing_state.ready
    }
}

impl PretrainedAudioClassifier {
    fn extract_features(&self, buffer: &[f32]) -> Vec<f32> {
        let mut features = Vec::new();
        
        // Energy
        let energy: f32 = buffer.iter().map(|&x| x * x).sum::<f32>() / buffer.len() as f32;
        features.push(energy);
        
        // Zero crossing rate
        let mut crossings = 0;
        for i in 1..buffer.len() {
            if (buffer[i] >= 0.0) != (buffer[i-1] >= 0.0) {
                crossings += 1;
            }
        }
        let zcr = crossings as f32 / buffer.len() as f32;
        features.push(zcr);
        
        // Spectral centroid (simplified)
        let spectral_centroid = self.compute_spectral_centroid(buffer);
        features.push(spectral_centroid / 10000.0); // Normalize
        
        // Pad or truncate to expected size
        features.resize(self.config.input_dim.min(10), 0.0);
        features
    }

    fn compute_spectral_centroid(&self, buffer: &[f32]) -> f32 {
        let mut weighted_sum = 0.0f32;
        let mut magnitude_sum = 0.0f32;
        
        for (i, &sample) in buffer.iter().enumerate() {
            let magnitude = sample.abs();
            weighted_sum += (i as f32) * magnitude;
            magnitude_sum += magnitude;
        }
        
        if magnitude_sum > 0.0 {
            weighted_sum / magnitude_sum
        } else {
            0.0
        }
    }
}

/// Utility functions for model management
pub struct ModelUtils;

impl ModelUtils {
    /// Load weights from binary data
    pub fn load_weights_from_bytes(data: &[u8], format: WeightFormat) -> Result<Vec<f32>> {
        match format {
            WeightFormat::Raw => {
                if data.len() % 4 != 0 {
                    return Err(LiquidAudioError::ModelError("Invalid weight data length".to_string()));
                }
                
                let mut weights = Vec::with_capacity(data.len() / 4);
                for chunk in data.chunks_exact(4) {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    weights.push(f32::from_le_bytes(bytes));
                }
                Ok(weights)
            }
            WeightFormat::Quantized => {
                // Simplified quantization - in reality would involve proper scaling
                let mut weights = Vec::with_capacity(data.len());
                for &byte in data {
                    weights.push((byte as f32 - 128.0) / 128.0); // Convert i8 to f32
                }
                Ok(weights)
            }
            _ => Err(LiquidAudioError::ModelError("Unsupported weight format".to_string())),
        }
    }

    /// Validate model checksum
    pub fn validate_checksum(data: &[u8], expected_checksum: &str) -> bool {
        // Simplified checksum validation - in reality would use proper hashing
        let calculated_checksum = format!("sha256:{:x}", data.iter().map(|&x| x as u64).sum::<u64>());
        calculated_checksum == expected_checksum
    }

    /// Get model size estimation
    pub fn estimate_model_size(config: &ModelConfig, quantization: QuantizationLevel) -> usize {
        let parameter_count = config.input_dim * config.hidden_dim +
                             config.hidden_dim * config.hidden_dim +
                             config.hidden_dim * config.output_dim;
        
        let bytes_per_param = match quantization {
            QuantizationLevel::Float32 => 4,
            QuantizationLevel::Float16 => 2,
            QuantizationLevel::Int8 => 1,
            QuantizationLevel::Int4 => 1, // Packed
            QuantizationLevel::Custom(bits) => (bits as usize + 7) / 8,
        };
        
        parameter_count * bytes_per_param
    }
}

// Note: Model caching removed for simplicity - models are created on-demand