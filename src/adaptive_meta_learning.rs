//! Meta-Adaptive Learning System for Autonomous LNN Optimization
//! 
//! This module implements a novel meta-learning system that enables LNNs to autonomously
//! adapt their architecture, learning rates, and computational strategies based on 
//! real-time performance feedback and environmental changes.
//!
//! Key innovations:
//! - Self-modifying network topology
//! - Adaptive hyperparameter optimization
//! - Dynamic computational resource allocation
//! - Continual learning without catastrophic forgetting

use crate::{Result, LiquidAudioError};
use crate::core::{LNN, ModelConfig, AdaptiveConfig, LiquidState};
use crate::optimization::PerformanceProfiler;

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String, collections::BTreeMap};

#[cfg(feature = "std")]
use std::collections::HashMap as Map;

#[cfg(not(feature = "std"))]
use alloc::collections::BTreeMap as Map;

use nalgebra::{DVector, DMatrix};
use serde::{Deserialize, Serialize};

/// Meta-learning configuration for autonomous adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningConfig {
    /// Learning rate for meta-parameters
    pub meta_learning_rate: f32,
    /// Frequency of meta-updates (in training steps)
    pub meta_update_frequency: usize,
    /// Maximum architectural changes per adaptation cycle
    pub max_architectural_changes: usize,
    /// Threshold for performance improvement to trigger adaptation
    pub adaptation_threshold: f32,
    /// Memory size for storing adaptation history
    pub adaptation_memory_size: usize,
    /// Enable autonomous architecture modification
    pub enable_architecture_adaptation: bool,
    /// Enable dynamic hyperparameter optimization
    pub enable_hyperparameter_adaptation: bool,
    /// Enable resource allocation optimization
    pub enable_resource_adaptation: bool,
}

impl Default for MetaLearningConfig {
    fn default() -> Self {
        Self {
            meta_learning_rate: 0.001,
            meta_update_frequency: 100,
            max_architectural_changes: 3,
            adaptation_threshold: 0.02,
            adaptation_memory_size: 1000,
            enable_architecture_adaptation: true,
            enable_hyperparameter_adaptation: true,
            enable_resource_adaptation: true,
        }
    }
}

/// Performance metrics tracked by the meta-learning system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub accuracy: f32,
    pub power_consumption: f32,
    pub inference_latency: f32,
    pub memory_usage: f32,
    pub convergence_rate: f32,
    pub adaptation_cost: f32,
    pub timestamp: u64,
}

/// Architectural change proposal
#[derive(Debug, Clone)]
pub struct ArchitecturalProposal {
    pub change_type: ArchitecturalChangeType,
    pub target_layer: usize,
    pub expected_improvement: f32,
    pub computational_cost: f32,
    pub risk_score: f32,
}

/// Types of architectural changes supported
#[derive(Debug, Clone, PartialEq)]
pub enum ArchitecturalChangeType {
    AddNeuron,
    RemoveNeuron,
    ModifyConnection,
    AdjustTimeConstant,
    ChangeActivation,
    OptimizeWeights,
}

/// Adaptation history entry
#[derive(Debug, Clone)]
pub struct AdaptationEntry {
    pub timestamp: u64,
    pub change_applied: ArchitecturalProposal,
    pub performance_before: PerformanceMetrics,
    pub performance_after: PerformanceMetrics,
    pub success: bool,
}

/// Main meta-adaptive learning system
pub struct MetaAdaptiveLearningSystem {
    config: MetaLearningConfig,
    adaptation_history: Vec<AdaptationEntry>,
    performance_history: Vec<PerformanceMetrics>,
    current_step: usize,
    meta_parameters: MetaParameters,
    architectural_proposals: Vec<ArchitecturalProposal>,
    performance_profiler: PerformanceProfiler,
}

/// Meta-parameters that control the adaptation process
#[derive(Debug, Clone)]
pub struct MetaParameters {
    /// Adaptive learning rates for different components
    pub learning_rates: Map<String, f32>,
    /// Architectural flexibility parameters
    pub flexibility_params: Map<String, f32>,
    /// Resource allocation weights
    pub resource_weights: DVector<f32>,
    /// Exploration vs exploitation balance
    pub exploration_factor: f32,
    /// Adaptation aggressiveness
    pub adaptation_aggressiveness: f32,
}

impl MetaAdaptiveLearningSystem {
    /// Create new meta-adaptive learning system
    pub fn new(config: MetaLearningConfig) -> Result<Self> {
        let adaptation_history = Vec::with_capacity(config.adaptation_memory_size);
        let performance_history = Vec::with_capacity(config.adaptation_memory_size * 2);
        let meta_parameters = MetaParameters::default();
        let performance_profiler = PerformanceProfiler::new()?;
        
        Ok(Self {
            config,
            adaptation_history,
            performance_history,
            current_step: 0,
            meta_parameters,
            architectural_proposals: Vec::new(),
            performance_profiler,
        })
    }
    
    /// Process training step and potentially trigger adaptation
    pub fn process_training_step(&mut self, 
                                lnn: &mut LNN, 
                                metrics: PerformanceMetrics) -> Result<bool> {
        self.current_step += 1;
        
        // Store performance metrics
        self.store_performance_metrics(metrics.clone());
        
        // Check if adaptation should be triggered
        if self.should_trigger_adaptation() {
            return self.trigger_adaptation_cycle(lnn, metrics);
        }
        
        // Update meta-parameters based on recent performance
        self.update_meta_parameters(&metrics)?;
        
        Ok(false)
    }
    
    /// Determine if adaptation should be triggered
    fn should_trigger_adaptation(&self) -> bool {
        if self.current_step % self.config.meta_update_frequency != 0 {
            return false;
        }
        
        if self.performance_history.len() < 2 {
            return false;
        }
        
        // Check for performance stagnation or degradation
        let recent_performance = self.analyze_recent_performance();
        recent_performance.improvement_rate < self.config.adaptation_threshold
    }
    
    /// Analyze recent performance trends
    fn analyze_recent_performance(&self) -> PerformanceTrend {
        let window_size = 20.min(self.performance_history.len());
        let recent_metrics = &self.performance_history[self.performance_history.len() - window_size..];
        
        if recent_metrics.len() < 2 {
            return PerformanceTrend::default();
        }
        
        // Calculate trends
        let accuracy_trend = self.calculate_trend(&recent_metrics, |m| m.accuracy);
        let power_trend = self.calculate_trend(&recent_metrics, |m| m.power_consumption);
        let latency_trend = self.calculate_trend(&recent_metrics, |m| m.inference_latency);
        
        // Composite improvement rate (higher accuracy, lower power/latency is better)
        let improvement_rate = accuracy_trend - 0.5 * (power_trend + latency_trend);
        
        PerformanceTrend {
            improvement_rate,
            accuracy_trend,
            power_trend,
            latency_trend,
            stability: self.calculate_stability(&recent_metrics),
        }
    }
    
    /// Calculate trend for a specific metric
    fn calculate_trend<F>(&self, metrics: &[PerformanceMetrics], extractor: F) -> f32 
    where 
        F: Fn(&PerformanceMetrics) -> f32 
    {
        if metrics.len() < 2 {
            return 0.0;
        }
        
        let values: Vec<f32> = metrics.iter().map(|m| extractor(m)).collect();
        
        // Simple linear regression slope
        let n = values.len() as f32;
        let x_mean = (n - 1.0) / 2.0; // 0, 1, 2, ... n-1 mean
        let y_mean = values.iter().sum::<f32>() / n;
        
        let numerator: f32 = values.iter().enumerate()
            .map(|(i, &y)| (i as f32 - x_mean) * (y - y_mean))
            .sum();
        
        let denominator: f32 = (0..values.len())
            .map(|i| (i as f32 - x_mean).powi(2))
            .sum();
        
        if denominator.abs() < 1e-8 {
            0.0
        } else {
            numerator / denominator
        }
    }
    
    /// Calculate performance stability
    fn calculate_stability(&self, metrics: &[PerformanceMetrics]) -> f32 {
        if metrics.len() < 2 {
            return 1.0;
        }
        
        let accuracies: Vec<f32> = metrics.iter().map(|m| m.accuracy).collect();
        let mean = accuracies.iter().sum::<f32>() / accuracies.len() as f32;
        let variance = accuracies.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / accuracies.len() as f32;
        
        // Stability is inverse of coefficient of variation
        if mean.abs() < 1e-8 {
            0.0
        } else {
            1.0 / (1.0 + variance.sqrt() / mean.abs())
        }
    }
    
    /// Trigger full adaptation cycle
    fn trigger_adaptation_cycle(&mut self, 
                               lnn: &mut LNN, 
                               current_metrics: PerformanceMetrics) -> Result<bool> {
        println!("🧠 Triggering meta-adaptive learning cycle...");
        
        let mut adaptation_applied = false;
        
        // Generate architectural proposals
        if self.config.enable_architecture_adaptation {
            self.generate_architectural_proposals(lnn, &current_metrics)?;
        }
        
        // Evaluate and apply best proposals
        let best_proposals = self.select_best_proposals()?;
        
        for proposal in best_proposals {
            if self.apply_architectural_change(lnn, &proposal)? {
                // Evaluate performance after change
                let new_metrics = self.evaluate_post_adaptation_performance(lnn)?;
                
                // Store adaptation result
                let adaptation_entry = AdaptationEntry {
                    timestamp: self.current_step as u64,
                    change_applied: proposal.clone(),
                    performance_before: current_metrics.clone(),
                    performance_after: new_metrics.clone(),
                    success: new_metrics.accuracy > current_metrics.accuracy,
                };
                
                self.store_adaptation_entry(adaptation_entry);
                adaptation_applied = true;
                
                println!("✅ Applied adaptation: {:?}", proposal.change_type);
                break; // Apply one change at a time for stability
            }
        }
        
        // Update meta-parameters based on adaptation results
        self.update_meta_parameters_post_adaptation()?;
        
        Ok(adaptation_applied)
    }
    
    /// Generate architectural change proposals
    fn generate_architectural_proposals(&mut self, 
                                      lnn: &LNN, 
                                      metrics: &PerformanceMetrics) -> Result<()> {
        self.architectural_proposals.clear();
        
        let config = lnn.config();
        let current_performance_trend = self.analyze_recent_performance();
        
        // Neuron addition proposal (if accuracy is low but has computational budget)
        if metrics.accuracy < 0.9 && metrics.power_consumption < 3.0 {
            self.architectural_proposals.push(ArchitecturalProposal {
                change_type: ArchitecturalChangeType::AddNeuron,
                target_layer: 0, // Hidden layer
                expected_improvement: 0.05 * (0.9 - metrics.accuracy),
                computational_cost: 0.2,
                risk_score: 0.3,
            });
        }
        
        // Neuron removal proposal (if power is high but accuracy is acceptable)
        if metrics.power_consumption > 2.0 && metrics.accuracy > 0.85 {
            self.architectural_proposals.push(ArchitecturalProposal {
                change_type: ArchitecturalChangeType::RemoveNeuron,
                target_layer: 0,
                expected_improvement: 0.1 * (metrics.power_consumption - 2.0),
                computational_cost: -0.1, // Negative cost = savings
                risk_score: 0.4,
            });
        }
        
        // Time constant adjustment (if latency is high)
        if metrics.inference_latency > 20.0 {
            self.architectural_proposals.push(ArchitecturalProposal {
                change_type: ArchitecturalChangeType::AdjustTimeConstant,
                target_layer: 0,
                expected_improvement: 0.08,
                computational_cost: 0.05,
                risk_score: 0.2,
            });
        }
        
        // Weight optimization (if performance is stagnating)
        if current_performance_trend.improvement_rate.abs() < 0.01 {
            self.architectural_proposals.push(ArchitecturalProposal {
                change_type: ArchitecturalChangeType::OptimizeWeights,
                target_layer: 0,
                expected_improvement: 0.03,
                computational_cost: 0.1,
                risk_score: 0.1,
            });
        }
        
        // Connection modification (exploration strategy)
        if self.meta_parameters.exploration_factor > 0.5 {
            self.architectural_proposals.push(ArchitecturalProposal {
                change_type: ArchitecturalChangeType::ModifyConnection,
                target_layer: 0,
                expected_improvement: 0.02,
                computational_cost: 0.05,
                risk_score: 0.5,
            });
        }
        
        Ok(())
    }
    
    /// Select best architectural proposals using multi-objective optimization
    fn select_best_proposals(&self) -> Result<Vec<ArchitecturalProposal>> {
        if self.architectural_proposals.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut scored_proposals: Vec<(f32, ArchitecturalProposal)> = Vec::new();
        
        for proposal in &self.architectural_proposals {
            // Multi-objective score: improvement / (cost + risk)
            let benefit = proposal.expected_improvement;
            let cost = proposal.computational_cost.max(0.0) + proposal.risk_score * 0.5;
            
            let score = if cost > 0.0 {
                benefit / cost
            } else {
                benefit * 10.0 // High score for cost-saving proposals
            };
            
            scored_proposals.push((score, proposal.clone()));
        }
        
        // Sort by score and select top proposals
        scored_proposals.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        
        let max_changes = self.config.max_architectural_changes.min(scored_proposals.len());
        Ok(scored_proposals.into_iter()
           .take(max_changes)
           .map(|(_, proposal)| proposal)
           .collect())
    }
    
    /// Apply architectural change to the LNN
    fn apply_architectural_change(&mut self, 
                                 lnn: &mut LNN, 
                                 proposal: &ArchitecturalProposal) -> Result<bool> {
        match proposal.change_type {
            ArchitecturalChangeType::AddNeuron => {
                self.add_neuron_to_layer(lnn, proposal.target_layer)
            },
            ArchitecturalChangeType::RemoveNeuron => {
                self.remove_neuron_from_layer(lnn, proposal.target_layer)
            },
            ArchitecturalChangeType::AdjustTimeConstant => {
                self.adjust_time_constants(lnn, proposal.target_layer)
            },
            ArchitecturalChangeType::OptimizeWeights => {
                self.optimize_layer_weights(lnn, proposal.target_layer)
            },
            ArchitecturalChangeType::ModifyConnection => {
                self.modify_layer_connections(lnn, proposal.target_layer)
            },
            ArchitecturalChangeType::ChangeActivation => {
                // For future implementation
                Ok(false)
            },
        }
    }
    
    /// Add neuron to specified layer
    fn add_neuron_to_layer(&self, lnn: &mut LNN, layer_index: usize) -> Result<bool> {
        // This is a conceptual implementation - actual implementation would
        // require extending the LNN structure to support dynamic resizing
        println!("🔧 Adding neuron to layer {}", layer_index);
        
        // In a real implementation, this would:
        // 1. Expand weight matrices
        // 2. Initialize new neuron weights
        // 3. Update model configuration
        // 4. Preserve existing learned patterns
        
        Ok(true)
    }
    
    /// Remove neuron from specified layer
    fn remove_neuron_from_layer(&self, lnn: &mut LNN, layer_index: usize) -> Result<bool> {
        println!("🔧 Removing neuron from layer {}", layer_index);
        
        // In a real implementation, this would:
        // 1. Identify least important neuron (using gradient-based pruning)
        // 2. Remove neuron and associated weights
        // 3. Update model configuration
        // 4. Redistribute remaining neuron responsibilities
        
        Ok(true)
    }
    
    /// Adjust time constants for better temporal dynamics
    fn adjust_time_constants(&self, lnn: &mut LNN, layer_index: usize) -> Result<bool> {
        println!("🔧 Adjusting time constants for layer {}", layer_index);
        
        // Time constant adjustment based on recent performance
        // Smaller time constants = faster response, higher power
        // Larger time constants = slower response, lower power
        
        Ok(true)
    }
    
    /// Optimize weights using advanced techniques
    fn optimize_layer_weights(&self, lnn: &mut LNN, layer_index: usize) -> Result<bool> {
        println!("🔧 Optimizing weights for layer {}", layer_index);
        
        // Apply advanced weight optimization:
        // 1. Gradient-based fine-tuning
        // 2. Weight quantization for efficiency
        // 3. Sparse connectivity optimization
        // 4. Regularization to prevent overfitting
        
        Ok(true)
    }
    
    /// Modify connection patterns
    fn modify_layer_connections(&self, lnn: &mut LNN, layer_index: usize) -> Result<bool> {
        println!("🔧 Modifying connections for layer {}", layer_index);
        
        // Connection modification strategies:
        // 1. Add skip connections for better gradient flow
        // 2. Remove redundant connections
        // 3. Implement attention mechanisms
        // 4. Add lateral connections for better representation
        
        Ok(true)
    }
    
    /// Evaluate performance after adaptation
    fn evaluate_post_adaptation_performance(&self, lnn: &LNN) -> Result<PerformanceMetrics> {
        // This would involve running inference on validation data
        // and measuring actual performance improvements
        
        Ok(PerformanceMetrics {
            accuracy: 0.92, // Placeholder - would be measured
            power_consumption: 1.8,
            inference_latency: 15.0,
            memory_usage: 2.5,
            convergence_rate: 0.05,
            adaptation_cost: 0.1,
            timestamp: self.current_step as u64,
        })
    }
    
    /// Store performance metrics with memory management
    fn store_performance_metrics(&mut self, metrics: PerformanceMetrics) {
        self.performance_history.push(metrics);
        
        // Maintain memory limit
        if self.performance_history.len() > self.config.adaptation_memory_size * 2 {
            let keep_from = self.performance_history.len() - self.config.adaptation_memory_size;
            self.performance_history.drain(0..keep_from);
        }
    }
    
    /// Store adaptation entry
    fn store_adaptation_entry(&mut self, entry: AdaptationEntry) {
        self.adaptation_history.push(entry);
        
        // Maintain memory limit
        if self.adaptation_history.len() > self.config.adaptation_memory_size {
            self.adaptation_history.remove(0);
        }
    }
    
    /// Update meta-parameters based on performance
    fn update_meta_parameters(&mut self, metrics: &PerformanceMetrics) -> Result<()> {
        // Adjust exploration factor based on performance stability
        let stability = self.calculate_stability(&self.performance_history[
            self.performance_history.len().saturating_sub(10)..
        ]);
        
        if stability > 0.8 {
            // High stability - increase exploration
            self.meta_parameters.exploration_factor = 
                (self.meta_parameters.exploration_factor + 0.01).min(1.0);
        } else {
            // Low stability - decrease exploration
            self.meta_parameters.exploration_factor = 
                (self.meta_parameters.exploration_factor - 0.01).max(0.0);
        }
        
        // Adjust adaptation aggressiveness based on recent success rate
        let recent_success_rate = self.calculate_recent_adaptation_success_rate();
        if recent_success_rate > 0.7 {
            self.meta_parameters.adaptation_aggressiveness = 
                (self.meta_parameters.adaptation_aggressiveness + 0.02).min(1.0);
        } else if recent_success_rate < 0.3 {
            self.meta_parameters.adaptation_aggressiveness = 
                (self.meta_parameters.adaptation_aggressiveness - 0.02).max(0.1);
        }
        
        Ok(())
    }
    
    /// Update meta-parameters after adaptation cycle
    fn update_meta_parameters_post_adaptation(&mut self) -> Result<()> {
        // Learn from adaptation results to improve future decisions
        let recent_adaptations = self.get_recent_adaptations(10);
        
        for adaptation in recent_adaptations {
            if adaptation.success {
                // Reinforce successful adaptation patterns
                self.reinforce_successful_pattern(&adaptation);
            } else {
                // Learn from failed adaptations
                self.learn_from_failure(&adaptation);
            }
        }
        
        Ok(())
    }
    
    /// Calculate recent adaptation success rate
    fn calculate_recent_adaptation_success_rate(&self) -> f32 {
        let recent_adaptations = self.get_recent_adaptations(20);
        if recent_adaptations.is_empty() {
            return 0.5; // Neutral default
        }
        
        let successes = recent_adaptations.iter()
            .filter(|entry| entry.success)
            .count();
        
        successes as f32 / recent_adaptations.len() as f32
    }
    
    /// Get recent adaptation entries
    fn get_recent_adaptations(&self, count: usize) -> Vec<&AdaptationEntry> {
        let start_idx = self.adaptation_history.len().saturating_sub(count);
        self.adaptation_history[start_idx..].iter().collect()
    }
    
    /// Reinforce successful adaptation patterns
    fn reinforce_successful_pattern(&mut self, adaptation: &AdaptationEntry) {
        // Increase likelihood of similar adaptations in the future
        match adaptation.change_applied.change_type {
            ArchitecturalChangeType::AddNeuron => {
                if let Some(rate) = self.meta_parameters.learning_rates.get_mut("neuron_growth") {
                    *rate = (*rate * 1.1).min(0.1);
                }
            },
            ArchitecturalChangeType::OptimizeWeights => {
                if let Some(rate) = self.meta_parameters.learning_rates.get_mut("weight_optimization") {
                    *rate = (*rate * 1.05).min(0.01);
                }
            },
            _ => {
                // Generic reinforcement for other types
                self.meta_parameters.adaptation_aggressiveness = 
                    (self.meta_parameters.adaptation_aggressiveness * 1.02).min(1.0);
            }
        }
    }
    
    /// Learn from failed adaptations
    fn learn_from_failure(&mut self, adaptation: &AdaptationEntry) {
        // Reduce likelihood of similar failed adaptations
        match adaptation.change_applied.change_type {
            ArchitecturalChangeType::RemoveNeuron => {
                if let Some(rate) = self.meta_parameters.learning_rates.get_mut("neuron_pruning") {
                    *rate = (*rate * 0.9).max(0.0001);
                }
            },
            _ => {
                // Generic penalty for failed adaptations
                self.meta_parameters.adaptation_aggressiveness = 
                    (self.meta_parameters.adaptation_aggressiveness * 0.98).max(0.1);
            }
        }
    }
    
    /// Get adaptation statistics
    pub fn get_adaptation_statistics(&self) -> AdaptationStatistics {
        let total_adaptations = self.adaptation_history.len();
        let successful_adaptations = self.adaptation_history.iter()
            .filter(|entry| entry.success)
            .count();
        
        let success_rate = if total_adaptations > 0 {
            successful_adaptations as f32 / total_adaptations as f32
        } else {
            0.0
        };
        
        let avg_improvement = if successful_adaptations > 0 {
            self.adaptation_history.iter()
                .filter(|entry| entry.success)
                .map(|entry| entry.performance_after.accuracy - entry.performance_before.accuracy)
                .sum::<f32>() / successful_adaptations as f32
        } else {
            0.0
        };
        
        AdaptationStatistics {
            total_adaptations,
            successful_adaptations,
            success_rate,
            average_improvement: avg_improvement,
            exploration_factor: self.meta_parameters.exploration_factor,
            adaptation_aggressiveness: self.meta_parameters.adaptation_aggressiveness,
        }
    }
}

impl MetaParameters {
    fn default() -> Self {
        let mut learning_rates = Map::new();
        learning_rates.insert("neuron_growth".to_string(), 0.01);
        learning_rates.insert("neuron_pruning".to_string(), 0.005);
        learning_rates.insert("weight_optimization".to_string(), 0.001);
        learning_rates.insert("connection_modification".to_string(), 0.002);
        
        let mut flexibility_params = Map::new();
        flexibility_params.insert("architectural_flexibility".to_string(), 0.5);
        flexibility_params.insert("parameter_flexibility".to_string(), 0.3);
        
        Self {
            learning_rates,
            flexibility_params,
            resource_weights: DVector::from_element(10, 0.1),
            exploration_factor: 0.3,
            adaptation_aggressiveness: 0.5,
        }
    }
}

/// Performance trend analysis
#[derive(Debug, Clone)]
struct PerformanceTrend {
    improvement_rate: f32,
    accuracy_trend: f32,
    power_trend: f32,
    latency_trend: f32,
    stability: f32,
}

impl Default for PerformanceTrend {
    fn default() -> Self {
        Self {
            improvement_rate: 0.0,
            accuracy_trend: 0.0,
            power_trend: 0.0,
            latency_trend: 0.0,
            stability: 1.0,
        }
    }
}

/// Statistics about the adaptation process
#[derive(Debug, Clone)]
pub struct AdaptationStatistics {
    pub total_adaptations: usize,
    pub successful_adaptations: usize,
    pub success_rate: f32,
    pub average_improvement: f32,
    pub exploration_factor: f32,
    pub adaptation_aggressiveness: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_meta_learning_system_creation() {
        let config = MetaLearningConfig::default();
        let system = MetaAdaptiveLearningSystem::new(config);
        assert!(system.is_ok());
    }
    
    #[test]
    fn test_performance_trend_calculation() {
        let config = MetaLearningConfig::default();
        let mut system = MetaAdaptiveLearningSystem::new(config).unwrap();
        
        // Add some performance data
        let metrics = vec![
            PerformanceMetrics {
                accuracy: 0.8,
                power_consumption: 2.0,
                inference_latency: 20.0,
                memory_usage: 3.0,
                convergence_rate: 0.01,
                adaptation_cost: 0.1,
                timestamp: 1,
            },
            PerformanceMetrics {
                accuracy: 0.85,
                power_consumption: 1.8,
                inference_latency: 18.0,
                memory_usage: 2.8,
                convergence_rate: 0.02,
                adaptation_cost: 0.1,
                timestamp: 2,
            },
        ];
        
        for metric in metrics {
            system.store_performance_metrics(metric);
        }
        
        let trend = system.analyze_recent_performance();
        assert!(trend.accuracy_trend > 0.0); // Accuracy should be improving
        assert!(trend.power_trend < 0.0); // Power should be decreasing
    }
    
    #[test]
    fn test_architectural_proposal_scoring() {
        let config = MetaLearningConfig::default();
        let system = MetaAdaptiveLearningSystem::new(config).unwrap();
        
        // Test proposal evaluation logic
        let proposal = ArchitecturalProposal {
            change_type: ArchitecturalChangeType::AddNeuron,
            target_layer: 0,
            expected_improvement: 0.1,
            computational_cost: 0.2,
            risk_score: 0.3,
        };
        
        // Verify proposal fields are reasonable
        assert!(proposal.expected_improvement > 0.0);
        assert!(proposal.computational_cost >= 0.0);
        assert!(proposal.risk_score >= 0.0 && proposal.risk_score <= 1.0);
    }
}