//! Load balancing and auto-scaling for Liquid Neural Networks
//!
//! Provides intelligent load distribution, auto-scaling triggers,
//! and system resource management for production deployments.

use crate::{Result, LiquidAudioError, ModelConfig, ProcessingResult};
use crate::concurrent::{ThreadPool, ThreadPoolConfig, WorkResult, EmbeddedScheduler};
use crate::diagnostics::{HealthStatus, HealthReport, DiagnosticsCollector};
#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String, collections::VecDeque};

#[cfg(feature = "std")]
use std::{vec::Vec, string::String, collections::VecDeque};
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

#[cfg(feature = "std")]
use std::sync::{Arc, Mutex};
#[cfg(feature = "std")]
use std::time::{Duration, Instant};

/// Load balancer for distributing work across multiple processing units
#[derive(Debug)]
pub struct LoadBalancer {
    /// Available processing nodes
    nodes: Vec<ProcessingNode>,
    /// Load balancing strategy
    strategy: LoadBalancingStrategy,
    /// Current node index (for round-robin)
    current_node: AtomicUsize,
    /// Load balancer statistics
    stats: LoadBalancerStats,
    /// Health monitoring enabled
    health_monitoring: bool,
}

/// Processing node representation
#[derive(Debug, Clone)]
pub struct ProcessingNode {
    /// Node ID
    pub id: String,
    /// Node capacity (requests per second)
    pub capacity: f32,
    /// Current load (0.0 to 1.0)
    pub current_load: f32,
    /// Average response time (ms)
    pub avg_response_time_ms: f32,
    /// Node health status
    pub health: HealthStatus,
    /// Node enabled
    pub enabled: bool,
    /// Total requests processed
    pub requests_processed: u64,
    /// Error count
    pub error_count: u64,
    /// Last health check
    pub last_health_check: u64,
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least connections
    LeastConnections,
    /// Least response time
    LeastResponseTime,
    /// Weighted round-robin
    WeightedRoundRobin,
    /// Least load
    LeastLoad,
    /// Health-aware selection
    HealthAware,
}

/// Load balancer statistics
#[derive(Debug, Clone, Default)]
pub struct LoadBalancerStats {
    /// Total requests distributed
    pub total_requests: u64,
    /// Requests per node
    pub requests_per_node: Vec<u64>,
    /// Failed requests
    pub failed_requests: u64,
    /// Average distribution time (ms)
    pub avg_distribution_time_ms: f32,
    /// Node utilization rates
    pub node_utilizations: Vec<f32>,
}

impl LoadBalancer {
    /// Create new load balancer
    pub fn new(
        nodes: Vec<ProcessingNode>,
        strategy: LoadBalancingStrategy,
        health_monitoring: bool,
    ) -> Self {
        let node_count = nodes.len();
        Self {
            nodes,
            strategy,
            current_node: AtomicUsize::new(0),
            stats: LoadBalancerStats {
                requests_per_node: vec![0; node_count],
                node_utilizations: vec![0.0; node_count],
                ..Default::default()
            },
            health_monitoring,
        }
    }

    /// Select best node for processing
    pub fn select_node(&mut self) -> Result<usize> {
        let healthy_nodes: Vec<_> = self.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.enabled && node.health != HealthStatus::Failed)
            .collect();

        if healthy_nodes.is_empty() {
            return Err(LiquidAudioError::ResourceExhausted("No healthy nodes available".to_string()));
        }

        let selected_index = match self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                self.round_robin_selection(&healthy_nodes)
            }
            LoadBalancingStrategy::LeastConnections => {
                self.least_connections_selection(&healthy_nodes)
            }
            LoadBalancingStrategy::LeastResponseTime => {
                self.least_response_time_selection(&healthy_nodes)
            }
            LoadBalancingStrategy::WeightedRoundRobin => {
                self.weighted_round_robin_selection(&healthy_nodes)
            }
            LoadBalancingStrategy::LeastLoad => {
                self.least_load_selection(&healthy_nodes)
            }
            LoadBalancingStrategy::HealthAware => {
                self.health_aware_selection(&healthy_nodes)
            }
        };

        // Update statistics
        self.stats.total_requests += 1;
        if selected_index < self.stats.requests_per_node.len() {
            self.stats.requests_per_node[selected_index] += 1;
        }

        Ok(selected_index)
    }

    /// Round-robin node selection
    fn round_robin_selection(&self, healthy_nodes: &[(usize, &ProcessingNode)]) -> usize {
        let current = self.current_node.load(Ordering::Relaxed);
        let next = (current + 1) % healthy_nodes.len();
        self.current_node.store(next, Ordering::Relaxed);
        healthy_nodes[current].0
    }

    /// Least connections selection
    fn least_connections_selection(&self, healthy_nodes: &[(usize, &ProcessingNode)]) -> usize {
        let min_node = healthy_nodes
            .iter()
            .min_by(|(_, a), (_, b)| a.current_load.partial_cmp(&b.current_load).unwrap())
            .unwrap();
        min_node.0
    }

    /// Least response time selection
    fn least_response_time_selection(&self, healthy_nodes: &[(usize, &ProcessingNode)]) -> usize {
        let min_node = healthy_nodes
            .iter()
            .min_by(|(_, a), (_, b)| a.avg_response_time_ms.partial_cmp(&b.avg_response_time_ms).unwrap())
            .unwrap();
        min_node.0
    }

    /// Weighted round-robin selection
    fn weighted_round_robin_selection(&self, healthy_nodes: &[(usize, &ProcessingNode)]) -> usize {
        // Simple implementation - select based on capacity
        let total_capacity: f32 = healthy_nodes.iter().map(|(_, node)| node.capacity).sum();
        let mut cumulative = 0.0;
        let threshold = (self.stats.total_requests as f32 % total_capacity) / total_capacity;

        for &(index, node) in healthy_nodes {
            cumulative += node.capacity / total_capacity;
            if threshold <= cumulative {
                return index;
            }
        }

        healthy_nodes[0].0 // Fallback
    }

    /// Least load selection
    fn least_load_selection(&self, healthy_nodes: &[(usize, &ProcessingNode)]) -> usize {
        let min_node = healthy_nodes
            .iter()
            .min_by(|(_, a), (_, b)| {
                let load_a = a.current_load / a.capacity;
                let load_b = b.current_load / b.capacity;
                load_a.partial_cmp(&load_b).unwrap()
            })
            .unwrap();
        min_node.0
    }

    /// Health-aware selection
    fn health_aware_selection(&self, healthy_nodes: &[(usize, &ProcessingNode)]) -> usize {
        // Prioritize healthy nodes, then use least load
        let healthy_only: Vec<_> = healthy_nodes
            .iter()
            .filter(|(_, node)| node.health == HealthStatus::Healthy)
            .collect();

        if !healthy_only.is_empty() {
            let min_node = healthy_only
                .iter()
                .min_by(|(_, a), (_, b)| a.current_load.partial_cmp(&b.current_load).unwrap())
                .unwrap();
            min_node.0
        } else {
            // Fall back to any available node
            self.least_load_selection(healthy_nodes)
        }
    }

    /// Update node statistics
    pub fn update_node_stats(&mut self, node_id: usize, response_time_ms: f32, success: bool) {
        if node_id < self.nodes.len() {
            let node = &mut self.nodes[node_id];
            
            // Update response time (exponential moving average)
            let alpha = 0.1;
            node.avg_response_time_ms = alpha * response_time_ms + (1.0 - alpha) * node.avg_response_time_ms;
            
            if success {
                node.requests_processed += 1;
            } else {
                node.error_count += 1;
            }

            // Update utilization
            if node_id < self.stats.node_utilizations.len() {
                self.stats.node_utilizations[node_id] = node.current_load;
            }
        }

        if !success {
            self.stats.failed_requests += 1;
        }
    }

    /// Perform health checks on all nodes
    pub fn health_check_nodes(&mut self) {
        if !self.health_monitoring {
            return;
        }

        let current_time = Self::current_timestamp();
        
        for node in &mut self.nodes {
            // Simple health check based on error rate and response time
            let error_rate = if node.requests_processed > 0 {
                node.error_count as f32 / node.requests_processed as f32
            } else {
                0.0
            };

            node.health = if error_rate > 0.1 {
                HealthStatus::Critical
            } else if error_rate > 0.05 || node.avg_response_time_ms > 100.0 {
                HealthStatus::Degraded
            } else if node.current_load > 0.9 {
                HealthStatus::Warning
            } else {
                HealthStatus::Healthy
            };

            node.last_health_check = current_time;
        }
    }

    /// Get load balancer statistics
    pub fn get_stats(&self) -> &LoadBalancerStats {
        &self.stats
    }

    /// Get current timestamp
    fn current_timestamp() -> u64 {
        static mut TIMESTAMP: u64 = 0;
        unsafe {
            TIMESTAMP += 1;
            TIMESTAMP
        }
    }
}

/// Auto-scaling controller
#[derive(Debug)]
pub struct AutoScaler {
    /// Scaling configuration
    config: ScalingConfig,
    /// Current scale level
    current_scale: AtomicUsize,
    /// Scaling history
    scaling_history: VecDeque<ScalingEvent>,
    /// Metrics buffer
    metrics_buffer: VecDeque<ScalingMetrics>,
    /// Last scaling action timestamp
    last_scaling_action: AtomicU64,
    /// Scaling enabled
    enabled: AtomicBool,
}

/// Scaling configuration
#[derive(Debug, Clone)]
pub struct ScalingConfig {
    /// Minimum scale (number of workers/nodes)
    pub min_scale: usize,
    /// Maximum scale
    pub max_scale: usize,
    /// Target CPU utilization (0.0 to 1.0)
    pub target_cpu_utilization: f32,
    /// Target queue depth
    pub target_queue_depth: usize,
    /// Scale-up threshold
    pub scale_up_threshold: f32,
    /// Scale-down threshold
    pub scale_down_threshold: f32,
    /// Cooldown period (ms)
    pub cooldown_ms: u64,
    /// Metrics window size
    pub metrics_window_size: usize,
    /// Aggressive scaling enabled
    pub aggressive_scaling: bool,
}

impl Default for ScalingConfig {
    fn default() -> Self {
        Self {
            min_scale: 1,
            max_scale: 10,
            target_cpu_utilization: 0.7,
            target_queue_depth: 100,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            cooldown_ms: 30000, // 30 seconds
            metrics_window_size: 10,
            aggressive_scaling: false,
        }
    }
}

/// Scaling event
#[derive(Debug, Clone)]
pub struct ScalingEvent {
    /// Event timestamp
    pub timestamp: u64,
    /// Scaling action
    pub action: ScalingAction,
    /// Previous scale
    pub previous_scale: usize,
    /// New scale
    pub new_scale: usize,
    /// Trigger reason
    pub reason: String,
    /// Metrics at time of scaling
    pub metrics: ScalingMetrics,
}

/// Scaling actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalingAction {
    ScaleUp,
    ScaleDown,
    NoAction,
}

/// Metrics for scaling decisions
#[derive(Debug, Clone)]
pub struct ScalingMetrics {
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f32,
    /// Queue depth
    pub queue_depth: usize,
    /// Average response time (ms)
    pub avg_response_time_ms: f32,
    /// Error rate
    pub error_rate: f32,
    /// Throughput (requests per second)
    pub throughput_rps: f32,
    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f32,
    /// Timestamp
    pub timestamp: u64,
}

impl AutoScaler {
    /// Create new auto-scaler
    pub fn new(config: ScalingConfig) -> Self {
        Self {
            current_scale: AtomicUsize::new(config.min_scale),
            scaling_history: VecDeque::with_capacity(100),
            metrics_buffer: VecDeque::with_capacity(config.metrics_window_size),
            last_scaling_action: AtomicU64::new(0),
            enabled: AtomicBool::new(true),
            config,
        }
    }

    /// Update metrics and check for scaling opportunities
    pub fn update_metrics(&mut self, metrics: ScalingMetrics) -> ScalingAction {
        if !self.enabled.load(Ordering::Relaxed) {
            return ScalingAction::NoAction;
        }

        // Add to metrics buffer
        if self.metrics_buffer.len() >= self.config.metrics_window_size {
            self.metrics_buffer.pop_front();
        }
        self.metrics_buffer.push_back(metrics.clone());

        // Check if we have enough metrics
        if self.metrics_buffer.len() < self.config.metrics_window_size / 2 {
            return ScalingAction::NoAction;
        }

        // Check cooldown period
        let current_time = Self::current_timestamp();
        let last_action = self.last_scaling_action.load(Ordering::Relaxed);
        if (current_time - last_action) < self.config.cooldown_ms {
            return ScalingAction::NoAction;
        }

        // Analyze metrics and make scaling decision
        let scaling_decision = self.analyze_scaling_need(&metrics);
        
        if scaling_decision != ScalingAction::NoAction {
            self.execute_scaling(scaling_decision, metrics);
        }

        scaling_decision
    }

    /// Analyze whether scaling is needed
    fn analyze_scaling_need(&self, current_metrics: &ScalingMetrics) -> ScalingAction {
        let current_scale = self.current_scale.load(Ordering::Relaxed);
        
        // Calculate average metrics over the window
        let avg_cpu = self.metrics_buffer.iter().map(|m| m.cpu_utilization).sum::<f32>() / self.metrics_buffer.len() as f32;
        let avg_queue_depth = self.metrics_buffer.iter().map(|m| m.queue_depth).sum::<usize>() / self.metrics_buffer.len();
        let avg_response_time = self.metrics_buffer.iter().map(|m| m.avg_response_time_ms).sum::<f32>() / self.metrics_buffer.len() as f32;

        // Scale-up conditions
        let should_scale_up = (avg_cpu > self.config.scale_up_threshold) ||
                             (avg_queue_depth > self.config.target_queue_depth * 2) ||
                             (avg_response_time > 50.0) ||
                             (self.config.aggressive_scaling && current_metrics.error_rate > 0.05);

        // Scale-down conditions
        let should_scale_down = (avg_cpu < self.config.scale_down_threshold) &&
                               (avg_queue_depth < self.config.target_queue_depth / 4) &&
                               (avg_response_time < 20.0) &&
                               (current_metrics.error_rate < 0.01);

        if should_scale_up && current_scale < self.config.max_scale {
            ScalingAction::ScaleUp
        } else if should_scale_down && current_scale > self.config.min_scale {
            ScalingAction::ScaleDown
        } else {
            ScalingAction::NoAction
        }
    }

    /// Execute scaling action
    fn execute_scaling(&mut self, action: ScalingAction, metrics: ScalingMetrics) {
        let current_scale = self.current_scale.load(Ordering::Relaxed);
        let new_scale = match action {
            ScalingAction::ScaleUp => {
                let increment = if self.config.aggressive_scaling { 2 } else { 1 };
                (current_scale + increment).min(self.config.max_scale)
            }
            ScalingAction::ScaleDown => {
                let decrement = if self.config.aggressive_scaling { 2 } else { 1 };
                current_scale.saturating_sub(decrement).max(self.config.min_scale)
            }
            ScalingAction::NoAction => current_scale,
        };

        if new_scale != current_scale {
            self.current_scale.store(new_scale, Ordering::Relaxed);
            self.last_scaling_action.store(Self::current_timestamp(), Ordering::Relaxed);

            let reason = match action {
                ScalingAction::ScaleUp => format!(
                    "High load: CPU {:.1}%, Queue {}, Response {:.1}ms",
                    metrics.cpu_utilization * 100.0,
                    metrics.queue_depth,
                    metrics.avg_response_time_ms
                ),
                ScalingAction::ScaleDown => format!(
                    "Low load: CPU {:.1}%, Queue {}, Response {:.1}ms",
                    metrics.cpu_utilization * 100.0,
                    metrics.queue_depth,
                    metrics.avg_response_time_ms
                ),
                ScalingAction::NoAction => "No action".to_string(),
            };

            let event = ScalingEvent {
                timestamp: Self::current_timestamp(),
                action,
                previous_scale: current_scale,
                new_scale,
                reason,
                metrics: metrics.clone(),
            };

            if self.scaling_history.len() >= 100 {
                self.scaling_history.pop_front();
            }
            self.scaling_history.push_back(event);
        }
    }

    /// Get current scale
    pub fn get_current_scale(&self) -> usize {
        self.current_scale.load(Ordering::Relaxed)
    }

    /// Get scaling history
    pub fn get_scaling_history(&self) -> Vec<ScalingEvent> {
        self.scaling_history.iter().cloned().collect()
    }

    /// Enable/disable auto-scaling
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Get scaling recommendations
    pub fn get_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if self.metrics_buffer.is_empty() {
            return recommendations;
        }

        let recent_metrics = self.metrics_buffer.back().unwrap();
        let current_scale = self.current_scale.load(Ordering::Relaxed);

        if recent_metrics.cpu_utilization > 0.9 {
            recommendations.push("Consider scaling up - high CPU utilization".to_string());
        }

        if recent_metrics.queue_depth > self.config.target_queue_depth * 3 {
            recommendations.push("Queue depth is very high - immediate scaling recommended".to_string());
        }

        if current_scale == self.config.max_scale && recent_metrics.cpu_utilization > 0.8 {
            recommendations.push("At maximum scale with high load - consider increasing max_scale".to_string());
        }

        if current_scale == self.config.min_scale && recent_metrics.cpu_utilization < 0.2 {
            recommendations.push("System may be over-provisioned".to_string());
        }

        if recent_metrics.error_rate > 0.1 {
            recommendations.push("High error rate detected - check system health".to_string());
        }

        recommendations
    }

    /// Get current timestamp
    fn current_timestamp() -> u64 {
        static mut TIMESTAMP: u64 = 0;
        unsafe {
            TIMESTAMP += 1;
            TIMESTAMP
        }
    }
}

/// Comprehensive scaling system combining load balancing and auto-scaling
#[derive(Debug)]
pub struct ScalingSystem {
    /// Load balancer
    load_balancer: LoadBalancer,
    /// Auto-scaler
    auto_scaler: AutoScaler,
    /// System enabled
    enabled: bool,
}

impl ScalingSystem {
    /// Create new scaling system
    pub fn new(
        nodes: Vec<ProcessingNode>,
        lb_strategy: LoadBalancingStrategy,
        scaling_config: ScalingConfig,
    ) -> Self {
        Self {
            load_balancer: LoadBalancer::new(nodes, lb_strategy, true),
            auto_scaler: AutoScaler::new(scaling_config),
            enabled: true,
        }
    }

    /// Process request with load balancing and scaling
    pub fn process_request(&mut self, metrics: ScalingMetrics) -> Result<usize> {
        if !self.enabled {
            return Err(LiquidAudioError::InvalidState("Scaling system disabled".to_string()));
        }

        // Update auto-scaler metrics
        let scaling_action = self.auto_scaler.update_metrics(metrics);
        
        // Handle scaling action if needed
        if scaling_action != ScalingAction::NoAction {
            // In a real implementation, this would actually spawn/destroy workers
            // or add/remove nodes based on the scaling decision
        }

        // Select node for processing
        let selected_node = self.load_balancer.select_node()?;
        
        Ok(selected_node)
    }

    /// Get comprehensive scaling statistics
    pub fn get_scaling_stats(&self) -> ScalingSystemStats {
        ScalingSystemStats {
            current_scale: self.auto_scaler.get_current_scale(),
            load_balancer_stats: self.load_balancer.get_stats().clone(),
            recent_scaling_events: self.auto_scaler.get_scaling_history().into_iter().rev().take(10).collect(),
            recommendations: self.auto_scaler.get_recommendations(),
            system_enabled: self.enabled,
        }
    }

    /// Enable/disable scaling system
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        self.auto_scaler.set_enabled(enabled);
    }
}

/// Comprehensive scaling system statistics
#[derive(Debug, Clone)]
pub struct ScalingSystemStats {
    /// Current system scale
    pub current_scale: usize,
    /// Load balancer statistics
    pub load_balancer_stats: LoadBalancerStats,
    /// Recent scaling events
    pub recent_scaling_events: Vec<ScalingEvent>,
    /// System recommendations
    pub recommendations: Vec<String>,
    /// Whether system is enabled
    pub system_enabled: bool,
}