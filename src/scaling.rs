//! Load balancing and auto-scaling for Liquid Neural Networks
//!
//! Provides intelligent load distribution, auto-scaling triggers,
//! and system resource management for production deployments.

use crate::{Result, LiquidAudioError};
use crate::diagnostics::HealthStatus;
// Note: TelemetrySystem would be imported from telemetry module when available

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String, collections::{VecDeque, BTreeMap}, boxed::Box};

#[cfg(feature = "std")]
use std::{vec::Vec, string::String, collections::{VecDeque, BTreeMap}, boxed::Box};
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};


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

/// Advanced predictive auto-scaling system with cost optimization
pub struct AdvancedAutoScaler {
    /// Base auto-scaler
    base_scaler: AutoScaler,
    /// Predictive engine for forecasting
    predictor: PredictiveScalingEngine,
    /// Cost optimizer
    cost_optimizer: CostOptimizer,
    /// Multi-region support
    regions: BTreeMap<String, RegionConfig>,
    /// Instance types available
    instance_types: Vec<InstanceType>,
    /// Advanced configuration
    advanced_config: AdvancedScalingConfig,
    /// Telemetry integration enabled
    telemetry_enabled: bool,
}

/// Advanced scaling configuration
#[derive(Debug, Clone)]
pub struct AdvancedScalingConfig {
    /// Enable predictive scaling
    pub predictive_scaling: bool,
    /// Enable cost optimization
    pub cost_optimization: bool,
    /// Maximum cost per hour (USD)
    pub max_cost_per_hour: f32,
    /// Prediction horizon (minutes)
    pub prediction_horizon: u32,
    /// Multi-region deployment
    pub multi_region: bool,
    /// Auto-migration between regions
    pub auto_migration: bool,
    /// Performance vs cost balance (0.0=cost, 1.0=performance)
    pub performance_cost_balance: f32,
}

impl Default for AdvancedScalingConfig {
    fn default() -> Self {
        Self {
            predictive_scaling: true,
            cost_optimization: true,
            max_cost_per_hour: 1000.0,
            prediction_horizon: 60,
            multi_region: false,
            auto_migration: false,
            performance_cost_balance: 0.7,
        }
    }
}

/// Region configuration for multi-region scaling
#[derive(Debug, Clone)]
pub struct RegionConfig {
    /// Region identifier
    pub region_id: String,
    /// Region display name
    pub region_name: String,
    /// Base latency to region (milliseconds)
    pub base_latency_ms: f32,
    /// Cost multiplier for this region
    pub cost_multiplier: f32,
    /// Maximum instances for this region
    pub max_instances: u32,
    /// Preferred instance types for this region
    pub preferred_instance_types: Vec<String>,
    /// Compliance requirements
    pub compliance_requirements: Vec<String>,
    /// Current deployment in this region
    pub current_deployment: RegionDeployment,
}

/// Current deployment state in a region
#[derive(Debug, Clone, Default)]
pub struct RegionDeployment {
    /// Active instances
    pub active_instances: u32,
    /// Instance types in use
    pub instance_types_in_use: BTreeMap<String, u32>,
    /// Total cost per hour for this region
    pub cost_per_hour: f32,
    /// Average latency from this region
    pub avg_latency_ms: f32,
    /// Health score for this region
    pub health_score: f32,
}

/// Instance type specification
#[derive(Debug, Clone)]
pub struct InstanceType {
    /// Instance type name
    pub type_name: String,
    /// CPU cores
    pub cpu_cores: u32,
    /// Memory in GB
    pub memory_gb: f32,
    /// Cost per hour (USD)
    pub cost_per_hour: f32,
    /// Performance score (relative)
    pub performance_score: f32,
    /// Power efficiency score
    pub power_efficiency: f32,
    /// Supported regions
    pub supported_regions: Vec<String>,
    /// Availability (0.0-1.0)
    pub availability: f32,
}

/// Predictive scaling engine
#[derive(Debug)]
pub struct PredictiveScalingEngine {
    /// Historical data
    historical_data: VecDeque<PredictionDataPoint>,
    /// Prediction models
    models: Vec<PredictionModel>,
    /// Current forecast
    current_forecast: Option<LoadForecast>,
    /// Prediction accuracy tracker
    accuracy: AccuracyTracker,
    /// Learning enabled
    learning_enabled: bool,
}

/// Data point for predictions
#[derive(Debug, Clone)]
pub struct PredictionDataPoint {
    /// Timestamp
    pub timestamp: u64,
    /// Load metrics
    pub load: f32,
    /// Response time
    pub response_time_ms: f32,
    /// Cost at this point
    pub cost: f32,
    /// External factors (hour of day, day of week, etc.)
    pub external_factors: BTreeMap<String, f32>,
}

/// Prediction model types
#[derive(Debug)]
pub enum PredictionModel {
    /// Linear trend analysis
    LinearTrend { weight: f32 },
    /// Seasonal pattern detection
    SeasonalPattern { 
        period_hours: f32,
        amplitude: f32,
        weight: f32 
    },
    /// Anomaly detection
    AnomalyDetection { 
        threshold: f32,
        sensitivity: f32 
    },
    /// Machine learning model
    MLModel { 
        model_type: String,
        accuracy: f32 
    },
}

/// Load forecast result
#[derive(Debug, Clone)]
pub struct LoadForecast {
    /// Predicted load over time
    pub predicted_loads: Vec<(u64, f32)>,
    /// Confidence intervals
    pub confidence_intervals: Vec<(f32, f32)>,
    /// Peak load prediction
    pub predicted_peak: PeakPrediction,
    /// Cost forecast
    pub cost_forecast: Vec<(u64, f32)>,
    /// Recommended actions
    pub recommended_actions: Vec<ScalingRecommendation>,
}

/// Peak load prediction
#[derive(Debug, Clone)]
pub struct PeakPrediction {
    /// Expected peak load
    pub peak_load: f32,
    /// Time until peak (minutes)
    pub time_to_peak_minutes: u32,
    /// Confidence level
    pub confidence: f32,
    /// Severity level
    pub severity: PeakSeverity,
}

/// Peak severity levels
#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub enum PeakSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Scaling recommendation
#[derive(Debug, Clone)]
pub struct ScalingRecommendation {
    /// Action type
    pub action: RecommendedAction,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Expected impact
    pub expected_impact: ExpectedImpact,
    /// Confidence in recommendation
    pub confidence: f32,
    /// Time sensitivity
    pub time_sensitive: bool,
    /// Cost impact
    pub cost_impact: f32,
}

/// Recommended actions
#[derive(Debug, Clone)]
pub enum RecommendedAction {
    ScaleUp { instances: u32, instance_type: String },
    ScaleDown { instances: u32 },
    MigrateRegion { from: String, to: String, instances: u32 },
    ChangeInstanceType { from: String, to: String, instances: u32 },
    EnableSpotInstances { percentage: f32 },
    ScheduleScaling { time: u64, action: Box<RecommendedAction> },
}

/// Recommendation priority
#[derive(Debug, Clone)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
}

/// Expected impact of recommendation
#[derive(Debug, Clone)]
pub struct ExpectedImpact {
    /// Performance improvement (0.0-1.0)
    pub performance_improvement: f32,
    /// Cost reduction (negative for cost increase)
    pub cost_reduction: f32,
    /// Latency improvement (negative for increase)
    pub latency_improvement_ms: f32,
    /// Reliability improvement
    pub reliability_improvement: f32,
}

/// Accuracy tracking for predictions
#[derive(Debug)]
pub struct AccuracyTracker {
    /// Historical accuracy scores
    accuracy_history: VecDeque<f32>,
    /// Current accuracy
    current_accuracy: f32,
    /// Model confidence
    confidence: f32,
    /// Total predictions made
    total_predictions: u64,
    /// Accurate predictions
    accurate_predictions: u64,
}

/// Cost optimizer for efficient resource allocation
#[derive(Debug)]
pub struct CostOptimizer {
    /// Optimization strategy
    strategy: OptimizationStrategy,
    /// Budget constraints
    budget: BudgetConstraints,
    /// Cost history
    cost_history: VecDeque<CostDataPoint>,
    /// Savings opportunities
    savings_opportunities: Vec<SavingsOpportunity>,
    /// Spending alerts
    spending_alerts: Vec<SpendingAlert>,
}

/// Cost optimization strategies
#[derive(Debug, Clone)]
pub enum OptimizationStrategy {
    MinimizeCost,
    MaximizePerformance,
    BalanceCostPerformance,
    MinimizeLatency,
    MaximizeEfficiency,
}

/// Budget constraints
#[derive(Debug, Clone)]
pub struct BudgetConstraints {
    /// Hourly budget limit
    pub hourly_limit: f32,
    /// Daily budget limit
    pub daily_limit: f32,
    /// Monthly budget limit
    pub monthly_limit: f32,
    /// Alert thresholds
    pub alert_thresholds: Vec<f32>,
    /// Auto-shutdown threshold
    pub emergency_threshold: f32,
}

/// Cost data point
#[derive(Debug, Clone)]
pub struct CostDataPoint {
    /// Timestamp
    pub timestamp: u64,
    /// Total cost
    pub total_cost: f32,
    /// Cost breakdown by region
    pub cost_by_region: BTreeMap<String, f32>,
    /// Cost breakdown by instance type
    pub cost_by_instance_type: BTreeMap<String, f32>,
    /// Performance metrics at this cost
    pub performance_metrics: PerformanceMetrics,
}

/// Performance metrics for cost analysis
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Average response time
    pub avg_response_time_ms: f32,
    /// Throughput (requests per second)
    pub throughput_rps: f32,
    /// Error rate
    pub error_rate: f32,
    /// CPU utilization
    pub cpu_utilization: f32,
    /// Memory utilization
    pub memory_utilization: f32,
}

/// Savings opportunity
#[derive(Debug, Clone)]
pub struct SavingsOpportunity {
    /// Opportunity type
    pub opportunity_type: SavingsType,
    /// Potential savings per hour
    pub savings_per_hour: f32,
    /// Implementation complexity
    pub complexity: ImplementationComplexity,
    /// Risk level
    pub risk_level: RiskLevel,
    /// Description
    pub description: String,
}

/// Types of savings opportunities
#[derive(Debug, Clone)]
pub enum SavingsType {
    RightsizeInstances,
    UseSpotInstances,
    ConsolidateRegions,
    ScheduleDowntime,
    OptimizeInstanceMix,
    EnableAutoShutdown,
}

/// Implementation complexity levels
#[derive(Debug, Clone, PartialEq)]
pub enum ImplementationComplexity {
    Low,
    Medium,
    High,
}

/// Risk levels for cost optimizations
#[derive(Debug, Clone, PartialEq)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
}

/// Spending alert
#[derive(Debug, Clone)]
pub struct SpendingAlert {
    /// Alert type
    pub alert_type: AlertType,
    /// Severity level
    pub severity: AlertSeverity,
    /// Current spending rate
    pub current_rate: f32,
    /// Threshold breached
    pub threshold: f32,
    /// Projected impact
    pub projected_impact: String,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
}

/// Alert types for spending
#[derive(Debug, Clone)]
pub enum AlertType {
    BudgetExceeded,
    UnexpectedSpike,
    InefficiencyDetected,
    CostAnomaly,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Trait for telemetry collection
pub trait TelemetryCollector {
    /// Record scaling event
    fn record_scaling_event(&mut self, event: &ScalingEvent);
    /// Record cost data
    fn record_cost_data(&mut self, cost_data: &CostDataPoint);
    /// Record prediction accuracy
    fn record_prediction_accuracy(&mut self, accuracy: f32);
}

impl AdvancedAutoScaler {
    /// Create new advanced auto-scaler
    pub fn new(
        base_config: ScalingConfig,
        advanced_config: AdvancedScalingConfig,
        regions: Vec<RegionConfig>,
        instance_types: Vec<InstanceType>,
    ) -> Result<Self> {
        let base_scaler = AutoScaler::new(base_config);
        let predictor = PredictiveScalingEngine::new(advanced_config.prediction_horizon);
        let cost_optimizer = CostOptimizer::new(advanced_config.max_cost_per_hour);
        
        let mut region_map = BTreeMap::new();
        for region in regions {
            region_map.insert(region.region_id.clone(), region);
        }
        
        Ok(Self {
            base_scaler,
            predictor,
            cost_optimizer,
            regions: region_map,
            instance_types,
            advanced_config,
            telemetry_enabled: false,
        })
    }
    
    /// Update with new metrics and generate scaling recommendations
    pub fn update_and_recommend(&mut self, metrics: ScalingMetrics) -> Result<Vec<ScalingRecommendation>> {
        let mut recommendations = Vec::new();
        
        // Update base scaler
        let base_action = self.base_scaler.update_metrics(metrics.clone());
        
        // Add prediction data point
        self.predictor.add_data_point(PredictionDataPoint {
            timestamp: metrics.timestamp,
            load: metrics.cpu_utilization,
            response_time_ms: metrics.avg_response_time_ms,
            cost: self.calculate_current_cost(),
            external_factors: self.get_external_factors(),
        });
        
        // Generate predictive recommendations
        if self.advanced_config.predictive_scaling {
            let forecast = self.predictor.generate_forecast(self.advanced_config.prediction_horizon)?;
            let predictive_recommendations = self.generate_predictive_recommendations(&forecast)?;
            recommendations.extend(predictive_recommendations);
        }
        
        // Generate cost optimization recommendations
        if self.advanced_config.cost_optimization {
            let cost_recommendations = self.cost_optimizer.generate_recommendations(&metrics)?;
            recommendations.extend(cost_recommendations);
        }
        
        // Multi-region recommendations
        if self.advanced_config.multi_region {
            let region_recommendations = self.generate_region_recommendations(&metrics)?;
            recommendations.extend(region_recommendations);
        }
        
        // Convert base action to recommendation if needed
        if base_action != ScalingAction::NoAction {
            recommendations.push(self.convert_base_action_to_recommendation(base_action, &metrics));
        }
        
        // Sort by priority and filter
        recommendations.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then(
                b.confidence.partial_cmp(&a.confidence).unwrap_or(core::cmp::Ordering::Equal)
            )
        });
        
        Ok(recommendations)
    }
    
    /// Get comprehensive scaling statistics
    pub fn get_advanced_statistics(&self) -> AdvancedScalingStatistics {
        AdvancedScalingStatistics {
            base_stats: self.base_scaler.get_scaling_history(),
            prediction_accuracy: self.predictor.accuracy.current_accuracy,
            cost_efficiency: self.cost_optimizer.get_efficiency_score(),
            regional_deployments: self.regions.values().map(|r| r.current_deployment.clone()).collect(),
            savings_opportunities: self.cost_optimizer.savings_opportunities.clone(),
            current_forecast: self.predictor.current_forecast.clone(),
        }
    }
    
    // Private helper methods
    
    fn calculate_current_cost(&self) -> f32 {
        self.regions.values()
            .map(|region| region.current_deployment.cost_per_hour)
            .sum()
    }
    
    fn get_external_factors(&self) -> BTreeMap<String, f32> {
        let mut factors = BTreeMap::new();
        
        // Simple time-based factors (in real implementation, these would be more sophisticated)
        let current_time = Self::current_timestamp();
        let hour_of_day = (current_time / 3600) % 24;
        let day_of_week = (current_time / 86400) % 7;
        
        factors.insert("hour_of_day".to_string(), hour_of_day as f32);
        factors.insert("day_of_week".to_string(), day_of_week as f32);
        
        factors
    }
    
    fn generate_predictive_recommendations(&self, forecast: &LoadForecast) -> Result<Vec<ScalingRecommendation>> {
        let mut recommendations = Vec::new();
        
        // Check for predicted peaks
        if forecast.predicted_peak.severity >= PeakSeverity::Medium {
            let instances_needed = self.calculate_instances_for_peak(&forecast.predicted_peak);
            
            recommendations.push(ScalingRecommendation {
                action: RecommendedAction::ScaleUp {
                    instances: instances_needed,
                    instance_type: self.select_optimal_instance_type("performance"),
                },
                priority: match forecast.predicted_peak.severity {
                    PeakSeverity::Critical => RecommendationPriority::Critical,
                    PeakSeverity::High => RecommendationPriority::High,
                    _ => RecommendationPriority::Medium,
                },
                expected_impact: ExpectedImpact {
                    performance_improvement: 0.3,
                    cost_reduction: -(instances_needed as f32) * 0.5,
                    latency_improvement_ms: -20.0,
                    reliability_improvement: 0.2,
                },
                confidence: forecast.predicted_peak.confidence,
                time_sensitive: forecast.predicted_peak.time_to_peak_minutes <= 15,
                cost_impact: instances_needed as f32 * 0.5,
            });
        }
        
        Ok(recommendations)
    }
    
    fn generate_region_recommendations(&self, _metrics: &ScalingMetrics) -> Result<Vec<ScalingRecommendation>> {
        let mut recommendations = Vec::new();
        
        // Find opportunities for regional optimization
        for (region_id, region) in &self.regions {
            if region.current_deployment.health_score < 0.7 {
                // Recommend migration from unhealthy region
                if let Some(best_region) = self.find_best_alternative_region(region_id) {
                    recommendations.push(ScalingRecommendation {
                        action: RecommendedAction::MigrateRegion {
                            from: region_id.clone(),
                            to: best_region,
                            instances: region.current_deployment.active_instances,
                        },
                        priority: RecommendationPriority::High,
                        expected_impact: ExpectedImpact {
                            performance_improvement: 0.4,
                            cost_reduction: 0.0,
                            latency_improvement_ms: -50.0,
                            reliability_improvement: 0.5,
                        },
                        confidence: 0.8,
                        time_sensitive: true,
                        cost_impact: 0.0,
                    });
                }
            }
        }
        
        Ok(recommendations)
    }
    
    fn convert_base_action_to_recommendation(&self, action: ScalingAction, metrics: &ScalingMetrics) -> ScalingRecommendation {
        match action {
            ScalingAction::ScaleUp => ScalingRecommendation {
                action: RecommendedAction::ScaleUp {
                    instances: 1,
                    instance_type: self.select_optimal_instance_type("general"),
                },
                priority: if metrics.error_rate > 0.05 {
                    RecommendationPriority::Critical
                } else {
                    RecommendationPriority::High
                },
                expected_impact: ExpectedImpact {
                    performance_improvement: 0.2,
                    cost_reduction: -0.5,
                    latency_improvement_ms: -10.0,
                    reliability_improvement: 0.1,
                },
                confidence: 0.9,
                time_sensitive: metrics.cpu_utilization > 0.9,
                cost_impact: 0.5,
            },
            ScalingAction::ScaleDown => ScalingRecommendation {
                action: RecommendedAction::ScaleDown { instances: 1 },
                priority: RecommendationPriority::Low,
                expected_impact: ExpectedImpact {
                    performance_improvement: -0.1,
                    cost_reduction: 0.5,
                    latency_improvement_ms: 5.0,
                    reliability_improvement: 0.0,
                },
                confidence: 0.8,
                time_sensitive: false,
                cost_impact: -0.5,
            },
            ScalingAction::NoAction => ScalingRecommendation {
                action: RecommendedAction::ScaleUp { instances: 0, instance_type: "none".to_string() },
                priority: RecommendationPriority::Low,
                expected_impact: ExpectedImpact {
                    performance_improvement: 0.0,
                    cost_reduction: 0.0,
                    latency_improvement_ms: 0.0,
                    reliability_improvement: 0.0,
                },
                confidence: 1.0,
                time_sensitive: false,
                cost_impact: 0.0,
            },
        }
    }
    
    fn calculate_instances_for_peak(&self, peak: &PeakPrediction) -> u32 {
        // Simple calculation - in practice this would be more sophisticated
        let current_scale = self.base_scaler.get_current_scale() as f32;
        let scale_factor = peak.peak_load / 0.7; // Target 70% utilization
        (scale_factor * current_scale).ceil() as u32 - current_scale as u32
    }
    
    fn select_optimal_instance_type(&self, workload_type: &str) -> String {
        // Select best instance type based on current config and workload
        for instance_type in &self.instance_types {
            match workload_type {
                "performance" => {
                    if instance_type.performance_score >= 0.8 {
                        return instance_type.type_name.clone();
                    }
                },
                "cost" => {
                    if instance_type.cost_per_hour <= 1.0 {
                        return instance_type.type_name.clone();
                    }
                },
                _ => {
                    return instance_type.type_name.clone();
                }
            }
        }
        "m5.large".to_string() // Default fallback
    }
    
    fn find_best_alternative_region(&self, current_region: &str) -> Option<String> {
        // Find best region alternative based on cost, latency, and health
        let mut best_region = None;
        let mut best_score = f32::NEG_INFINITY;
        
        for (region_id, region) in &self.regions {
            if region_id == current_region {
                continue;
            }
            
            // Simple scoring - in practice this would be more sophisticated
            let score = region.current_deployment.health_score * 0.5 - 
                       region.cost_multiplier * 0.3 - 
                       region.base_latency_ms * 0.002;
            
            if score > best_score {
                best_score = score;
                best_region = Some(region_id.clone());
            }
        }
        
        best_region
    }
    
    fn current_timestamp() -> u64 {
        static mut TIMESTAMP: u64 = 0;
        unsafe {
            TIMESTAMP += 1;
            TIMESTAMP
        }
    }
}

/// Advanced scaling statistics
#[derive(Debug, Clone)]
pub struct AdvancedScalingStatistics {
    /// Base scaling events
    pub base_stats: Vec<ScalingEvent>,
    /// Prediction accuracy
    pub prediction_accuracy: f32,
    /// Cost efficiency score
    pub cost_efficiency: f32,
    /// Regional deployment status
    pub regional_deployments: Vec<RegionDeployment>,
    /// Available cost savings
    pub savings_opportunities: Vec<SavingsOpportunity>,
    /// Current forecast
    pub current_forecast: Option<LoadForecast>,
}

// Implementation stubs for complex components

impl PredictiveScalingEngine {
    fn new(horizon: u32) -> Self {
        Self {
            historical_data: VecDeque::with_capacity(1000),
            models: vec![
                PredictionModel::LinearTrend { weight: 0.3 },
                PredictionModel::SeasonalPattern { 
                    period_hours: 24.0,
                    amplitude: 0.2,
                    weight: 0.4 
                },
            ],
            current_forecast: None,
            accuracy: AccuracyTracker {
                accuracy_history: VecDeque::with_capacity(100),
                current_accuracy: 0.5,
                confidence: 0.5,
                total_predictions: 0,
                accurate_predictions: 0,
            },
            learning_enabled: true,
        }
    }
    
    fn add_data_point(&mut self, data_point: PredictionDataPoint) {
        if self.historical_data.len() >= 1000 {
            self.historical_data.pop_front();
        }
        self.historical_data.push_back(data_point);
    }
    
    fn generate_forecast(&mut self, horizon_minutes: u32) -> Result<LoadForecast> {
        // Simplified forecasting - in production, this would use sophisticated ML
        let current_load = self.historical_data.back()
            .map(|dp| dp.load)
            .unwrap_or(0.5);
        
        let mut predicted_loads = Vec::new();
        let current_time = AdvancedAutoScaler::current_timestamp();
        
        for i in 0..horizon_minutes {
            let time = current_time + (i as u64 * 60);
            let predicted_load = current_load * (1.0 + (i as f32 * 0.01)); // Simple linear growth
            predicted_loads.push((time, predicted_load));
        }
        
        let forecast = LoadForecast {
            predicted_loads,
            confidence_intervals: vec![(0.4, 0.6); horizon_minutes as usize],
            predicted_peak: PeakPrediction {
                peak_load: current_load * 1.3,
                time_to_peak_minutes: horizon_minutes / 2,
                confidence: self.accuracy.confidence,
                severity: if current_load > 0.8 { PeakSeverity::High } else { PeakSeverity::Medium },
            },
            cost_forecast: vec![(current_time, 100.0)],
            recommended_actions: Vec::new(),
        };
        
        self.current_forecast = Some(forecast.clone());
        Ok(forecast)
    }
}

impl CostOptimizer {
    fn new(max_cost_per_hour: f32) -> Self {
        Self {
            strategy: OptimizationStrategy::BalanceCostPerformance,
            budget: BudgetConstraints {
                hourly_limit: max_cost_per_hour,
                daily_limit: max_cost_per_hour * 24.0,
                monthly_limit: max_cost_per_hour * 24.0 * 30.0,
                alert_thresholds: vec![0.8, 0.9, 0.95],
                emergency_threshold: 1.0,
            },
            cost_history: VecDeque::with_capacity(1000),
            savings_opportunities: Vec::new(),
            spending_alerts: Vec::new(),
        }
    }
    
    fn generate_recommendations(&mut self, _metrics: &ScalingMetrics) -> Result<Vec<ScalingRecommendation>> {
        // Generate cost optimization recommendations
        Ok(vec![
            ScalingRecommendation {
                action: RecommendedAction::EnableSpotInstances { percentage: 0.3 },
                priority: RecommendationPriority::Medium,
                expected_impact: ExpectedImpact {
                    performance_improvement: 0.0,
                    cost_reduction: 30.0,
                    latency_improvement_ms: 0.0,
                    reliability_improvement: -0.1,
                },
                confidence: 0.8,
                time_sensitive: false,
                cost_impact: -30.0,
            }
        ])
    }
    
    fn get_efficiency_score(&self) -> f32 {
        // Calculate cost efficiency score
        0.75 // Placeholder
    }
}

impl PartialOrd for RecommendationPriority {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for RecommendationPriority {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        match (self, other) {
            (RecommendationPriority::Critical, RecommendationPriority::Critical) => core::cmp::Ordering::Equal,
            (RecommendationPriority::Critical, _) => core::cmp::Ordering::Greater,
            (_, RecommendationPriority::Critical) => core::cmp::Ordering::Less,
            (RecommendationPriority::High, RecommendationPriority::High) => core::cmp::Ordering::Equal,
            (RecommendationPriority::High, _) => core::cmp::Ordering::Greater,
            (_, RecommendationPriority::High) => core::cmp::Ordering::Less,
            (RecommendationPriority::Medium, RecommendationPriority::Medium) => core::cmp::Ordering::Equal,
            (RecommendationPriority::Medium, RecommendationPriority::Low) => core::cmp::Ordering::Greater,
            (RecommendationPriority::Low, RecommendationPriority::Medium) => core::cmp::Ordering::Less,
            (RecommendationPriority::Low, RecommendationPriority::Low) => core::cmp::Ordering::Equal,
        }
    }
}

impl PartialEq for RecommendationPriority {
    fn eq(&self, other: &Self) -> bool {
        core::mem::discriminant(self) == core::mem::discriminant(other)
    }
}

impl Eq for RecommendationPriority {}

#[cfg(test)]
mod advanced_tests {
    use super::*;
    
    #[test]
    fn test_advanced_auto_scaler_creation() {
        let base_config = ScalingConfig::default();
        let advanced_config = AdvancedScalingConfig::default();
        let regions = vec![];
        let instance_types = vec![
            InstanceType {
                type_name: "m5.large".to_string(),
                cpu_cores: 2,
                memory_gb: 8.0,
                cost_per_hour: 0.096,
                performance_score: 0.7,
                power_efficiency: 0.8,
                supported_regions: vec!["us-east-1".to_string()],
                availability: 0.99,
            }
        ];
        
        let scaler = AdvancedAutoScaler::new(base_config, advanced_config, regions, instance_types);
        assert!(scaler.is_ok());
    }
    
    #[test]
    fn test_prediction_engine() {
        let mut engine = PredictiveScalingEngine::new(60);
        
        let data_point = PredictionDataPoint {
            timestamp: 1000,
            load: 0.5,
            response_time_ms: 100.0,
            cost: 50.0,
            external_factors: BTreeMap::new(),
        };
        
        engine.add_data_point(data_point);
        let forecast = engine.generate_forecast(30);
        assert!(forecast.is_ok());
    }
    
    #[test]
    fn test_cost_optimizer() {
        let mut optimizer = CostOptimizer::new(100.0);
        let metrics = ScalingMetrics {
            cpu_utilization: 0.8,
            queue_depth: 50,
            avg_response_time_ms: 80.0,
            error_rate: 0.02,
            throughput_rps: 100.0,
            memory_utilization: 0.7,
            timestamp: 1000,
        };
        
        let recommendations = optimizer.generate_recommendations(&metrics);
        assert!(recommendations.is_ok());
    }
}