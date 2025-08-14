//! Edge computing and deployment automation
//! 
//! Advanced edge deployment system with automatic provisioning, dynamic scaling,
//! and distributed orchestration for global edge computing infrastructure.

pub mod orchestrator;
pub mod provisioning;
pub mod discovery;
pub mod sync;
pub mod federation;

pub use orchestrator::{EdgeOrchestrator, EdgeNode, EdgeCluster};
pub use provisioning::{EdgeProvisioner, ProvisioningConfig, DeploymentPipeline};
pub use discovery::{ServiceDiscovery, EdgeRegistry, HealthChecker};
pub use sync::{EdgeSync, DataSync, ModelSync, SyncConfig};
pub use federation::{EdgeFederation, FederatedLearning, ConsensusProtocol};

use crate::{Result, LiquidAudioError};
use crate::regions::{Region, RegionalConfig};

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String, boxed::Box, collections::BTreeMap};

#[cfg(feature = "std")]
use std::collections::BTreeMap;

/// Edge computing configuration for global deployment
#[derive(Debug, Clone)]
pub struct EdgeConfig {
    /// Edge deployment strategy
    pub deployment_strategy: DeploymentStrategy,
    /// Geographic distribution requirements
    pub geographic_distribution: GeographicDistribution,
    /// Resource allocation per edge node
    pub resource_allocation: ResourceAllocation,
    /// Synchronization configuration
    pub sync_config: SyncConfig,
    /// Federation configuration
    pub federation_config: FederationConfig,
    /// Auto-scaling settings
    pub auto_scaling: AutoScalingConfig,
    /// Health monitoring configuration
    pub health_monitoring: HealthMonitoringConfig,
    /// Security configuration
    pub security_config: EdgeSecurityConfig,
}

/// Deployment strategies for edge infrastructure
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeploymentStrategy {
    /// Replicate to all edge nodes
    FullReplication,
    /// Deploy based on geographic proximity
    GeographicProximity,
    /// Deploy based on demand patterns
    DemandBased,
    /// Hierarchical deployment (regional hubs + local nodes)
    Hierarchical,
    /// Hybrid strategy combining multiple approaches
    Hybrid,
}

/// Geographic distribution requirements
#[derive(Debug, Clone)]
pub struct GeographicDistribution {
    /// Target regions for deployment
    pub target_regions: Vec<Region>,
    /// Minimum nodes per region
    pub min_nodes_per_region: usize,
    /// Maximum latency between nodes (milliseconds)
    pub max_inter_node_latency_ms: u64,
    /// Data residency requirements
    pub data_residency: BTreeMap<Region, DataResidencyRequirement>,
    /// Compliance requirements per region
    pub compliance_requirements: BTreeMap<Region, Vec<String>>,
}

/// Data residency requirements
#[derive(Debug, Clone)]
pub struct DataResidencyRequirement {
    /// Data must remain in region
    pub data_must_stay_in_region: bool,
    /// Processing must occur in region
    pub processing_must_stay_in_region: bool,
    /// Backup locations allowed
    pub backup_regions_allowed: Vec<Region>,
    /// Encryption requirements
    pub encryption_required: bool,
}

/// Resource allocation per edge node
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// CPU cores per node
    pub cpu_cores: u32,
    /// Memory in MB per node
    pub memory_mb: u64,
    /// Storage in GB per node
    pub storage_gb: u64,
    /// Network bandwidth in Mbps
    pub network_bandwidth_mbps: u64,
    /// GPU resources (optional)
    pub gpu_resources: Option<GpuResources>,
    /// Power budget in watts
    pub power_budget_watts: f32,
}

/// GPU resource specification
#[derive(Debug, Clone)]
pub struct GpuResources {
    /// GPU type (e.g., "NVIDIA T4", "ARM Mali")
    pub gpu_type: String,
    /// Number of GPU units
    pub gpu_count: u32,
    /// GPU memory in MB
    pub gpu_memory_mb: u64,
    /// CUDA compute capability (for NVIDIA)
    pub compute_capability: Option<String>,
}

/// Auto-scaling configuration
#[derive(Debug, Clone)]
pub struct AutoScalingConfig {
    /// Enable auto-scaling
    pub enabled: bool,
    /// Minimum number of nodes
    pub min_nodes: usize,
    /// Maximum number of nodes
    pub max_nodes: usize,
    /// CPU utilization threshold for scaling up (0.0-1.0)
    pub scale_up_cpu_threshold: f32,
    /// CPU utilization threshold for scaling down (0.0-1.0)
    pub scale_down_cpu_threshold: f32,
    /// Memory utilization threshold for scaling up (0.0-1.0)
    pub scale_up_memory_threshold: f32,
    /// Latency threshold for scaling up (milliseconds)
    pub scale_up_latency_threshold_ms: u64,
    /// Cool-down period between scaling actions (seconds)
    pub cooldown_period_seconds: u64,
}

/// Health monitoring configuration
#[derive(Debug, Clone)]
pub struct HealthMonitoringConfig {
    /// Health check interval (seconds)
    pub check_interval_seconds: u64,
    /// Health check timeout (seconds)
    pub check_timeout_seconds: u64,
    /// Number of failed checks before marking unhealthy
    pub failure_threshold: u32,
    /// Number of successful checks to mark healthy again
    pub success_threshold: u32,
    /// Enable predictive health monitoring
    pub predictive_monitoring: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
}

/// Alert threshold configuration
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// CPU usage alert threshold (0.0-1.0)
    pub cpu_usage_alert: f32,
    /// Memory usage alert threshold (0.0-1.0)
    pub memory_usage_alert: f32,
    /// Disk usage alert threshold (0.0-1.0)
    pub disk_usage_alert: f32,
    /// Network latency alert threshold (milliseconds)
    pub network_latency_alert_ms: u64,
    /// Error rate alert threshold (0.0-1.0)
    pub error_rate_alert: f32,
}

/// Edge security configuration
#[derive(Debug, Clone)]
pub struct EdgeSecurityConfig {
    /// Enable mutual TLS between nodes
    pub mutual_tls_enabled: bool,
    /// Certificate rotation interval (days)
    pub cert_rotation_interval_days: u32,
    /// Enable data encryption at rest
    pub data_encryption_at_rest: bool,
    /// Enable data encryption in transit
    pub data_encryption_in_transit: bool,
    /// Access control configuration
    pub access_control: AccessControlConfig,
    /// Audit logging configuration
    pub audit_logging: AuditLoggingConfig,
}

/// Access control configuration
#[derive(Debug, Clone)]
pub struct AccessControlConfig {
    /// Enable role-based access control
    pub rbac_enabled: bool,
    /// Default access level for new nodes
    pub default_access_level: AccessLevel,
    /// API key authentication required
    pub api_key_required: bool,
    /// JWT token expiration (seconds)
    pub jwt_expiration_seconds: u64,
}

/// Access levels for edge nodes
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum AccessLevel {
    /// Read-only access
    ReadOnly,
    /// Read and execute access
    ReadExecute,
    /// Read, execute, and write access
    ReadWriteExecute,
    /// Full administrative access
    Admin,
}

/// Audit logging configuration
#[derive(Debug, Clone)]
pub struct AuditLoggingConfig {
    /// Enable audit logging
    pub enabled: bool,
    /// Log level for audit events
    pub log_level: AuditLogLevel,
    /// Retention period for audit logs (days)
    pub retention_days: u32,
    /// Remote logging endpoint
    pub remote_endpoint: Option<String>,
}

/// Audit log levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AuditLogLevel {
    /// Log only critical security events
    Critical,
    /// Log important security events
    Important,
    /// Log all security events
    All,
    /// Log everything including debug information
    Debug,
}

/// Federation configuration for distributed learning
#[derive(Debug, Clone)]
pub struct FederationConfig {
    /// Enable federated learning
    pub enabled: bool,
    /// Consensus protocol for coordination
    pub consensus_protocol: ConsensusProtocol,
    /// Minimum participants for consensus
    pub min_participants: usize,
    /// Model aggregation strategy
    pub aggregation_strategy: AggregationStrategy,
    /// Communication protocol
    pub communication_protocol: CommunicationProtocol,
    /// Privacy preservation settings
    pub privacy_preservation: PrivacyPreservationConfig,
}

/// Model aggregation strategies
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggregationStrategy {
    /// Simple federated averaging
    FederatedAveraging,
    /// Weighted federated averaging
    WeightedAveraging,
    /// Secure aggregation with differential privacy
    SecureAggregation,
    /// Hierarchical aggregation
    HierarchicalAggregation,
}

/// Communication protocols for federation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CommunicationProtocol {
    /// gRPC-based communication
    Grpc,
    /// REST API communication
    Rest,
    /// Peer-to-peer networking
    P2P,
    /// Blockchain-based coordination
    Blockchain,
}

/// Privacy preservation configuration
#[derive(Debug, Clone)]
pub struct PrivacyPreservationConfig {
    /// Enable differential privacy
    pub differential_privacy_enabled: bool,
    /// Privacy budget (epsilon)
    pub privacy_budget: f32,
    /// Enable secure multi-party computation
    pub secure_multiparty_computation: bool,
    /// Enable homomorphic encryption
    pub homomorphic_encryption: bool,
    /// Data anonymization level
    pub anonymization_level: AnonymizationLevel,
}

/// Data anonymization levels
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum AnonymizationLevel {
    /// No anonymization
    None,
    /// Basic anonymization (remove direct identifiers)
    Basic,
    /// Advanced anonymization (k-anonymity)
    Advanced,
    /// Full anonymization (differential privacy)
    Full,
}

impl Default for EdgeConfig {
    fn default() -> Self {
        Self {
            deployment_strategy: DeploymentStrategy::GeographicProximity,
            geographic_distribution: GeographicDistribution {
                target_regions: vec![
                    Region::NorthAmerica,
                    Region::Europe,
                    Region::AsiaPacific,
                ],
                min_nodes_per_region: 2,
                max_inter_node_latency_ms: 100,
                data_residency: BTreeMap::new(),
                compliance_requirements: BTreeMap::new(),
            },
            resource_allocation: ResourceAllocation {
                cpu_cores: 4,
                memory_mb: 8192,
                storage_gb: 100,
                network_bandwidth_mbps: 1000,
                gpu_resources: None,
                power_budget_watts: 50.0,
            },
            sync_config: SyncConfig::default(),
            federation_config: FederationConfig {
                enabled: false,
                consensus_protocol: ConsensusProtocol::Raft,
                min_participants: 3,
                aggregation_strategy: AggregationStrategy::FederatedAveraging,
                communication_protocol: CommunicationProtocol::Grpc,
                privacy_preservation: PrivacyPreservationConfig {
                    differential_privacy_enabled: true,
                    privacy_budget: 1.0,
                    secure_multiparty_computation: false,
                    homomorphic_encryption: false,
                    anonymization_level: AnonymizationLevel::Advanced,
                },
            },
            auto_scaling: AutoScalingConfig {
                enabled: true,
                min_nodes: 1,
                max_nodes: 10,
                scale_up_cpu_threshold: 0.8,
                scale_down_cpu_threshold: 0.3,
                scale_up_memory_threshold: 0.8,
                scale_up_latency_threshold_ms: 100,
                cooldown_period_seconds: 300,
            },
            health_monitoring: HealthMonitoringConfig {
                check_interval_seconds: 30,
                check_timeout_seconds: 10,
                failure_threshold: 3,
                success_threshold: 2,
                predictive_monitoring: true,
                alert_thresholds: AlertThresholds {
                    cpu_usage_alert: 0.9,
                    memory_usage_alert: 0.9,
                    disk_usage_alert: 0.85,
                    network_latency_alert_ms: 200,
                    error_rate_alert: 0.05,
                },
            },
            security_config: EdgeSecurityConfig {
                mutual_tls_enabled: true,
                cert_rotation_interval_days: 30,
                data_encryption_at_rest: true,
                data_encryption_in_transit: true,
                access_control: AccessControlConfig {
                    rbac_enabled: true,
                    default_access_level: AccessLevel::ReadExecute,
                    api_key_required: true,
                    jwt_expiration_seconds: 3600,
                },
                audit_logging: AuditLoggingConfig {
                    enabled: true,
                    log_level: AuditLogLevel::Important,
                    retention_days: 90,
                    remote_endpoint: None,
                },
            },
        }
    }
}

impl EdgeConfig {
    /// Create configuration optimized for IoT edge devices
    pub fn iot_optimized() -> Self {
        Self {
            deployment_strategy: DeploymentStrategy::DemandBased,
            resource_allocation: ResourceAllocation {
                cpu_cores: 2,
                memory_mb: 2048,
                storage_gb: 32,
                network_bandwidth_mbps: 100,
                gpu_resources: None,
                power_budget_watts: 10.0,
            },
            auto_scaling: AutoScalingConfig {
                enabled: false, // Limited scaling for IoT
                min_nodes: 1,
                max_nodes: 3,
                scale_up_cpu_threshold: 0.9,
                scale_down_cpu_threshold: 0.2,
                scale_up_memory_threshold: 0.9,
                scale_up_latency_threshold_ms: 200,
                cooldown_period_seconds: 600,
            },
            security_config: EdgeSecurityConfig {
                mutual_tls_enabled: false, // Simplified for IoT
                cert_rotation_interval_days: 90,
                data_encryption_at_rest: false,
                data_encryption_in_transit: true,
                access_control: AccessControlConfig {
                    rbac_enabled: false,
                    default_access_level: AccessLevel::ReadExecute,
                    api_key_required: true,
                    jwt_expiration_seconds: 7200,
                },
                audit_logging: AuditLoggingConfig {
                    enabled: false, // Reduced logging for resource constraints
                    log_level: AuditLogLevel::Critical,
                    retention_days: 7,
                    remote_endpoint: None,
                },
            },
            ..Default::default()
        }
    }
    
    /// Create configuration for high-security enterprise deployment
    pub fn enterprise_secure() -> Self {
        Self {
            deployment_strategy: DeploymentStrategy::Hierarchical,
            security_config: EdgeSecurityConfig {
                mutual_tls_enabled: true,
                cert_rotation_interval_days: 7, // Weekly rotation
                data_encryption_at_rest: true,
                data_encryption_in_transit: true,
                access_control: AccessControlConfig {
                    rbac_enabled: true,
                    default_access_level: AccessLevel::ReadOnly,
                    api_key_required: true,
                    jwt_expiration_seconds: 1800, // 30 minutes
                },
                audit_logging: AuditLoggingConfig {
                    enabled: true,
                    log_level: AuditLogLevel::All,
                    retention_days: 365, // 1 year retention
                    remote_endpoint: Some("https://audit.company.com/api/logs".to_string()),
                },
            },
            federation_config: FederationConfig {
                enabled: true,
                consensus_protocol: ConsensusProtocol::Pbft, // Byzantine fault tolerance
                min_participants: 5,
                aggregation_strategy: AggregationStrategy::SecureAggregation,
                communication_protocol: CommunicationProtocol::Grpc,
                privacy_preservation: PrivacyPreservationConfig {
                    differential_privacy_enabled: true,
                    privacy_budget: 0.1, // Strict privacy
                    secure_multiparty_computation: true,
                    homomorphic_encryption: true,
                    anonymization_level: AnonymizationLevel::Full,
                },
            },
            ..Default::default()
        }
    }
    
    /// Validate edge configuration
    pub fn validate(&self) -> Result<()> {
        // Validate resource allocation
        if self.resource_allocation.cpu_cores == 0 {
            return Err(LiquidAudioError::ConfigError(
                "CPU cores must be greater than zero".to_string()
            ));
        }
        
        if self.resource_allocation.memory_mb < 512 {
            return Err(LiquidAudioError::ConfigError(
                "Memory must be at least 512MB".to_string()
            ));
        }
        
        // Validate geographic distribution
        if self.geographic_distribution.target_regions.is_empty() {
            return Err(LiquidAudioError::ConfigError(
                "At least one target region must be specified".to_string()
            ));
        }
        
        if self.geographic_distribution.min_nodes_per_region == 0 {
            return Err(LiquidAudioError::ConfigError(
                "Minimum nodes per region must be greater than zero".to_string()
            ));
        }
        
        // Validate auto-scaling configuration
        if self.auto_scaling.min_nodes > self.auto_scaling.max_nodes {
            return Err(LiquidAudioError::ConfigError(
                "Minimum nodes cannot exceed maximum nodes".to_string()
            ));
        }
        
        if self.auto_scaling.scale_up_cpu_threshold <= self.auto_scaling.scale_down_cpu_threshold {
            return Err(LiquidAudioError::ConfigError(
                "Scale up threshold must be greater than scale down threshold".to_string()
            ));
        }
        
        // Validate federation configuration
        if self.federation_config.enabled && self.federation_config.min_participants < 2 {
            return Err(LiquidAudioError::ConfigError(
                "Federation requires at least 2 participants".to_string()
            ));
        }
        
        if self.federation_config.privacy_preservation.privacy_budget <= 0.0 {
            return Err(LiquidAudioError::ConfigError(
                "Privacy budget must be positive".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Estimate total resource requirements
    pub fn estimate_total_resources(&self) -> ResourceEstimate {
        let total_nodes = self.geographic_distribution.target_regions.len() * 
                         self.geographic_distribution.min_nodes_per_region;
        
        ResourceEstimate {
            total_nodes,
            total_cpu_cores: total_nodes as u32 * self.resource_allocation.cpu_cores,
            total_memory_gb: (total_nodes as u64 * self.resource_allocation.memory_mb) / 1024,
            total_storage_gb: total_nodes as u64 * self.resource_allocation.storage_gb,
            total_bandwidth_gbps: (total_nodes as u64 * self.resource_allocation.network_bandwidth_mbps) / 1000,
            total_power_kw: (total_nodes as f32 * self.resource_allocation.power_budget_watts) / 1000.0,
            estimated_monthly_cost_usd: self.estimate_monthly_cost(total_nodes),
        }
    }
    
    /// Estimate monthly operational cost
    fn estimate_monthly_cost(&self, total_nodes: usize) -> f32 {
        // Rough cost estimation (actual costs depend on cloud provider)
        let cpu_cost_per_core_per_month = 20.0; // $20 per core per month
        let memory_cost_per_gb_per_month = 5.0;  // $5 per GB per month
        let storage_cost_per_gb_per_month = 0.1; // $0.10 per GB per month
        let bandwidth_cost_per_mbps_per_month = 1.0; // $1 per Mbps per month
        
        let cpu_cost = total_nodes as f32 * self.resource_allocation.cpu_cores as f32 * cpu_cost_per_core_per_month;
        let memory_cost = total_nodes as f32 * (self.resource_allocation.memory_mb as f32 / 1024.0) * memory_cost_per_gb_per_month;
        let storage_cost = total_nodes as f32 * self.resource_allocation.storage_gb as f32 * storage_cost_per_gb_per_month;
        let bandwidth_cost = total_nodes as f32 * self.resource_allocation.network_bandwidth_mbps as f32 * bandwidth_cost_per_mbps_per_month;
        
        cpu_cost + memory_cost + storage_cost + bandwidth_cost
    }
}

/// Resource estimation for edge deployment
#[derive(Debug, Clone)]
pub struct ResourceEstimate {
    /// Total number of edge nodes
    pub total_nodes: usize,
    /// Total CPU cores across all nodes
    pub total_cpu_cores: u32,
    /// Total memory in GB
    pub total_memory_gb: u64,
    /// Total storage in GB
    pub total_storage_gb: u64,
    /// Total bandwidth in Gbps
    pub total_bandwidth_gbps: u64,
    /// Total power consumption in kW
    pub total_power_kw: f32,
    /// Estimated monthly cost in USD
    pub estimated_monthly_cost_usd: f32,
}

impl ResourceEstimate {
    /// Get resource utilization efficiency
    pub fn efficiency_score(&self) -> f32 {
        // Calculate efficiency based on resource balance
        let cpu_memory_ratio = self.total_cpu_cores as f32 / (self.total_memory_gb as f32).max(1.0);
        let optimal_ratio = 0.5; // 1 core per 2GB memory
        let ratio_efficiency = 1.0 - (cpu_memory_ratio - optimal_ratio).abs() / optimal_ratio;
        
        // Factor in power efficiency
        let power_per_core = self.total_power_kw / (self.total_cpu_cores as f32).max(1.0);
        let power_efficiency = if power_per_core < 0.05 { 1.0 } else { 0.05 / power_per_core };
        
        (ratio_efficiency + power_efficiency) / 2.0
    }
    
    /// Get cost efficiency (performance per dollar)
    pub fn cost_efficiency(&self) -> f32 {
        if self.estimated_monthly_cost_usd > 0.0 {
            (self.total_cpu_cores as f32 * self.total_memory_gb as f32) / self.estimated_monthly_cost_usd
        } else {
            0.0
        }
    }
    
    /// Generate summary report
    pub fn summary(&self) -> String {
        format!(
            "Edge Deployment: {} nodes, {} cores, {:.1}GB RAM, {:.1}GB storage, {:.2}kW power, ${:.0}/month",
            self.total_nodes,
            self.total_cpu_cores,
            self.total_memory_gb,
            self.total_storage_gb,
            self.total_power_kw,
            self.estimated_monthly_cost_usd
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_edge_config_validation() {
        let config = EdgeConfig::default();
        assert!(config.validate().is_ok());
        
        let mut invalid_config = EdgeConfig::default();
        invalid_config.resource_allocation.cpu_cores = 0;
        assert!(invalid_config.validate().is_err());
    }
    
    #[test]
    fn test_edge_config_presets() {
        let iot_config = EdgeConfig::iot_optimized();
        assert_eq!(iot_config.resource_allocation.cpu_cores, 2);
        assert_eq!(iot_config.resource_allocation.memory_mb, 2048);
        assert!(!iot_config.auto_scaling.enabled);
        
        let enterprise_config = EdgeConfig::enterprise_secure();
        assert_eq!(enterprise_config.deployment_strategy, DeploymentStrategy::Hierarchical);
        assert!(enterprise_config.security_config.mutual_tls_enabled);
        assert_eq!(enterprise_config.security_config.cert_rotation_interval_days, 7);
    }
    
    #[test]
    fn test_resource_estimation() {
        let config = EdgeConfig::default();
        let estimate = config.estimate_total_resources();
        
        assert!(estimate.total_nodes > 0);
        assert!(estimate.total_cpu_cores > 0);
        assert!(estimate.estimated_monthly_cost_usd > 0.0);
        
        let efficiency = estimate.efficiency_score();
        assert!(efficiency >= 0.0 && efficiency <= 1.0);
    }
    
    #[test]
    fn test_auto_scaling_validation() {
        let mut config = EdgeConfig::default();
        config.auto_scaling.min_nodes = 10;
        config.auto_scaling.max_nodes = 5; // Invalid: min > max
        
        assert!(config.validate().is_err());
        
        config.auto_scaling.max_nodes = 15;
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_federation_config() {
        let config = EdgeConfig::enterprise_secure();
        assert!(config.federation_config.enabled);
        assert_eq!(config.federation_config.consensus_protocol, ConsensusProtocol::Pbft);
        assert_eq!(config.federation_config.aggregation_strategy, AggregationStrategy::SecureAggregation);
        assert!(config.federation_config.privacy_preservation.differential_privacy_enabled);
    }
    
    #[test]
    fn test_geographic_distribution() {
        let config = EdgeConfig::default();
        assert!(!config.geographic_distribution.target_regions.is_empty());
        assert!(config.geographic_distribution.min_nodes_per_region > 0);
        assert!(config.geographic_distribution.max_inter_node_latency_ms > 0);
    }
    
    #[test]
    fn test_security_config() {
        let enterprise_config = EdgeConfig::enterprise_secure();
        let security = &enterprise_config.security_config;
        
        assert!(security.mutual_tls_enabled);
        assert!(security.data_encryption_at_rest);
        assert!(security.data_encryption_in_transit);
        assert!(security.access_control.rbac_enabled);
        assert!(security.audit_logging.enabled);
        assert_eq!(security.audit_logging.log_level, AuditLogLevel::All);
    }
}