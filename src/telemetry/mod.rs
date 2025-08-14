//! Production telemetry and monitoring system
//! 
//! Comprehensive telemetry framework with OpenTelemetry integration,
//! real-time metrics collection, and distributed tracing capabilities.

pub mod metrics;
pub mod tracing;
pub mod monitoring;
pub mod alerts;
pub mod exporters;

pub use metrics::{MetricsCollector, MetricType, MetricValue, MetricsRegistry};
pub use tracing::{TracingContext, Span, SpanProcessor};
pub use monitoring::{HealthMonitor, SystemMonitor, PerformanceMonitor};
pub use alerts::{AlertManager, AlertRule, AlertSeverity, AlertChannel};
pub use exporters::{TelemetryExporter, PrometheusExporter, OtelExporter};

use crate::{Result, LiquidAudioError};

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String, boxed::Box, collections::BTreeMap};

#[cfg(feature = "std")]
use std::collections::BTreeMap;

/// Telemetry configuration for production monitoring
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Enable metrics collection
    pub metrics_enabled: bool,
    /// Enable distributed tracing
    pub tracing_enabled: bool,
    /// Enable health monitoring
    pub health_monitoring: bool,
    /// Enable alerting
    pub alerting_enabled: bool,
    /// Metrics collection interval (milliseconds)
    pub metrics_interval_ms: u64,
    /// Trace sampling rate (0.0-1.0)
    pub trace_sampling_rate: f32,
    /// Maximum number of metrics to retain
    pub max_metrics: usize,
    /// Maximum number of spans to retain
    pub max_spans: usize,
    /// Service name for telemetry
    pub service_name: String,
    /// Service version
    pub service_version: String,
    /// Environment (dev, staging, production)
    pub environment: String,
    /// Additional tags
    pub tags: BTreeMap<String, String>,
    /// Export endpoints
    pub export_endpoints: Vec<ExportEndpoint>,
}

/// Export endpoint configuration
#[derive(Debug, Clone)]
pub struct ExportEndpoint {
    /// Endpoint type (prometheus, otel, jaeger, etc.)
    pub endpoint_type: String,
    /// Endpoint URL
    pub url: String,
    /// Authentication headers
    pub headers: BTreeMap<String, String>,
    /// Export interval (milliseconds)
    pub interval_ms: u64,
    /// Batch size for exports
    pub batch_size: usize,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        let mut tags = BTreeMap::new();
        tags.insert("service".to_string(), "liquid-audio-nets".to_string());
        
        Self {
            metrics_enabled: true,
            tracing_enabled: true,
            health_monitoring: true,
            alerting_enabled: true,
            metrics_interval_ms: 5000,  // 5 seconds
            trace_sampling_rate: 0.1,   // 10% sampling
            max_metrics: 10000,
            max_spans: 1000,
            service_name: "liquid-audio-nets".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            environment: "production".to_string(),
            tags,
            export_endpoints: Vec::new(),
        }
    }
}

impl TelemetryConfig {
    /// Create configuration for development environment
    pub fn development() -> Self {
        Self {
            environment: "development".to_string(),
            trace_sampling_rate: 1.0,  // 100% sampling in dev
            metrics_interval_ms: 1000, // 1 second
            alerting_enabled: false,   // No alerts in dev
            ..Default::default()
        }
    }
    
    /// Create configuration for production environment
    pub fn production() -> Self {
        Self {
            environment: "production".to_string(),
            trace_sampling_rate: 0.01, // 1% sampling in production
            metrics_interval_ms: 10000, // 10 seconds
            max_metrics: 50000,
            max_spans: 5000,
            ..Default::default()
        }
    }
    
    /// Add Prometheus export endpoint
    pub fn with_prometheus(mut self, url: String) -> Self {
        self.export_endpoints.push(ExportEndpoint {
            endpoint_type: "prometheus".to_string(),
            url,
            headers: BTreeMap::new(),
            interval_ms: 15000,  // 15 seconds
            batch_size: 1000,
        });
        self
    }
    
    /// Add OpenTelemetry export endpoint
    pub fn with_otel(mut self, url: String) -> Self {
        self.export_endpoints.push(ExportEndpoint {
            endpoint_type: "otel".to_string(),
            url,
            headers: BTreeMap::new(),
            interval_ms: 10000,  // 10 seconds
            batch_size: 500,
        });
        self
    }
    
    /// Add Jaeger tracing endpoint
    pub fn with_jaeger(mut self, url: String) -> Self {
        self.export_endpoints.push(ExportEndpoint {
            endpoint_type: "jaeger".to_string(),
            url,
            headers: BTreeMap::new(),
            interval_ms: 5000,   // 5 seconds
            batch_size: 100,
        });
        self
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.trace_sampling_rate < 0.0 || self.trace_sampling_rate > 1.0 {
            return Err(LiquidAudioError::ConfigError(
                "Trace sampling rate must be between 0.0 and 1.0".to_string()
            ));
        }
        
        if self.metrics_interval_ms == 0 {
            return Err(LiquidAudioError::ConfigError(
                "Metrics interval must be greater than zero".to_string()
            ));
        }
        
        if self.service_name.is_empty() {
            return Err(LiquidAudioError::ConfigError(
                "Service name cannot be empty".to_string()
            ));
        }
        
        Ok(())
    }
}

/// Comprehensive telemetry system for production monitoring
pub struct TelemetrySystem {
    /// Configuration
    config: TelemetryConfig,
    /// Metrics collector
    metrics: MetricsCollector,
    /// Tracing context
    tracing: TracingContext,
    /// Health monitor
    health_monitor: HealthMonitor,
    /// System monitor
    system_monitor: SystemMonitor,
    /// Performance monitor
    performance_monitor: PerformanceMonitor,
    /// Alert manager
    alert_manager: AlertManager,
    /// Telemetry exporters
    exporters: Vec<Box<dyn TelemetryExporter>>,
    /// System state
    state: TelemetryState,
}

/// Telemetry system state
#[derive(Debug, Clone)]
struct TelemetryState {
    /// System start time
    start_time: u64,
    /// Total metrics collected
    metrics_collected: u64,
    /// Total spans created
    spans_created: u64,
    /// Total alerts fired
    alerts_fired: u64,
    /// Last export time
    last_export_time: u64,
    /// Export success count
    export_successes: u64,
    /// Export failure count
    export_failures: u64,
}

impl TelemetrySystem {
    /// Create new telemetry system
    pub fn new(config: TelemetryConfig) -> Result<Self> {
        config.validate()?;
        
        let metrics = MetricsCollector::new(&config)?;
        let tracing = TracingContext::new(&config)?;
        let health_monitor = HealthMonitor::new(&config)?;
        let system_monitor = SystemMonitor::new(&config)?;
        let performance_monitor = PerformanceMonitor::new(&config)?;
        let alert_manager = AlertManager::new(&config)?;
        
        // Initialize exporters based on configuration
        let mut exporters: Vec<Box<dyn TelemetryExporter>> = Vec::new();
        
        for endpoint in &config.export_endpoints {
            match endpoint.endpoint_type.as_str() {
                "prometheus" => {
                    exporters.push(Box::new(PrometheusExporter::new(endpoint.clone())?));
                },
                "otel" => {
                    exporters.push(Box::new(OtelExporter::new(endpoint.clone())?));
                },
                _ => {
                    // Log warning about unknown exporter type
                }
            }
        }
        
        let state = TelemetryState {
            start_time: get_current_time_ms(),
            metrics_collected: 0,
            spans_created: 0,
            alerts_fired: 0,
            last_export_time: 0,
            export_successes: 0,
            export_failures: 0,
        };
        
        Ok(Self {
            config,
            metrics,
            tracing,
            health_monitor,
            system_monitor,
            performance_monitor,
            alert_manager,
            exporters,
            state,
        })
    }
    
    /// Initialize telemetry system and start background tasks
    pub fn initialize(&mut self) -> Result<()> {
        // Start metrics collection
        if self.config.metrics_enabled {
            self.metrics.start_collection(self.config.metrics_interval_ms)?;
        }
        
        // Start health monitoring
        if self.config.health_monitoring {
            self.health_monitor.start_monitoring()?;
        }
        
        // Configure alert rules
        if self.config.alerting_enabled {
            self.setup_default_alerts()?;
        }
        
        // Start exporters
        for exporter in &mut self.exporters {
            exporter.start()?;
        }
        
        Ok(())
    }
    
    /// Record processing metrics for LNN inference
    pub fn record_processing_metrics(
        &mut self,
        processing_time_ms: f64,
        input_size: usize,
        output_size: usize,
        power_mw: f32,
        complexity: f32,
    ) -> Result<()> {
        if !self.config.metrics_enabled {
            return Ok(());
        }
        
        // Core processing metrics
        self.metrics.record_histogram(
            "lnn_processing_time_ms",
            processing_time_ms,
            &[
                ("environment", &self.config.environment),
                ("service", &self.config.service_name),
            ],
        )?;
        
        self.metrics.record_histogram(
            "lnn_power_consumption_mw",
            power_mw as f64,
            &[("environment", &self.config.environment)],
        )?;
        
        self.metrics.record_histogram(
            "lnn_complexity_score",
            complexity as f64,
            &[("environment", &self.config.environment)],
        )?;
        
        // Throughput metrics
        self.metrics.record_counter(
            "lnn_samples_processed_total",
            input_size as f64,
            &[("environment", &self.config.environment)],
        )?;
        
        self.metrics.record_counter(
            "lnn_inferences_total",
            1.0,
            &[("environment", &self.config.environment)],
        )?;
        
        // Efficiency metrics
        let efficiency = (output_size as f64) / (processing_time_ms * power_mw as f64);
        self.metrics.record_histogram(
            "lnn_efficiency_samples_per_mw_ms",
            efficiency,
            &[("environment", &self.config.environment)],
        )?;
        
        self.state.metrics_collected += 6;
        
        Ok(())
    }
    
    /// Start tracing span for operation
    pub fn start_span(&mut self, operation_name: &str) -> Result<Span> {
        if !self.config.tracing_enabled {
            return Ok(Span::noop());
        }
        
        let span = self.tracing.start_span(operation_name)?;
        self.state.spans_created += 1;
        
        Ok(span)
    }
    
    /// Record system health metrics
    pub fn record_health_metrics(&mut self) -> Result<()> {
        if !self.config.health_monitoring {
            return Ok(());
        }
        
        let health_status = self.health_monitor.check_health()?;
        
        // Record health score
        self.metrics.record_gauge(
            "system_health_score",
            health_status.overall_score as f64,
            &[("environment", &self.config.environment)],
        )?;
        
        // Record component health
        for (component, healthy) in &health_status.components {
            self.metrics.record_gauge(
                "component_health",
                if *healthy { 1.0 } else { 0.0 },
                &[
                    ("environment", &self.config.environment),
                    ("component", component),
                ],
            )?;
        }
        
        // Record system metrics
        let system_metrics = self.system_monitor.collect_metrics()?;
        
        self.metrics.record_gauge(
            "system_cpu_usage_percent",
            system_metrics.cpu_usage_percent as f64,
            &[("environment", &self.config.environment)],
        )?;
        
        self.metrics.record_gauge(
            "system_memory_usage_bytes",
            system_metrics.memory_usage_bytes as f64,
            &[("environment", &self.config.environment)],
        )?;
        
        self.metrics.record_gauge(
            "system_memory_usage_percent",
            system_metrics.memory_usage_percent as f64,
            &[("environment", &self.config.environment)],
        )?;
        
        Ok(())
    }
    
    /// Check and fire alerts based on current metrics
    pub fn check_alerts(&mut self) -> Result<()> {
        if !self.config.alerting_enabled {
            return Ok(());
        }
        
        let fired_alerts = self.alert_manager.check_alerts(&self.metrics)?;
        self.state.alerts_fired += fired_alerts.len() as u64;
        
        // Record alert metrics
        for alert in fired_alerts {
            self.metrics.record_counter(
                "alerts_fired_total",
                1.0,
                &[
                    ("environment", &self.config.environment),
                    ("severity", &format!("{:?}", alert.severity)),
                    ("rule", &alert.rule_name),
                ],
            )?;
        }
        
        Ok(())
    }
    
    /// Export metrics to configured endpoints
    pub fn export_telemetry(&mut self) -> Result<()> {
        let current_time = get_current_time_ms();
        
        for exporter in &mut self.exporters {
            match exporter.export(&self.metrics, &self.tracing) {
                Ok(_) => {
                    self.state.export_successes += 1;
                },
                Err(e) => {
                    self.state.export_failures += 1;
                    // Log error but continue with other exporters
                    eprintln!("Failed to export telemetry: {}", e);
                }
            }
        }
        
        self.state.last_export_time = current_time;
        
        Ok(())
    }
    
    /// Get telemetry system statistics
    pub fn get_statistics(&self) -> TelemetryStatistics {
        let uptime_ms = get_current_time_ms().saturating_sub(self.state.start_time);
        
        TelemetryStatistics {
            uptime_ms,
            metrics_collected: self.state.metrics_collected,
            spans_created: self.state.spans_created,
            alerts_fired: self.state.alerts_fired,
            export_successes: self.state.export_successes,
            export_failures: self.state.export_failures,
            export_success_rate: if self.state.export_successes + self.state.export_failures > 0 {
                self.state.export_successes as f64 / 
                (self.state.export_successes + self.state.export_failures) as f64
            } else {
                0.0
            },
            metrics_rate_per_second: if uptime_ms > 0 {
                (self.state.metrics_collected as f64) / (uptime_ms as f64 / 1000.0)
            } else {
                0.0
            },
            spans_rate_per_second: if uptime_ms > 0 {
                (self.state.spans_created as f64) / (uptime_ms as f64 / 1000.0)
            } else {
                0.0
            },
        }
    }
    
    /// Setup default alert rules for LNN monitoring
    fn setup_default_alerts(&mut self) -> Result<()> {
        use alerts::{AlertRule, AlertSeverity, AlertCondition, ComparisonOp};
        
        // High processing time alert
        let high_latency_rule = AlertRule {
            name: "HighProcessingLatency".to_string(),
            description: "LNN processing latency is too high".to_string(),
            severity: AlertSeverity::Warning,
            condition: AlertCondition {
                metric_name: "lnn_processing_time_ms".to_string(),
                comparison: ComparisonOp::GreaterThan,
                threshold: 100.0, // 100ms
                duration_ms: 60000, // 1 minute
            },
            enabled: true,
        };
        
        // High power consumption alert
        let high_power_rule = AlertRule {
            name: "HighPowerConsumption".to_string(),
            description: "LNN power consumption is too high".to_string(),
            severity: AlertSeverity::Warning,
            condition: AlertCondition {
                metric_name: "lnn_power_consumption_mw".to_string(),
                comparison: ComparisonOp::GreaterThan,
                threshold: 10.0, // 10mW
                duration_ms: 300000, // 5 minutes
            },
            enabled: true,
        };
        
        // Low system health alert
        let low_health_rule = AlertRule {
            name: "LowSystemHealth".to_string(),
            description: "System health score is low".to_string(),
            severity: AlertSeverity::Critical,
            condition: AlertCondition {
                metric_name: "system_health_score".to_string(),
                comparison: ComparisonOp::LessThan,
                threshold: 0.5, // 50%
                duration_ms: 120000, // 2 minutes
            },
            enabled: true,
        };
        
        // High CPU usage alert
        let high_cpu_rule = AlertRule {
            name: "HighCpuUsage".to_string(),
            description: "CPU usage is too high".to_string(),
            severity: AlertSeverity::Warning,
            condition: AlertCondition {
                metric_name: "system_cpu_usage_percent".to_string(),
                comparison: ComparisonOp::GreaterThan,
                threshold: 80.0, // 80%
                duration_ms: 180000, // 3 minutes
            },
            enabled: true,
        };
        
        self.alert_manager.add_rule(high_latency_rule)?;
        self.alert_manager.add_rule(high_power_rule)?;
        self.alert_manager.add_rule(low_health_rule)?;
        self.alert_manager.add_rule(high_cpu_rule)?;
        
        Ok(())
    }
    
    /// Shutdown telemetry system gracefully
    pub fn shutdown(&mut self) -> Result<()> {
        // Export final metrics
        self.export_telemetry()?;
        
        // Stop all exporters
        for exporter in &mut self.exporters {
            exporter.stop()?;
        }
        
        // Stop monitoring
        self.health_monitor.stop()?;
        self.system_monitor.stop()?;
        
        Ok(())
    }
}

/// Telemetry system statistics
#[derive(Debug, Clone)]
pub struct TelemetryStatistics {
    /// System uptime (milliseconds)
    pub uptime_ms: u64,
    /// Total metrics collected
    pub metrics_collected: u64,
    /// Total spans created
    pub spans_created: u64,
    /// Total alerts fired
    pub alerts_fired: u64,
    /// Successful exports
    pub export_successes: u64,
    /// Failed exports
    pub export_failures: u64,
    /// Export success rate (0.0-1.0)
    pub export_success_rate: f64,
    /// Metrics collection rate (per second)
    pub metrics_rate_per_second: f64,
    /// Span creation rate (per second)
    pub spans_rate_per_second: f64,
}

impl TelemetryStatistics {
    /// Get formatted summary string
    pub fn summary(&self) -> String {
        format!(
            "Telemetry: {}s uptime, {:.1} metrics/s, {:.1} spans/s, {:.1}% export success",
            self.uptime_ms / 1000,
            self.metrics_rate_per_second,
            self.spans_rate_per_second,
            self.export_success_rate * 100.0
        )
    }
}

/// Get current time in milliseconds
fn get_current_time_ms() -> u64 {
    #[cfg(feature = "std")]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
    
    #[cfg(not(feature = "std"))]
    {
        // Simple counter for embedded systems
        static mut COUNTER: u64 = 0;
        unsafe {
            COUNTER += 1000; // Increment by 1 second
            COUNTER
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_telemetry_config_validation() {
        let config = TelemetryConfig::default();
        assert!(config.validate().is_ok());
        
        let invalid_config = TelemetryConfig {
            trace_sampling_rate: 1.5, // Invalid > 1.0
            ..Default::default()
        };
        assert!(invalid_config.validate().is_err());
    }
    
    #[test]
    fn test_telemetry_config_presets() {
        let dev_config = TelemetryConfig::development();
        assert_eq!(dev_config.environment, "development");
        assert_eq!(dev_config.trace_sampling_rate, 1.0);
        assert!(!dev_config.alerting_enabled);
        
        let prod_config = TelemetryConfig::production();
        assert_eq!(prod_config.environment, "production");
        assert_eq!(prod_config.trace_sampling_rate, 0.01);
        assert!(prod_config.alerting_enabled);
    }
    
    #[test]
    fn test_telemetry_config_endpoints() {
        let config = TelemetryConfig::default()
            .with_prometheus("http://localhost:9090".to_string())
            .with_otel("http://localhost:4317".to_string());
        
        assert_eq!(config.export_endpoints.len(), 2);
        assert_eq!(config.export_endpoints[0].endpoint_type, "prometheus");
        assert_eq!(config.export_endpoints[1].endpoint_type, "otel");
    }
    
    #[test]
    fn test_telemetry_statistics() {
        let stats = TelemetryStatistics {
            uptime_ms: 60000,  // 1 minute
            metrics_collected: 120,
            spans_created: 30,
            alerts_fired: 2,
            export_successes: 8,
            export_failures: 2,
            export_success_rate: 0.8,
            metrics_rate_per_second: 2.0,
            spans_rate_per_second: 0.5,
        };
        
        let summary = stats.summary();
        assert!(summary.contains("60s uptime"));
        assert!(summary.contains("2.0 metrics/s"));
        assert!(summary.contains("80.0% export success"));
    }
}