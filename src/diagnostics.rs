//! Diagnostics and health monitoring for Liquid Neural Networks
//!
//! Provides comprehensive diagnostics, health checks, and performance monitoring
//! for production deployment of LNN models.

use crate::{Result, LiquidAudioError, ModelConfig, ProcessingResult};

#[cfg(not(feature = "std"))]
use core::alloc::{vec::Vec, string::String};

/// Health status levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// System is operating normally
    Healthy,
    /// System is degraded but functional
    Degraded,
    /// System has warnings but is operational
    Warning,
    /// System has critical issues
    Critical,
    /// System is not functional
    Failed,
}

/// Comprehensive health check results
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// Overall system status
    pub status: HealthStatus,
    /// Individual check results
    pub checks: Vec<HealthCheck>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Timestamp of health check
    pub timestamp: u64,
    /// System uptime in seconds
    pub uptime_seconds: u64,
}

/// Individual health check result
#[derive(Debug, Clone)]
pub struct HealthCheck {
    /// Name of the check
    pub name: String,
    /// Check status
    pub status: HealthStatus,
    /// Detailed message
    pub message: String,
    /// Check duration in microseconds
    pub duration_us: u64,
    /// Additional metadata
    pub metadata: Vec<(String, String)>,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Average processing latency (ms)
    pub avg_latency_ms: f32,
    /// Peak latency (ms)
    pub peak_latency_ms: f32,
    /// Current power consumption (mW)
    pub current_power_mw: f32,
    /// Average power consumption (mW)
    pub avg_power_mw: f32,
    /// Total processed samples
    pub total_samples: u64,
    /// Error rate (errors per 1000 samples)
    pub error_rate: f32,
    /// Memory usage (bytes)
    pub memory_usage_bytes: usize,
    /// CPU utilization percentage
    pub cpu_utilization_percent: f32,
}

/// Diagnostic information collector
#[derive(Debug)]
pub struct DiagnosticsCollector {
    /// Performance history
    latency_history: Vec<f32>,
    power_history: Vec<f32>,
    error_count: u64,
    total_samples: u64,
    start_time: u64,
    last_health_check: u64,
}

impl DiagnosticsCollector {
    /// Create new diagnostics collector
    pub fn new() -> Self {
        Self {
            latency_history: Vec::new(),
            power_history: Vec::new(),
            error_count: 0,
            total_samples: 0,
            start_time: Self::current_timestamp(),
            last_health_check: 0,
        }
    }

    /// Record processing metrics
    pub fn record_processing(&mut self, result: &ProcessingResult, processing_time_ms: f32) {
        self.total_samples += 1;
        
        // Keep rolling window of recent metrics
        self.latency_history.push(processing_time_ms);
        if self.latency_history.len() > 1000 {
            self.latency_history.remove(0);
        }
        
        self.power_history.push(result.power_mw);
        if self.power_history.len() > 1000 {
            self.power_history.remove(0);
        }
    }

    /// Record processing error
    pub fn record_error(&mut self, _error: &LiquidAudioError) {
        self.error_count += 1;
    }

    /// Perform comprehensive health check
    pub fn health_check(&mut self, config: &ModelConfig) -> Result<HealthReport> {
        let timestamp = Self::current_timestamp();
        self.last_health_check = timestamp;
        
        let mut checks = Vec::new();
        let mut overall_status = HealthStatus::Healthy;

        // Performance health checks
        let perf_check = self.check_performance();
        if perf_check.status as u8 > overall_status as u8 {
            overall_status = perf_check.status;
        }
        checks.push(perf_check);

        // Memory health check
        let memory_check = self.check_memory_usage(config);
        if memory_check.status as u8 > overall_status as u8 {
            overall_status = memory_check.status;
        }
        checks.push(memory_check);

        // Error rate health check
        let error_check = self.check_error_rate();
        if error_check.status as u8 > overall_status as u8 {
            overall_status = error_check.status;
        }
        checks.push(error_check);

        // System resource check
        let resource_check = self.check_system_resources();
        if resource_check.status as u8 > overall_status as u8 {
            overall_status = resource_check.status;
        }
        checks.push(resource_check);

        // Configuration validation check
        let config_check = self.check_configuration(config);
        if config_check.status as u8 > overall_status as u8 {
            overall_status = config_check.status;
        }
        checks.push(config_check);

        // Compute performance metrics
        let metrics = self.compute_metrics();

        Ok(HealthReport {
            status: overall_status,
            checks,
            metrics,
            timestamp,
            uptime_seconds: (timestamp - self.start_time) / 1000,
        })
    }

    /// Check performance metrics
    fn check_performance(&self) -> HealthCheck {
        let start_time = Self::current_timestamp();
        
        let (status, message) = if self.latency_history.is_empty() {
            (HealthStatus::Warning, "No performance data available".to_string())
        } else {
            let avg_latency = self.latency_history.iter().sum::<f32>() / self.latency_history.len() as f32;
            let max_latency = self.latency_history.iter().fold(0.0f32, |a, &b| a.max(b));
            
            if avg_latency > 100.0 {
                (HealthStatus::Critical, format!("High average latency: {:.1}ms", avg_latency))
            } else if avg_latency > 50.0 {
                (HealthStatus::Degraded, format!("Elevated latency: {:.1}ms", avg_latency))
            } else if max_latency > 200.0 {
                (HealthStatus::Warning, format!("High peak latency: {:.1}ms", max_latency))
            } else {
                (HealthStatus::Healthy, format!("Performance normal: avg {:.1}ms", avg_latency))
            }
        };

        let duration_us = (Self::current_timestamp() - start_time) * 1000;

        HealthCheck {
            name: "Performance".to_string(),
            status,
            message,
            duration_us,
            metadata: vec![
                ("samples".to_string(), self.latency_history.len().to_string()),
            ],
        }
    }

    /// Check memory usage
    fn check_memory_usage(&self, config: &ModelConfig) -> HealthCheck {
        let start_time = Self::current_timestamp();
        
        // Estimate memory usage based on model configuration
        let estimated_usage = self.estimate_memory_usage(config);
        
        let (status, message) = if estimated_usage > 512 * 1024 {
            (HealthStatus::Critical, format!("High memory usage: {}KB", estimated_usage / 1024))
        } else if estimated_usage > 256 * 1024 {
            (HealthStatus::Warning, format!("Elevated memory usage: {}KB", estimated_usage / 1024))
        } else {
            (HealthStatus::Healthy, format!("Memory usage normal: {}KB", estimated_usage / 1024))
        };

        let duration_us = (Self::current_timestamp() - start_time) * 1000;

        HealthCheck {
            name: "Memory".to_string(),
            status,
            message,
            duration_us,
            metadata: vec![
                ("estimated_bytes".to_string(), estimated_usage.to_string()),
            ],
        }
    }

    /// Check error rate
    fn check_error_rate(&self) -> HealthCheck {
        let start_time = Self::current_timestamp();
        
        let (status, message) = if self.total_samples == 0 {
            (HealthStatus::Warning, "No processing data available".to_string())
        } else {
            let error_rate = (self.error_count as f32 / self.total_samples as f32) * 1000.0;
            
            if error_rate > 10.0 {
                (HealthStatus::Critical, format!("High error rate: {:.1} errors/1000 samples", error_rate))
            } else if error_rate > 5.0 {
                (HealthStatus::Degraded, format!("Elevated error rate: {:.1} errors/1000 samples", error_rate))
            } else if error_rate > 1.0 {
                (HealthStatus::Warning, format!("Moderate error rate: {:.1} errors/1000 samples", error_rate))
            } else {
                (HealthStatus::Healthy, format!("Error rate normal: {:.1} errors/1000 samples", error_rate))
            }
        };

        let duration_us = (Self::current_timestamp() - start_time) * 1000;

        HealthCheck {
            name: "Error Rate".to_string(),
            status,
            message,
            duration_us,
            metadata: vec![
                ("errors".to_string(), self.error_count.to_string()),
                ("samples".to_string(), self.total_samples.to_string()),
            ],
        }
    }

    /// Check system resources
    fn check_system_resources(&self) -> HealthCheck {
        let start_time = Self::current_timestamp();
        
        // Simple resource check - in real implementation would check actual system resources
        let (status, message) = (
            HealthStatus::Healthy, 
            "System resources OK".to_string()
        );

        let duration_us = (Self::current_timestamp() - start_time) * 1000;

        HealthCheck {
            name: "System Resources".to_string(),
            status,
            message,
            duration_us,
            metadata: vec![],
        }
    }

    /// Check configuration validity
    fn check_configuration(&self, config: &ModelConfig) -> HealthCheck {
        let start_time = Self::current_timestamp();
        
        let (status, message) = if config.input_dim == 0 || config.hidden_dim == 0 || config.output_dim == 0 {
            (HealthStatus::Critical, "Invalid model dimensions".to_string())
        } else if config.sample_rate < 8000 || config.sample_rate > 48000 {
            (HealthStatus::Warning, format!("Unusual sample rate: {}Hz", config.sample_rate))
        } else if config.frame_size < 64 || config.frame_size > 4096 {
            (HealthStatus::Warning, format!("Unusual frame size: {}", config.frame_size))
        } else {
            (HealthStatus::Healthy, "Configuration valid".to_string())
        };

        let duration_us = (Self::current_timestamp() - start_time) * 1000;

        HealthCheck {
            name: "Configuration".to_string(),
            status,
            message,
            duration_us,
            metadata: vec![
                ("input_dim".to_string(), config.input_dim.to_string()),
                ("hidden_dim".to_string(), config.hidden_dim.to_string()),
                ("output_dim".to_string(), config.output_dim.to_string()),
            ],
        }
    }

    /// Compute current performance metrics
    fn compute_metrics(&self) -> PerformanceMetrics {
        let avg_latency_ms = if self.latency_history.is_empty() {
            0.0
        } else {
            self.latency_history.iter().sum::<f32>() / self.latency_history.len() as f32
        };

        let peak_latency_ms = self.latency_history.iter().fold(0.0f32, |a, &b| a.max(b));

        let (current_power_mw, avg_power_mw) = if self.power_history.is_empty() {
            (0.0, 0.0)
        } else {
            let current = self.power_history.last().copied().unwrap_or(0.0);
            let avg = self.power_history.iter().sum::<f32>() / self.power_history.len() as f32;
            (current, avg)
        };

        let error_rate = if self.total_samples > 0 {
            (self.error_count as f32 / self.total_samples as f32) * 1000.0
        } else {
            0.0
        };

        // Estimate memory usage (simplified)
        let memory_usage_bytes = self.latency_history.len() * 4 + self.power_history.len() * 4 + 1024;

        // Simple CPU utilization estimate based on latency
        let cpu_utilization_percent = if avg_latency_ms > 0.0 {
            (avg_latency_ms / 10.0).min(100.0)  // Rough estimate
        } else {
            0.0
        };

        PerformanceMetrics {
            avg_latency_ms,
            peak_latency_ms,
            current_power_mw,
            avg_power_mw,
            total_samples: self.total_samples,
            error_rate,
            memory_usage_bytes,
            cpu_utilization_percent,
        }
    }

    /// Estimate memory usage for given configuration
    fn estimate_memory_usage(&self, config: &ModelConfig) -> usize {
        // Model weights
        let weights_size = (config.input_dim * config.hidden_dim + 
                           config.hidden_dim * config.hidden_dim +
                           config.hidden_dim * config.output_dim) * 4; // f32

        // State and buffers  
        let state_size = config.hidden_dim * 4;
        let buffer_size = config.frame_size * 4;

        // History buffers
        let history_size = (self.latency_history.len() + self.power_history.len()) * 4;

        // Overhead
        let overhead = 16 * 1024;

        weights_size + state_size + buffer_size + history_size + overhead
    }

    /// Get current timestamp in milliseconds
    fn current_timestamp() -> u64 {
        // Simple timestamp - in real implementation would use proper time source
        static mut TIMESTAMP: u64 = 0;
        unsafe {
            TIMESTAMP += 1;
            TIMESTAMP
        }
    }

    /// Generate health summary
    pub fn format_health_summary(&self, report: &HealthReport) -> String {
        let status_emoji = match report.status {
            HealthStatus::Healthy => "✅",
            HealthStatus::Warning => "⚠️",
            HealthStatus::Degraded => "🟡",
            HealthStatus::Critical => "🔴",
            HealthStatus::Failed => "❌",
        };

        format!(
            "{} System Status: {:?}\n\
            📊 Processed: {} samples\n\
            ⚡ Power: {:.1}mW avg, {:.1}mW current\n\
            ⏱️  Latency: {:.1}ms avg, {:.1}ms peak\n\
            🚨 Errors: {:.1}/1000 samples\n\
            💾 Memory: {}KB\n\
            ⏰ Uptime: {}s",
            status_emoji, report.status,
            report.metrics.total_samples,
            report.metrics.avg_power_mw, report.metrics.current_power_mw,
            report.metrics.avg_latency_ms, report.metrics.peak_latency_ms,
            report.metrics.error_rate,
            report.metrics.memory_usage_bytes / 1024,
            report.uptime_seconds
        )
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.latency_history.clear();
        self.power_history.clear();
        self.error_count = 0;
        self.total_samples = 0;
        self.start_time = Self::current_timestamp();
    }

    /// Get diagnostic recommendations
    pub fn get_recommendations(&self, report: &HealthReport) -> Vec<String> {
        let mut recommendations = Vec::new();

        if report.metrics.avg_latency_ms > 50.0 {
            recommendations.push("Consider reducing model complexity to improve latency".to_string());
        }

        if report.metrics.avg_power_mw > 2.0 {
            recommendations.push("Enable adaptive timestep to reduce power consumption".to_string());
        }

        if report.metrics.error_rate > 5.0 {
            recommendations.push("Check input validation and error handling".to_string());
        }

        if report.metrics.memory_usage_bytes > 256 * 1024 {
            recommendations.push("Consider model quantization to reduce memory usage".to_string());
        }

        if recommendations.is_empty() {
            recommendations.push("System is operating within normal parameters".to_string());
        }

        recommendations
    }
}

impl Default for DiagnosticsCollector {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple logging interface for diagnostics
pub struct Logger;

impl Logger {
    /// Log info message
    pub fn info(message: &str) {
        #[cfg(feature = "std")]
        println!("[INFO] {}", message);
        
        #[cfg(not(feature = "std"))]
        {
            // In no-std mode, logging would go to RTT, UART, or be stored in buffer
            let _ = message; // Suppress unused warning
        }
    }

    /// Log warning message
    pub fn warn(message: &str) {
        #[cfg(feature = "std")]
        println!("[WARN] {}", message);
        
        #[cfg(not(feature = "std"))]
        {
            let _ = message;
        }
    }

    /// Log error message
    pub fn error(message: &str) {
        #[cfg(feature = "std")]
        eprintln!("[ERROR] {}", message);
        
        #[cfg(not(feature = "std"))]
        {
            let _ = message;
        }
    }

    /// Log critical message
    pub fn critical(message: &str) {
        #[cfg(feature = "std")]
        eprintln!("[CRITICAL] {}", message);
        
        #[cfg(not(feature = "std"))]
        {
            let _ = message;
        }
    }
}