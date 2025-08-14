//! Performance optimization utilities for Liquid Neural Networks
//!
//! Provides advanced optimization techniques including vectorization,
//! memory pooling, batch processing, and adaptive computation.

use crate::{Result, LiquidAudioError, ProcessingResult};
use crate::cache::{ModelCache, CacheConfig};
#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String, collections::VecDeque};

#[cfg(feature = "std")]
use std::{vec::Vec, string::String, collections::VecDeque};

#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

/// Vectorized mathematical operations
pub struct VectorOps;

impl VectorOps {
    /// Vectorized dot product
    pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(LiquidAudioError::InvalidInput("Vector length mismatch".to_string()));
        }

        let mut sum = 0.0f32;
        
        // Process in chunks for better cache performance
        const CHUNK_SIZE: usize = 8;
        let chunks = a.len() / CHUNK_SIZE;
        
        // Vectorized chunk processing
        for i in 0..chunks {
            let base = i * CHUNK_SIZE;
            let mut chunk_sum = 0.0f32;
            
            for j in 0..CHUNK_SIZE {
                chunk_sum += a[base + j] * b[base + j];
            }
            sum += chunk_sum;
        }
        
        // Handle remainder
        for i in (chunks * CHUNK_SIZE)..a.len() {
            sum += a[i] * b[i];
        }
        
        Ok(sum)
    }

    /// Vectorized matrix-vector multiplication
    pub fn matvec_multiply(matrix: &[Vec<f32>], vector: &[f32]) -> Result<Vec<f32>> {
        if matrix.is_empty() || matrix[0].len() != vector.len() {
            return Err(LiquidAudioError::InvalidInput("Matrix-vector dimension mismatch".to_string()));
        }

        let mut result = Vec::with_capacity(matrix.len());
        
        for row in matrix {
            result.push(Self::dot_product(row, vector)?);
        }
        
        Ok(result)
    }

    /// Vectorized activation function (tanh)
    pub fn tanh_activation(input: &mut [f32]) {
        for value in input.iter_mut() {
            // Fast tanh approximation for embedded systems
            *value = Self::fast_tanh(*value);
        }
    }

    /// Fast tanh approximation
    fn fast_tanh(x: f32) -> f32 {
        if x > 2.0 {
            1.0
        } else if x < -2.0 {
            -1.0
        } else {
            // Rational approximation
            let x2 = x * x;
            x * (27.0 + x2) / (27.0 + 9.0 * x2)
        }
    }

    /// Vectorized softmax
    pub fn softmax(input: &mut [f32]) -> Result<()> {
        if input.is_empty() {
            return Err(LiquidAudioError::InvalidInput("Empty input for softmax".to_string()));
        }

        // Find maximum for numerical stability
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exponentials and sum
        let mut sum = 0.0f32;
        for value in input.iter_mut() {
            *value = (*value - max_val).exp();
            sum += *value;
        }
        
        // Normalize
        if sum > f32::EPSILON {
            for value in input.iter_mut() {
                *value /= sum;
            }
        }
        
        Ok(())
    }

    /// Element-wise vector operations
    pub fn vector_add(a: &mut [f32], b: &[f32]) -> Result<()> {
        if a.len() != b.len() {
            return Err(LiquidAudioError::InvalidInput("Vector length mismatch".to_string()));
        }

        for (ai, &bi) in a.iter_mut().zip(b.iter()) {
            *ai += bi;
        }
        
        Ok(())
    }

    /// Scalar multiplication
    pub fn scalar_multiply(vector: &mut [f32], scalar: f32) {
        for value in vector.iter_mut() {
            *value *= scalar;
        }
    }
}

/// Memory pool for efficient allocation
#[derive(Debug)]
pub struct MemoryPool {
    /// Pool of reusable float vectors
    float_vectors: Vec<Vec<f32>>,
    /// Pool configuration
    config: PoolConfig,
    /// Allocation statistics
    stats: PoolStats,
}

/// Memory pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Initial pool size
    pub initial_size: usize,
    /// Maximum pool size
    pub max_size: usize,
    /// Default vector capacity
    pub default_capacity: usize,
    /// Enable pool monitoring
    pub enable_monitoring: bool,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            initial_size: 10,
            max_size: 100,
            default_capacity: 512,
            enable_monitoring: true,
        }
    }
}

/// Pool allocation statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Number of allocations
    pub allocations: u64,
    /// Number of returns
    pub returns: u64,
    /// Current pool size
    pub pool_size: usize,
    /// Peak pool size
    pub peak_pool_size: usize,
}

impl MemoryPool {
    /// Create new memory pool
    pub fn new(config: PoolConfig) -> Self {
        let mut pool = Self {
            float_vectors: Vec::with_capacity(config.max_size),
            config,
            stats: PoolStats::default(),
        };

        // Pre-allocate initial vectors
        for _ in 0..pool.config.initial_size {
            pool.float_vectors.push(Vec::with_capacity(pool.config.default_capacity));
        }

        pool.stats.pool_size = pool.config.initial_size;
        pool.stats.peak_pool_size = pool.config.initial_size;
        
        pool
    }

    /// Get a vector from the pool
    pub fn get_vector(&mut self, capacity: usize) -> Vec<f32> {
        self.stats.allocations += 1;

        if let Some(mut vector) = self.float_vectors.pop() {
            // Reuse existing vector
            vector.clear();
            if vector.capacity() < capacity {
                vector.reserve(capacity - vector.capacity());
            }
            self.stats.pool_size = self.stats.pool_size.saturating_sub(1);
            vector
        } else {
            // Create new vector
            Vec::with_capacity(capacity)
        }
    }

    /// Return a vector to the pool
    pub fn return_vector(&mut self, mut vector: Vec<f32>) {
        self.stats.returns += 1;

        if self.float_vectors.len() < self.config.max_size {
            vector.clear();
            self.float_vectors.push(vector);
            self.stats.pool_size += 1;
            self.stats.peak_pool_size = self.stats.peak_pool_size.max(self.stats.pool_size);
        }
        // Otherwise let it be dropped
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Clear the pool
    pub fn clear(&mut self) {
        self.float_vectors.clear();
        self.stats.pool_size = 0;
    }
}

/// Batch processing engine
#[derive(Debug)]
pub struct BatchProcessor {
    /// Batch buffer
    batch: Vec<Vec<f32>>,
    /// Maximum batch size
    max_batch_size: usize,
    /// Current batch size
    current_size: usize,
    /// Memory pool
    memory_pool: MemoryPool,
    /// Processing statistics
    stats: BatchStats,
}

/// Batch processing statistics
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Processed batches
    pub batches_processed: u64,
    /// Total samples processed
    pub total_samples: u64,
    /// Average batch size
    pub avg_batch_size: f32,
    /// Processing time (ms)
    pub total_processing_time_ms: f32,
}

impl BatchProcessor {
    /// Create new batch processor
    pub fn new(max_batch_size: usize, pool_config: PoolConfig) -> Self {
        Self {
            batch: Vec::with_capacity(max_batch_size),
            max_batch_size,
            current_size: 0,
            memory_pool: MemoryPool::new(pool_config),
            stats: BatchStats::default(),
        }
    }

    /// Add sample to batch
    pub fn add_sample(&mut self, sample: Vec<f32>) -> bool {
        if self.current_size >= self.max_batch_size {
            return false; // Batch full
        }

        self.batch.push(sample);
        self.current_size += 1;
        true
    }

    /// Check if batch is ready for processing
    pub fn is_batch_ready(&self) -> bool {
        self.current_size >= self.max_batch_size
    }

    /// Process current batch
    pub fn process_batch<F>(&mut self, processor: F) -> Result<Vec<ProcessingResult>>
    where
        F: Fn(&[Vec<f32>]) -> Result<Vec<ProcessingResult>>,
    {
        if self.current_size == 0 {
            return Ok(Vec::new());
        }

        let start_time = Self::current_timestamp();
        
        // Process the batch
        let results = processor(&self.batch[..self.current_size])?;

        let end_time = Self::current_timestamp();
        let processing_time = (end_time - start_time) as f32;

        // Update statistics
        self.stats.batches_processed += 1;
        self.stats.total_samples += self.current_size as u64;
        self.stats.total_processing_time_ms += processing_time;
        
        // Update average batch size
        let total_batches = self.stats.batches_processed as f32;
        self.stats.avg_batch_size = (self.stats.avg_batch_size * (total_batches - 1.0) + self.current_size as f32) / total_batches;

        // Return vectors to pool and clear batch
        for sample in self.batch.drain(..self.current_size) {
            self.memory_pool.return_vector(sample);
        }
        self.current_size = 0;

        Ok(results)
    }

    /// Force process partial batch
    pub fn flush_batch<F>(&mut self, processor: F) -> Result<Vec<ProcessingResult>>
    where
        F: Fn(&[Vec<f32>]) -> Result<Vec<ProcessingResult>>,
    {
        self.process_batch(processor)
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> &BatchStats {
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

/// Adaptive computation controller
#[derive(Debug)]
pub struct AdaptiveComputation {
    /// Current computation level (0.0 to 1.0)
    computation_level: f32,
    /// Performance history
    performance_history: VecDeque<PerformanceMetric>,
    /// Adaptation parameters
    config: AdaptationConfig,
}

/// Performance metric for adaptation
#[derive(Debug, Clone)]
struct PerformanceMetric {
    /// Processing latency (ms)
    latency_ms: f32,
    /// Power consumption (mW)
    power_mw: f32,
    /// Accuracy estimate
    accuracy: f32,
    /// Timestamp
    timestamp: u64,
}

/// Adaptation configuration
#[derive(Debug, Clone)]
pub struct AdaptationConfig {
    /// Target latency (ms)
    pub target_latency_ms: f32,
    /// Target power (mW)
    pub target_power_mw: f32,
    /// Minimum computation level
    pub min_computation_level: f32,
    /// Maximum computation level
    pub max_computation_level: f32,
    /// Adaptation rate
    pub adaptation_rate: f32,
    /// History window size
    pub history_window: usize,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            target_latency_ms: 10.0,
            target_power_mw: 1.0,
            min_computation_level: 0.1,
            max_computation_level: 1.0,
            adaptation_rate: 0.05,
            history_window: 20,
        }
    }
}

impl AdaptiveComputation {
    /// Create new adaptive computation controller
    pub fn new(config: AdaptationConfig) -> Self {
        Self {
            computation_level: 0.5, // Start at medium level
            performance_history: VecDeque::with_capacity(config.history_window),
            config,
        }
    }

    /// Update performance and adapt computation level
    pub fn update_performance(&mut self, latency_ms: f32, power_mw: f32, accuracy: f32) {
        let metric = PerformanceMetric {
            latency_ms,
            power_mw,
            accuracy,
            timestamp: Self::current_timestamp(),
        };

        // Add to history
        if self.performance_history.len() >= self.config.history_window {
            self.performance_history.pop_front();
        }
        self.performance_history.push_back(metric.clone());

        // Adapt computation level
        self.adapt_computation_level(&metric);
    }

    /// Get current computation level
    pub fn get_computation_level(&self) -> f32 {
        self.computation_level
    }

    /// Adapt computation level based on performance
    fn adapt_computation_level(&mut self, current: &PerformanceMetric) {
        let latency_error = current.latency_ms - self.config.target_latency_ms;
        let power_error = current.power_mw - self.config.target_power_mw;

        // Combined error signal (latency is more critical)
        let combined_error = 0.7 * latency_error + 0.3 * power_error;

        // Adapt computation level
        let adaptation = -self.config.adaptation_rate * combined_error / self.config.target_latency_ms;
        
        self.computation_level += adaptation;
        self.computation_level = self.computation_level
            .max(self.config.min_computation_level)
            .min(self.config.max_computation_level);
    }

    /// Get adaptation recommendations
    pub fn get_recommendations(&self) -> AdaptationRecommendations {
        if self.performance_history.is_empty() {
            return AdaptationRecommendations::default();
        }

        let recent_metrics: Vec<_> = self.performance_history.iter().rev().take(5).collect();
        let avg_latency = recent_metrics.iter().map(|m| m.latency_ms).sum::<f32>() / recent_metrics.len() as f32;
        let avg_power = recent_metrics.iter().map(|m| m.power_mw).sum::<f32>() / recent_metrics.len() as f32;
        let avg_accuracy = recent_metrics.iter().map(|m| m.accuracy).sum::<f32>() / recent_metrics.len() as f32;

        let mut recommendations = Vec::new();

        if avg_latency > self.config.target_latency_ms * 1.2 {
            recommendations.push("Consider reducing model complexity".to_string());
        }
        if avg_power > self.config.target_power_mw * 1.2 {
            recommendations.push("Enable power-saving features".to_string());
        }
        if avg_accuracy < 0.8 {
            recommendations.push("Increase computation level for better accuracy".to_string());
        }
        if self.computation_level < 0.3 {
            recommendations.push("System running at very low computation level".to_string());
        }

        AdaptationRecommendations {
            current_level: self.computation_level,
            avg_latency_ms: avg_latency,
            avg_power_mw: avg_power,
            avg_accuracy,
            recommendations,
        }
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

/// Adaptation recommendations
#[derive(Debug, Clone)]
pub struct AdaptationRecommendations {
    /// Current computation level
    pub current_level: f32,
    /// Average latency
    pub avg_latency_ms: f32,
    /// Average power
    pub avg_power_mw: f32,
    /// Average accuracy
    pub avg_accuracy: f32,
    /// Recommendations
    pub recommendations: Vec<String>,
}

impl Default for AdaptationRecommendations {
    fn default() -> Self {
        Self {
            current_level: 0.5,
            avg_latency_ms: 0.0,
            avg_power_mw: 0.0,
            avg_accuracy: 0.0,
            recommendations: Vec::new(),
        }
    }
}

/// Performance optimizer combining all optimization techniques
#[derive(Debug)]
pub struct PerformanceOptimizer {
    /// Model cache
    cache: ModelCache,
    /// Memory pool
    memory_pool: MemoryPool,
    /// Batch processor
    batch_processor: BatchProcessor,
    /// Adaptive computation
    adaptive_computation: AdaptiveComputation,
    /// Optimization enabled
    enabled: bool,
}

impl PerformanceOptimizer {
    /// Create new performance optimizer
    pub fn new(
        cache_config: CacheConfig,
        pool_config: PoolConfig,
        batch_size: usize,
        adaptation_config: AdaptationConfig,
    ) -> Self {
        Self {
            cache: ModelCache::new(cache_config),
            memory_pool: MemoryPool::new(pool_config.clone()),
            batch_processor: BatchProcessor::new(batch_size, pool_config),
            adaptive_computation: AdaptiveComputation::new(adaptation_config),
            enabled: true,
        }
    }

    /// Enable/disable optimization
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Get mutable cache reference
    pub fn cache_mut(&mut self) -> &mut ModelCache {
        &mut self.cache
    }

    /// Get mutable memory pool reference
    pub fn memory_pool_mut(&mut self) -> &mut MemoryPool {
        &mut self.memory_pool
    }

    /// Get mutable batch processor reference
    pub fn batch_processor_mut(&mut self) -> &mut BatchProcessor {
        &mut self.batch_processor
    }

    /// Get mutable adaptive computation reference
    pub fn adaptive_computation_mut(&mut self) -> &mut AdaptiveComputation {
        &mut self.adaptive_computation
    }

    /// Update performance metrics
    pub fn update_performance(&mut self, latency_ms: f32, power_mw: f32, accuracy: f32) {
        if self.enabled {
            self.adaptive_computation.update_performance(latency_ms, power_mw, accuracy);
        }
    }

    /// Get comprehensive optimization statistics
    pub fn get_optimization_stats(&self) -> OptimizationStats {
        OptimizationStats {
            cache_stats: self.cache.get_stats(),
            pool_stats: self.memory_pool.get_stats().clone(),
            batch_stats: self.batch_processor.get_stats().clone(),
            adaptation_recommendations: self.adaptive_computation.get_recommendations(),
            optimization_enabled: self.enabled,
        }
    }

    /// Cleanup and maintenance
    pub fn maintenance(&mut self) {
        if self.enabled {
            self.cache.cleanup_all();
        }
    }
}

/// Comprehensive optimization statistics
#[derive(Debug, Clone)]
pub struct OptimizationStats {
    /// Cache statistics
    pub cache_stats: crate::cache::ModelCacheStats,
    /// Memory pool statistics
    pub pool_stats: PoolStats,
    /// Batch processing statistics
    pub batch_stats: BatchStats,
    /// Adaptation recommendations
    pub adaptation_recommendations: AdaptationRecommendations,
    /// Whether optimization is enabled
    pub optimization_enabled: bool,
}