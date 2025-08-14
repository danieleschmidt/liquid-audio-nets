//! High-performance caching system for Liquid Neural Networks
//!
//! Provides intelligent caching of model weights, intermediate computations,
//! and frequently accessed data to optimize performance and reduce latency.

use crate::{ModelConfig, ProcessingResult};

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, collections::BTreeMap, string::String, boxed::Box};

#[cfg(feature = "std")]
use std::{vec::Vec, collections::BTreeMap};
use core::hash::Hash;

/// Cache entry with metadata
#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    /// Cached data
    pub data: T,
    /// Access count for LRU
    pub access_count: u64,
    /// Last access timestamp
    pub last_access: u64,
    /// Entry size in bytes (estimated)
    pub size_bytes: usize,
    /// Time to live (0 = permanent)
    pub ttl_ms: u64,
    /// Creation timestamp
    pub created_at: u64,
}

/// High-performance LRU cache with size limits
#[derive(Debug)]
pub struct LRUCache<K, V> 
where
    K: Clone + Ord + Hash,
    V: Clone,
{
    /// Storage map
    entries: BTreeMap<K, CacheEntry<V>>,
    /// Maximum number of entries
    max_entries: usize,
    /// Maximum total size in bytes
    max_size_bytes: usize,
    /// Current total size
    current_size_bytes: usize,
    /// Global access counter
    access_counter: u64,
    /// Cache hit count
    hit_count: u64,
    /// Cache miss count
    miss_count: u64,
}

impl<K, V> LRUCache<K, V>
where
    K: Clone + Ord + Hash,
    V: Clone,
{
    /// Create new LRU cache
    pub fn new(max_entries: usize, max_size_bytes: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            max_entries,
            max_size_bytes,
            current_size_bytes: 0,
            access_counter: 0,
            hit_count: 0,
            miss_count: 0,
        }
    }

    /// Insert item into cache
    pub fn insert(&mut self, key: K, value: V, size_bytes: usize, ttl_ms: u64) {
        let timestamp = Self::current_timestamp();
        
        // Remove existing entry if present
        if let Some(old_entry) = self.entries.remove(&key) {
            self.current_size_bytes = self.current_size_bytes.saturating_sub(old_entry.size_bytes);
        }

        // Ensure we have space
        while (self.entries.len() >= self.max_entries) || 
              (self.current_size_bytes + size_bytes > self.max_size_bytes) {
            self.evict_lru();
        }

        let entry = CacheEntry {
            data: value,
            access_count: 1,
            last_access: timestamp,
            size_bytes,
            ttl_ms,
            created_at: timestamp,
        };

        self.entries.insert(key, entry);
        self.current_size_bytes += size_bytes;
    }

    /// Get item from cache
    pub fn get(&mut self, key: &K) -> Option<V> {
        let timestamp = Self::current_timestamp();
        
        if let Some(entry) = self.entries.get_mut(key) {
            // Check TTL
            if entry.ttl_ms > 0 && (timestamp - entry.created_at) > entry.ttl_ms {
                let expired_entry = self.entries.remove(key).unwrap();
                self.current_size_bytes = self.current_size_bytes.saturating_sub(expired_entry.size_bytes);
                self.miss_count += 1;
                return None;
            }

            // Update access info
            self.access_counter += 1;
            entry.access_count += 1;
            entry.last_access = timestamp;
            
            self.hit_count += 1;
            Some(entry.data.clone())
        } else {
            self.miss_count += 1;
            None
        }
    }

    /// Remove item from cache
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(entry) = self.entries.remove(key) {
            self.current_size_bytes = self.current_size_bytes.saturating_sub(entry.size_bytes);
            Some(entry.data)
        } else {
            None
        }
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
        self.current_size_bytes = 0;
        self.hit_count = 0;
        self.miss_count = 0;
    }

    /// Get cache statistics
    pub fn stats(&self) -> CacheStats {
        let hit_rate = if (self.hit_count + self.miss_count) > 0 {
            self.hit_count as f32 / (self.hit_count + self.miss_count) as f32
        } else {
            0.0
        };

        CacheStats {
            entries: self.entries.len(),
            max_entries: self.max_entries,
            size_bytes: self.current_size_bytes,
            max_size_bytes: self.max_size_bytes,
            hit_count: self.hit_count,
            miss_count: self.miss_count,
            hit_rate,
        }
    }

    /// Evict least recently used entry
    fn evict_lru(&mut self) {
        if self.entries.is_empty() {
            return;
        }

        // Find LRU entry
        let mut lru_key = None;
        let mut oldest_access = u64::MAX;

        for (key, entry) in &self.entries {
            if entry.last_access < oldest_access {
                oldest_access = entry.last_access;
                lru_key = Some(key.clone());
            }
        }

        if let Some(key) = lru_key {
            let removed_entry = self.entries.remove(&key).unwrap();
            self.current_size_bytes = self.current_size_bytes.saturating_sub(removed_entry.size_bytes);
        }
    }

    /// Clean expired entries
    pub fn cleanup_expired(&mut self) {
        let timestamp = Self::current_timestamp();
        let mut expired_keys = Vec::new();

        for (key, entry) in &self.entries {
            if entry.ttl_ms > 0 && (timestamp - entry.created_at) > entry.ttl_ms {
                expired_keys.push(key.clone());
            }
        }

        for key in expired_keys {
            if let Some(entry) = self.entries.remove(&key) {
                self.current_size_bytes = self.current_size_bytes.saturating_sub(entry.size_bytes);
            }
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

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Current number of entries
    pub entries: usize,
    /// Maximum allowed entries
    pub max_entries: usize,
    /// Current size in bytes
    pub size_bytes: usize,
    /// Maximum allowed size in bytes
    pub max_size_bytes: usize,
    /// Number of cache hits
    pub hit_count: u64,
    /// Number of cache misses
    pub miss_count: u64,
    /// Hit rate (0.0 to 1.0)
    pub hit_rate: f32,
}

/// Model cache key for efficient lookups
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelCacheKey {
    /// Model configuration hash
    pub config_hash: u64,
    /// Input hash for memoization
    pub input_hash: u64,
    /// Cache type identifier
    pub cache_type: CacheType,
}

/// Types of cached data
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CacheType {
    /// Preprocessed audio features
    Features,
    /// Model weights
    Weights,
    /// Intermediate computations
    Intermediate,
    /// Final predictions
    Predictions,
    /// Optimization metadata
    Metadata,
}

/// Specialized cache for model operations
#[derive(Debug)]
pub struct ModelCache {
    /// Feature cache
    feature_cache: LRUCache<ModelCacheKey, Vec<f32>>,
    /// Weight cache
    weight_cache: LRUCache<ModelCacheKey, Vec<f32>>,
    /// Prediction cache
    prediction_cache: LRUCache<ModelCacheKey, ProcessingResult>,
    /// Cache configuration
    config: CacheConfig,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum feature cache entries
    pub max_feature_entries: usize,
    /// Maximum weight cache entries
    pub max_weight_entries: usize,
    /// Maximum prediction cache entries
    pub max_prediction_entries: usize,
    /// Feature cache size limit (bytes)
    pub feature_cache_size_bytes: usize,
    /// Weight cache size limit (bytes)
    pub weight_cache_size_bytes: usize,
    /// Prediction cache size limit (bytes)
    pub prediction_cache_size_bytes: usize,
    /// Default TTL for cached items (ms)
    pub default_ttl_ms: u64,
    /// Enable aggressive caching
    pub aggressive_caching: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_feature_entries: 100,
            max_weight_entries: 10,
            max_prediction_entries: 50,
            feature_cache_size_bytes: 1024 * 1024,     // 1MB
            weight_cache_size_bytes: 4 * 1024 * 1024,  // 4MB
            prediction_cache_size_bytes: 512 * 1024,   // 512KB
            default_ttl_ms: 60000,                     // 1 minute
            aggressive_caching: false,
        }
    }
}

impl ModelCache {
    /// Create new model cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            feature_cache: LRUCache::new(config.max_feature_entries, config.feature_cache_size_bytes),
            weight_cache: LRUCache::new(config.max_weight_entries, config.weight_cache_size_bytes),
            prediction_cache: LRUCache::new(config.max_prediction_entries, config.prediction_cache_size_bytes),
            config,
        }
    }

    /// Cache preprocessed features
    pub fn cache_features(&mut self, key: ModelCacheKey, features: Vec<f32>) {
        let size = features.len() * 4; // f32 size
        self.feature_cache.insert(key, features, size, self.config.default_ttl_ms);
    }

    /// Get cached features
    pub fn get_features(&mut self, key: &ModelCacheKey) -> Option<Vec<f32>> {
        self.feature_cache.get(key)
    }

    /// Cache model weights
    pub fn cache_weights(&mut self, key: ModelCacheKey, weights: Vec<f32>) {
        let size = weights.len() * 4; // f32 size
        self.weight_cache.insert(key, weights, size, 0); // Weights don't expire
    }

    /// Get cached weights
    pub fn get_weights(&mut self, key: &ModelCacheKey) -> Option<Vec<f32>> {
        self.weight_cache.get(key)
    }

    /// Cache prediction result
    pub fn cache_prediction(&mut self, key: ModelCacheKey, result: ProcessingResult) {
        let size = result.output.len() * 4 + 64; // Estimate size
        let ttl = if self.config.aggressive_caching {
            self.config.default_ttl_ms * 2
        } else {
            self.config.default_ttl_ms / 2
        };
        self.prediction_cache.insert(key, result, size, ttl);
    }

    /// Get cached prediction
    pub fn get_prediction(&mut self, key: &ModelCacheKey) -> Option<ProcessingResult> {
        self.prediction_cache.get(key)
    }

    /// Get comprehensive cache statistics
    pub fn get_stats(&self) -> ModelCacheStats {
        ModelCacheStats {
            feature_cache: self.feature_cache.stats(),
            weight_cache: self.weight_cache.stats(),
            prediction_cache: self.prediction_cache.stats(),
        }
    }

    /// Cleanup all expired entries
    pub fn cleanup_all(&mut self) {
        self.feature_cache.cleanup_expired();
        self.weight_cache.cleanup_expired();
        self.prediction_cache.cleanup_expired();
    }

    /// Clear all caches
    pub fn clear_all(&mut self) {
        self.feature_cache.clear();
        self.weight_cache.clear();
        self.prediction_cache.clear();
    }

    /// Generate cache key for model configuration
    pub fn generate_config_key(config: &ModelConfig, cache_type: CacheType) -> ModelCacheKey {
        // Simple hash of configuration
        let config_hash = Self::hash_config(config);
        
        ModelCacheKey {
            config_hash,
            input_hash: 0, // Will be set for input-specific keys
            cache_type,
        }
    }

    /// Generate cache key for input data
    pub fn generate_input_key(&self, config: &ModelConfig, input: &[f32], cache_type: CacheType) -> ModelCacheKey {
        let config_hash = Self::hash_config(config);
        let input_hash = Self::hash_input(input);
        
        ModelCacheKey {
            config_hash,
            input_hash,
            cache_type,
        }
    }

    /// Simple hash function for model config
    fn hash_config(config: &ModelConfig) -> u64 {
        // Simple hash based on key parameters
        let mut hash = 0u64;
        hash = hash.wrapping_mul(31).wrapping_add(config.input_dim as u64);
        hash = hash.wrapping_mul(31).wrapping_add(config.hidden_dim as u64);
        hash = hash.wrapping_mul(31).wrapping_add(config.output_dim as u64);
        hash = hash.wrapping_mul(31).wrapping_add(config.sample_rate as u64);
        hash = hash.wrapping_mul(31).wrapping_add(config.frame_size as u64);
        hash
    }

    /// Simple hash function for input data
    fn hash_input(input: &[f32]) -> u64 {
        let mut hash = 0u64;
        for (i, &value) in input.iter().enumerate().take(16) { // Hash first 16 values
            let bits = value.to_bits() as u64;
            hash = hash.wrapping_mul(31).wrapping_add(bits).wrapping_add(i as u64);
        }
        hash = hash.wrapping_mul(31).wrapping_add(input.len() as u64);
        hash
    }
}

/// Combined cache statistics
#[derive(Debug, Clone)]
pub struct ModelCacheStats {
    /// Feature cache stats
    pub feature_cache: CacheStats,
    /// Weight cache stats
    pub weight_cache: CacheStats,
    /// Prediction cache stats
    pub prediction_cache: CacheStats,
}

impl ModelCacheStats {
    /// Get overall hit rate
    pub fn overall_hit_rate(&self) -> f32 {
        let total_hits = self.feature_cache.hit_count + 
                        self.weight_cache.hit_count + 
                        self.prediction_cache.hit_count;
        let total_requests = total_hits + 
                            self.feature_cache.miss_count + 
                            self.weight_cache.miss_count + 
                            self.prediction_cache.miss_count;
        
        if total_requests > 0 {
            total_hits as f32 / total_requests as f32
        } else {
            0.0
        }
    }

    /// Get total memory usage
    pub fn total_memory_bytes(&self) -> usize {
        self.feature_cache.size_bytes + 
        self.weight_cache.size_bytes + 
        self.prediction_cache.size_bytes
    }
}

/// Cache performance monitor
#[derive(Debug)]
pub struct CacheMonitor {
    /// Performance history
    hit_rate_history: Vec<f32>,
    /// Memory usage history
    memory_history: Vec<usize>,
    /// Monitoring enabled
    enabled: bool,
}

impl CacheMonitor {
    /// Create new cache monitor
    pub fn new(enabled: bool) -> Self {
        Self {
            hit_rate_history: Vec::new(),
            memory_history: Vec::new(),
            enabled,
        }
    }

    /// Record cache performance
    pub fn record_performance(&mut self, stats: &ModelCacheStats) {
        if !self.enabled {
            return;
        }

        self.hit_rate_history.push(stats.overall_hit_rate());
        self.memory_history.push(stats.total_memory_bytes());

        // Keep rolling window
        if self.hit_rate_history.len() > 100 {
            self.hit_rate_history.remove(0);
        }
        if self.memory_history.len() > 100 {
            self.memory_history.remove(0);
        }
    }

    /// Get performance analysis
    pub fn get_analysis(&self) -> CacheAnalysis {
        if self.hit_rate_history.is_empty() {
            return CacheAnalysis::default();
        }

        let avg_hit_rate = self.hit_rate_history.iter().sum::<f32>() / self.hit_rate_history.len() as f32;
        let min_hit_rate = self.hit_rate_history.iter().fold(1.0f32, |a, &b| a.min(b));
        let max_hit_rate = self.hit_rate_history.iter().fold(0.0f32, |a, &b| a.max(b));

        let avg_memory = if !self.memory_history.is_empty() {
            self.memory_history.iter().sum::<usize>() / self.memory_history.len()
        } else {
            0
        };

        CacheAnalysis {
            avg_hit_rate,
            min_hit_rate,
            max_hit_rate,
            avg_memory_bytes: avg_memory,
            samples: self.hit_rate_history.len(),
        }
    }
}

/// Cache performance analysis
#[derive(Debug, Clone)]
pub struct CacheAnalysis {
    /// Average hit rate
    pub avg_hit_rate: f32,
    /// Minimum hit rate
    pub min_hit_rate: f32,
    /// Maximum hit rate
    pub max_hit_rate: f32,
    /// Average memory usage
    pub avg_memory_bytes: usize,
    /// Number of samples
    pub samples: usize,
}

impl Default for CacheAnalysis {
    fn default() -> Self {
        Self {
            avg_hit_rate: 0.0,
            min_hit_rate: 0.0,
            max_hit_rate: 0.0,
            avg_memory_bytes: 0,
            samples: 0,
        }
    }
}