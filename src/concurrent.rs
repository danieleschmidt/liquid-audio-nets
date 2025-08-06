//! Concurrent processing and resource pooling for Liquid Neural Networks
//!
//! Provides thread-safe concurrent processing, worker pools, and
//! resource management for high-throughput audio processing.

use crate::{Result, LiquidAudioError, ModelConfig, ProcessingResult};
use crate::optimization::{PerformanceOptimizer, MemoryPool, PoolConfig};
#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String, boxed::Box, collections::VecDeque};

#[cfg(feature = "std")]
use std::{vec::Vec, string::String, boxed::Box, collections::VecDeque};
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

#[cfg(feature = "std")]
use std::sync::{Arc, Mutex, Condvar};
#[cfg(feature = "std")]
use std::thread;

/// Thread-safe work item for processing
#[derive(Debug, Clone)]
pub struct WorkItem {
    /// Unique work ID
    pub id: u64,
    /// Audio data to process
    pub audio_data: Vec<f32>,
    /// Model configuration
    pub config: ModelConfig,
    /// Priority level (higher = more urgent)
    pub priority: u8,
    /// Creation timestamp
    pub created_at: u64,
    /// Optional metadata
    pub metadata: Option<String>,
}

/// Processing result with work ID
#[derive(Debug, Clone)]
pub struct WorkResult {
    /// Work item ID
    pub work_id: u64,
    /// Processing result
    pub result: Result<ProcessingResult>,
    /// Processing duration (ms)
    pub processing_time_ms: f32,
    /// Worker ID that processed this item
    pub worker_id: usize,
    /// Completion timestamp
    pub completed_at: u64,
}

/// Worker thread state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerState {
    /// Worker is idle
    Idle,
    /// Worker is processing
    Busy,
    /// Worker is paused
    Paused,
    /// Worker is shutting down
    Shutdown,
}

/// Worker statistics
#[derive(Debug, Clone)]
pub struct WorkerStats {
    /// Worker ID
    pub worker_id: usize,
    /// Current state
    pub state: WorkerState,
    /// Items processed
    pub items_processed: u64,
    /// Total processing time (ms)
    pub total_processing_time_ms: f32,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f32,
    /// Last activity timestamp
    pub last_activity: u64,
    /// Error count
    pub error_count: u64,
}

/// Thread pool configuration
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    /// Number of worker threads
    pub num_workers: usize,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Enable work stealing
    pub enable_work_stealing: bool,
    /// Worker timeout (ms)
    pub worker_timeout_ms: u64,
    /// Priority queue enabled
    pub priority_queue: bool,
    /// Enable worker statistics
    pub enable_stats: bool,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            num_workers: 4,
            max_queue_size: 1000,
            enable_work_stealing: true,
            worker_timeout_ms: 1000,
            priority_queue: true,
            enable_stats: true,
        }
    }
}

#[cfg(feature = "std")]
/// High-performance thread pool for concurrent audio processing
pub struct ThreadPool {
    /// Worker threads
    workers: Vec<thread::JoinHandle<()>>,
    /// Work queue
    work_queue: Arc<Mutex<WorkQueue>>,
    /// Result queue
    result_queue: Arc<Mutex<VecDeque<WorkResult>>>,
    /// Pool configuration
    config: ThreadPoolConfig,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// Worker statistics
    worker_stats: Arc<Mutex<Vec<WorkerStats>>>,
    /// Next work ID
    next_work_id: Arc<AtomicU64>,
    /// Pool statistics
    pool_stats: Arc<Mutex<PoolStats>>,
}

/// Work queue with priority support
#[derive(Debug)]
struct WorkQueue {
    /// High priority items
    high_priority: VecDeque<WorkItem>,
    /// Normal priority items
    normal_priority: VecDeque<WorkItem>,
    /// Low priority items
    low_priority: VecDeque<WorkItem>,
    /// Queue size limit
    max_size: usize,
    /// Current size
    current_size: usize,
}

impl WorkQueue {
    fn new(max_size: usize) -> Self {
        Self {
            high_priority: VecDeque::new(),
            normal_priority: VecDeque::new(),
            low_priority: VecDeque::new(),
            max_size,
            current_size: 0,
        }
    }

    fn push(&mut self, item: WorkItem) -> bool {
        if self.current_size >= self.max_size {
            return false;
        }

        match item.priority {
            200..=255 => self.high_priority.push_back(item),
            100..=199 => self.normal_priority.push_back(item),
            _ => self.low_priority.push_back(item),
        }

        self.current_size += 1;
        true
    }

    fn pop(&mut self) -> Option<WorkItem> {
        // Process high priority first
        if let Some(item) = self.high_priority.pop_front() {
            self.current_size -= 1;
            return Some(item);
        }

        // Then normal priority
        if let Some(item) = self.normal_priority.pop_front() {
            self.current_size -= 1;
            return Some(item);
        }

        // Finally low priority
        if let Some(item) = self.low_priority.pop_front() {
            self.current_size -= 1;
            return Some(item);
        }

        None
    }

    fn is_empty(&self) -> bool {
        self.current_size == 0
    }

    fn size(&self) -> usize {
        self.current_size
    }
}

/// Thread pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total items queued
    pub items_queued: u64,
    /// Total items processed
    pub items_processed: u64,
    /// Total items failed
    pub items_failed: u64,
    /// Current queue size
    pub current_queue_size: usize,
    /// Peak queue size
    pub peak_queue_size: usize,
    /// Total processing time (ms)
    pub total_processing_time_ms: f32,
    /// Average queue wait time (ms)
    pub avg_queue_wait_time_ms: f32,
    /// Pool uptime (ms)
    pub uptime_ms: u64,
}

impl Default for PoolStats {
    fn default() -> Self {
        Self {
            items_queued: 0,
            items_processed: 0,
            items_failed: 0,
            current_queue_size: 0,
            peak_queue_size: 0,
            total_processing_time_ms: 0.0,
            avg_queue_wait_time_ms: 0.0,
            uptime_ms: 0,
        }
    }
}

#[cfg(feature = "std")]
impl ThreadPool {
    /// Create new thread pool
    pub fn new(config: ThreadPoolConfig) -> Result<Self> {
        let work_queue = Arc::new(Mutex::new(WorkQueue::new(config.max_queue_size)));
        let result_queue = Arc::new(Mutex::new(VecDeque::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let worker_stats = Arc::new(Mutex::new(Vec::new()));
        let next_work_id = Arc::new(AtomicU64::new(1));
        let pool_stats = Arc::new(Mutex::new(PoolStats::default()));

        let mut workers = Vec::new();

        // Initialize worker statistics
        {
            let mut stats = worker_stats.lock().unwrap();
            for i in 0..config.num_workers {
                stats.push(WorkerStats {
                    worker_id: i,
                    state: WorkerState::Idle,
                    items_processed: 0,
                    total_processing_time_ms: 0.0,
                    avg_processing_time_ms: 0.0,
                    last_activity: Self::current_timestamp(),
                    error_count: 0,
                });
            }
        }

        // Spawn worker threads
        for worker_id in 0..config.num_workers {
            let work_queue_clone = Arc::clone(&work_queue);
            let result_queue_clone = Arc::clone(&result_queue);
            let shutdown_clone = Arc::clone(&shutdown);
            let worker_stats_clone = Arc::clone(&worker_stats);
            let pool_stats_clone = Arc::clone(&pool_stats);
            let timeout_ms = config.worker_timeout_ms;

            let handle = thread::spawn(move || {
                Self::worker_thread(
                    worker_id,
                    work_queue_clone,
                    result_queue_clone,
                    shutdown_clone,
                    worker_stats_clone,
                    pool_stats_clone,
                    timeout_ms,
                );
            });

            workers.push(handle);
        }

        Ok(Self {
            workers,
            work_queue,
            result_queue,
            config,
            shutdown,
            worker_stats,
            next_work_id,
            pool_stats,
        })
    }

    /// Submit work item for processing
    pub fn submit_work(
        &self,
        audio_data: Vec<f32>,
        config: ModelConfig,
        priority: u8,
        metadata: Option<String>,
    ) -> Result<u64> {
        if self.shutdown.load(Ordering::Relaxed) {
            return Err(LiquidAudioError::InvalidState("Thread pool is shutting down".to_string()));
        }

        let work_id = self.next_work_id.fetch_add(1, Ordering::Relaxed);
        let work_item = WorkItem {
            id: work_id,
            audio_data,
            config,
            priority,
            created_at: Self::current_timestamp(),
            metadata,
        };

        {
            let mut queue = self.work_queue.lock().unwrap();
            if !queue.push(work_item) {
                return Err(LiquidAudioError::ResourceExhausted("Work queue is full".to_string()));
            }
        }

        // Update statistics
        {
            let mut stats = self.pool_stats.lock().unwrap();
            stats.items_queued += 1;
            let current_size = {
                let queue = self.work_queue.lock().unwrap();
                queue.size()
            };
            stats.current_queue_size = current_size;
            stats.peak_queue_size = stats.peak_queue_size.max(current_size);
        }

        Ok(work_id)
    }

    /// Get completed results
    pub fn get_results(&self) -> Vec<WorkResult> {
        let mut results = self.result_queue.lock().unwrap();
        let mut completed = Vec::new();
        
        while let Some(result) = results.pop_front() {
            completed.push(result);
        }
        
        completed
    }

    /// Get worker statistics
    pub fn get_worker_stats(&self) -> Vec<WorkerStats> {
        self.worker_stats.lock().unwrap().clone()
    }

    /// Get pool statistics
    pub fn get_pool_stats(&self) -> PoolStats {
        self.pool_stats.lock().unwrap().clone()
    }

    /// Shutdown the thread pool
    pub fn shutdown(self) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);

        for handle in self.workers {
            handle.join().map_err(|_| LiquidAudioError::ThreadError("Failed to join worker thread".to_string()))?;
        }

        Ok(())
    }

    /// Worker thread function
    fn worker_thread(
        worker_id: usize,
        work_queue: Arc<Mutex<WorkQueue>>,
        result_queue: Arc<Mutex<VecDeque<WorkResult>>>,
        shutdown: Arc<AtomicBool>,
        worker_stats: Arc<Mutex<Vec<WorkerStats>>>,
        pool_stats: Arc<Mutex<PoolStats>>,
        timeout_ms: u64,
    ) {
        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }

            // Get work item
            let work_item = {
                let mut queue = work_queue.lock().unwrap();
                queue.pop()
            };

            match work_item {
                Some(item) => {
                    // Update worker state
                    {
                        let mut stats = worker_stats.lock().unwrap();
                        stats[worker_id].state = WorkerState::Busy;
                        stats[worker_id].last_activity = Self::current_timestamp();
                    }

                    // Process the work item
                    let start_time = Self::current_timestamp();
                    let result = Self::process_work_item(&item);
                    let end_time = Self::current_timestamp();
                    let processing_time_ms = (end_time - start_time) as f32;

                    // Update worker statistics
                    {
                        let mut stats = worker_stats.lock().unwrap();
                        let worker_stat = &mut stats[worker_id];
                        worker_stat.items_processed += 1;
                        worker_stat.total_processing_time_ms += processing_time_ms;
                        worker_stat.avg_processing_time_ms = 
                            worker_stat.total_processing_time_ms / worker_stat.items_processed as f32;
                        
                        if result.is_err() {
                            worker_stat.error_count += 1;
                        }
                        
                        worker_stat.state = WorkerState::Idle;
                    }

                    // Update pool statistics
                    {
                        let mut pool_stat = pool_stats.lock().unwrap();
                        if result.is_ok() {
                            pool_stat.items_processed += 1;
                        } else {
                            pool_stat.items_failed += 1;
                        }
                        pool_stat.total_processing_time_ms += processing_time_ms;
                        
                        // Update queue size
                        let current_size = {
                            let queue = work_queue.lock().unwrap();
                            queue.size()
                        };
                        pool_stat.current_queue_size = current_size;
                    }

                    // Store result
                    let work_result = WorkResult {
                        work_id: item.id,
                        result,
                        processing_time_ms,
                        worker_id,
                        completed_at: end_time,
                    };

                    {
                        let mut results = result_queue.lock().unwrap();
                        results.push_back(work_result);
                    }
                }
                None => {
                    // No work available, sleep briefly
                    thread::sleep(std::time::Duration::from_millis(timeout_ms / 10));
                }
            }
        }

        // Update worker state on shutdown
        {
            let mut stats = worker_stats.lock().unwrap();
            stats[worker_id].state = WorkerState::Shutdown;
        }
    }

    /// Process a work item (placeholder implementation)
    fn process_work_item(item: &WorkItem) -> Result<ProcessingResult> {
        // This is a placeholder - in a real implementation, this would
        // use the actual LNN processing logic
        if item.audio_data.is_empty() {
            return Err(LiquidAudioError::InvalidInput("Empty audio data".to_string()));
        }

        // Simulate processing
        Ok(ProcessingResult {
            output: vec![0.5, 0.3], // Mock output
            confidence: 0.8,
            timestep_ms: 10.0,
            power_mw: 0.8,
            complexity: 0.5,
            liquid_energy: 0.25,
            metadata: item.metadata.clone(),
        })
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

/// No-std compatible work scheduler for embedded systems
#[derive(Debug)]
pub struct EmbeddedScheduler {
    /// Work queue
    work_queue: VecDeque<WorkItem>,
    /// Maximum queue size
    max_queue_size: usize,
    /// Next work ID
    next_work_id: u64,
    /// Processing statistics
    stats: EmbeddedStats,
    /// Scheduler enabled
    enabled: bool,
}

/// Embedded scheduler statistics
#[derive(Debug, Clone, Default)]
pub struct EmbeddedStats {
    /// Items processed
    pub items_processed: u64,
    /// Items dropped (queue full)
    pub items_dropped: u64,
    /// Total processing time (ms)
    pub total_processing_time_ms: f32,
    /// Average processing time (ms)
    pub avg_processing_time_ms: f32,
    /// Queue utilization (0.0 to 1.0)
    pub queue_utilization: f32,
}

impl EmbeddedScheduler {
    /// Create new embedded scheduler
    pub fn new(max_queue_size: usize) -> Self {
        Self {
            work_queue: VecDeque::with_capacity(max_queue_size),
            max_queue_size,
            next_work_id: 1,
            stats: EmbeddedStats::default(),
            enabled: true,
        }
    }

    /// Submit work for processing
    pub fn submit_work(
        &mut self,
        audio_data: Vec<f32>,
        config: ModelConfig,
        priority: u8,
    ) -> Result<u64> {
        if !self.enabled {
            return Err(LiquidAudioError::InvalidState("Scheduler disabled".to_string()));
        }

        if self.work_queue.len() >= self.max_queue_size {
            self.stats.items_dropped += 1;
            return Err(LiquidAudioError::ResourceExhausted("Queue full".to_string()));
        }

        let work_id = self.next_work_id;
        self.next_work_id = self.next_work_id.wrapping_add(1);

        let work_item = WorkItem {
            id: work_id,
            audio_data,
            config,
            priority,
            created_at: Self::current_timestamp(),
            metadata: None,
        };

        // Insert based on priority
        let mut insert_index = None;
        for (i, item) in self.work_queue.iter().enumerate() {
            if work_item.priority > item.priority {
                insert_index = Some(i);
                break;
            }
        }

        if let Some(i) = insert_index {
            self.work_queue.insert(i, work_item);
        } else {
            self.work_queue.push_back(work_item);
        }

        // Update utilization
        self.stats.queue_utilization = self.work_queue.len() as f32 / self.max_queue_size as f32;

        Ok(work_id)
    }

    /// Process next work item
    pub fn process_next<F>(&mut self, processor: F) -> Option<WorkResult>
    where
        F: FnOnce(&WorkItem) -> Result<ProcessingResult>,
    {
        if let Some(item) = self.work_queue.pop_front() {
            let start_time = Self::current_timestamp();
            let result = processor(&item);
            let end_time = Self::current_timestamp();
            let processing_time_ms = (end_time - start_time) as f32;

            // Update statistics
            self.stats.items_processed += 1;
            self.stats.total_processing_time_ms += processing_time_ms;
            self.stats.avg_processing_time_ms = 
                self.stats.total_processing_time_ms / self.stats.items_processed as f32;
            self.stats.queue_utilization = self.work_queue.len() as f32 / self.max_queue_size as f32;

            Some(WorkResult {
                work_id: item.id,
                result,
                processing_time_ms,
                worker_id: 0, // Single worker in embedded
                completed_at: end_time,
            })
        } else {
            None
        }
    }

    /// Get queue size
    pub fn queue_size(&self) -> usize {
        self.work_queue.len()
    }

    /// Get statistics
    pub fn get_stats(&self) -> &EmbeddedStats {
        &self.stats
    }

    /// Enable/disable scheduler
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Clear all work items
    pub fn clear(&mut self) {
        self.work_queue.clear();
        self.stats.queue_utilization = 0.0;
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

/// Resource pool for managing shared resources
pub struct ResourcePool<T> {
    /// Available resources
    resources: Vec<T>,
    /// Resource factory
    factory: Box<dyn Fn() -> T>,
    /// Maximum pool size
    max_size: usize,
    /// Current allocations
    allocations: AtomicUsize,
    /// Pool statistics
    stats: ResourcePoolStats,
}

/// Resource pool statistics
#[derive(Debug, Clone, Default)]
pub struct ResourcePoolStats {
    /// Total allocations
    pub total_allocations: u64,
    /// Total returns
    pub total_returns: u64,
    /// Current active allocations
    pub active_allocations: usize,
    /// Peak allocations
    pub peak_allocations: usize,
    /// Pool size
    pub pool_size: usize,
}

impl<T> ResourcePool<T> {
    /// Create new resource pool
    pub fn new<F>(factory: F, initial_size: usize, max_size: usize) -> Self 
    where
        F: Fn() -> T + 'static,
    {
        let mut resources = Vec::with_capacity(max_size);
        
        // Pre-allocate resources
        for _ in 0..initial_size {
            resources.push(factory());
        }

        Self {
            resources,
            factory: Box::new(factory),
            max_size,
            allocations: AtomicUsize::new(0),
            stats: ResourcePoolStats {
                pool_size: initial_size,
                ..Default::default()
            },
        }
    }

    /// Acquire resource from pool
    pub fn acquire(&mut self) -> T {
        self.stats.total_allocations += 1;
        
        let active = self.allocations.fetch_add(1, Ordering::Relaxed) + 1;
        self.stats.active_allocations = active;
        self.stats.peak_allocations = self.stats.peak_allocations.max(active);
        
        if let Some(resource) = self.resources.pop() {
            self.stats.pool_size -= 1;
            resource
        } else {
            // Create new resource
            (self.factory)()
        }
    }

    /// Return resource to pool
    pub fn return_resource(&mut self, resource: T) {
        self.stats.total_returns += 1;
        self.allocations.fetch_sub(1, Ordering::Relaxed);
        self.stats.active_allocations = self.allocations.load(Ordering::Relaxed);

        if self.resources.len() < self.max_size {
            self.resources.push(resource);
            self.stats.pool_size += 1;
        }
        // Otherwise drop the resource
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> &ResourcePoolStats {
        &self.stats
    }
}