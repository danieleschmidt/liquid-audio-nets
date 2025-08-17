//! Concurrent processing and resource pooling for Liquid Neural Networks
//!
//! Provides thread-safe concurrent processing, worker pools, and
//! resource management for high-throughput audio processing.

use crate::{Result, LiquidAudioError, ModelConfig, ProcessingResult};
#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String, boxed::Box, collections::VecDeque};

#[cfg(feature = "std")]
use std::{vec::Vec, string::String, boxed::Box, collections::VecDeque};
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

#[cfg(feature = "std")]
use std::sync::{Arc, Mutex};
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
    #[allow(dead_code)]
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

    #[allow(dead_code)]
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

/// Advanced work stealing for high-performance concurrent processing
#[derive(Debug)]
pub struct WorkStealingScheduler {
    /// Worker queues for work stealing
    worker_queues: Vec<VecDeque<WorkItem>>,
    /// Number of workers
    num_workers: usize,
    /// Steal attempts counter
    steal_attempts: AtomicU64,
    /// Successful steals counter
    successful_steals: AtomicU64,
    /// Random seed for stealing strategy
    random_seed: AtomicU64,
}

impl WorkStealingScheduler {
    /// Create new work stealing scheduler
    pub fn new(num_workers: usize) -> Self {
        let mut worker_queues = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            worker_queues.push(VecDeque::new());
        }
        
        Self {
            worker_queues,
            num_workers,
            steal_attempts: AtomicU64::new(0),
            successful_steals: AtomicU64::new(0),
            random_seed: AtomicU64::new(42),
        }
    }
    
    /// Submit work to least loaded worker
    pub fn submit_work(&mut self, item: WorkItem) -> Result<()> {
        let mut min_queue_size = usize::MAX;
        let mut target_worker = 0;
        
        // Find least loaded worker
        for (i, queue) in self.worker_queues.iter().enumerate() {
            if queue.len() < min_queue_size {
                min_queue_size = queue.len();
                target_worker = i;
            }
        }
        
        self.worker_queues[target_worker].push_back(item);
        Ok(())
    }
    
    /// Try to steal work from another worker
    pub fn try_steal_work(&mut self, worker_id: usize) -> Option<WorkItem> {
        if worker_id >= self.num_workers {
            return None;
        }
        
        self.steal_attempts.fetch_add(1, Ordering::Relaxed);
        
        // Try to find a worker with work to steal
        let seed = self.random_seed.fetch_add(1, Ordering::Relaxed);
        let start_worker = (seed as usize) % self.num_workers;
        
        for i in 0..self.num_workers {
            let target_worker = (start_worker + i) % self.num_workers;
            if target_worker != worker_id && !self.worker_queues[target_worker].is_empty() {
                if let Some(item) = self.worker_queues[target_worker].pop_back() {
                    self.successful_steals.fetch_add(1, Ordering::Relaxed);
                    return Some(item);
                }
            }
        }
        
        None
    }
    
    /// Get work from worker's own queue
    pub fn get_local_work(&mut self, worker_id: usize) -> Option<WorkItem> {
        if worker_id >= self.num_workers {
            return None;
        }
        
        self.worker_queues[worker_id].pop_front()
    }
    
    /// Get work stealing statistics
    pub fn get_steal_stats(&self) -> WorkStealingStats {
        let attempts = self.steal_attempts.load(Ordering::Relaxed);
        let successes = self.successful_steals.load(Ordering::Relaxed);
        
        WorkStealingStats {
            steal_attempts: attempts,
            successful_steals: successes,
            steal_success_rate: if attempts > 0 {
                successes as f64 / attempts as f64
            } else {
                0.0
            },
            queue_sizes: self.worker_queues.iter().map(|q| q.len()).collect(),
        }
    }
    
    /// Get total pending work across all queues
    pub fn total_pending_work(&self) -> usize {
        self.worker_queues.iter().map(|q| q.len()).sum()
    }
}

/// Work stealing statistics
#[derive(Debug, Clone)]
pub struct WorkStealingStats {
    pub steal_attempts: u64,
    pub successful_steals: u64,
    pub steal_success_rate: f64,
    pub queue_sizes: Vec<usize>,
}

/// NUMA-aware processing for high-performance systems
#[derive(Debug)]
pub struct NumaAwareProcessor {
    /// NUMA node configurations
    numa_nodes: Vec<NumaNode>,
    /// Worker thread assignments
    worker_assignments: Vec<usize>, // worker_id -> numa_node
    /// Memory affinity settings
    #[allow(dead_code)]
    memory_affinity: bool,
    /// Processing statistics per NUMA node
    numa_stats: Vec<NumaStats>,
}

/// NUMA node configuration
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// Node ID
    pub node_id: usize,
    /// CPU cores on this node
    pub cpu_cores: Vec<usize>,
    /// Memory size (bytes)
    pub memory_size: usize,
    /// Node latency characteristics
    pub latency_profile: LatencyProfile,
}

/// Latency characteristics of NUMA node
#[derive(Debug, Clone)]
pub struct LatencyProfile {
    /// Local memory access latency (nanoseconds)
    pub local_latency_ns: u32,
    /// Remote memory access latency (nanoseconds)
    pub remote_latency_ns: u32,
    /// Bandwidth (GB/s)
    pub bandwidth_gbps: f32,
}

/// NUMA node statistics
#[derive(Debug, Clone, Default)]
pub struct NumaStats {
    /// Total processing time on this node
    pub total_processing_time_ms: f64,
    /// Number of tasks processed
    pub tasks_processed: u64,
    /// Average task completion time
    pub avg_completion_time_ms: f64,
    /// Memory utilization
    pub memory_utilization: f32,
    /// CPU utilization per core
    pub cpu_utilization: Vec<f32>,
}

impl NumaAwareProcessor {
    /// Create NUMA-aware processor
    pub fn new(numa_nodes: Vec<NumaNode>) -> Self {
        let numa_stats = vec![NumaStats::default(); numa_nodes.len()];
        
        Self {
            numa_nodes,
            worker_assignments: Vec::new(),
            memory_affinity: true,
            numa_stats,
        }
    }
    
    /// Assign worker to optimal NUMA node
    pub fn assign_worker(&mut self, worker_id: usize) -> usize {
        // Simple round-robin assignment for now
        // In production, this would consider current load
        let numa_node = worker_id % self.numa_nodes.len();
        
        if worker_id >= self.worker_assignments.len() {
            self.worker_assignments.resize(worker_id + 1, 0);
        }
        
        self.worker_assignments[worker_id] = numa_node;
        numa_node
    }
    
    /// Get optimal NUMA node for work item
    pub fn get_optimal_node(&self, _work_item: &WorkItem) -> usize {
        // Find least loaded NUMA node
        let mut min_load = f64::MAX;
        let mut optimal_node = 0;
        
        for (i, stats) in self.numa_stats.iter().enumerate() {
            let current_load = stats.avg_completion_time_ms * stats.tasks_processed as f64;
            if current_load < min_load {
                min_load = current_load;
                optimal_node = i;
            }
        }
        
        optimal_node
    }
    
    /// Update NUMA statistics
    pub fn update_stats(&mut self, numa_node: usize, processing_time_ms: f64) {
        if numa_node < self.numa_stats.len() {
            let stats = &mut self.numa_stats[numa_node];
            stats.total_processing_time_ms += processing_time_ms;
            stats.tasks_processed += 1;
            stats.avg_completion_time_ms = 
                stats.total_processing_time_ms / stats.tasks_processed as f64;
        }
    }
    
    /// Get NUMA statistics
    pub fn get_numa_stats(&self) -> &[NumaStats] {
        &self.numa_stats
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

/// Advanced batch processing for high-throughput scenarios
#[derive(Debug)]
pub struct BatchProcessor {
    /// Batch size configuration
    batch_size: usize,
    /// Current batch buffer
    current_batch: Vec<WorkItem>,
    /// Batch timeout (microseconds)
    batch_timeout_us: u64,
    /// Last batch creation time
    last_batch_time: u64,
    /// Batch processing statistics
    batch_stats: BatchStats,
}

/// Batch processing statistics
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total batches processed
    pub total_batches: u64,
    /// Total items processed
    pub total_items: u64,
    /// Average batch size
    pub avg_batch_size: f32,
    /// Average batch processing time
    pub avg_batch_time_ms: f64,
    /// Batch efficiency (items/ms)
    pub batch_efficiency: f64,
    /// Timeout-triggered batches
    pub timeout_batches: u64,
}

impl BatchProcessor {
    /// Create new batch processor
    pub fn new(batch_size: usize, batch_timeout_us: u64) -> Self {
        Self {
            batch_size,
            current_batch: Vec::with_capacity(batch_size),
            batch_timeout_us,
            last_batch_time: Self::current_timestamp(),
            batch_stats: BatchStats::default(),
        }
    }
    
    /// Add item to current batch
    pub fn add_item(&mut self, item: WorkItem) -> Option<Vec<WorkItem>> {
        self.current_batch.push(item);
        
        // Check if batch is ready
        if self.current_batch.len() >= self.batch_size {
            self.finalize_batch(false)
        } else {
            None
        }
    }
    
    /// Check if batch should be flushed due to timeout
    pub fn check_timeout(&mut self) -> Option<Vec<WorkItem>> {
        let current_time = Self::current_timestamp();
        if !self.current_batch.is_empty() && 
           (current_time - self.last_batch_time) > self.batch_timeout_us {
            self.finalize_batch(true)
        } else {
            None
        }
    }
    
    /// Force flush current batch
    pub fn flush(&mut self) -> Option<Vec<WorkItem>> {
        if !self.current_batch.is_empty() {
            self.finalize_batch(false)
        } else {
            None
        }
    }
    
    /// Finalize current batch
    fn finalize_batch(&mut self, timeout_triggered: bool) -> Option<Vec<WorkItem>> {
        if self.current_batch.is_empty() {
            return None;
        }
        
        let batch = core::mem::replace(&mut self.current_batch, Vec::with_capacity(self.batch_size));
        let batch_size = batch.len();
        
        // Update statistics
        self.batch_stats.total_batches += 1;
        self.batch_stats.total_items += batch_size as u64;
        self.batch_stats.avg_batch_size = 
            self.batch_stats.total_items as f32 / self.batch_stats.total_batches as f32;
        
        if timeout_triggered {
            self.batch_stats.timeout_batches += 1;
        }
        
        self.last_batch_time = Self::current_timestamp();
        
        Some(batch)
    }
    
    /// Get batch statistics
    pub fn get_stats(&self) -> &BatchStats {
        &self.batch_stats
    }
    
    /// Update batch processing time
    pub fn record_batch_time(&mut self, processing_time_ms: f64) {
        let alpha = 0.1;
        self.batch_stats.avg_batch_time_ms = 
            self.batch_stats.avg_batch_time_ms * (1.0 - alpha) + processing_time_ms * alpha;
        
        // Calculate efficiency
        if self.batch_stats.avg_batch_time_ms > 0.0 {
            self.batch_stats.batch_efficiency = 
                self.batch_stats.avg_batch_size as f64 / self.batch_stats.avg_batch_time_ms;
        }
    }
    
    fn current_timestamp() -> u64 {
        static mut TIMESTAMP: u64 = 0;
        unsafe {
            TIMESTAMP += 1000; // Increment by 1ms
            TIMESTAMP
        }
    }
}

/// Adaptive load balancer for dynamic workload distribution
#[derive(Debug)]
pub struct AdaptiveLoadBalancer {
    /// Worker load tracking
    worker_loads: Vec<WorkerLoad>,
    /// Load balancing algorithm
    algorithm: LoadBalancingAlgorithm,
    /// Performance history
    performance_history: Vec<PerformanceSnapshot>,
    /// Adaptation parameters
    adaptation_config: AdaptationConfig,
}

/// Worker load information
#[derive(Debug, Clone)]
pub struct WorkerLoad {
    /// Worker ID
    pub worker_id: usize,
    /// Current queue size
    pub queue_size: usize,
    /// Recent processing time (ms)
    pub recent_processing_time_ms: f64,
    /// Success rate
    pub success_rate: f64,
    /// CPU utilization estimate
    pub cpu_utilization: f32,
    /// Load score (higher = more loaded)
    pub load_score: f64,
}

/// Load balancing algorithms
#[derive(Debug, Clone, Copy)]
pub enum LoadBalancingAlgorithm {
    /// Round robin distribution
    RoundRobin,
    /// Least loaded worker
    LeastLoaded,
    /// Weighted round robin
    WeightedRoundRobin,
    /// Consistent hashing
    ConsistentHashing,
    /// Machine learning based
    MLBased,
}

/// Performance snapshot for adaptation
#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    /// Timestamp
    pub timestamp: u64,
    /// Overall throughput
    pub throughput: f64,
    /// Average latency
    pub avg_latency_ms: f64,
    /// Load variance across workers
    pub load_variance: f64,
    /// Algorithm used
    pub algorithm: LoadBalancingAlgorithm,
}

/// Adaptation configuration
#[derive(Debug, Clone)]
pub struct AdaptationConfig {
    /// Enable adaptive algorithm switching
    pub adaptive_switching: bool,
    /// Performance evaluation window
    pub evaluation_window_size: usize,
    /// Minimum performance improvement threshold
    pub improvement_threshold: f64,
    /// Algorithm switching cooldown
    pub switching_cooldown_ms: u64,
}

impl AdaptiveLoadBalancer {
    /// Create new adaptive load balancer
    pub fn new(num_workers: usize, algorithm: LoadBalancingAlgorithm) -> Self {
        let worker_loads = (0..num_workers).map(|id| WorkerLoad {
            worker_id: id,
            queue_size: 0,
            recent_processing_time_ms: 0.0,
            success_rate: 1.0,
            cpu_utilization: 0.0,
            load_score: 0.0,
        }).collect();
        
        Self {
            worker_loads,
            algorithm,
            performance_history: Vec::with_capacity(1000),
            adaptation_config: AdaptationConfig {
                adaptive_switching: true,
                evaluation_window_size: 100,
                improvement_threshold: 0.05,
                switching_cooldown_ms: 30000,
            },
        }
    }
    
    /// Select optimal worker for new work item
    pub fn select_worker(&mut self, _work_item: &WorkItem) -> usize {
        match self.algorithm {
            LoadBalancingAlgorithm::RoundRobin => {
                static mut COUNTER: usize = 0;
                unsafe {
                    let worker = COUNTER % self.worker_loads.len();
                    COUNTER = COUNTER.wrapping_add(1);
                    worker
                }
            },
            LoadBalancingAlgorithm::LeastLoaded => {
                self.worker_loads.iter()
                    .min_by(|a, b| a.load_score.partial_cmp(&b.load_score).unwrap())
                    .map(|w| w.worker_id)
                    .unwrap_or(0)
            },
            LoadBalancingAlgorithm::WeightedRoundRobin => {
                // Select based on inverse load score
                let total_inverse_load: f64 = self.worker_loads.iter()
                    .map(|w| 1.0 / (w.load_score + 0.1))
                    .sum();
                
                let mut target = Self::random_f64() * total_inverse_load;
                for worker in &self.worker_loads {
                    target -= 1.0 / (worker.load_score + 0.1);
                    if target <= 0.0 {
                        return worker.worker_id;
                    }
                }
                0 // Fallback
            },
            _ => 0, // Simplified for other algorithms
        }
    }
    
    /// Update worker load information
    pub fn update_worker_load(&mut self, worker_id: usize, 
                              queue_size: usize, 
                              processing_time_ms: f64,
                              success: bool) {
        if worker_id < self.worker_loads.len() {
            let worker = &mut self.worker_loads[worker_id];
            worker.queue_size = queue_size;
            
            // Update processing time with exponential moving average
            let alpha = 0.2;
            worker.recent_processing_time_ms = 
                worker.recent_processing_time_ms * (1.0 - alpha) + processing_time_ms * alpha;
            
            // Update success rate
            worker.success_rate = worker.success_rate * 0.95 + (if success { 1.0 } else { 0.0 }) * 0.05;
            
            // Calculate load score
            worker.load_score = 
                worker.queue_size as f64 * 0.4 +
                worker.recent_processing_time_ms * 0.3 +
                (1.0 - worker.success_rate) * 100.0 * 0.2 +
                worker.cpu_utilization as f64 * 0.1;
        }
    }
    
    /// Record performance snapshot
    pub fn record_performance(&mut self, throughput: f64, avg_latency_ms: f64) {
        let load_variance = self.calculate_load_variance();
        
        let snapshot = PerformanceSnapshot {
            timestamp: Self::current_timestamp(),
            throughput,
            avg_latency_ms,
            load_variance,
            algorithm: self.algorithm,
        };
        
        self.performance_history.push(snapshot);
        
        // Keep only recent history
        if self.performance_history.len() > 1000 {
            self.performance_history.remove(0);
        }
        
        // Check if we should adapt algorithm
        if self.adaptation_config.adaptive_switching {
            self.consider_algorithm_adaptation();
        }
    }
    
    /// Calculate load variance across workers
    fn calculate_load_variance(&self) -> f64 {
        let mean_load = self.worker_loads.iter().map(|w| w.load_score).sum::<f64>() / 
                       self.worker_loads.len() as f64;
        
        let variance = self.worker_loads.iter()
            .map(|w| (w.load_score - mean_load).powi(2))
            .sum::<f64>() / self.worker_loads.len() as f64;
        
        variance.sqrt()
    }
    
    /// Consider switching to a better algorithm
    fn consider_algorithm_adaptation(&mut self) {
        if self.performance_history.len() < self.adaptation_config.evaluation_window_size {
            return;
        }
        
        // Evaluate current algorithm performance
        let recent_performance = &self.performance_history
            [self.performance_history.len() - self.adaptation_config.evaluation_window_size..];
        
        let avg_throughput = recent_performance.iter().map(|p| p.throughput).sum::<f64>() / 
                            recent_performance.len() as f64;
        let avg_latency = recent_performance.iter().map(|p| p.avg_latency_ms).sum::<f64>() / 
                         recent_performance.len() as f64;
        
        // Simple adaptation logic - switch to least loaded if performance is poor
        if avg_latency > 50.0 && !matches!(self.algorithm, LoadBalancingAlgorithm::LeastLoaded) {
            self.algorithm = LoadBalancingAlgorithm::LeastLoaded;
        } else if avg_throughput > 1000.0 && avg_latency < 20.0 {
            self.algorithm = LoadBalancingAlgorithm::WeightedRoundRobin;
        }
    }
    
    /// Get current worker loads
    pub fn get_worker_loads(&self) -> &[WorkerLoad] {
        &self.worker_loads
    }
    
    fn current_timestamp() -> u64 {
        static mut TIMESTAMP: u64 = 0;
        unsafe {
            TIMESTAMP += 1000;
            TIMESTAMP
        }
    }
    
    /// Simple random number generator for load balancing
    fn random_f64() -> f64 {
        static mut SEED: u64 = 1;
        unsafe {
            SEED = SEED.wrapping_mul(1103515245).wrapping_add(12345);
            (SEED as f64) / (u64::MAX as f64)
        }
    }
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