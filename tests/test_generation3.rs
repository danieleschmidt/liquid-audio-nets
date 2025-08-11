//! Tests for Generation 3 scaling and optimization features

use liquid_audio_nets::*;
use liquid_audio_nets::cache::*;
use liquid_audio_nets::optimization::*;
use liquid_audio_nets::scaling::*;
use liquid_audio_nets::pretrained::*;
use liquid_audio_nets::deployment::*;
use liquid_audio_nets::benchmark::*;

/// Helper function to create LNN with proper permissions for testing
fn create_test_lnn_with_permissions(config: ModelConfig) -> LNN {
    let security_context = liquid_audio_nets::SecurityContext {
        session_id: "test_session".to_string(),
        permissions: vec!["basic_processing".to_string(), "audio_processing".to_string()],
        rate_limits: vec![],
        security_level: liquid_audio_nets::SecurityLevel::Authenticated,
        last_auth_time: 0,
        failed_attempts: 0,
    };
    
    LNN::new_with_security(config, security_context).expect("Should create LNN")
}

#[test]
fn test_model_cache_basic_operations() {
    let cache_config = CacheConfig::default();
    let mut cache = ModelCache::new(cache_config);
    
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    // Test feature caching
    let key = ModelCache::generate_config_key(&config, CacheType::Features);
    let features = vec![0.1, 0.2, 0.3, 0.4];
    
    cache.cache_features(key.clone(), features.clone());
    let cached_features = cache.get_features(&key).unwrap();
    assert_eq!(cached_features, features);
    
    // Test cache statistics
    let stats = cache.get_stats();
    assert_eq!(stats.feature_cache.hit_count, 1);
    assert_eq!(stats.feature_cache.miss_count, 0);
}

#[test]
fn test_model_cache_lru_eviction() {
    let cache_config = CacheConfig {
        max_feature_entries: 2,
        feature_cache_size_bytes: 1000,
        ..Default::default()
    };
    let mut cache = ModelCache::new(cache_config);
    
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    // Fill cache beyond capacity
    for i in 0..5 {
        let mut key = ModelCache::generate_config_key(&config, CacheType::Features);
        key.input_hash = i; // Make keys unique
        let features = vec![i as f32; 10];
        cache.cache_features(key, features);
    }
    
    let stats = cache.get_stats();
    assert!(stats.feature_cache.entries <= 2); // Should not exceed max entries
}

#[test]
fn test_memory_pool() {
    let pool_config = PoolConfig {
        initial_size: 3,
        max_size: 5,
        default_capacity: 100,
        enable_monitoring: true,
    };
    let mut pool = MemoryPool::new(pool_config);
    
    // Test vector acquisition
    let vec1 = pool.get_vector(100);
    let vec2 = pool.get_vector(200);
    
    let stats = pool.get_stats();
    let initial_pool_size = stats.pool_size;
    assert_eq!(stats.allocations, 2);
    assert!(stats.pool_size <= 3);
    
    // Test vector return
    pool.return_vector(vec1);
    pool.return_vector(vec2);
    
    let stats_after = pool.get_stats();
    assert_eq!(stats_after.returns, 2);
    assert!(stats_after.pool_size >= initial_pool_size);
}

#[test]
fn test_vectorized_operations() {
    let a = vec![1.0, 2.0, 3.0, 4.0];
    let b = vec![2.0, 1.0, 4.0, 3.0];
    
    // Test dot product
    let dot_result = VectorOps::dot_product(&a, &b).unwrap();
    let expected = 1.0*2.0 + 2.0*1.0 + 3.0*4.0 + 4.0*3.0; // 2 + 2 + 12 + 12 = 28
    assert!((dot_result - expected).abs() < 1e-6);
    
    // Test softmax
    let mut input = vec![1.0, 2.0, 3.0];
    VectorOps::softmax(&mut input).unwrap();
    
    // Check softmax properties
    let sum: f32 = input.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    assert!(input.iter().all(|&x| x > 0.0 && x < 1.0));
    
    // Check ordering is preserved (larger inputs -> larger outputs)
    assert!(input[2] > input[1]);
    assert!(input[1] > input[0]);
}

#[test]
fn test_batch_processor() {
    let pool_config = PoolConfig::default();
    let mut processor = BatchProcessor::new(3, pool_config);
    
    // Add samples to batch
    let sample1 = vec![0.1, 0.2, 0.3];
    let sample2 = vec![0.4, 0.5, 0.6];
    let sample3 = vec![0.7, 0.8, 0.9];
    
    assert!(processor.add_sample(sample1));
    assert!(processor.add_sample(sample2));
    assert!(!processor.is_batch_ready()); // Not ready yet
    
    assert!(processor.add_sample(sample3));
    assert!(processor.is_batch_ready()); // Now ready
    
    // Test batch processing
    let results = processor.process_batch(|batch| {
        // Mock processor that returns dummy results
        Ok(vec![
            ProcessingResult {
                output: vec![0.8, 0.2],
                confidence: 0.8,
                timestep_ms: 10.0,
                power_mw: 1.0,
                complexity: 0.5,
                liquid_energy: 0.25,
                metadata: Some("batch_test".to_string()),
            };
            batch.len()
        ])
    }).unwrap();
    
    assert_eq!(results.len(), 3);
    
    let stats = processor.get_stats();
    assert_eq!(stats.batches_processed, 1);
    assert_eq!(stats.total_samples, 3);
}

#[test]
fn test_adaptive_computation() {
    let config = AdaptationConfig {
        target_latency_ms: 10.0,
        target_power_mw: 1.0,
        adaptation_rate: 0.1,
        ..Default::default()
    };
    let mut controller = AdaptiveComputation::new(config);
    
    let initial_level = controller.get_computation_level();
    assert!(initial_level > 0.0 && initial_level <= 1.0);
    
    // Update with high latency - should reduce computation level
    let high_latency_metrics = ScalingMetrics {
        cpu_utilization: 0.9,
        queue_depth: 200,
        avg_response_time_ms: 25.0, // High latency
        error_rate: 0.0,
        throughput_rps: 50.0,
        memory_utilization: 0.6,
        timestamp: 1000,
    };
    
    controller.update_performance(
        high_latency_metrics.avg_response_time_ms,
        high_latency_metrics.throughput_rps / 100.0, // Convert to rough power estimate
        0.9 // accuracy
    );
    
    // Should have adapted (though specific direction depends on algorithm)
    let adapted_level = controller.get_computation_level();
    assert!(adapted_level >= 0.1 && adapted_level <= 1.0);
}

#[test]
fn test_load_balancer() {
    let nodes = vec![
        ProcessingNode {
            id: "node1".to_string(),
            capacity: 100.0,
            current_load: 0.3,
            avg_response_time_ms: 5.0,
            health: HealthStatus::Healthy,
            enabled: true,
            requests_processed: 1000,
            error_count: 10,
            last_health_check: 1000,
        },
        ProcessingNode {
            id: "node2".to_string(),
            capacity: 100.0,
            current_load: 0.8, // Higher load
            avg_response_time_ms: 12.0, // Higher latency
            health: HealthStatus::Healthy,
            enabled: true,
            requests_processed: 1500,
            error_count: 5,
            last_health_check: 1000,
        },
    ];
    
    let mut balancer = LoadBalancer::new(
        nodes,
        LoadBalancingStrategy::LeastLoad,
        true,
    );
    
    // Test node selection - should prefer node1 (lower load)
    let selected = balancer.select_node().unwrap();
    assert_eq!(selected, 0); // node1 index
    
    // Update node statistics
    balancer.update_node_stats(0, 8.0, true);
    
    let stats = balancer.get_stats();
    assert_eq!(stats.total_requests, 1);
}

#[test]
fn test_auto_scaler() {
    let config = ScalingConfig {
        min_scale: 1,
        max_scale: 5,
        target_cpu_utilization: 0.7,
        target_queue_depth: 100,
        scale_up_threshold: 0.8,
        scale_down_threshold: 0.3,
        cooldown_ms: 0, // No cooldown for testing
        metrics_window_size: 3,
        aggressive_scaling: false,
    };
    let mut scaler = AutoScaler::new(config);
    
    assert_eq!(scaler.get_current_scale(), 1);
    
    // Feed high CPU utilization metrics
    let high_load_metrics = ScalingMetrics {
        cpu_utilization: 0.9, // Above scale-up threshold
        queue_depth: 150,
        avg_response_time_ms: 15.0,
        error_rate: 0.01,
        throughput_rps: 200.0,
        memory_utilization: 0.6,
        timestamp: 1000,
    };
    
    // Need to feed multiple metrics to fill window
    for _ in 0..3 {
        scaler.update_metrics(high_load_metrics.clone());
    }
    
    let action = scaler.update_metrics(high_load_metrics);
    assert_eq!(action, ScalingAction::ScaleUp);
    assert!(scaler.get_current_scale() > 1);
}

#[test] 
fn test_pretrained_model_registry() {
    let mut registry = ModelRegistry::new();
    
    // Check that built-in models are registered
    let models = registry.list_models();
    assert!(models.contains(&"keyword_spotter_v1".to_string()));
    assert!(models.contains(&"vad_ultra_low_power".to_string()));
    assert!(models.contains(&"audio_classifier_general".to_string()));
    
    // Get model info
    let model_info = registry.get_model_info("keyword_spotter_v1").unwrap();
    assert_eq!(model_info.model_type, ModelType::KeywordSpotter);
    assert_eq!(model_info.metadata.accuracy, 0.95);
    
    // Load a model
    let model = registry.load_model("keyword_spotter_v1", None).unwrap();
    assert_eq!(model.model_type(), "keyword_spotter");
    assert!(model.is_ready());
}

#[test]
fn test_pretrained_model_processing() {
    let mut registry = ModelRegistry::new();
    let mut model = registry.load_model("vad_ultra_low_power", None).unwrap();
    
    // Test with voice-like signal
    let voice_signal = vec![0.1, -0.05, 0.12, -0.08, 0.15]; // Varying amplitudes
    let result = model.process_audio(&voice_signal).unwrap();
    
    assert_eq!(result.output.len(), 2); // VAD outputs voice/non-voice probabilities
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    assert!(result.power_mw > 0.0);
    assert!(result.timestep_ms > 0.0);
    
    // Test with silence
    let silence = vec![0.0; 10];
    let silence_result = model.process_audio(&silence).unwrap();
    
    // Should detect silence (non-voice)
    assert!(silence_result.output[0] < silence_result.output[1]); // Non-voice probability higher
}

#[test]
fn test_deployment_config() {
    let config = DeploymentConfig::default();
    
    assert_eq!(config.environment, Environment::Development);
    assert_eq!(config.service_config.service_name, "liquid-audio-nets");
    assert_eq!(config.service_config.port, 8080);
    assert!(config.monitoring_config.metrics_enabled);
    assert!(config.security_config.tls.enabled);
}

#[test]
fn test_deployment_manager() {
    let config = DeploymentConfig::default();
    let mut manager = DeploymentManager::new(config).unwrap();
    
    assert_eq!(manager.get_status(), DeploymentStatus::NotDeployed);
    
    // Initialize deployment
    manager.initialize().unwrap();
    assert_eq!(manager.get_status(), DeploymentStatus::Running);
    
    // Update metrics
    manager.update_metrics(true, 15.5);
    manager.update_metrics(false, 25.0);
    
    let metrics = manager.get_metrics();
    assert_eq!(metrics.total_requests, 2);
    assert_eq!(metrics.successful_requests, 1);
    assert_eq!(metrics.failed_requests, 1);
    assert!(metrics.avg_response_time_ms > 0.0);
    
    // Health check
    let health = manager.health_check().unwrap();
    assert!(health.checks.len() > 0);
}

#[test]
fn test_deployment_manifest_generation() {
    let config = DeploymentConfig::default();
    let manager = DeploymentManager::new(config).unwrap();
    
    let manifest = manager.generate_manifest().unwrap();
    
    // Check that manifest contains key Kubernetes elements
    assert!(manifest.contains("apiVersion: apps/v1"));
    assert!(manifest.contains("kind: Deployment"));
    assert!(manifest.contains("liquid-audio-nets"));
    assert!(manifest.contains("containerPort: 8080"));
    assert!(manifest.contains("livenessProbe"));
    assert!(manifest.contains("readinessProbe"));
}

#[test]
fn test_dockerfile_generation() {
    let config = DeploymentConfig::default();
    let manager = DeploymentManager::new(config).unwrap();
    
    let dockerfile = manager.generate_dockerfile().unwrap();
    
    // Check Dockerfile structure
    assert!(dockerfile.contains("FROM ubuntu:22.04"));
    assert!(dockerfile.contains("EXPOSE 8080"));
    assert!(dockerfile.contains("HEALTHCHECK"));
    assert!(dockerfile.contains("USER appuser"));
    assert!(dockerfile.contains("CMD"));
}

#[test]
fn test_benchmark_suite_basic() {
    let scenario = BenchmarkScenario {
        name: "basic_test".to_string(),
        model_config: ModelConfig {
            input_dim: 40,
            hidden_dim: 64,
            output_dim: 10,
            sample_rate: 16000,
            frame_size: 512,
            model_type: "test".to_string(),
        },
        input_specs: InputSpecs {
            data_type: InputDataType::SineWave,
            size_params: DataSizeParams {
                min_buffer_size: 100,
                max_buffer_size: 500,
                sample_rates: vec![16000],
                duration_range: (0.1, 1.0),
            },
            characteristics: DataCharacteristics {
                amplitude_range: (0.0, 1.0),
                frequency_range: (100.0, 8000.0),
                snr_db: Some(20.0),
                dynamic_range: Some(40.0),
            },
            num_samples: 5,
        },
        performance_criteria: PerformanceCriteria {
            max_latency_ms: 20.0,
            max_power_mw: 2.0,
            min_accuracy: 0.8,
            max_memory_kb: 1024,
            min_throughput_sps: 1000.0,
            max_error_rate: 0.05,
        },
        stress_test: None,
        platform_settings: PlatformSettings {
            platform: TargetPlatform::X86_64,
            cpu_affinity: None,
            memory_strategy: MemoryStrategy::System,
            power_management: PowerManagement {
                dynamic_frequency: false,
                voltage_scaling: false,
                sleep_states: false,
                power_monitoring: true,
            },
            hardware_acceleration: HardwareAcceleration {
                simd_enabled: false,
                dsp_acceleration: false,
                gpu_acceleration: false,
                custom_accelerators: vec![],
            },
        },
    };
    
    let config = BenchmarkConfig {
        name: "test_benchmark".to_string(),
        scenarios: vec![scenario],
        warmup_iterations: 1,
        test_iterations: 3,
        statistical_analysis: true,
        comparative_analysis: false,
        output_format: OutputFormat::Json,
        timeout_ms: 10000,
    };
    
    let mut suite = BenchmarkSuite::new(config);
    let results = suite.run_benchmarks().unwrap();
    
    assert_eq!(results.scenario_results.len(), 1);
    assert!(results.overall_stats.total_tests > 0);
    assert!(results.overall_stats.successful_tests > 0);
}

#[test]
fn test_benchmark_export() {
    let scenario = BenchmarkScenario {
        name: "export_test".to_string(),
        model_config: ModelConfig {
            input_dim: 10,
            hidden_dim: 20,
            output_dim: 5,
            sample_rate: 16000,
            frame_size: 256,
            model_type: "test".to_string(),
        },
        input_specs: InputSpecs {
            data_type: InputDataType::WhiteNoise,
            size_params: DataSizeParams {
                min_buffer_size: 50,
                max_buffer_size: 100,
                sample_rates: vec![16000],
                duration_range: (0.1, 0.5),
            },
            characteristics: DataCharacteristics {
                amplitude_range: (0.0, 0.5),
                frequency_range: (20.0, 8000.0),
                snr_db: None,
                dynamic_range: None,
            },
            num_samples: 2,
        },
        performance_criteria: PerformanceCriteria {
            max_latency_ms: 10.0,
            max_power_mw: 1.5,
            min_accuracy: 0.7,
            max_memory_kb: 512,
            min_throughput_sps: 2000.0,
            max_error_rate: 0.1,
        },
        stress_test: None,
        platform_settings: PlatformSettings {
            platform: TargetPlatform::CortexM,
            cpu_affinity: None,
            memory_strategy: MemoryStrategy::Pool,
            power_management: PowerManagement {
                dynamic_frequency: true,
                voltage_scaling: true,
                sleep_states: true,
                power_monitoring: true,
            },
            hardware_acceleration: HardwareAcceleration {
                simd_enabled: false,
                dsp_acceleration: true,
                gpu_acceleration: false,
                custom_accelerators: vec![],
            },
        },
    };
    
    let config = BenchmarkConfig {
        name: "export_test".to_string(),
        scenarios: vec![scenario],
        warmup_iterations: 1,
        test_iterations: 2,
        statistical_analysis: true,
        comparative_analysis: false,
        output_format: OutputFormat::Json,
        timeout_ms: 5000,
    };
    
    let mut suite = BenchmarkSuite::new(config);
    let results = suite.run_benchmarks().unwrap();
    
    // Test JSON export
    let json_output = suite.export_results(&results, OutputFormat::Json).unwrap();
    assert!(json_output.contains("export_test"));
    assert!(json_output.contains("overall_score"));
    
    // Test CSV export
    let csv_output = suite.export_results(&results, OutputFormat::Csv).unwrap();
    assert!(csv_output.contains("scenario,latency_ms,power_mw,accuracy,memory_kb"));
    
    // Test Markdown export
    let md_output = suite.export_results(&results, OutputFormat::Markdown).unwrap();
    assert!(md_output.contains("# Benchmark Results"));
    assert!(md_output.contains("## Summary"));
    assert!(md_output.contains("| Scenario |"));
}

#[test]
fn test_integration_lnn_with_optimization() {
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "integration_test".to_string(),
    };
    
    let mut lnn = create_test_lnn_with_permissions(config.clone());
    
    // Create performance optimizer
    let cache_config = CacheConfig::default();
    let pool_config = PoolConfig::default();
    let adaptation_config = AdaptationConfig::default();
    
    let mut optimizer = PerformanceOptimizer::new(
        cache_config,
        pool_config,
        4, // batch size
        adaptation_config,
    );
    
    // Process audio with optimization
    let audio_data = vec![0.1, 0.2, -0.1, 0.3, -0.2];
    let result = lnn.process(&audio_data).unwrap();
    
    // Update performance metrics
    optimizer.update_performance(result.timestep_ms, result.power_mw, result.confidence);
    
    // Get optimization statistics
    let stats = optimizer.get_optimization_stats();
    assert!(stats.optimization_enabled);
    
    // Test caching
    let cache_key = optimizer.cache_mut().generate_input_key(&config, &audio_data, CacheType::Features);
    let features = vec![result.confidence, 1.0 - result.confidence];
    optimizer.cache_mut().cache_features(cache_key.clone(), features.clone());
    
    let cached_features = optimizer.cache_mut().get_features(&cache_key).unwrap();
    assert_eq!(cached_features, features);
}

#[test]
fn test_performance_optimizer_maintenance() {
    let cache_config = CacheConfig::default();
    let pool_config = PoolConfig::default();
    let adaptation_config = AdaptationConfig::default();
    
    let mut optimizer = PerformanceOptimizer::new(
        cache_config,
        pool_config,
        4,
        adaptation_config,
    );
    
    // Add some data to cache
    let config = ModelConfig {
        input_dim: 10,
        hidden_dim: 20,
        output_dim: 5,
        sample_rate: 16000,
        frame_size: 256,
        model_type: "maintenance_test".to_string(),
    };
    
    let cache_key = ModelCache::generate_config_key(&config, CacheType::Weights);
    let weights = vec![0.5; 100];
    optimizer.cache_mut().cache_weights(cache_key, weights);
    
    // Perform maintenance
    optimizer.maintenance();
    
    // Check that system is still functional
    let stats = optimizer.get_optimization_stats();
    assert!(stats.optimization_enabled);
}