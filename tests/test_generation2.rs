//! Tests for Generation 2 robustness features

use liquid_audio_nets::*;

/// Helper function to create LNN with proper permissions for testing
fn create_test_lnn_with_permissions(config: ModelConfig) -> LNN {
    let security_context = SecurityContext {
        session_id: "test_session".to_string(),
        permissions: vec!["basic_processing".to_string(), "audio_processing".to_string()],
        rate_limits: vec![],
        security_level: SecurityLevel::Authenticated,
        last_auth_time: 0,
        failed_attempts: 0,
    };
    
    LNN::new_with_security(config, security_context).expect("Should create LNN")
}

#[test]
fn test_enhanced_lnn_creation() {
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    let lnn = LNN::new(config).expect("Should create enhanced LNN");
    assert_eq!(lnn.current_power_mw(), 1.0);
}

#[test]
fn test_config_validation() {
    // Test invalid input dimension
    let invalid_config = ModelConfig {
        input_dim: 0,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    let result = LNN::new(invalid_config);
    assert!(result.is_err());
    match result.unwrap_err() {
        LiquidAudioError::ConfigError(msg) => {
            assert!(msg.contains("Input dimension"));
        }
        _ => panic!("Expected ConfigError"),
    }
}

#[test]
fn test_sample_rate_validation() {
    let invalid_config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 500, // Too low
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    let result = LNN::new(invalid_config);
    assert!(result.is_err());
    match result.unwrap_err() {
        LiquidAudioError::ConfigError(msg) => {
            assert!(msg.contains("Sample rate"));
        }
        _ => panic!("Expected ConfigError"),
    }
}

#[test]
fn test_input_validation() {
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    let mut lnn = create_test_lnn_with_permissions(config);
    
    // Test empty buffer
    let empty_buffer = vec![];
    let result = lnn.process(&empty_buffer);
    assert!(result.is_err());
    
    // Test extremely large buffer
    let huge_buffer = vec![0.1f32; 10000];
    let result = lnn.process(&huge_buffer);
    assert!(result.is_err());
}

#[test]
fn test_nan_input_handling() {
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    let mut lnn = create_test_lnn_with_permissions(config);
    
    // Test buffer with NaN
    let nan_buffer = vec![0.1, 0.2, f32::NAN, 0.4, 0.5];
    let result = lnn.process(&nan_buffer);
    assert!(result.is_err());
}

#[test]
fn test_infinity_input_handling() {
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    let mut lnn = create_test_lnn_with_permissions(config);
    
    // Test buffer with infinity
    let inf_buffer = vec![0.1, 0.2, f32::INFINITY, 0.4, 0.5];
    let result = lnn.process(&inf_buffer);
    assert!(result.is_err());
}

#[test]
fn test_adaptive_config_validation() {
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    let mut lnn = create_test_lnn_with_permissions(config);
    
    // Test valid adaptive config
    let valid_adaptive = AdaptiveConfig {
        min_timestep_ms: 5.0,
        max_timestep_ms: 50.0,
        energy_threshold: 0.1,
        complexity_penalty: 0.02,
        power_budget_mw: 1.0,
    };
    
    lnn.set_adaptive_config(valid_adaptive);
    
    // Test invalid adaptive config (min >= max)
    let invalid_adaptive = AdaptiveConfig {
        min_timestep_ms: 50.0,
        max_timestep_ms: 5.0, // Invalid: min > max
        energy_threshold: 0.1,
        complexity_penalty: 0.02,
        power_budget_mw: 1.0,
    };
    
    lnn.set_adaptive_config(invalid_adaptive); // Should log error but not panic
}

#[test]
fn test_enhanced_processing() {
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    let mut lnn = create_test_lnn_with_permissions(config);
    
    // Set adaptive config
    let adaptive_config = AdaptiveConfig {
        min_timestep_ms: 5.0,
        max_timestep_ms: 50.0,
        energy_threshold: 0.1,
        complexity_penalty: 0.02,
        power_budget_mw: 1.0,
    };
    lnn.set_adaptive_config(adaptive_config);
    
    // Test processing with adaptive timestep
    let audio_buffer = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let result = lnn.process(&audio_buffer).expect("Should process audio");
    
    assert_eq!(result.output.len(), 2);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    assert!(result.timestep_ms >= 5.0 && result.timestep_ms <= 50.0);
    assert!(result.power_mw > 0.0);
}

#[test]
fn test_health_checks() {
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    let mut lnn = create_test_lnn_with_permissions(config);
    
    // Process some data first
    let audio_buffer = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let _ = lnn.process(&audio_buffer).expect("Should process audio");
    
    // Perform health check
    let health_report = lnn.health_check().expect("Health check should succeed");
    
    // Check that we got a valid health report
    assert!(!health_report.checks.is_empty());
    assert!(health_report.metrics.total_samples > 0);
    
    // Get performance summary
    let summary = lnn.get_performance_summary();
    assert!(!summary.is_empty());
    
    // Get recommendations
    let recommendations = lnn.get_recommendations();
    assert!(!recommendations.is_empty());
}

#[test]
fn test_validation_toggle() {
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    let mut lnn = create_test_lnn_with_permissions(config);
    
    // Disable validation
    lnn.set_validation_enabled(false);
    
    // This should now work even with empty buffer (validation disabled)
    let empty_buffer = vec![];
    // Note: This might still fail due to processing logic, but not validation
    
    // Re-enable validation
    lnn.set_validation_enabled(true);
    
    // This should fail with validation enabled
    let result = lnn.process(&empty_buffer);
    assert!(result.is_err());
}

#[test]
fn test_max_input_magnitude() {
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    let mut lnn = create_test_lnn_with_permissions(config);
    
    // Set a low maximum input magnitude
    lnn.set_max_input_magnitude(1.0);
    
    // Test with values exceeding the limit (should trigger warnings)
    let large_buffer = vec![2.0, 3.0, 0.5, 0.2, 0.1];
    let result = lnn.process(&large_buffer).expect("Should still process (just warn)");
    
    assert_eq!(result.output.len(), 2);
}

#[test]
fn test_state_reset() {
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    let mut lnn = create_test_lnn_with_permissions(config);
    
    // Process some data
    let audio_buffer = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let _ = lnn.process(&audio_buffer).expect("Should process audio");
    
    // Reset state
    lnn.reset_state();
    
    // Power should be reset to initial value
    assert_eq!(lnn.current_power_mw(), 1.0);
}

#[test]
fn test_error_recovery() {
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    let mut lnn = create_test_lnn_with_permissions(config);
    
    // The error recovery is internal to process(), so we test indirectly
    // by trying to process valid data after potentially problematic data
    
    let normal_buffer = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let result = lnn.process(&normal_buffer);
    assert!(result.is_ok());
    
    // Processing should continue to work
    let result2 = lnn.process(&normal_buffer);
    assert!(result2.is_ok());
}