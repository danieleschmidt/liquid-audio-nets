//! Basic tests for Rust API

use liquid_audio_nets::*;

#[test]
fn test_lnn_creation() {
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    let lnn = LNN::new(config).expect("Should create LNN");
    assert_eq!(lnn.current_power_mw(), 1.0);
}

#[test]
fn test_adaptive_config() {
    let config = AdaptiveConfig::default();
    assert_eq!(config.min_timestep_ms, 5.0);
    assert_eq!(config.max_timestep_ms, 50.0);
    assert_eq!(config.energy_threshold, 0.1);
}

#[test]
fn test_audio_processing() {
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    // Create security context with proper permissions
    let security_context = SecurityContext {
        session_id: "test_session".to_string(),
        permissions: vec!["basic_processing".to_string(), "audio_processing".to_string()],
        rate_limits: vec![],
        security_level: SecurityLevel::Authenticated,
        last_auth_time: 0,
        failed_attempts: 0,
    };
    
    let mut lnn = LNN::new_with_security(config, security_context).expect("Should create LNN");
    
    // Test with some dummy audio data
    let audio_buffer = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let result = lnn.process(&audio_buffer).expect("Should process audio");
    
    assert_eq!(result.output.len(), 2);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
    assert_eq!(result.timestep_ms, 10.0);
    assert!(result.power_mw > 0.0);
}

#[test]
fn test_empty_buffer_error() {
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    // Create security context with proper permissions for testing
    let security_context = SecurityContext {
        session_id: "test_session".to_string(),
        permissions: vec!["basic_processing".to_string(), "audio_processing".to_string()],
        rate_limits: vec![],
        security_level: SecurityLevel::Authenticated,
        last_auth_time: 0,
        failed_attempts: 0,
    };
    
    let mut lnn = LNN::new_with_security(config, security_context).expect("Should create LNN");
    
    // Test with empty buffer
    let empty_buffer = vec![];
    let result = lnn.process(&empty_buffer);
    
    assert!(result.is_err());
    match result.unwrap_err() {
        LiquidAudioError::InvalidInput(msg) => {
            assert_eq!(msg, "Empty audio buffer");
        }
        _ => panic!("Expected InvalidInput error"),
    }
}

#[test]
fn test_error_display() {
    let error = LiquidAudioError::ModelError("Test error".to_string());
    let display_str = format!("{}", error);
    assert_eq!(display_str, "Model error: Test error");
}

#[test]
fn test_version() {
    assert_eq!(VERSION, env!("CARGO_PKG_VERSION"));
}