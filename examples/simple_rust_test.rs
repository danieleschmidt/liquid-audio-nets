use liquid_audio_nets::{ModelConfig, LNN, Result};

fn main() -> Result<()> {
    println!("🦀 Simple Rust LNN Test");
    
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 8,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "test".to_string(),
    };
    
    let mut lnn = LNN::new(config)?;
    println!("✅ Created LNN successfully");
    
    // Test with simple audio
    let audio = vec![0.1, 0.2, 0.1, -0.1, 0.0, 0.3, -0.2, 0.1];
    let result = lnn.process(&audio)?;
    
    println!("🔄 Processed audio:");
    println!("   Confidence: {:.3}", result.confidence);
    println!("   Power: {:.2}mW", result.power_mw);
    println!("   Timestep: {:.1}ms", result.timestep_ms);
    
    println!("✅ Test completed!");
    Ok(())
}