#!/usr/bin/env rust-script
//! Basic LNN demonstration in Rust
//! 
//! This example shows the core Liquid Neural Network functionality
//! implemented in Rust for maximum performance and embedded compatibility.

use liquid_audio_nets::{
    ModelConfig, AdaptiveConfig, LNN, 
    models::ModelFactory,
    Result
};
use std::time::Instant;

/// Generate synthetic audio for testing
fn generate_test_audio(pattern: &str, duration_ms: u32, sample_rate: u32) -> Vec<f32> {
    let samples = (duration_ms * sample_rate / 1000) as usize;
    let mut audio = vec![0.0; samples];
    
    let dt = 1.0 / sample_rate as f32;
    
    match pattern {
        "wake" => {
            // Two-tone burst pattern
            let half = samples / 2;
            for (i, sample) in audio.iter_mut().take(half).enumerate() {
                let t = i as f32 * dt;
                *sample = 0.3 * (2.0 * std::f32::consts::PI * 800.0 * t).sin();
            }
            for (i, sample) in audio.iter_mut().skip(half).enumerate() {
                let t = i as f32 * dt;
                *sample = 0.3 * (2.0 * std::f32::consts::PI * 1200.0 * t).sin();
            }
        }
        "stop" => {
            // Exponentially decaying tone
            for (i, sample) in audio.iter_mut().enumerate() {
                let t = i as f32 * dt;
                *sample = 0.4 * (2.0 * std::f32::consts::PI * 600.0 * t).sin() * (-t * 3.0).exp();
            }
        }
        "noise" => {
            // Simple noise pattern
            for (i, sample) in audio.iter_mut().enumerate() {
                *sample = 0.05 * ((i as f32 * 12345.67) % 2.0 - 1.0); // Pseudo-random
            }
        }
        _ => {
            // Silence - already initialized to zeros
        }
    }
    
    // Add small amount of noise
    for (i, sample) in audio.iter_mut().enumerate() {
        let noise = 0.02 * (((i as f32 * 987.654) % 2.0) - 1.0);
        *sample += noise;
    }
    
    audio
}

fn demo_basic_processing() -> Result<()> {
    println!("🦀 Rust LNN Basic Processing Demo");
    println!("==================================");
    
    // Create model configuration
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 8,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "keyword_spotting".to_string(),
    };
    
    // Create LNN instance
    let mut lnn = LNN::new(config)?;
    println!("✅ Created LNN: {:?}", lnn.config());
    
    // Configure adaptive timestep
    let adaptive_config = AdaptiveConfig {
        min_timestep_ms: 5.0,
        max_timestep_ms: 40.0,
        energy_threshold: 0.1,
        complexity_penalty: 0.02,
        power_budget_mw: 1.0,
    };
    lnn.set_adaptive_config(adaptive_config);
    println!("🔧 Set adaptive config: {:.1}-{:.1}ms timestep range", 
             adaptive_config.min_timestep_ms, adaptive_config.max_timestep_ms);
    
    // Test different audio patterns
    let test_cases = [
        ("wake", "Wake word pattern"),
        ("stop", "Stop word pattern"),  
        ("noise", "Background noise"),
        ("silence", "Silent audio"),
    ];
    
    println!("\n📊 Processing Results:");
    println!("Pattern    Confidence  Power(mW)  Timestep(ms)  Complexity  Energy");
    println!("---------------------------------------------------------------");
    
    for (pattern, _description) in &test_cases {
        // Generate test audio
        let audio = generate_test_audio(pattern, 500, 16000); // 500ms
        
        // Process with timing
        let start = Instant::now();
        let result = lnn.process(&audio)?;
        let processing_time = start.elapsed();
        
        // Display results
        println!("{:10} {:10.3} {:9.2} {:12.1} {:10.3} {:6.3}", 
                 pattern,
                 result.confidence,
                 result.power_mw,
                 result.timestep_ms,
                 result.complexity,
                 result.liquid_energy);
        
        // Additional info for interesting results
        if result.confidence > 0.5 {
            println!("    🎯 High confidence detection!");
        }
        if result.power_mw < 1.0 {
            println!("    ⚡ Ultra-low power: {:.2}mW", result.power_mw);
        }
        
        println!("    ⏱️  Processing time: {:?}", processing_time);
    }
    
    println!("\n💡 Final power consumption: {:.2}mW", lnn.current_power_mw());
    Ok(())
}

fn demo_adaptive_optimization() -> Result<()> {
    println!("\n\n⚡ Adaptive Optimization Demo");
    println!("=============================");
    
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 8,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "power_optimized".to_string(),
    };
    
    let mut lnn = LNN::new(config)?;
    
    // Test without adaptive config first
    println!("\n🔸 Without adaptive timestep:");
    let audio_silence = generate_test_audio("silence", 250, 16000);
    let audio_complex = generate_test_audio("wake", 250, 16000);
    
    let result_silence = lnn.process(&audio_silence)?;
    let result_complex = lnn.process(&audio_complex)?;
    
    println!("  Silence:  {:.2}mW, {:.1}ms timestep", 
             result_silence.power_mw, result_silence.timestep_ms);
    println!("  Complex:  {:.2}mW, {:.1}ms timestep", 
             result_complex.power_mw, result_complex.timestep_ms);
    
    // Now with adaptive config
    let adaptive_config = AdaptiveConfig {
        min_timestep_ms: 3.0,
        max_timestep_ms: 50.0,
        energy_threshold: 0.05,
        complexity_penalty: 0.01,
        power_budget_mw: 1.0,
    };
    lnn.set_adaptive_config(adaptive_config);
    
    println!("\n🔹 With adaptive timestep:");
    let result_silence_adaptive = lnn.process(&audio_silence)?;
    let result_complex_adaptive = lnn.process(&audio_complex)?;
    
    println!("  Silence:  {:.2}mW, {:.1}ms timestep", 
             result_silence_adaptive.power_mw, result_silence_adaptive.timestep_ms);
    println!("  Complex:  {:.2}mW, {:.1}ms timestep", 
             result_complex_adaptive.power_mw, result_complex_adaptive.timestep_ms);
    
    // Calculate power savings
    let silence_saving = (result_silence.power_mw - result_silence_adaptive.power_mw) / result_silence.power_mw * 100.0;
    let complex_saving = (result_complex.power_mw - result_complex_adaptive.power_mw) / result_complex.power_mw * 100.0;
    
    println!("\n📊 Power Savings:");
    println!("  Silence:  {:.1}%", silence_saving);
    println!("  Complex:  {:.1}%", complex_saving);
    println!("  Average:  {:.1}%", (silence_saving + complex_saving) / 2.0);
    
    Ok(())
}

fn demo_memory_efficiency() -> Result<()> {
    println!("\n\n🧠 Memory Efficiency Demo");
    println!("========================");
    
    // Test different model sizes
    let model_sizes = [
        (20, 32, "Tiny"),
        (40, 64, "Standard"), 
        (80, 128, "Large"),
    ];
    
    for (input_dim, hidden_dim, size_name) in &model_sizes {
        let config = ModelConfig {
            input_dim: *input_dim,
            hidden_dim: *hidden_dim,
            output_dim: 8,
            sample_rate: 16000,
            frame_size: 256,
            model_type: format!("size_test_{}", size_name.to_lowercase()),
        };
        
        let mut lnn = LNN::new(config)?;
        
        // Estimate memory usage
        let param_count = input_dim * hidden_dim + hidden_dim * hidden_dim + hidden_dim * 8;
        let memory_kb = (param_count * 4 + 1024) / 1024; // Rough estimate in KB
        
        // Test processing
        let audio = generate_test_audio("wake", 100, 16000);
        let result = lnn.process(&audio)?;
        
        println!("{:8} model: {}→{}→8, ~{}KB, {:.2}mW", 
                 size_name, input_dim, hidden_dim, memory_kb, result.power_mw);
    }
    
    Ok(())
}

fn demo_model_factory() -> Result<()> {
    println!("\n\n🏭 Model Factory Demo");
    println!("====================");
    
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 10,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "factory_test".to_string(),
    };
    
    // Create model via factory
    let mut model = ModelFactory::create_by_type("lnn", config)?;
    
    println!("✅ Created model via factory");
    println!("   Type: {}", model.model_type());
    println!("   Ready: {}", model.is_ready());
    
    // Test processing
    let audio = generate_test_audio("wake", 200, 16000);
    let result = model.process_audio(&audio)?;
    
    println!("🔄 Processed audio: {:.2}mW power, {:.3} confidence", 
             result.power_mw, result.confidence);
    
    // Reset and test again
    model.reset();
    println!("🔄 Model reset");
    
    let result2 = model.process_audio(&audio)?;
    println!("🔄 Post-reset processing: {:.2}mW power", result2.power_mw);
    
    Ok(())
}

fn demo_error_handling() -> Result<()> {
    println!("\n\n🚨 Error Handling Demo");
    println!("=====================");
    
    let config = ModelConfig {
        input_dim: 40,
        hidden_dim: 64,
        output_dim: 8,
        sample_rate: 16000,
        frame_size: 512,
        model_type: "error_test".to_string(),
    };
    
    let mut lnn = LNN::new(config)?;
    
    // Test empty buffer
    match lnn.process(&[]) {
        Ok(_) => println!("❌ Empty buffer should have failed"),
        Err(e) => println!("✅ Empty buffer correctly rejected: {}", e),
    }
    
    // Test very large buffer
    let huge_buffer = vec![0.1; 100000];
    match lnn.process(&huge_buffer) {
        Ok(result) => println!("✅ Large buffer processed: {:.2}mW", result.power_mw),
        Err(e) => println!("⚠️  Large buffer failed: {}", e),
    }
    
    // Test NaN values
    let bad_buffer = vec![f32::NAN, 0.1, f32::INFINITY, 0.2];
    match lnn.process(&bad_buffer) {
        Ok(result) => println!("⚠️  NaN buffer processed (unexpected): {:.2}mW", result.power_mw),
        Err(e) => println!("✅ NaN buffer correctly rejected: {}", e),
    }
    
    Ok(())
}

fn main() -> Result<()> {
    println!("🚀 Liquid Audio Networks - Rust Core Demo");
    println!("This demonstrates the high-performance Rust implementation\n");
    
    // Run all demonstrations
    demo_basic_processing()?;
    demo_adaptive_optimization()?;
    demo_memory_efficiency()?;
    demo_model_factory()?;
    demo_error_handling()?;
    
    println!("\n\n✅ All Rust demos completed successfully!");
    
    println!("\n💡 Key Rust Advantages Demonstrated:");
    println!("   • Zero-cost abstractions for embedded deployment");
    println!("   • Memory safety without garbage collection overhead");
    println!("   • Cross-compilation to ARM Cortex-M targets");
    println!("   • Optimal performance with adaptive algorithms");
    println!("   • Comprehensive error handling with Result types");
    
    println!("\n🔗 Next Steps:");
    println!("   • Cross-compile for embedded targets (cargo build --target thumbv7em-none-eabihf)");
    println!("   • Profile with embedded hardware simulators");
    println!("   • Integrate with RTOS and interrupt-driven systems");
    println!("   • Optimize for specific MCU architectures");
    
    Ok(())
}