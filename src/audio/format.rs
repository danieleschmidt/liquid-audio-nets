//! Audio format definitions and validation

use crate::{Result, LiquidAudioError};
use serde::{Deserialize, Serialize};

/// Audio format specification with validation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioFormat {
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u16,
    /// Sample format
    pub sample_format: SampleFormat,
    /// Bits per sample
    pub bits_per_sample: u16,
    /// Block alignment
    pub block_align: u16,
    /// Average bytes per second
    pub byte_rate: u32,
}

impl AudioFormat {
    /// Create new audio format with validation
    pub fn new(
        sample_rate: u32,
        channels: u16,
        sample_format: SampleFormat,
        bits_per_sample: u16,
    ) -> Result<Self> {
        // Validate sample rate
        if sample_rate < 8000 || sample_rate > 192000 {
            return Err(LiquidAudioError::ConfigError(
                format!("Invalid sample rate: {} Hz (must be 8000-192000)", sample_rate)
            ));
        }
        
        // Validate channels
        if channels == 0 || channels > 8 {
            return Err(LiquidAudioError::ConfigError(
                format!("Invalid channel count: {} (must be 1-8)", channels)
            ));
        }
        
        // Validate bits per sample
        if ![8, 16, 24, 32].contains(&bits_per_sample) {
            return Err(LiquidAudioError::ConfigError(
                format!("Invalid bits per sample: {} (must be 8, 16, 24, or 32)", bits_per_sample)
            ));
        }
        
        // Validate sample format compatibility
        sample_format.validate_with_bit_depth(bits_per_sample)?;
        
        // Calculate derived values
        let block_align = channels * (bits_per_sample / 8);
        let byte_rate = sample_rate * block_align as u32;
        
        Ok(Self {
            sample_rate,
            channels,
            sample_format,
            bits_per_sample,
            block_align,
            byte_rate,
        })
    }
    
    /// Create common formats with predefined settings
    pub fn pcm_16khz_mono() -> Result<Self> {
        Self::new(16000, 1, SampleFormat::SignedInt, 16)
    }
    
    pub fn pcm_44khz_stereo() -> Result<Self> {
        Self::new(44100, 2, SampleFormat::SignedInt, 16)
    }
    
    pub fn pcm_48khz_mono() -> Result<Self> {
        Self::new(48000, 1, SampleFormat::SignedInt, 16)
    }
    
    pub fn float32_16khz_mono() -> Result<Self> {
        Self::new(16000, 1, SampleFormat::Float, 32)
    }
    
    /// Get maximum frame size for this format
    pub fn max_frame_size(&self) -> usize {
        // Conservative limit: 1 second of audio
        self.sample_rate as usize * self.channels as usize
    }
    
    /// Calculate buffer size in bytes for given number of samples
    pub fn buffer_size_bytes(&self, samples: usize) -> usize {
        samples * self.channels as usize * (self.bits_per_sample as usize / 8)
    }
    
    /// Calculate number of samples for given buffer size in bytes
    pub fn samples_from_bytes(&self, bytes: usize) -> usize {
        bytes / (self.channels as usize * (self.bits_per_sample as usize / 8))
    }
    
    /// Get duration in seconds for given number of samples
    pub fn duration_seconds(&self, samples: usize) -> f64 {
        samples as f64 / self.sample_rate as f64
    }
    
    /// Get number of samples for given duration
    pub fn samples_for_duration(&self, duration_seconds: f64) -> usize {
        (duration_seconds * self.sample_rate as f64) as usize
    }
    
    /// Check if format is suitable for embedded systems
    pub fn is_embedded_suitable(&self) -> bool {
        self.sample_rate <= 48000 &&
        self.channels <= 2 &&
        self.bits_per_sample <= 16 &&
        matches!(self.sample_format, SampleFormat::SignedInt | SampleFormat::UnsignedInt)
    }
    
    /// Check if format supports real-time processing
    pub fn supports_realtime(&self) -> bool {
        // Heuristic: formats with reasonable computational requirements
        let samples_per_second = self.sample_rate as u64 * self.channels as u64;
        let bits_per_second = samples_per_second * self.bits_per_sample as u64;
        
        // Limit to ~1.5 Mbps for real-time processing
        bits_per_second <= 1_500_000
    }
    
    /// Get Nyquist frequency (maximum representable frequency)
    pub fn nyquist_frequency(&self) -> f32 {
        self.sample_rate as f32 / 2.0
    }
    
    /// Validate buffer for this format
    pub fn validate_buffer(&self, buffer: &[u8]) -> Result<()> {
        // Check alignment
        if buffer.len() % self.block_align as usize != 0 {
            return Err(LiquidAudioError::InvalidInput(
                format!("Buffer size {} not aligned to block size {}", 
                       buffer.len(), self.block_align)
            ));
        }
        
        // Check reasonable size limits
        let max_duration_seconds = 60.0; // 1 minute maximum
        let max_bytes = (self.byte_rate as f64 * max_duration_seconds) as usize;
        
        if buffer.len() > max_bytes {
            return Err(LiquidAudioError::InvalidInput(
                format!("Buffer too large: {} bytes (max {} for 1 minute)", 
                       buffer.len(), max_bytes)
            ));
        }
        
        Ok(())
    }
    
    /// Convert to different sample rate (returns new format)
    pub fn with_sample_rate(&self, new_rate: u32) -> Result<Self> {
        Self::new(new_rate, self.channels, self.sample_format, self.bits_per_sample)
    }
    
    /// Convert to different channel count
    pub fn with_channels(&self, new_channels: u16) -> Result<Self> {
        Self::new(self.sample_rate, new_channels, self.sample_format, self.bits_per_sample)
    }
    
    /// Convert to mono (if stereo)
    pub fn to_mono(&self) -> Result<Self> {
        if self.channels == 1 {
            Ok(self.clone())
        } else {
            self.with_channels(1)
        }
    }
    
    /// Check compatibility with another format
    pub fn is_compatible_with(&self, other: &AudioFormat) -> bool {
        self.sample_rate == other.sample_rate &&
        self.sample_format == other.sample_format &&
        self.bits_per_sample == other.bits_per_sample
        // Note: channels can differ (mono/stereo conversion possible)
    }
    
    /// Get format description string  
    pub fn description(&self) -> String {
        format!("{} Hz, {} channel{}, {}-bit {}",
                self.sample_rate,
                self.channels,
                if self.channels == 1 { "" } else { "s" },
                self.bits_per_sample,
                self.sample_format.name())
    }
}

/// Sample format enumeration with validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SampleFormat {
    /// Signed integer samples
    SignedInt,
    /// Unsigned integer samples  
    UnsignedInt,
    /// IEEE floating point samples
    Float,
    /// A-law compressed
    ALaw,
    /// μ-law compressed
    MuLaw,
}

impl SampleFormat {
    /// Validate compatibility with bit depth
    pub fn validate_with_bit_depth(&self, bits_per_sample: u16) -> Result<()> {
        match self {
            SampleFormat::SignedInt | SampleFormat::UnsignedInt => {
                if ![8, 16, 24, 32].contains(&bits_per_sample) {
                    return Err(LiquidAudioError::ConfigError(
                        format!("Invalid bit depth {} for integer format", bits_per_sample)
                    ));
                }
            },
            SampleFormat::Float => {
                if ![32, 64].contains(&bits_per_sample) {
                    return Err(LiquidAudioError::ConfigError(
                        format!("Invalid bit depth {} for float format (must be 32 or 64)", bits_per_sample)
                    ));
                }
            },
            SampleFormat::ALaw | SampleFormat::MuLaw => {
                if bits_per_sample != 8 {
                    return Err(LiquidAudioError::ConfigError(
                        format!("Invalid bit depth {} for compressed format (must be 8)", bits_per_sample)
                    ));
                }
            },
        }
        Ok(())
    }
    
    /// Get format name
    pub fn name(&self) -> &'static str {
        match self {
            SampleFormat::SignedInt => "PCM signed",
            SampleFormat::UnsignedInt => "PCM unsigned",
            SampleFormat::Float => "IEEE float",
            SampleFormat::ALaw => "A-law",
            SampleFormat::MuLaw => "μ-law",
        }
    }
    
    /// Check if format requires special handling
    pub fn needs_conversion(&self) -> bool {
        matches!(self, SampleFormat::ALaw | SampleFormat::MuLaw)
    }
    
    /// Get dynamic range in dB
    pub fn dynamic_range_db(&self, bits_per_sample: u16) -> f32 {
        match self {
            SampleFormat::SignedInt | SampleFormat::UnsignedInt => {
                // Dynamic range = 6.02 * bits + 1.76 dB
                6.02 * bits_per_sample as f32 + 1.76
            },
            SampleFormat::Float => {
                match bits_per_sample {
                    32 => 144.0, // ~24-bit equivalent
                    64 => 192.0, // ~32-bit equivalent
                    _ => 144.0,
                }
            },
            SampleFormat::ALaw | SampleFormat::MuLaw => {
                // Approximately 13-bit dynamic range
                78.0
            },
        }
    }
}

/// Audio format converter for safe format transitions
#[derive(Debug)]
pub struct FormatConverter {
    source_format: AudioFormat,
    target_format: AudioFormat,
    conversion_buffer: Vec<f32>,
}

impl FormatConverter {
    /// Create new format converter with validation
    pub fn new(source: AudioFormat, target: AudioFormat) -> Result<Self> {
        // Validate conversion possibility
        if source.sample_rate != target.sample_rate {
            return Err(LiquidAudioError::ConfigError(
                "Sample rate conversion not implemented".to_string()
            ));
        }
        
        // Check for lossy conversions and warn
        if source.bits_per_sample > target.bits_per_sample {
            // This would be lossy - allow but could add warning mechanism
        }
        
        let conversion_buffer = Vec::new();
        
        Ok(Self {
            source_format: source,
            target_format: target,
            conversion_buffer,
        })
    }
    
    /// Convert audio data from source to target format
    pub fn convert(&mut self, input: &[u8]) -> Result<Vec<u8>> {
        // Validate input
        self.source_format.validate_buffer(input)?;
        
        // For now, implement basic conversions
        match (&self.source_format.sample_format, &self.target_format.sample_format) {
            (SampleFormat::SignedInt, SampleFormat::Float) => {
                self.convert_int_to_float(input)
            },
            (SampleFormat::Float, SampleFormat::SignedInt) => {
                self.convert_float_to_int(input)
            },
            (format_a, format_b) if format_a == format_b => {
                // Same format, just copy
                Ok(input.to_vec())
            },
            _ => {
                Err(LiquidAudioError::ConfigError(
                    format!("Conversion from {} to {} not implemented", 
                           self.source_format.sample_format.name(),
                           self.target_format.sample_format.name())
                ))
            }
        }
    }
    
    /// Convert signed integer to float
    fn convert_int_to_float(&mut self, input: &[u8]) -> Result<Vec<u8>> {
        let samples_count = self.source_format.samples_from_bytes(input.len());
        self.conversion_buffer.clear();
        self.conversion_buffer.reserve(samples_count);
        
        // Convert based on bit depth
        match self.source_format.bits_per_sample {
            16 => {
                for chunk in input.chunks_exact(2) {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                    let float_sample = sample as f32 / i16::MAX as f32;
                    self.conversion_buffer.push(float_sample);
                }
            },
            32 => {
                for chunk in input.chunks_exact(4) {
                    let sample = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    let float_sample = sample as f32 / i32::MAX as f32;
                    self.conversion_buffer.push(float_sample);
                }
            },
            _ => {
                return Err(LiquidAudioError::ConfigError(
                    format!("Unsupported bit depth for int->float: {}", 
                           self.source_format.bits_per_sample)
                ));
            }
        }
        
        // Convert float buffer to bytes
        let output_bytes: Vec<u8> = self.conversion_buffer
            .iter()
            .flat_map(|&f| f.to_le_bytes())
            .collect();
        
        Ok(output_bytes)
    }
    
    /// Convert float to signed integer
    fn convert_float_to_int(&mut self, input: &[u8]) -> Result<Vec<u8>> {
        if input.len() % 4 != 0 {
            return Err(LiquidAudioError::InvalidInput(
                "Input not aligned to 32-bit floats".to_string()
            ));
        }
        
        let mut output = Vec::new();
        
        for chunk in input.chunks_exact(4) {
            let float_bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let float_sample = f32::from_le_bytes(float_bytes);
            
            // Clamp to [-1.0, 1.0] to prevent overflow
            let clamped = float_sample.clamp(-1.0, 1.0);
            
            // Convert based on target bit depth
            match self.target_format.bits_per_sample {
                16 => {
                    let int_sample = (clamped * i16::MAX as f32) as i16;
                    output.extend_from_slice(&int_sample.to_le_bytes());
                },
                32 => {
                    let int_sample = (clamped * i32::MAX as f32) as i32;
                    output.extend_from_slice(&int_sample.to_le_bytes());
                },
                _ => {
                    return Err(LiquidAudioError::ConfigError(
                        format!("Unsupported bit depth for float->int: {}", 
                               self.target_format.bits_per_sample)
                    ));
                }
            }
        }
        
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_audio_format_creation() {
        let format = AudioFormat::pcm_16khz_mono().unwrap();
        assert_eq!(format.sample_rate, 16000);
        assert_eq!(format.channels, 1);
        assert_eq!(format.bits_per_sample, 16);
    }
    
    #[test]
    fn test_invalid_sample_rate() {
        let result = AudioFormat::new(1000, 1, SampleFormat::SignedInt, 16);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_buffer_size_calculation() {
        let format = AudioFormat::pcm_16khz_mono().unwrap();
        let bytes = format.buffer_size_bytes(1000);
        assert_eq!(bytes, 2000); // 1000 samples * 1 channel * 2 bytes
    }
    
    #[test]
    fn test_duration_calculation() {
        let format = AudioFormat::pcm_16khz_mono().unwrap();
        let duration = format.duration_seconds(16000);
        assert!((duration - 1.0).abs() < 1e-6); // 1 second
    }
    
    #[test]
    fn test_format_compatibility() {
        let format1 = AudioFormat::pcm_16khz_mono().unwrap();
        let format2 = AudioFormat::new(16000, 2, SampleFormat::SignedInt, 16).unwrap();
        
        assert!(format1.is_compatible_with(&format2)); // Channels can differ
    }
    
    #[test]
    fn test_sample_format_validation() {
        assert!(SampleFormat::Float.validate_with_bit_depth(32).is_ok());
        assert!(SampleFormat::Float.validate_with_bit_depth(16).is_err());
        
        assert!(SampleFormat::ALaw.validate_with_bit_depth(8).is_ok());
        assert!(SampleFormat::ALaw.validate_with_bit_depth(16).is_err());
    }
    
    #[test]
    fn test_embedded_suitability() {
        let embedded_format = AudioFormat::pcm_16khz_mono().unwrap();
        assert!(embedded_format.is_embedded_suitable());
        
        let high_res_format = AudioFormat::new(192000, 8, SampleFormat::Float, 32).unwrap();
        assert!(!high_res_format.is_embedded_suitable());
    }
}