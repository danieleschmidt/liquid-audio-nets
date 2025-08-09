"""
Input validation and sanitization for Liquid Audio Networks.

This module provides comprehensive validation for audio data, model parameters,
and system configurations to ensure robust operation in production environments.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for audio and model validation."""
    
    # Audio validation parameters
    max_audio_duration_s: float = 30.0
    min_audio_duration_s: float = 0.001
    max_sample_rate: int = 192000
    min_sample_rate: int = 1000
    max_amplitude: float = 10.0
    min_amplitude: float = 1e-8
    max_dc_offset: float = 0.5
    
    # Frequency domain validation
    max_frequency_hz: float = 24000.0
    min_frequency_resolution_hz: float = 0.1
    
    # Model validation parameters
    max_model_size_mb: int = 100
    min_confidence_threshold: float = 0.0
    max_confidence_threshold: float = 1.0
    max_processing_time_s: float = 1.0
    
    # Security parameters
    max_file_size_mb: int = 50
    allowed_extensions: List[str] = None
    enable_content_scanning: bool = True
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ['.lnn', '.wav', '.mp3', '.flac', '.m4a']

class AudioValidator:
    """Comprehensive audio data validation and sanitization."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.validation_stats = {
            'samples_validated': 0,
            'samples_rejected': 0,
            'samples_sanitized': 0,
            'common_issues': {}
        }
    
    def validate_audio_buffer(self, audio_buffer: np.ndarray, sample_rate: int = 16000) -> Tuple[bool, List[str]]:
        """
        Validate audio buffer for common issues and security concerns.
        
        Args:
            audio_buffer: Audio samples to validate
            sample_rate: Sample rate in Hz
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Basic array validation
            if not isinstance(audio_buffer, np.ndarray):
                issues.append(f"Audio buffer must be numpy array, got {type(audio_buffer)}")
                return False, issues
            
            if len(audio_buffer.shape) != 1:
                issues.append(f"Audio buffer must be 1D, got shape {audio_buffer.shape}")
                return False, issues
                
            if audio_buffer.size == 0:
                issues.append("Audio buffer is empty")
                return False, issues
            
            # Duration validation
            duration_s = len(audio_buffer) / sample_rate
            if duration_s > self.config.max_audio_duration_s:
                issues.append(f"Audio duration {duration_s:.3f}s exceeds maximum {self.config.max_audio_duration_s}s")
                return False, issues
                
            if duration_s < self.config.min_audio_duration_s:
                issues.append(f"Audio duration {duration_s:.6f}s below minimum {self.config.min_audio_duration_s}s")
                return False, issues
            
            # Sample rate validation
            if sample_rate > self.config.max_sample_rate:
                issues.append(f"Sample rate {sample_rate}Hz exceeds maximum {self.config.max_sample_rate}Hz")
                return False, issues
                
            if sample_rate < self.config.min_sample_rate:
                issues.append(f"Sample rate {sample_rate}Hz below minimum {self.config.min_sample_rate}Hz")
                return False, issues
            
            # Amplitude validation
            max_amplitude = np.max(np.abs(audio_buffer))
            if max_amplitude > self.config.max_amplitude:
                issues.append(f"Maximum amplitude {max_amplitude:.3f} exceeds limit {self.config.max_amplitude}")
                return False, issues
                
            if max_amplitude < self.config.min_amplitude:
                issues.append(f"Maximum amplitude {max_amplitude:.6e} below minimum {self.config.min_amplitude}")
                return False, issues
            
            # NaN and infinity validation
            nan_count = np.sum(np.isnan(audio_buffer))
            if nan_count > 0:
                issues.append(f"Audio contains {nan_count} NaN values")
                return False, issues
                
            inf_count = np.sum(np.isinf(audio_buffer))
            if inf_count > 0:
                issues.append(f"Audio contains {inf_count} infinite values")
                return False, issues
            
            # DC offset validation
            dc_offset = np.mean(audio_buffer)
            if abs(dc_offset) > self.config.max_dc_offset:
                issues.append(f"DC offset {dc_offset:.3f} exceeds limit {self.config.max_dc_offset}")
                # This is a warning, not a failure
            
            # Dynamic range validation
            dynamic_range = max_amplitude / (np.std(audio_buffer) + 1e-8)
            if dynamic_range > 1000:
                issues.append(f"Suspiciously high dynamic range: {dynamic_range:.1f}")
            
            # Clipping detection
            clipping_threshold = 0.95 * max_amplitude
            clipped_samples = np.sum(np.abs(audio_buffer) >= clipping_threshold)
            if clipped_samples > len(audio_buffer) * 0.1:  # More than 10% clipped
                issues.append(f"Possible clipping detected: {clipped_samples}/{len(audio_buffer)} samples")
            
            # Frequency domain validation
            if len(audio_buffer) >= 64:  # Minimum for meaningful FFT
                fft = np.fft.rfft(audio_buffer)
                magnitude = np.abs(fft)
                freqs = np.fft.rfftfreq(len(audio_buffer), 1/sample_rate)
                
                # Check for suspicious frequency content
                nyquist = sample_rate / 2
                high_freq_energy = np.sum(magnitude[freqs > nyquist * 0.8]) / np.sum(magnitude)
                if high_freq_energy > 0.5:
                    issues.append(f"Suspicious high-frequency content: {high_freq_energy:.2%}")
                
                # Check for artificial patterns
                peak_indices = np.where(magnitude > np.max(magnitude) * 0.8)[0]
                if len(peak_indices) == 1 and np.max(magnitude) > 100 * np.median(magnitude):
                    issues.append("Possible artificial tone detected")
            
            # Update statistics
            self.validation_stats['samples_validated'] += 1
            if issues:
                self.validation_stats['samples_rejected'] += 1
                for issue in issues:
                    category = issue.split()[0]  # First word as category
                    self.validation_stats['common_issues'][category] = self.validation_stats['common_issues'].get(category, 0) + 1
            
            # Validation passed if no critical issues
            critical_keywords = ['exceeds', 'below', 'empty', 'NaN', 'infinite']
            has_critical_issues = any(keyword in issue for issue in issues for keyword in critical_keywords)
            
            return not has_critical_issues, issues
            
        except Exception as e:
            logger.error(f"Audio validation failed with exception: {e}")
            issues.append(f"Validation exception: {str(e)}")
            return False, issues
    
    def sanitize_audio_buffer(self, audio_buffer: np.ndarray, sample_rate: int = 16000) -> Tuple[np.ndarray, List[str]]:
        """
        Sanitize audio buffer by fixing common issues.
        
        Args:
            audio_buffer: Audio samples to sanitize
            sample_rate: Sample rate in Hz
            
        Returns:
            Tuple of (sanitized_audio, list_of_fixes_applied)
        """
        fixes_applied = []
        sanitized = audio_buffer.copy()
        
        try:
            # Fix NaN values
            nan_mask = np.isnan(sanitized)
            if np.any(nan_mask):
                sanitized[nan_mask] = 0.0
                fixes_applied.append(f"Replaced {np.sum(nan_mask)} NaN values with zeros")
            
            # Fix infinite values
            inf_mask = np.isinf(sanitized)
            if np.any(inf_mask):
                sanitized[inf_mask] = np.sign(sanitized[inf_mask]) * self.config.max_amplitude * 0.9
                fixes_applied.append(f"Clipped {np.sum(inf_mask)} infinite values")
            
            # Clamp amplitude
            max_amp = np.max(np.abs(sanitized))
            if max_amp > self.config.max_amplitude:
                scale_factor = self.config.max_amplitude * 0.9 / max_amp
                sanitized *= scale_factor
                fixes_applied.append(f"Scaled amplitude by {scale_factor:.3f} to prevent overflow")
            
            # Remove DC offset
            dc_offset = np.mean(sanitized)
            if abs(dc_offset) > self.config.max_dc_offset:
                sanitized -= dc_offset
                fixes_applied.append(f"Removed DC offset of {dc_offset:.3f}")
            
            # Apply soft limiting to prevent harsh clipping
            soft_limit_threshold = self.config.max_amplitude * 0.8
            over_threshold = np.abs(sanitized) > soft_limit_threshold
            if np.any(over_threshold):
                # Apply soft limiting using tanh
                sanitized[over_threshold] = np.sign(sanitized[over_threshold]) * soft_limit_threshold * np.tanh(
                    np.abs(sanitized[over_threshold]) / soft_limit_threshold
                )
                fixes_applied.append(f"Applied soft limiting to {np.sum(over_threshold)} samples")
            
            # Normalize if signal is too quiet
            if max_amp < self.config.min_amplitude * 10:
                if max_amp > 0:
                    scale_factor = self.config.min_amplitude * 10 / max_amp
                    sanitized *= scale_factor
                    fixes_applied.append(f"Amplified quiet signal by {scale_factor:.1f}x")
            
            if fixes_applied:
                self.validation_stats['samples_sanitized'] += 1
            
            return sanitized, fixes_applied
            
        except Exception as e:
            logger.error(f"Audio sanitization failed: {e}")
            return audio_buffer, [f"Sanitization failed: {str(e)}"]

class ModelValidator:
    """Validation for model parameters and configurations."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
    
    def validate_model_config(self, model_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate model configuration parameters."""
        issues = []
        
        # Required fields
        required_fields = ['input_dim', 'hidden_dim', 'output_dim', 'sample_rate']
        for field in required_fields:
            if field not in model_config:
                issues.append(f"Missing required field: {field}")
                continue
            
            value = model_config[field]
            if not isinstance(value, int) or value <= 0:
                issues.append(f"Field {field} must be positive integer, got {value}")
        
        # Validate dimensions
        if 'input_dim' in model_config:
            input_dim = model_config['input_dim']
            if input_dim > 1024:
                issues.append(f"Input dimension {input_dim} may be too large for embedded deployment")
            elif input_dim < 8:
                issues.append(f"Input dimension {input_dim} may be too small for meaningful features")
        
        if 'hidden_dim' in model_config:
            hidden_dim = model_config['hidden_dim']
            if hidden_dim > 512:
                issues.append(f"Hidden dimension {hidden_dim} may exceed memory constraints")
            elif hidden_dim < 16:
                issues.append(f"Hidden dimension {hidden_dim} may be too small for complex patterns")
        
        if 'output_dim' in model_config:
            output_dim = model_config['output_dim']
            if output_dim > 100:
                issues.append(f"Output dimension {output_dim} seems unusually large")
            elif output_dim < 2:
                issues.append(f"Output dimension {output_dim} must be at least 2 for classification")
        
        # Validate sample rate
        if 'sample_rate' in model_config:
            sample_rate = model_config['sample_rate']
            if sample_rate not in [8000, 16000, 22050, 44100, 48000]:
                issues.append(f"Unusual sample rate {sample_rate}Hz - consider standard rates")
        
        # Validate memory requirements
        if all(field in model_config for field in ['input_dim', 'hidden_dim', 'output_dim']):
            total_params = (model_config['input_dim'] * model_config['hidden_dim'] + 
                           model_config['hidden_dim'] * model_config['hidden_dim'] +
                           model_config['hidden_dim'] * model_config['output_dim'])
            memory_mb = (total_params * 4) / (1024 * 1024)  # 4 bytes per float32
            
            if memory_mb > self.config.max_model_size_mb:
                issues.append(f"Model size {memory_mb:.1f}MB exceeds limit {self.config.max_model_size_mb}MB")
        
        return len(issues) == 0, issues
    
    def validate_adaptive_config(self, adaptive_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate adaptive timestep configuration."""
        issues = []
        
        if 'min_timestep' in adaptive_config and 'max_timestep' in adaptive_config:
            min_ts = adaptive_config['min_timestep']
            max_ts = adaptive_config['max_timestep']
            
            if min_ts >= max_ts:
                issues.append(f"min_timestep ({min_ts}) must be less than max_timestep ({max_ts})")
            
            if min_ts <= 0:
                issues.append(f"min_timestep ({min_ts}) must be positive")
            
            if max_ts > 1.0:
                issues.append(f"max_timestep ({max_ts}) seems too large (>1 second)")
            
            if min_ts < 0.001:
                issues.append(f"min_timestep ({min_ts}) may be too small for stable integration")
        
        if 'energy_threshold' in adaptive_config:
            threshold = adaptive_config['energy_threshold']
            if not 0 <= threshold <= 1:
                issues.append(f"energy_threshold ({threshold}) should be between 0 and 1")
        
        return len(issues) == 0, issues

class SecurityValidator:
    """Security validation for inputs and model files."""
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
    
    def validate_file_path(self, file_path: Union[str, Path]) -> Tuple[bool, List[str]]:
        """Validate file path for security issues."""
        issues = []
        path = Path(file_path)
        
        # Path traversal protection
        try:
            resolved_path = path.resolve()
            if '..' in str(path) or str(resolved_path) != str(path.resolve(strict=False)):
                issues.append("Potential path traversal detected")
        except Exception as e:
            issues.append(f"Path resolution failed: {e}")
        
        # File extension validation
        if path.suffix.lower() not in self.config.allowed_extensions:
            issues.append(f"File extension {path.suffix} not in allowed list: {self.config.allowed_extensions}")
        
        # File size validation
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > self.config.max_file_size_mb:
                issues.append(f"File size {size_mb:.1f}MB exceeds limit {self.config.max_file_size_mb}MB")
        
        return len(issues) == 0, issues
    
    def validate_model_file(self, file_data: bytes) -> Tuple[bool, List[str]]:
        """Validate model file content for security issues."""
        issues = []
        
        # File size check
        size_mb = len(file_data) / (1024 * 1024)
        if size_mb > self.config.max_file_size_mb:
            issues.append(f"File size {size_mb:.1f}MB exceeds limit")
        
        # Magic number check for .lnn files
        if len(file_data) >= 4:
            magic = file_data[:4]
            if magic != b'LNN\x01':
                issues.append("Invalid magic number - not a valid LNN file")
        else:
            issues.append("File too small to be valid")
        
        # Basic entropy check (detect highly compressed or encrypted content)
        if len(file_data) >= 1024 and self.config.enable_content_scanning:
            entropy = self._calculate_entropy(file_data[:1024])
            if entropy > 7.5:  # Very high entropy might indicate encryption/compression
                issues.append(f"Suspicious file entropy: {entropy:.2f}")
        
        return len(issues) == 0, issues
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of data."""
        if len(data) == 0:
            return 0.0
        
        # Count byte frequencies
        counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
        probabilities = counts / len(data)
        
        # Calculate entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy

class ValidationReporter:
    """Generate validation reports and recommendations."""
    
    def __init__(self):
        self.validation_results = []
    
    def add_result(self, validator_type: str, is_valid: bool, issues: List[str], metadata: Dict[str, Any] = None):
        """Add validation result to report."""
        self.validation_results.append({
            'validator_type': validator_type,
            'is_valid': is_valid,
            'issues': issues,
            'metadata': metadata or {},
            'timestamp': np.datetime64('now')
        })
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        if not self.validation_results:
            return {'status': 'no_validations_performed'}
        
        total_validations = len(self.validation_results)
        passed_validations = sum(1 for result in self.validation_results if result['is_valid'])
        
        # Categorize issues
        issue_categories = {}
        for result in self.validation_results:
            for issue in result['issues']:
                category = result['validator_type']
                if category not in issue_categories:
                    issue_categories[category] = []
                issue_categories[category].append(issue)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issue_categories)
        
        return {
            'summary': {
                'total_validations': total_validations,
                'passed_validations': passed_validations,
                'success_rate': passed_validations / total_validations,
                'failed_validations': total_validations - passed_validations
            },
            'issue_categories': issue_categories,
            'recommendations': recommendations,
            'detailed_results': self.validation_results
        }
    
    def _generate_recommendations(self, issue_categories: Dict[str, List[str]]) -> List[str]:
        """Generate recommendations based on validation issues."""
        recommendations = []
        
        for category, issues in issue_categories.items():
            if category == 'audio':
                if any('amplitude' in issue for issue in issues):
                    recommendations.append("Consider normalizing audio amplitude before processing")
                if any('NaN' in issue for issue in issues):
                    recommendations.append("Implement NaN detection and replacement in audio pipeline")
                if any('clipping' in issue for issue in issues):
                    recommendations.append("Add clipping detection and soft limiting")
            
            elif category == 'model':
                if any('dimension' in issue for issue in issues):
                    recommendations.append("Review model architecture for embedded deployment constraints")
                if any('memory' in issue for issue in issues):
                    recommendations.append("Consider model quantization or pruning to reduce memory usage")
            
            elif category == 'security':
                if any('path traversal' in issue for issue in issues):
                    recommendations.append("Implement strict path validation and sandboxing")
                if any('entropy' in issue for issue in issues):
                    recommendations.append("Add content analysis for uploaded model files")
        
        if not recommendations:
            recommendations.append("All validations passed - no specific recommendations")
        
        return recommendations

# Convenience functions for common validation tasks
def validate_audio(audio_buffer: np.ndarray, sample_rate: int = 16000, config: Optional[ValidationConfig] = None) -> Dict[str, Any]:
    """Convenience function to validate audio with comprehensive reporting."""
    validator = AudioValidator(config)
    reporter = ValidationReporter()
    
    # Validate
    is_valid, issues = validator.validate_audio_buffer(audio_buffer, sample_rate)
    reporter.add_result('audio', is_valid, issues, {
        'buffer_length': len(audio_buffer),
        'sample_rate': sample_rate,
        'duration_s': len(audio_buffer) / sample_rate,
        'max_amplitude': float(np.max(np.abs(audio_buffer)))
    })
    
    # Sanitize if needed
    sanitized_audio = None
    fixes_applied = []
    if not is_valid or issues:
        sanitized_audio, fixes_applied = validator.sanitize_audio_buffer(audio_buffer, sample_rate)
        if fixes_applied:
            reporter.add_result('audio_sanitization', True, [], {'fixes': fixes_applied})
    
    return {
        'is_valid': is_valid,
        'issues': issues,
        'sanitized_audio': sanitized_audio,
        'fixes_applied': fixes_applied,
        'validation_stats': validator.validation_stats,
        'report': reporter.generate_report()
    }

def validate_model_config(config: Dict[str, Any], validation_config: Optional[ValidationConfig] = None) -> Dict[str, Any]:
    """Convenience function to validate model configuration."""
    validator = ModelValidator(validation_config)
    reporter = ValidationReporter()
    
    is_valid, issues = validator.validate_model_config(config)
    reporter.add_result('model_config', is_valid, issues, config)
    
    return {
        'is_valid': is_valid,
        'issues': issues,
        'report': reporter.generate_report()
    }