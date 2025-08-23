"""Generation 2: Comprehensive validation and error handling for Liquid Neural Networks."""

from typing import Any, Dict, List, Optional, Tuple, Union
import math
from enum import Enum
from dataclasses import dataclass

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


class ValidationSeverity(Enum):
    """Severity levels for validation warnings and errors."""
    INFO = "info"
    WARNING = "warning"  
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    passed: bool
    severity: ValidationSeverity
    message: str
    suggestion: Optional[str] = None


class AudioValidator:
    """Comprehensive audio input validation."""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
    def validate_audio_buffer(self, audio_buffer: Union[List, Any]) -> List[ValidationResult]:
        """Validate audio buffer input."""
        results = []
        
        # Convert to list if needed
        if HAS_NUMPY and hasattr(audio_buffer, 'tolist'):
            audio_data = audio_buffer.tolist()
        else:
            audio_data = list(audio_buffer) if hasattr(audio_buffer, '__iter__') else [audio_buffer]
        
        # Check buffer length
        if len(audio_data) == 0:
            results.append(ValidationResult(
                passed=False,
                severity=ValidationSeverity.ERROR,
                message="Audio buffer is empty",
                suggestion="Provide non-empty audio data"
            ))
            return results
        
        # Check for valid numeric values
        invalid_count = 0
        for sample in audio_data:
            try:
                val = float(sample)
                if math.isinf(val) or math.isnan(val):
                    invalid_count += 1
            except (ValueError, TypeError):
                invalid_count += 1
        
        if invalid_count > 0:
            results.append(ValidationResult(
                passed=False,
                severity=ValidationSeverity.ERROR,
                message=f"Found {invalid_count} invalid values in audio buffer",
                suggestion="Remove or fix invalid values before processing"
            ))
        
        # If no issues found, add success result  
        if not any(not r.passed for r in results):
            results.append(ValidationResult(
                passed=True,
                severity=ValidationSeverity.INFO,
                message="Audio buffer validation passed"
            ))
        
        return results


def validate_lnn_input(audio_buffer: Union[List, Any], 
                      config: Optional[Dict] = None) -> Tuple[bool, List[str]]:
    """Quick validation of LNN input.
    
    Returns:
        (is_valid, error_messages)
    """
    validator = AudioValidator()
    results = validator.validate_audio_buffer(audio_buffer)
    
    # Extract error issues
    errors = []
    is_valid = True
    
    for result in results:
        if not result.passed and result.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            errors.append(result.message)
            is_valid = False
    
    return is_valid, errors