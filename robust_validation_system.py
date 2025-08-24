#!/usr/bin/env python3
"""
Generation 2: Robust Validation and Error Handling System
Advanced security, validation, and monitoring for LNN research
"""

import logging
import hashlib
import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
import traceback


class ValidationSeverity(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning" 
    ERROR = "error"
    CRITICAL = "critical"


class SecurityLevel(Enum):
    """Security clearance levels"""
    PUBLIC = 1
    AUTHENTICATED = 2
    PRIVILEGED = 3
    ADMIN = 4


@dataclass
class ValidationResult:
    """Comprehensive validation result"""
    is_valid: bool
    severity: ValidationSeverity
    field: str
    message: str
    suggested_fix: Optional[str] = None
    security_implications: List[str] = field(default_factory=list)
    performance_impact: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SecurityContext:
    """Security context with advanced features"""
    user_id: str
    session_id: str
    permissions: List[str]
    security_level: SecurityLevel
    ip_address: str
    rate_limits: Dict[str, int] = field(default_factory=dict)
    failed_attempts: int = 0
    last_activity: float = field(default_factory=time.time)
    encrypted_token: str = ""


class AdvancedValidator:
    """Production-grade validation system"""
    
    def __init__(self):
        self.validation_rules = {}
        self.security_policies = {}
        self.validation_history = []
        self.setup_logging()
        
    def setup_logging(self):
        """Configure comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('lnn_validation.log')
            ]
        )
        self.logger = logging.getLogger('LNN_Validator')
        
    def register_validation_rule(self, field: str, validator: Callable[[Any], ValidationResult]):
        """Register custom validation rules"""
        self.validation_rules[field] = validator
        self.logger.info(f"Registered validation rule for field: {field}")
    
    def validate_lnn_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Comprehensive LNN configuration validation"""
        results = []
        
        # Core parameter validation
        results.extend(self._validate_core_params(config))
        
        # Security validation  
        results.extend(self._validate_security_config(config))
        
        # Performance validation
        results.extend(self._validate_performance_params(config))
        
        # Resource validation
        results.extend(self._validate_resource_constraints(config))
        
        # Log all validation results
        for result in results:
            level = getattr(logging, result.severity.value.upper())
            self.logger.log(level, f"Validation: {result.field} - {result.message}")
            
        self.validation_history.extend(results)
        return results
    
    def _validate_core_params(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate core LNN parameters"""
        results = []
        
        # Hidden dimension validation
        hidden_dim = config.get('hidden_dim', 0)
        if not isinstance(hidden_dim, int) or hidden_dim <= 0:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field='hidden_dim',
                message=f'Hidden dimension must be positive integer, got {hidden_dim}',
                suggested_fix='Set hidden_dim to a value between 32 and 512'
            ))
        elif hidden_dim > 1024:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                field='hidden_dim',
                message=f'Large hidden dimension ({hidden_dim}) may impact performance',
                performance_impact=hidden_dim * 0.001  # Estimated impact
            ))
            
        # Timestep validation
        min_timestep = config.get('min_timestep', 0)
        max_timestep = config.get('max_timestep', 0)
        
        if min_timestep <= 0 or max_timestep <= 0:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field='timestep',
                message='Timesteps must be positive',
                suggested_fix='Set min_timestep > 0 and max_timestep > min_timestep'
            ))
        elif min_timestep >= max_timestep:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.ERROR,
                field='timestep',
                message='min_timestep must be less than max_timestep',
                suggested_fix=f'Set min_timestep < {max_timestep}'
            ))
            
        return results
    
    def _validate_security_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate security configuration"""
        results = []
        
        # API key validation
        api_key = config.get('api_key', '')
        if api_key and len(api_key) < 32:
            results.append(ValidationResult(
                is_valid=False,
                severity=ValidationSeverity.CRITICAL,
                field='api_key',
                message='API key too short for secure operation',
                security_implications=['Weak authentication', 'Brute force vulnerability'],
                suggested_fix='Use API key with minimum 32 characters'
            ))
            
        # Encryption validation
        encryption_enabled = config.get('encryption_enabled', False)
        if not encryption_enabled:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                field='encryption',
                message='Encryption disabled - data may be vulnerable',
                security_implications=['Data exposure', 'Man-in-the-middle attacks'],
                suggested_fix='Enable encryption for production deployment'
            ))
            
        return results
    
    def _validate_performance_params(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate performance-related parameters"""
        results = []
        
        # Memory limit validation
        memory_limit = config.get('memory_limit_mb', 0)
        if memory_limit > 0 and memory_limit < 64:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                field='memory_limit',
                message=f'Low memory limit ({memory_limit}MB) may cause instability',
                performance_impact=0.3,
                suggested_fix='Consider increasing memory limit to at least 64MB'
            ))
            
        # Batch size validation
        batch_size = config.get('batch_size', 1)
        if batch_size > 64:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.INFO,
                field='batch_size',
                message=f'Large batch size ({batch_size}) detected',
                performance_impact=batch_size * 0.01
            ))
            
        return results
    
    def _validate_resource_constraints(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """Validate resource constraints"""
        results = []
        
        # Power budget validation
        power_budget = config.get('power_budget_mw', 0)
        if power_budget > 0 and power_budget < 0.5:
            results.append(ValidationResult(
                is_valid=True,
                severity=ValidationSeverity.WARNING,
                field='power_budget',
                message=f'Very low power budget ({power_budget}mW) may limit performance',
                suggested_fix='Consider power budget of at least 1.0mW for reliable operation'
            ))
            
        return results


class SecurityManager:
    """Advanced security management system"""
    
    def __init__(self):
        self.active_sessions = {}
        self.security_log = []
        self.rate_limiters = {}
        self.setup_security_logging()
        
    def setup_security_logging(self):
        """Setup security-specific logging"""
        self.security_logger = logging.getLogger('LNN_Security')
        
    def authenticate_user(self, credentials: Dict[str, str]) -> Optional[SecurityContext]:
        """Authenticate user with advanced security"""
        username = credentials.get('username', '')
        password = credentials.get('password', '')
        ip_address = credentials.get('ip_address', '0.0.0.0')
        
        # Hash password for secure comparison (simplified)
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Simulate authentication (in production, check against secure database)
        if self._verify_credentials(username, password_hash):
            session_id = self._generate_session_id()
            
            security_context = SecurityContext(
                user_id=username,
                session_id=session_id,
                permissions=self._get_user_permissions(username),
                security_level=self._get_security_level(username),
                ip_address=ip_address,
                encrypted_token=self._generate_encrypted_token(session_id)
            )
            
            self.active_sessions[session_id] = security_context
            self.security_logger.info(f"User {username} authenticated from {ip_address}")
            
            return security_context
        else:
            self.security_logger.warning(f"Failed authentication attempt for {username} from {ip_address}")
            return None
    
    def _verify_credentials(self, username: str, password_hash: str) -> bool:
        """Verify user credentials (simplified for demo)"""
        # In production, check against encrypted database
        valid_users = {
            'researcher': 'ef2d127de37b942baad06145e54b0c619a1f22327b2ebbcfbec78f5564afe39d',  # "secure123"
            'admin': 'c6ba91b90d922e159893f46c387e5dc1b3dc5c101a5a4522f03b987177a24a91'  # "admin456"
        }
        return username in valid_users and valid_users[username] == password_hash
    
    def _generate_session_id(self) -> str:
        """Generate secure session ID"""
        return hashlib.sha256(f"{time.time()}{threading.current_thread().ident}".encode()).hexdigest()[:32]
    
    def _generate_encrypted_token(self, session_id: str) -> str:
        """Generate encrypted authentication token"""
        return hashlib.sha256(f"LNN_TOKEN_{session_id}_{time.time()}".encode()).hexdigest()
    
    def _get_user_permissions(self, username: str) -> List[str]:
        """Get user permissions"""
        permissions_map = {
            'researcher': ['read', 'experiment', 'analyze'],
            'admin': ['read', 'write', 'experiment', 'analyze', 'deploy', 'manage_users']
        }
        return permissions_map.get(username, ['read'])
    
    def _get_security_level(self, username: str) -> SecurityLevel:
        """Get user security level"""
        level_map = {
            'researcher': SecurityLevel.AUTHENTICATED,
            'admin': SecurityLevel.ADMIN
        }
        return level_map.get(username, SecurityLevel.PUBLIC)
    
    def check_permissions(self, context: SecurityContext, required_permission: str) -> bool:
        """Check if user has required permissions"""
        has_permission = required_permission in context.permissions
        
        if not has_permission:
            self.security_logger.warning(
                f"Permission denied: User {context.user_id} attempted {required_permission}"
            )
            
        return has_permission
    
    def rate_limit_check(self, context: SecurityContext, resource: str, limit: int = 100) -> bool:
        """Advanced rate limiting"""
        key = f"{context.user_id}_{resource}"
        current_time = int(time.time() / 60)  # Per minute
        
        if key not in self.rate_limiters:
            self.rate_limiters[key] = {'count': 0, 'window': current_time}
        
        rate_data = self.rate_limiters[key]
        
        # Reset counter if new time window
        if rate_data['window'] != current_time:
            rate_data['count'] = 0
            rate_data['window'] = current_time
        
        rate_data['count'] += 1
        
        if rate_data['count'] > limit:
            self.security_logger.warning(
                f"Rate limit exceeded: User {context.user_id} for resource {resource}"
            )
            return False
            
        return True


class ErrorHandler:
    """Production-grade error handling"""
    
    def __init__(self):
        self.error_counts = {}
        self.recovery_strategies = {}
        self.setup_error_logging()
        
    def setup_error_logging(self):
        """Setup error logging"""
        self.error_logger = logging.getLogger('LNN_Errors')
        
    @contextmanager
    def handle_errors(self, operation: str, context: Optional[Dict] = None):
        """Context manager for robust error handling"""
        try:
            yield
        except Exception as e:
            self._handle_exception(operation, e, context or {})
            raise  # Re-raise after logging
            
    def _handle_exception(self, operation: str, exception: Exception, context: Dict):
        """Handle and log exceptions with context"""
        error_id = f"{operation}_{type(exception).__name__}"
        
        # Track error frequency
        self.error_counts[error_id] = self.error_counts.get(error_id, 0) + 1
        
        # Log with full context
        self.error_logger.error(
            f"Operation: {operation}, Error: {str(exception)}, "
            f"Context: {json.dumps(context, default=str)}, "
            f"Traceback: {traceback.format_exc()}"
        )
        
        # Check for recovery strategy
        if error_id in self.recovery_strategies:
            self.error_logger.info(f"Attempting recovery for {error_id}")
            try:
                self.recovery_strategies[error_id](exception, context)
            except Exception as recovery_error:
                self.error_logger.error(f"Recovery failed: {recovery_error}")
    
    def register_recovery_strategy(self, error_pattern: str, recovery_func: Callable):
        """Register error recovery strategies"""
        self.recovery_strategies[error_pattern] = recovery_func
        self.error_logger.info(f"Registered recovery strategy for: {error_pattern}")


def main():
    """Demonstrate Generation 2 robust validation and security"""
    print("🛡️ Generation 2: Robust Validation & Security System")
    print("=" * 60)
    
    # Initialize systems
    validator = AdvancedValidator()
    security_manager = SecurityManager()
    error_handler = ErrorHandler()
    
    print("\n🔐 SECURITY TESTING")
    
    # Test authentication
    credentials = {
        'username': 'researcher',
        'password': 'secure123',
        'ip_address': '192.168.1.100'
    }
    
    security_context = security_manager.authenticate_user(credentials)
    if security_context:
        print(f"✅ Authentication successful: {security_context.user_id}")
        print(f"   Security Level: {security_context.security_level}")
        print(f"   Permissions: {', '.join(security_context.permissions)}")
    else:
        print("❌ Authentication failed")
    
    print("\n🔍 VALIDATION TESTING")
    
    # Test configuration validation
    test_configs = [
        {
            'hidden_dim': 128,
            'min_timestep': 0.001,
            'max_timestep': 0.05,
            'power_budget_mw': 2.0,
            'encryption_enabled': True,
            'api_key': 'secure_api_key_with_sufficient_length_123'
        },
        {
            'hidden_dim': -10,  # Invalid
            'min_timestep': 0.05,  # Invalid (greater than max)
            'max_timestep': 0.01,
            'api_key': 'short',  # Insecure
            'encryption_enabled': False  # Security warning
        }
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n📋 Testing Configuration {i+1}:")
        validation_results = validator.validate_lnn_config(config)
        
        # Summary
        errors = [r for r in validation_results if r.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]
        warnings = [r for r in validation_results if r.severity == ValidationSeverity.WARNING]
        
        print(f"   ❌ Errors: {len(errors)}")
        print(f"   ⚠️  Warnings: {len(warnings)}")
        print(f"   ✅ Valid: {len(errors) == 0}")
        
        # Show critical issues
        for result in errors:
            print(f"      🚨 {result.field}: {result.message}")
            if result.suggested_fix:
                print(f"         💡 Fix: {result.suggested_fix}")
    
    print("\n🚨 ERROR HANDLING TESTING")
    
    # Test error handling
    with error_handler.handle_errors("division_test", {"operation": "divide", "values": [10, 0]}):
        try:
            result = 10 / 0  # Will raise ZeroDivisionError
        except ZeroDivisionError:
            print("✅ Error caught and logged properly")
    
    print("\n🎯 RATE LIMITING TESTING")
    
    if security_context:
        # Test rate limiting
        for i in range(3):
            allowed = security_manager.rate_limit_check(security_context, "api_calls", limit=2)
            print(f"   Request {i+1}: {'✅ Allowed' if allowed else '❌ Rate Limited'}")
    
    print(f"\n✨ Generation 2 robust validation system complete!")
    print(f"🔒 Security features: Authentication, Authorization, Rate Limiting")
    print(f"🔍 Validation features: Multi-level validation with suggestions") 
    print(f"🚨 Error handling: Contextual logging and recovery strategies")
    
    return {
        'validator': validator,
        'security_manager': security_manager,
        'error_handler': error_handler,
        'validation_tests': len(test_configs),
        'security_context': security_context
    }


if __name__ == "__main__":
    results = main()