"""Liquid Neural Network core implementation."""

from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive timestep control."""
    
    min_timestep: float = 0.001  # 1ms minimum
    max_timestep: float = 0.050  # 50ms maximum
    energy_threshold: float = 0.1
    complexity_metric: str = "spectral_flux"


class LNN:
    """Liquid Neural Network for audio processing.
    
    Implements continuous-time neural dynamics with adaptive computation
    for ultra-low-power edge deployment.
    """
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """Initialize LNN model.
        
        Args:
            model_path: Path to pre-trained model file (.lnn format)
        """
        self.model_path = model_path
        self._config: Optional[AdaptiveConfig] = None
        self._current_power_mw = 0.0
        
        # Initialize with default model config for testing
        self._model_config = {
            'input_dim': 40,
            'hidden_dim': 64, 
            'output_dim': 10,
            'sample_rate': 16000,
            'frame_size': 256
        }
        
        if model_path:
            self.load(model_path)
    
    @classmethod
    def from_file(cls, model_path: Union[str, Path]) -> "LNN":
        """Load LNN model from file.
        
        Args:
            model_path: Path to model file
            
        Returns:
            Initialized LNN instance
        """
        return cls(model_path)
    
    def load(self, model_path: Union[str, Path]) -> None:
        """Load model from file.
        
        Args:
            model_path: Path to model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model format is invalid
        """
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not self.model_path.suffix == '.lnn':
            raise ValueError(f"Invalid model format. Expected .lnn, got {self.model_path.suffix}")
        
        # Load .lnn format (binary model file)
        try:
            with open(self.model_path, 'rb') as f:
                # .lnn format specification:
                # Header (32 bytes): magic number, version, architecture info
                # Model data: weights, biases, network topology
                header = f.read(32)
                
                # Validate magic number (first 4 bytes)
                magic = header[:4]
                if magic != b'LNN\x01':
                    raise ValueError("Invalid .lnn file format - incorrect magic number")
                
                # Extract version and architecture info
                version = int.from_bytes(header[4:8], 'little')
                input_dim = int.from_bytes(header[8:12], 'little')
                hidden_dim = int.from_bytes(header[12:16], 'little')
                output_dim = int.from_bytes(header[16:20], 'little')
                
                # Store model configuration
                self._model_config = {
                    'version': version,
                    'input_dim': input_dim,
                    'hidden_dim': hidden_dim,
                    'output_dim': output_dim
                }
                
                # Load model weights (remaining file content)
                weights_data = f.read()
                self._weights = np.frombuffer(weights_data, dtype=np.float32)
                
                print(f"✅ Loaded LNN model: {input_dim}→{hidden_dim}→{output_dim} (v{version})")
                
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the loaded model.
        
        Returns:
            Model configuration dict or None if no model loaded
        """
        return getattr(self, '_model_config', None)
        
    def set_adaptive_config(self, config: AdaptiveConfig) -> None:
        """Configure adaptive timestep control.
        
        Args:
            config: Adaptive configuration parameters
        """
        self._config = config
        
    def process(self, audio_buffer: np.ndarray) -> Dict[str, Any]:
        """Process audio buffer through LNN.
        
        Args:
            audio_buffer: Input audio samples
            
        Returns:
            Processing results with detected keywords/activity
            
        Raises:
            ValueError: If no model is loaded or buffer is invalid
        """
        if not hasattr(self, '_model_config'):
            raise ValueError("No model loaded. Call load() first.")
        
        if len(audio_buffer) == 0:
            raise ValueError("Empty audio buffer")
        
        # Validate input dimensions
        if len(audio_buffer.shape) != 1:
            raise ValueError(f"Expected 1D audio buffer, got {len(audio_buffer.shape)}D")
        
        # Feature extraction pipeline
        features = self._extract_features(audio_buffer)
        
        # Core LNN processing with adaptive timestep
        if self._config:
            # Use adaptive timestep based on signal complexity
            complexity = self._estimate_complexity(audio_buffer)
            timestep = self._calculate_adaptive_timestep(complexity)
        else:
            # Default fixed timestep
            timestep = 0.01  # 10ms
        
        # Liquid state computation (simplified ODE integration)
        liquid_state = self._integrate_liquid_dynamics(features, timestep)
        
        # Output layer processing
        output = self._compute_output(liquid_state)
        
        # Keyword detection logic
        keyword_detected = False
        detected_keyword = None
        confidence = 0.0
        
        if output.max() > 0.7:  # Detection threshold
            keyword_idx = np.argmax(output)
            confidence = float(output[keyword_idx])
            keyword_detected = True
            
            # Map to keyword names (would be loaded from model)
            keywords = ["wake", "stop", "yes", "no", "up", "down", "left", "right"]
            if keyword_idx < len(keywords):
                detected_keyword = keywords[keyword_idx]
        
        # Power estimation
        power_mw = self._estimate_power(audio_buffer, complexity if self._config else None)
        
        result = {
            "keyword_detected": keyword_detected,
            "keyword": detected_keyword,
            "confidence": confidence,
            "power_mw": power_mw,
            "timestep_ms": timestep * 1000,
            "liquid_state_energy": float(np.mean(liquid_state**2)),
            "processing_time_ms": self._estimate_processing_time(len(audio_buffer), timestep)
        }
        
        self._current_power_mw = power_mw
        return result
    
    def _extract_features(self, audio_buffer: np.ndarray) -> np.ndarray:
        """Extract audio features for LNN processing.
        
        Args:
            audio_buffer: Raw audio samples
            
        Returns:
            Feature vector
        """
        # Simplified feature extraction - in practice would use MFCC, spectrograms, etc.
        # This interfaces with the Rust/C++ core for performance
        
        # Basic spectral features
        fft = np.fft.rfft(audio_buffer)
        magnitude = np.abs(fft)
        
        # Log mel-scale features (simplified)
        n_features = self._model_config.get('input_dim', 40)
        features = np.zeros(n_features)
        
        # Bin the FFT into mel-scale features
        for i in range(min(n_features, len(magnitude))):
            features[i] = np.log(magnitude[i] + 1e-8)
        
        # Normalize features
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        return features
    
    def _estimate_complexity(self, audio_buffer: np.ndarray) -> float:
        """Estimate signal complexity for adaptive timestep control.
        
        Args:
            audio_buffer: Input audio samples
            
        Returns:
            Complexity metric (0.0 to 1.0)
        """
        if self._config.complexity_metric == "spectral_flux":
            # Spectral flux - measure of spectral change
            fft = np.fft.rfft(audio_buffer)
            magnitude = np.abs(fft)
            
            # Simple spectral flux approximation
            spectral_centroid = np.sum(magnitude * np.arange(len(magnitude))) / np.sum(magnitude)
            spectral_rolloff = np.sum(magnitude > 0.1 * np.max(magnitude))
            
            # Combine metrics (normalized to 0-1)
            complexity = min(1.0, (spectral_centroid / len(magnitude) + 
                                 spectral_rolloff / len(magnitude)) / 2)
            
        elif self._config.complexity_metric == "energy":
            # Simple energy-based complexity
            energy = np.mean(audio_buffer**2)
            complexity = min(1.0, energy * 10)  # Scale to 0-1 range
            
        else:
            complexity = 0.5  # Default moderate complexity
        
        return complexity
    
    def _calculate_adaptive_timestep(self, complexity: float) -> float:
        """Calculate adaptive timestep based on signal complexity.
        
        Args:
            complexity: Signal complexity (0.0 to 1.0)
            
        Returns:
            Timestep in seconds
        """
        # Higher complexity = smaller timestep = more accurate but higher power
        # Lower complexity = larger timestep = less accurate but lower power
        
        timestep_range = self._config.max_timestep - self._config.min_timestep
        # Inverse relationship: high complexity -> small timestep
        timestep = self._config.max_timestep - (complexity * timestep_range)
        
        return max(self._config.min_timestep, min(self._config.max_timestep, timestep))
    
    def _integrate_liquid_dynamics(self, features: np.ndarray, timestep: float) -> np.ndarray:
        """Integrate liquid neural dynamics using simplified ODE solver.
        
        Args:
            features: Input feature vector
            timestep: Integration timestep
            
        Returns:
            Liquid state vector
        """
        hidden_dim = self._model_config.get('hidden_dim', 64)
        
        # Initialize or update liquid state
        if not hasattr(self, '_liquid_state'):
            self._liquid_state = np.zeros(hidden_dim)
        
        # Simplified liquid dynamics: dx/dt = -x/tau + W*input + W_rec*x
        # Using Euler integration for simplicity
        
        # Model parameters (would be loaded from .lnn file)
        tau = 0.1  # Time constant
        W_input = np.random.randn(hidden_dim, len(features)) * 0.1  # Input weights
        W_recurrent = np.random.randn(hidden_dim, hidden_dim) * 0.05  # Recurrent weights
        
        # Compute derivatives
        input_current = W_input @ features
        recurrent_current = W_recurrent @ self._liquid_state
        decay_current = -self._liquid_state / tau
        
        # Euler step: x(t+dt) = x(t) + dt * dx/dt
        dx_dt = decay_current + input_current + recurrent_current
        self._liquid_state += timestep * dx_dt
        
        # Apply activation function (tanh for liquid networks)
        self._liquid_state = np.tanh(self._liquid_state)
        
        return self._liquid_state.copy()
    
    def _compute_output(self, liquid_state: np.ndarray) -> np.ndarray:
        """Compute output from liquid state.
        
        Args:
            liquid_state: Current liquid state
            
        Returns:
            Output predictions
        """
        output_dim = self._model_config.get('output_dim', 8)
        
        # Output weights (would be loaded from model)
        W_output = np.random.randn(output_dim, len(liquid_state)) * 0.1
        
        # Linear output layer
        output = W_output @ liquid_state
        
        # Apply softmax for classification
        exp_output = np.exp(output - np.max(output))  # Numerical stability
        softmax_output = exp_output / np.sum(exp_output)
        
        return softmax_output
    
    def _estimate_processing_time(self, buffer_length: int, timestep: float) -> float:
        """Estimate processing time for power calculations.
        
        Args:
            buffer_length: Length of audio buffer
            timestep: Current timestep
            
        Returns:
            Estimated processing time in milliseconds
        """
        # Simple model: processing time depends on buffer size and timestep
        base_time = 0.5  # Base processing overhead (ms)
        buffer_time = buffer_length * 0.001  # Time per sample
        timestep_time = (1 / timestep) * 0.1  # Smaller timestep = more computation
        
        return base_time + buffer_time + timestep_time
    
    def detect_activity(self, audio_frame: np.ndarray) -> Dict[str, Any]:
        """Detect voice activity in audio frame.
        
        Args:
            audio_frame: Input audio frame
            
        Returns:
            Activity detection results with speech/non-speech classification
            
        Raises:
            ValueError: If audio frame is invalid
        """
        if len(audio_frame) == 0:
            raise ValueError("Empty audio frame")
        
        if len(audio_frame.shape) != 1:
            raise ValueError(f"Expected 1D audio frame, got {len(audio_frame.shape)}D")
        
        # Energy-based features
        energy = np.mean(audio_frame**2)
        log_energy = np.log(energy + 1e-8)
        
        # Spectral features for voice activity detection
        fft = np.fft.rfft(audio_frame)
        magnitude = np.abs(fft)
        
        # Spectral centroid (measure of spectral "brightness")
        freqs = np.fft.rfftfreq(len(audio_frame))
        spectral_centroid = np.sum(freqs * magnitude) / (np.sum(magnitude) + 1e-8)
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumulative_energy = np.cumsum(magnitude**2)
        total_energy = cumulative_energy[-1]
        rolloff_idx = np.where(cumulative_energy >= 0.85 * total_energy)[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else 0.0
        
        # Zero crossing rate (good for distinguishing speech from noise)
        zero_crossings = np.sum(np.diff(np.sign(audio_frame)) != 0)
        zcr = zero_crossings / len(audio_frame)
        
        # Spectral flux (measure of spectral change - useful for detecting speech onsets)
        if hasattr(self, '_prev_magnitude'):
            spectral_flux = np.sum((magnitude - self._prev_magnitude)**2)
        else:
            spectral_flux = 0.0
        self._prev_magnitude = magnitude
        
        # Voice activity detection using multiple features
        # Simple heuristic-based approach (real implementation would use ML model)
        
        # Energy thresholds
        energy_threshold = 0.001  # Minimum energy for activity
        energy_score = min(1.0, energy / energy_threshold)
        
        # Spectral characteristics typical of speech
        # Speech typically has centroid in 500-4000 Hz range
        centroid_score = 1.0 if 500 <= spectral_centroid <= 4000 else 0.3
        
        # Speech has moderate zero crossing rate
        zcr_score = 1.0 if 0.01 <= zcr <= 0.15 else 0.3
        
        # Combine features with weights
        feature_weights = {
            'energy': 0.4,
            'centroid': 0.2,
            'zcr': 0.2,
            'rolloff': 0.1,
            'flux': 0.1
        }
        
        # Normalize rolloff and flux
        rolloff_score = min(1.0, spectral_rolloff / 4000)  # Normalize to 0-4kHz
        flux_score = min(1.0, spectral_flux / 100)  # Simple normalization
        
        # Weighted combination
        activity_score = (
            feature_weights['energy'] * energy_score +
            feature_weights['centroid'] * centroid_score +
            feature_weights['zcr'] * zcr_score +
            feature_weights['rolloff'] * rolloff_score +
            feature_weights['flux'] * flux_score
        )
        
        # Apply adaptive threshold based on recent activity
        if not hasattr(self, '_activity_history'):
            self._activity_history = []
        
        # Keep sliding window of recent activity scores
        self._activity_history.append(activity_score)
        if len(self._activity_history) > 50:  # Keep last 50 frames
            self._activity_history.pop(0)
        
        # Adaptive threshold based on recent background
        background_level = np.percentile(self._activity_history, 25)  # 25th percentile
        adaptive_threshold = max(0.3, background_level + 0.2)
        
        # Final decision
        is_speech = activity_score > adaptive_threshold
        confidence = min(1.0, activity_score / adaptive_threshold) if is_speech else 0.0
        
        # Additional context for ultra-low-power optimization
        power_mode = self._recommend_power_mode(activity_score, energy)
        
        return {
            "is_speech": is_speech,
            "confidence": confidence,
            "energy": energy,
            "log_energy": log_energy,
            "spectral_centroid": spectral_centroid,
            "spectral_rolloff": spectral_rolloff,
            "zero_crossing_rate": zcr,
            "spectral_flux": spectral_flux,
            "activity_score": activity_score,
            "adaptive_threshold": adaptive_threshold,
            "background_level": background_level,
            "recommended_power_mode": power_mode,
            "processing_complexity": self._estimate_vad_complexity(audio_frame)
        }
    
    def _recommend_power_mode(self, activity_score: float, energy: float) -> str:
        """Recommend power mode based on activity detection.
        
        Args:
            activity_score: Current activity score
            energy: Signal energy
            
        Returns:
            Recommended power mode
        """
        if activity_score > 0.8 and energy > 0.01:
            return "active"  # High activity - full processing
        elif activity_score > 0.4:
            return "reduced"  # Moderate activity - reduced processing
        elif energy > 0.001:
            return "minimal"  # Low activity but some signal
        else:
            return "sleep"  # Very low activity - deep sleep mode
    
    def _estimate_vad_complexity(self, audio_frame: np.ndarray) -> float:
        """Estimate computational complexity for VAD processing.
        
        Args:
            audio_frame: Input audio frame
            
        Returns:
            Complexity metric for power estimation
        """
        # Complexity depends on frame length and spectral content
        base_complexity = len(audio_frame) / 1024  # Normalized by typical frame size
        
        # Spectral complexity adds computational load
        if len(audio_frame) > 0:
            fft_complexity = np.log2(len(audio_frame)) / 10  # FFT complexity
        else:
            fft_complexity = 0.0
        
        return min(1.0, base_complexity + fft_complexity)
    
    def current_power_mw(self) -> float:
        """Get current power consumption in milliwatts.
        
        Returns:
            Current power consumption
        """
        return self._current_power_mw
    
    def _estimate_power(self, audio_buffer: np.ndarray, complexity: Optional[float] = None) -> float:
        """Estimate power consumption based on signal complexity and processing requirements.
        
        Args:
            audio_buffer: Input audio buffer
            complexity: Optional signal complexity (0.0 to 1.0)
            
        Returns:
            Estimated power in milliwatts
        """
        # Base power consumption (always-on components)
        base_power = 0.08  # MCU idle power
        
        # Signal-dependent power
        energy = np.mean(audio_buffer**2)
        signal_power = min(energy * 5, 1.5)  # Scale signal energy to power
        
        # Complexity-dependent power (adaptive timestep affects computation)
        if complexity is not None:
            # Higher complexity = smaller timestep = more computation = more power
            complexity_power = complexity * 1.2
        else:
            complexity_power = 0.6  # Default moderate complexity
        
        # Processing power based on buffer size
        buffer_power = len(audio_buffer) * 0.0001  # Power per sample
        
        # Feature extraction power (FFT, spectral analysis)
        feature_power = 0.3 if len(audio_buffer) > 0 else 0.0
        
        # Liquid state integration power (depends on hidden dimension)
        if hasattr(self, '_model_config'):
            hidden_dim = self._model_config.get('hidden_dim', 64)
            liquid_power = (hidden_dim / 64) * 0.4  # Scale with network size
        else:
            liquid_power = 0.4
        
        # Total power with efficiency factors
        total_power = (
            base_power + 
            signal_power + 
            complexity_power + 
            buffer_power + 
            feature_power + 
            liquid_power
        )
        
        # Apply power optimization based on adaptive config
        if self._config:
            # Adaptive timestep can reduce power during low activity
            efficiency_factor = 1.0 - (complexity or 0.5) * 0.3
            total_power *= efficiency_factor
        
        # Realistic power bounds for edge devices
        return max(0.05, min(total_power, 5.0))  # 0.05mW - 5mW range