"""
Reproducible Experimental Framework for Liquid Neural Networks Research.

This module provides comprehensive tools for conducting reproducible research,
including experiment configuration, dataset generation, result tracking,
and statistical analysis with proper controls.
"""

import numpy as np
import json
import hashlib
import time
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import pickle
import random
import os
from datetime import datetime
import platform
import psutil
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a reproducible experiment."""
    experiment_name: str
    description: str
    random_seed: int = 42
    data_config: Dict[str, Any] = field(default_factory=dict)
    model_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    optimization_config: Dict[str, Any] = field(default_factory=dict)
    hardware_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate unique experiment ID and set defaults."""
        if 'experiment_id' not in self.metadata:
            # Create deterministic ID based on config
            config_str = json.dumps(asdict(self), sort_keys=True)
            self.metadata['experiment_id'] = hashlib.md5(config_str.encode()).hexdigest()[:8]
        
        if 'timestamp' not in self.metadata:
            self.metadata['timestamp'] = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary."""
        return cls(**data)
    
    def save(self, filepath: Path) -> None:
        """Save configuration to file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'ExperimentConfig':
        """Load configuration from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


@dataclass
class SystemInfo:
    """System information for reproducibility."""
    platform: str
    architecture: str
    cpu_model: str
    cpu_cores: int
    memory_gb: float
    python_version: str
    numpy_version: str
    pytorch_version: Optional[str] = None
    gpu_info: Optional[Dict[str, Any]] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    
    @classmethod
    def collect(cls) -> 'SystemInfo':
        """Collect current system information."""
        import sys
        
        # Basic system info
        system_info = {
            'platform': platform.platform(),
            'architecture': platform.architecture()[0],
            'cpu_model': platform.processor() or 'Unknown',
            'cpu_cores': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': sys.version,
            'numpy_version': np.__version__,
        }
        
        # PyTorch version
        try:
            import torch
            system_info['pytorch_version'] = torch.__version__
            
            # GPU info
            if torch.cuda.is_available():
                system_info['gpu_info'] = {
                    'available': True,
                    'device_count': torch.cuda.device_count(),
                    'current_device': torch.cuda.current_device(),
                    'device_name': torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None,
                    'memory_total': torch.cuda.get_device_properties(0).total_memory if torch.cuda.device_count() > 0 else None
                }
            else:
                system_info['gpu_info'] = {'available': False}
        except ImportError:
            system_info['pytorch_version'] = None
            system_info['gpu_info'] = None
        
        # Relevant environment variables
        env_vars = {}
        important_vars = ['CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 
                         'PYTHONHASHSEED', 'PYTHONPATH']
        for var in important_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]
        
        system_info['environment_variables'] = env_vars
        
        return cls(**system_info)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    experiment_id: str
    config: ExperimentConfig
    system_info: SystemInfo
    results: Dict[str, Any]
    metrics: Dict[str, float]
    artifacts: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    def save(self, filepath: Path) -> None:
        """Save results to file."""
        # Convert to JSON-serializable format
        data = {
            'experiment_id': self.experiment_id,
            'config': self.config.to_dict(),
            'system_info': asdict(self.system_info),
            'results': self.results,
            'metrics': self.metrics,
            'execution_time': self.execution_time,
            'success': self.success,
            'error_message': self.error_message,
            # Artifacts are saved separately due to size
            'artifacts_available': list(self.artifacts.keys())
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Save artifacts separately
        artifacts_dir = filepath.parent / f"{filepath.stem}_artifacts"
        if self.artifacts:
            artifacts_dir.mkdir(exist_ok=True)
            for name, artifact in self.artifacts.items():
                artifact_path = artifacts_dir / f"{name}.pkl"
                with open(artifact_path, 'wb') as f:
                    pickle.dump(artifact, f)
    
    @classmethod
    def load(cls, filepath: Path) -> 'ExperimentResult':
        """Load results from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load artifacts
        artifacts = {}
        artifacts_dir = filepath.parent / f"{filepath.stem}_artifacts"
        if artifacts_dir.exists():
            for artifact_name in data.get('artifacts_available', []):
                artifact_path = artifacts_dir / f"{artifact_name}.pkl"
                if artifact_path.exists():
                    with open(artifact_path, 'rb') as f:
                        artifacts[artifact_name] = pickle.load(f)
        
        return cls(
            experiment_id=data['experiment_id'],
            config=ExperimentConfig.from_dict(data['config']),
            system_info=SystemInfo(**data['system_info']),
            results=data['results'],
            metrics=data['metrics'],
            artifacts=artifacts,
            execution_time=data['execution_time'],
            success=data['success'],
            error_message=data.get('error_message')
        )


class ReproducibilityManager:
    """Manages reproducibility aspects of experiments."""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.seed_counter = 0
        
    def set_global_seed(self, seed: int) -> None:
        """Set global random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Set PyTorch seeds if available
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except ImportError:
            pass
    
    def get_next_seed(self) -> int:
        """Get next deterministic seed for subsystems."""
        self.seed_counter += 1
        return self.base_seed + self.seed_counter
    
    def create_data_split_seeds(self, n_splits: int) -> List[int]:
        """Create deterministic seeds for data splits."""
        return [self.get_next_seed() for _ in range(n_splits)]
    
    def create_model_init_seed(self) -> int:
        """Create seed for model initialization."""
        return self.get_next_seed()
    
    def create_training_seed(self) -> int:
        """Create seed for training process."""
        return self.get_next_seed()
    
    def verify_reproducibility(self, experiment_func: Callable, 
                             config: ExperimentConfig, n_runs: int = 3) -> Dict[str, Any]:
        """Verify that experiment is reproducible across multiple runs."""
        results = []
        
        for run in range(n_runs):
            logger.info(f"Reproducibility check run {run + 1}/{n_runs}")
            
            # Reset seeds
            self.seed_counter = 0
            self.set_global_seed(config.random_seed)
            
            # Run experiment
            try:
                result = experiment_func(config)
                results.append(result)
            except Exception as e:
                logger.error(f"Reproducibility check failed on run {run + 1}: {e}")
                return {'reproducible': False, 'error': str(e)}
        
        # Compare results
        if len(results) < 2:
            return {'reproducible': False, 'error': 'Insufficient runs'}
        
        # Check if key metrics are identical
        first_metrics = results[0].metrics
        reproducible = True
        differences = {}
        
        for key, value in first_metrics.items():
            values = [r.metrics.get(key) for r in results]
            
            if not all(v is not None for v in values):
                reproducible = False
                differences[key] = f"Missing values: {values}"
                continue
            
            # Check for numerical differences
            if isinstance(value, (int, float)):
                max_diff = max(values) - min(values)
                rel_diff = max_diff / abs(np.mean(values)) if np.mean(values) != 0 else max_diff
                
                if rel_diff > 1e-6:  # Tolerance for floating point errors
                    reproducible = False
                    differences[key] = f"Values: {values}, max_diff: {max_diff}, rel_diff: {rel_diff}"
            else:
                # For non-numeric values, check exact equality
                if not all(v == value for v in values):
                    reproducible = False
                    differences[key] = f"Values: {values}"
        
        return {
            'reproducible': reproducible,
            'n_runs': n_runs,
            'differences': differences,
            'results': results
        }


class DatasetGenerator:
    """Generate synthetic datasets for audio processing experiments."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        
    def generate_audio_classification_dataset(self, 
                                            n_samples: int = 1000,
                                            n_features: int = 40,
                                            n_classes: int = 8,
                                            noise_level: float = 0.1,
                                            class_separation: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic audio classification dataset."""
        # Create class centers in feature space
        class_centers = np.random.randn(n_classes, n_features) * class_separation
        
        # Generate samples
        X = []
        y = []
        
        samples_per_class = n_samples // n_classes
        
        for class_idx in range(n_classes):
            # Generate samples around class center
            center = class_centers[class_idx]
            
            for _ in range(samples_per_class):
                # Base sample from class center
                sample = center + np.random.randn(n_features) * noise_level
                
                # Add some audio-like features
                sample = self._add_audio_characteristics(sample)
                
                X.append(sample)
                y.append(class_idx)
        
        # Add remaining samples to reach exact count
        remaining = n_samples - len(X)
        for _ in range(remaining):
            class_idx = np.random.randint(n_classes)
            center = class_centers[class_idx]
            sample = center + np.random.randn(n_features) * noise_level
            sample = self._add_audio_characteristics(sample)
            X.append(sample)
            y.append(class_idx)
        
        X = np.array(X)
        y = np.array(y)
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def _add_audio_characteristics(self, sample: np.ndarray) -> np.ndarray:
        """Add audio-like characteristics to feature vector."""
        # Simulate MFCC-like features with realistic ranges
        
        # Energy concentration in lower frequencies
        n_features = len(sample)
        frequency_weights = np.exp(-np.arange(n_features) / (n_features / 4))
        sample = sample * frequency_weights
        
        # Add spectral rolloff characteristics
        rolloff_point = np.random.randint(n_features // 4, 3 * n_features // 4)
        sample[rolloff_point:] *= 0.3
        
        # Normalize to reasonable range for MFCC features
        sample = sample / (np.std(sample) + 1e-8) * 2.0
        
        return sample
    
    def generate_keyword_spotting_dataset(self,
                                        n_samples: int = 2000,
                                        sequence_length: int = 100,
                                        n_features: int = 40,
                                        keywords: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic keyword spotting dataset with temporal structure."""
        if keywords is None:
            keywords = ['wake', 'stop', 'go', 'yes', 'no', 'up', 'down', 'silence']
        
        n_classes = len(keywords)
        
        X = []
        y = []
        
        samples_per_class = n_samples // n_classes
        
        for class_idx, keyword in enumerate(keywords):
            for _ in range(samples_per_class):
                if keyword == 'silence':
                    # Generate silence/noise pattern
                    sequence = self._generate_silence_sequence(sequence_length, n_features)
                else:
                    # Generate keyword pattern
                    sequence = self._generate_keyword_sequence(keyword, sequence_length, n_features)
                
                X.append(sequence)
                y.append(class_idx)
        
        # Add remaining samples
        remaining = n_samples - len(X)
        for _ in range(remaining):
            class_idx = np.random.randint(n_classes)
            keyword = keywords[class_idx]
            
            if keyword == 'silence':
                sequence = self._generate_silence_sequence(sequence_length, n_features)
            else:
                sequence = self._generate_keyword_sequence(keyword, sequence_length, n_features)
            
            X.append(sequence)
            y.append(class_idx)
        
        X = np.array(X)
        y = np.array(y)
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def _generate_silence_sequence(self, length: int, n_features: int) -> np.ndarray:
        """Generate silence/background noise sequence."""
        # Low-level random noise
        noise = np.random.randn(length, n_features) * 0.1
        
        # Add some low-frequency components for realistic background
        for i in range(length):
            noise[i, :5] += np.random.randn(5) * 0.2
        
        return noise
    
    def _generate_keyword_sequence(self, keyword: str, length: int, n_features: int) -> np.ndarray:
        """Generate realistic keyword sequence with temporal structure."""
        sequence = np.zeros((length, n_features))
        
        # Create keyword signature based on hash
        keyword_hash = hashlib.md5(keyword.encode()).digest()
        signature = np.frombuffer(keyword_hash[:n_features * 4], dtype=np.float32)[:n_features]
        signature = signature / np.linalg.norm(signature)
        
        # Determine keyword timing (when the keyword occurs in the sequence)
        keyword_start = np.random.randint(length // 4, 3 * length // 4)
        keyword_duration = min(np.random.randint(length // 8, length // 4), length - keyword_start)
        
        # Add background noise throughout
        sequence += np.random.randn(length, n_features) * 0.1
        
        # Add keyword pattern
        for t in range(keyword_start, keyword_start + keyword_duration):
            # Intensity envelope (starts low, peaks in middle, ends low)
            relative_pos = (t - keyword_start) / keyword_duration
            intensity = 4 * relative_pos * (1 - relative_pos)  # Parabolic envelope
            
            # Add keyword signature with temporal variation
            temporal_variation = np.sin(2 * np.pi * relative_pos) * 0.3
            sequence[t] += signature * intensity * (1 + temporal_variation)
        
        # Add formant-like frequency structure
        self._add_formant_structure(sequence, keyword_start, keyword_duration)
        
        return sequence
    
    def _add_formant_structure(self, sequence: np.ndarray, start: int, duration: int) -> None:
        """Add formant-like frequency structure to speech sequence."""
        length, n_features = sequence.shape
        
        # Define formant frequencies (as feature indices)
        formants = [5, 12, 20, 28]  # Typical formant positions in MFCC features
        
        for t in range(start, min(start + duration, length)):
            relative_pos = (t - start) / duration
            
            for formant_idx in formants:
                if formant_idx < n_features:
                    # Add formant energy with some temporal variation
                    formant_energy = np.sin(2 * np.pi * relative_pos * 3) * 0.5 + 0.5
                    sequence[t, formant_idx] += formant_energy * 0.8
    
    def generate_power_efficiency_test_data(self,
                                          complexity_levels: List[str] = None,
                                          samples_per_level: int = 100) -> Dict[str, Tuple[np.ndarray, Dict[str, float]]]:
        """Generate test data for power efficiency analysis."""
        if complexity_levels is None:
            complexity_levels = ['low', 'medium', 'high', 'extreme']
        
        test_data = {}
        
        for level in complexity_levels:
            # Generate data with specific complexity characteristics
            X, expected_metrics = self._generate_complexity_data(level, samples_per_level)
            test_data[level] = (X, expected_metrics)
        
        return test_data
    
    def _generate_complexity_data(self, complexity_level: str, n_samples: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate data with specific computational complexity."""
        n_features = 40
        
        if complexity_level == 'low':
            # Simple, sparse patterns
            X = np.random.randn(n_samples, n_features) * 0.5
            # Add simple patterns
            for i in range(n_samples):
                X[i, :10] += np.sin(np.arange(10)) * 2
            
            expected_metrics = {
                'expected_power_mw': 0.8,
                'expected_latency_ms': 5.0,
                'pattern_complexity': 0.2
            }
        
        elif complexity_level == 'medium':
            # Moderate complexity patterns
            X = np.random.randn(n_samples, n_features) * 1.0
            for i in range(n_samples):
                # Add multiple frequency components
                for freq in [1, 3, 5]:
                    X[i] += np.sin(np.arange(n_features) * freq / 10) * 0.5
            
            expected_metrics = {
                'expected_power_mw': 1.5,
                'expected_latency_ms': 10.0,
                'pattern_complexity': 0.5
            }
        
        elif complexity_level == 'high':
            # Complex, high-variance patterns
            X = np.random.randn(n_samples, n_features) * 2.0
            for i in range(n_samples):
                # Add noise and complex patterns
                X[i] += np.random.randn(n_features) * 1.5
                # Add chaotic components
                for j in range(n_features):
                    X[i, j] += np.sin(j * X[i, j] * 0.1) * 0.8
            
            expected_metrics = {
                'expected_power_mw': 3.2,
                'expected_latency_ms': 18.0,
                'pattern_complexity': 0.8
            }
        
        else:  # extreme
            # Very complex, rapidly changing patterns
            X = np.random.randn(n_samples, n_features) * 3.0
            for i in range(n_samples):
                # Add extreme complexity
                for freq in range(1, 20):
                    phase = np.random.random() * 2 * np.pi
                    X[i] += np.sin(np.arange(n_features) * freq / 5 + phase) * np.random.random()
                
                # Add white noise
                X[i] += np.random.randn(n_features) * 2.0
            
            expected_metrics = {
                'expected_power_mw': 6.5,
                'expected_latency_ms': 35.0,
                'pattern_complexity': 1.0
            }
        
        return X, expected_metrics


class ExperimentalFramework:
    """Main framework for conducting reproducible experiments."""
    
    def __init__(self, results_dir: Path = None):
        self.results_dir = results_dir or Path("experiment_results")
        self.results_dir.mkdir(exist_ok=True)
        
        self.reproducibility_manager = ReproducibilityManager()
        self.dataset_generator = DatasetGenerator()
        
        # Experiment tracking
        self.experiment_registry = {}
        self.current_experiment = None
        
    def register_experiment(self, config: ExperimentConfig) -> str:
        """Register an experiment configuration."""
        exp_id = config.metadata['experiment_id']
        self.experiment_registry[exp_id] = config
        
        # Save configuration
        config_path = self.results_dir / f"{exp_id}_config.json"
        config.save(config_path)
        
        logger.info(f"Registered experiment {exp_id}: {config.experiment_name}")
        return exp_id
    
    def run_experiment(self, 
                      config: ExperimentConfig,
                      experiment_function: Callable[[ExperimentConfig], Dict[str, Any]],
                      verify_reproducibility: bool = False,
                      save_artifacts: bool = True) -> ExperimentResult:
        """Run a complete experiment with full tracking."""
        start_time = time.time()
        exp_id = config.metadata['experiment_id']
        
        logger.info(f"Starting experiment {exp_id}: {config.experiment_name}")
        
        # Set up reproducibility
        self.reproducibility_manager.set_global_seed(config.random_seed)
        
        # Collect system information
        system_info = SystemInfo.collect()
        
        try:
            # Verify reproducibility if requested
            if verify_reproducibility:
                logger.info("Verifying reproducibility...")
                repro_result = self.reproducibility_manager.verify_reproducibility(
                    experiment_function, config, n_runs=3
                )
                
                if not repro_result['reproducible']:
                    logger.warning(f"Reproducibility check failed: {repro_result}")
                else:
                    logger.info("Reproducibility verified ✓")
            
            # Run the main experiment
            self.current_experiment = config
            results = experiment_function(config)
            
            # Extract metrics
            metrics = self._extract_metrics(results)
            
            # Create result object
            execution_time = time.time() - start_time
            
            experiment_result = ExperimentResult(
                experiment_id=exp_id,
                config=config,
                system_info=system_info,
                results=results,
                metrics=metrics,
                execution_time=execution_time,
                success=True
            )
            
            # Save artifacts if requested
            if save_artifacts and 'artifacts' in results:
                experiment_result.artifacts = results['artifacts']
            
        except Exception as e:
            logger.error(f"Experiment {exp_id} failed: {e}")
            execution_time = time.time() - start_time
            
            experiment_result = ExperimentResult(
                experiment_id=exp_id,
                config=config,
                system_info=system_info,
                results={},
                metrics={},
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
        
        finally:
            self.current_experiment = None
        
        # Save results
        result_path = self.results_dir / f"{exp_id}_results.json"
        experiment_result.save(result_path)
        
        logger.info(f"Experiment {exp_id} completed in {execution_time:.2f}s")
        return experiment_result
    
    def _extract_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Extract numeric metrics from experiment results."""
        metrics = {}
        
        def extract_recursive(obj, prefix=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    extract_recursive(value, new_prefix)
            elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                metrics[prefix] = float(obj)
            elif isinstance(obj, (list, tuple)) and len(obj) > 0:
                # Extract statistics from numeric arrays
                if all(isinstance(x, (int, float)) for x in obj):
                    metrics[f"{prefix}.mean"] = float(np.mean(obj))
                    metrics[f"{prefix}.std"] = float(np.std(obj))
                    metrics[f"{prefix}.min"] = float(np.min(obj))
                    metrics[f"{prefix}.max"] = float(np.max(obj))
        
        extract_recursive(results)
        return metrics
    
    def run_comparative_study(self,
                            configs: List[ExperimentConfig],
                            experiment_function: Callable,
                            study_name: str = "Comparative Study") -> Dict[str, Any]:
        """Run a comparative study across multiple configurations."""
        logger.info(f"Starting comparative study: {study_name}")
        
        study_results = {
            'study_name': study_name,
            'timestamp': datetime.now().isoformat(),
            'experiments': {},
            'comparisons': {},
            'summary': {}
        }
        
        # Run all experiments
        experiment_results = []
        for i, config in enumerate(configs):
            logger.info(f"Running experiment {i+1}/{len(configs)}: {config.experiment_name}")
            
            result = self.run_experiment(config, experiment_function)
            experiment_results.append(result)
            study_results['experiments'][result.experiment_id] = {
                'config': config.to_dict(),
                'metrics': result.metrics,
                'success': result.success,
                'execution_time': result.execution_time
            }
        
        # Perform comparisons
        study_results['comparisons'] = self._compare_experiments(experiment_results)
        
        # Generate summary
        study_results['summary'] = self._generate_study_summary(experiment_results)
        
        # Save study results
        study_path = self.results_dir / f"study_{study_name.replace(' ', '_').lower()}.json"
        with open(study_path, 'w') as f:
            json.dump(study_results, f, indent=2, default=str)
        
        logger.info(f"Comparative study completed. Results saved to {study_path}")
        return study_results
    
    def _compare_experiments(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Compare experimental results statistically."""
        comparisons = {}
        
        # Extract common metrics
        common_metrics = set()
        for result in results:
            if result.success:
                common_metrics.update(result.metrics.keys())
        
        # Remove metrics that aren't present in all successful experiments
        for result in results:
            if result.success:
                common_metrics.intersection_update(result.metrics.keys())
        
        # Statistical comparisons
        for metric in common_metrics:
            values = []
            experiment_ids = []
            
            for result in results:
                if result.success and metric in result.metrics:
                    values.append(result.metrics[metric])
                    experiment_ids.append(result.experiment_id)
            
            if len(values) > 1:
                comparisons[metric] = {
                    'values': values,
                    'experiment_ids': experiment_ids,
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'range': float(np.max(values) - np.min(values)),
                    'coefficient_of_variation': float(np.std(values) / np.mean(values)) if np.mean(values) != 0 else float('inf')
                }
                
                # Best and worst performing experiments for this metric
                best_idx = np.argmax(values) if 'accuracy' in metric or 'score' in metric else np.argmin(values)
                worst_idx = np.argmin(values) if 'accuracy' in metric or 'score' in metric else np.argmax(values)
                
                comparisons[metric]['best_experiment'] = experiment_ids[best_idx]
                comparisons[metric]['worst_experiment'] = experiment_ids[worst_idx]
                comparisons[metric]['best_value'] = values[best_idx]
                comparisons[metric]['worst_value'] = values[worst_idx]
        
        return comparisons
    
    def _generate_study_summary(self, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Generate summary of the comparative study."""
        successful_results = [r for r in results if r.success]
        
        summary = {
            'total_experiments': len(results),
            'successful_experiments': len(successful_results),
            'failed_experiments': len(results) - len(successful_results),
            'total_execution_time': sum(r.execution_time for r in results),
            'avg_execution_time': np.mean([r.execution_time for r in results]) if results else 0,
        }
        
        if successful_results:
            # Find overall best performing experiment
            # Use a simple scoring based on multiple metrics
            scores = []
            for result in successful_results:
                score = 0
                count = 0
                
                # Positive metrics (higher is better)
                positive_metrics = ['accuracy', 'f1_score', 'precision', 'recall', 'throughput']
                for metric in positive_metrics:
                    if any(pm in metric for pm in positive_metrics):
                        for key, value in result.metrics.items():
                            if any(pm in key.lower() for pm in positive_metrics):
                                score += value
                                count += 1
                
                # Negative metrics (lower is better) - subtract normalized values
                negative_metrics = ['power', 'latency', 'error']
                for key, value in result.metrics.items():
                    if any(nm in key.lower() for nm in negative_metrics):
                        # Normalize by inverting (smaller values get higher scores)
                        if value > 0:
                            score += 1.0 / value
                            count += 1
                
                scores.append(score / count if count > 0 else 0)
            
            if scores:
                best_idx = np.argmax(scores)
                best_result = successful_results[best_idx]
                
                summary['best_experiment'] = {
                    'experiment_id': best_result.experiment_id,
                    'experiment_name': best_result.config.experiment_name,
                    'score': scores[best_idx],
                    'key_metrics': {k: v for k, v in best_result.metrics.items() 
                                  if any(metric in k.lower() for metric in 
                                        ['accuracy', 'power', 'latency', 'f1_score'])}
                }
        
        return summary
    
    def load_experiment_results(self, experiment_ids: List[str] = None) -> List[ExperimentResult]:
        """Load experiment results from disk."""
        if experiment_ids is None:
            # Load all available results
            result_files = list(self.results_dir.glob("*_results.json"))
            experiment_ids = [f.stem.replace("_results", "") for f in result_files]
        
        results = []
        for exp_id in experiment_ids:
            result_path = self.results_dir / f"{exp_id}_results.json"
            if result_path.exists():
                try:
                    result = ExperimentResult.load(result_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to load result {exp_id}: {e}")
        
        return results
    
    def generate_experiment_report(self, experiment_ids: List[str] = None) -> str:
        """Generate comprehensive experiment report."""
        results = self.load_experiment_results(experiment_ids)
        
        if not results:
            return "No experiment results found."
        
        report = []
        
        # Header
        report.append("# Experimental Results Report")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Total Experiments:** {len(results)}")
        report.append("")
        
        # Summary statistics
        successful_results = [r for r in results if r.success]
        report.append("## Summary")
        report.append(f"- **Successful Experiments:** {len(successful_results)}/{len(results)}")
        report.append(f"- **Total Execution Time:** {sum(r.execution_time for r in results):.2f} seconds")
        if successful_results:
            report.append(f"- **Average Execution Time:** {np.mean([r.execution_time for r in successful_results]):.2f} seconds")
        report.append("")
        
        # Individual experiment results
        report.append("## Individual Experiments")
        report.append("")
        
        for result in results:
            report.append(f"### {result.config.experiment_name} ({result.experiment_id})")
            report.append(f"**Status:** {'✓ Success' if result.success else '✗ Failed'}")
            report.append(f"**Execution Time:** {result.execution_time:.2f}s")
            
            if result.success and result.metrics:
                report.append("**Key Metrics:**")
                # Show top 10 most important metrics
                important_metrics = {}
                for key, value in result.metrics.items():
                    if any(term in key.lower() for term in 
                          ['accuracy', 'power', 'latency', 'f1', 'precision', 'recall', 'throughput']):
                        important_metrics[key] = value
                
                for key, value in sorted(important_metrics.items())[:10]:
                    report.append(f"- {key}: {value:.4f}")
            
            if not result.success and result.error_message:
                report.append(f"**Error:** {result.error_message}")
            
            report.append("")
        
        # System information
        if results:
            sample_system = results[0].system_info
            report.append("## System Information")
            report.append(f"- **Platform:** {sample_system.platform}")
            report.append(f"- **CPU:** {sample_system.cpu_model}")
            report.append(f"- **Memory:** {sample_system.memory_gb:.1f} GB")
            report.append(f"- **Python:** {sample_system.python_version.split()[0]}")
            report.append("")
        
        return "\n".join(report)
    
    def cleanup_old_results(self, days_old: int = 30) -> None:
        """Clean up old experiment results."""
        cutoff_time = time.time() - (days_old * 24 * 3600)
        
        cleaned_count = 0
        for result_file in self.results_dir.glob("*_results.json"):
            if result_file.stat().st_mtime < cutoff_time:
                # Remove result file and associated artifacts
                result_file.unlink()
                
                # Remove artifacts directory if it exists
                artifacts_dir = result_file.parent / f"{result_file.stem}_artifacts"
                if artifacts_dir.exists():
                    import shutil
                    shutil.rmtree(artifacts_dir)
                
                cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} old experiment results")


# Example usage and integration
if __name__ == "__main__":
    # This would be run as a standalone script for testing
    pass