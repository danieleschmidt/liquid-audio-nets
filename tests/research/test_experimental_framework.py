"""
Comprehensive tests for the experimental framework.
"""

import pytest
import numpy as np
import tempfile
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from liquid_audio_nets.research.experimental_framework import (
    ExperimentConfig,
    ExperimentResult,
    SystemInfo,
    ReproducibilityManager,
    DatasetGenerator,
    ExperimentalFramework
)


class TestExperimentConfig:
    """Test ExperimentConfig dataclass."""
    
    def test_config_creation(self):
        """Test creating experiment configuration."""
        config = ExperimentConfig(
            experiment_name="Test Experiment",
            description="A test experiment",
            random_seed=42
        )
        
        assert config.experiment_name == "Test Experiment"
        assert config.description == "A test experiment"
        assert config.random_seed == 42
        assert isinstance(config.data_config, dict)
        assert isinstance(config.model_config, dict)
        assert 'experiment_id' in config.metadata
        assert 'timestamp' in config.metadata
    
    def test_config_deterministic_id(self):
        """Test that same config produces same ID."""
        config1 = ExperimentConfig(
            experiment_name="Test",
            description="Test",
            random_seed=42
        )
        
        # Create identical config
        config2 = ExperimentConfig(
            experiment_name="Test", 
            description="Test",
            random_seed=42
        )
        
        # Remove timestamps to make them identical
        config1.metadata.pop('timestamp', None)
        config2.metadata.pop('timestamp', None)
        
        # Should produce same ID for identical configs
        id1 = config1.metadata['experiment_id']
        config2.metadata = config1.metadata.copy()  # Force same metadata
        config2_dict = config2.to_dict()
        config2_dict['metadata'] = config1.metadata
        
        # The IDs should be deterministic based on content
        assert len(id1) == 8  # MD5 hash truncated to 8 chars
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = ExperimentConfig(
            experiment_name="Test",
            description="Test description",
            data_config={'param': 'value'},
            model_config={'hidden_dim': 64}
        )
        
        # Test to_dict
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict['experiment_name'] == "Test"
        assert config_dict['data_config']['param'] == 'value'
        
        # Test from_dict
        config_restored = ExperimentConfig.from_dict(config_dict)
        assert config_restored.experiment_name == config.experiment_name
        assert config_restored.description == config.description
        assert config_restored.data_config == config.data_config
    
    def test_config_save_load(self):
        """Test saving and loading configuration."""
        config = ExperimentConfig(
            experiment_name="Save Test",
            description="Test saving",
            random_seed=123
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Save configuration
            config.save(temp_path)
            assert temp_path.exists()
            
            # Load configuration
            config_loaded = ExperimentConfig.load(temp_path)
            
            assert config_loaded.experiment_name == config.experiment_name
            assert config_loaded.description == config.description
            assert config_loaded.random_seed == config.random_seed
            
        finally:
            temp_path.unlink(missing_ok=True)


class TestSystemInfo:
    """Test SystemInfo dataclass."""
    
    @patch('platform.platform')
    @patch('platform.architecture')
    @patch('platform.processor')
    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    def test_system_info_collection(self, mock_vmem, mock_cpu_count, mock_processor, 
                                  mock_arch, mock_platform):
        """Test system information collection."""
        # Mock system info
        mock_platform.return_value = "Linux-5.4.0"
        mock_arch.return_value = ("64bit", "ELF")
        mock_processor.return_value = "Intel Core i7"
        mock_cpu_count.return_value = 8
        
        # Mock memory (psutil returns object with total attribute)
        mock_memory = MagicMock()
        mock_memory.total = 16 * 1024**3  # 16GB
        mock_vmem.return_value = mock_memory
        
        system_info = SystemInfo.collect()
        
        assert "Linux-5.4.0" in system_info.platform
        assert system_info.architecture == "64bit"
        assert system_info.cpu_model == "Intel Core i7"
        assert system_info.cpu_cores == 8
        assert system_info.memory_gb == 16.0
        assert system_info.python_version is not None
        assert system_info.numpy_version is not None
    
    def test_system_info_with_torch(self):
        """Test system info collection with PyTorch available."""
        try:
            import torch
            system_info = SystemInfo.collect()
            assert system_info.pytorch_version is not None
            assert system_info.gpu_info is not None
        except ImportError:
            # PyTorch not available, should handle gracefully
            system_info = SystemInfo.collect()
            assert system_info.pytorch_version is None


class TestReproducibilityManager:
    """Test ReproducibilityManager."""
    
    def test_manager_creation(self):
        """Test creating reproducibility manager."""
        manager = ReproducibilityManager(base_seed=123)
        assert manager.base_seed == 123
        assert manager.seed_counter == 0
    
    def test_global_seed_setting(self):
        """Test setting global seeds."""
        manager = ReproducibilityManager()
        
        # Should not raise errors
        manager.set_global_seed(42)
        
        # Check that numpy seed is set
        np.random.seed(42)
        val1 = np.random.random()
        
        manager.set_global_seed(42)
        val2 = np.random.random()
        
        assert val1 == val2  # Should produce same value
    
    def test_seed_generation(self):
        """Test deterministic seed generation."""
        manager = ReproducibilityManager(base_seed=100)
        
        # Should generate deterministic sequence
        seed1 = manager.get_next_seed()
        seed2 = manager.get_next_seed()
        seed3 = manager.get_next_seed()
        
        assert seed1 == 101  # base_seed + 1
        assert seed2 == 102  # base_seed + 2
        assert seed3 == 103  # base_seed + 3
        
        # Reset and check reproducibility
        manager.seed_counter = 0
        new_seed1 = manager.get_next_seed()
        assert new_seed1 == seed1
    
    def test_data_split_seeds(self):
        """Test generating seeds for data splits."""
        manager = ReproducibilityManager(base_seed=50)
        
        seeds = manager.create_data_split_seeds(5)
        
        assert len(seeds) == 5
        assert seeds == [51, 52, 53, 54, 55]  # Sequential from base_seed
    
    def test_reproducibility_verification(self):
        """Test reproducibility verification."""
        manager = ReproducibilityManager(base_seed=42)
        
        # Mock experiment function that should be reproducible
        def reproducible_experiment(config):
            manager.set_global_seed(config.random_seed)
            value = np.random.random()
            
            # Mock result structure
            result = MagicMock()
            result.metrics = {'test_metric': value}
            return result
        
        config = ExperimentConfig(
            experiment_name="Reproducibility Test",
            description="Test",
            random_seed=42
        )
        
        verification_result = manager.verify_reproducibility(
            reproducible_experiment, config, n_runs=3
        )
        
        assert verification_result['reproducible'] == True
        assert verification_result['n_runs'] == 3
        assert len(verification_result['differences']) == 0
    
    def test_reproducibility_verification_failure(self):
        """Test reproducibility verification with non-reproducible function."""
        manager = ReproducibilityManager()
        
        # Non-reproducible experiment (uses time)
        def non_reproducible_experiment(config):
            value = time.time() % 1  # Will always be different
            
            result = MagicMock()
            result.metrics = {'test_metric': value}
            return result
        
        config = ExperimentConfig(
            experiment_name="Non-reproducible Test",
            description="Test",
            random_seed=42
        )
        
        verification_result = manager.verify_reproducibility(
            non_reproducible_experiment, config, n_runs=3
        )
        
        assert verification_result['reproducible'] == False
        assert len(verification_result['differences']) > 0


class TestDatasetGenerator:
    """Test DatasetGenerator."""
    
    def test_generator_creation(self):
        """Test creating dataset generator."""
        generator = DatasetGenerator(seed=42)
        assert generator.seed == 42
    
    def test_audio_classification_dataset(self):
        """Test generating audio classification dataset."""
        generator = DatasetGenerator(seed=42)
        
        X, y = generator.generate_audio_classification_dataset(
            n_samples=100,
            n_features=20,
            n_classes=4,
            noise_level=0.1
        )
        
        assert X.shape == (100, 20)
        assert y.shape == (100,)
        assert np.min(y) >= 0
        assert np.max(y) < 4
        assert len(np.unique(y)) <= 4
        
        # Check that data has reasonable range
        assert np.min(X) > -10  # Not extremely negative
        assert np.max(X) < 10   # Not extremely positive
    
    def test_keyword_spotting_dataset(self):
        """Test generating keyword spotting dataset."""
        generator = DatasetGenerator(seed=42)
        
        keywords = ['wake', 'stop', 'go', 'silence']
        X, y = generator.generate_keyword_spotting_dataset(
            n_samples=40,  # 10 per class
            sequence_length=50,
            n_features=20,
            keywords=keywords
        )
        
        assert X.shape == (40, 50, 20)  # samples, time, features
        assert y.shape == (40,)
        assert np.min(y) >= 0
        assert np.max(y) < 4
        
        # Check that each class is represented
        unique_classes = np.unique(y)
        assert len(unique_classes) == 4
    
    def test_power_efficiency_test_data(self):
        """Test generating power efficiency test data."""
        generator = DatasetGenerator(seed=42)
        
        test_data = generator.generate_power_efficiency_test_data(
            complexity_levels=['low', 'medium', 'high'],
            samples_per_level=10
        )
        
        assert 'low' in test_data
        assert 'medium' in test_data
        assert 'high' in test_data
        
        # Check data structure
        for level, (X, metrics) in test_data.items():
            assert X.shape == (10, 40)  # Default feature dim
            assert 'expected_power_mw' in metrics
            assert 'expected_latency_ms' in metrics
            assert 'pattern_complexity' in metrics
            
            # Check that complexity increases across levels
            assert metrics['pattern_complexity'] >= 0
            assert metrics['pattern_complexity'] <= 1
        
        # Verify that complexity increases
        low_complexity = test_data['low'][1]['pattern_complexity']
        medium_complexity = test_data['medium'][1]['pattern_complexity']
        high_complexity = test_data['high'][1]['pattern_complexity']
        
        assert low_complexity < medium_complexity < high_complexity
    
    def test_audio_characteristics(self):
        """Test audio characteristic generation."""
        generator = DatasetGenerator(seed=42)
        
        sample = np.random.randn(40)
        enhanced_sample = generator._add_audio_characteristics(sample)
        
        assert enhanced_sample.shape == sample.shape
        assert isinstance(enhanced_sample, np.ndarray)
        
        # Should modify the sample (very unlikely to be identical)
        assert not np.array_equal(sample, enhanced_sample)


class TestExperimentResult:
    """Test ExperimentResult dataclass."""
    
    def test_result_creation(self):
        """Test creating experiment result."""
        config = ExperimentConfig("Test", "Description")
        system_info = SystemInfo(
            platform="Test", architecture="64bit", cpu_model="Test CPU",
            cpu_cores=4, memory_gb=8.0, python_version="3.8",
            numpy_version="1.21"
        )
        
        result = ExperimentResult(
            experiment_id="test123",
            config=config,
            system_info=system_info,
            results={'accuracy': 0.95},
            metrics={'accuracy': 0.95, 'loss': 0.05},
            execution_time=10.5,
            success=True
        )
        
        assert result.experiment_id == "test123"
        assert result.success == True
        assert result.execution_time == 10.5
        assert result.metrics['accuracy'] == 0.95
    
    def test_result_serialization(self):
        """Test result saving and loading."""
        config = ExperimentConfig("Test", "Description")
        system_info = SystemInfo(
            platform="Test", architecture="64bit", cpu_model="Test CPU",
            cpu_cores=4, memory_gb=8.0, python_version="3.8",
            numpy_version="1.21"
        )
        
        result = ExperimentResult(
            experiment_id="test123",
            config=config,
            system_info=system_info,
            results={'test': 'value'},
            metrics={'metric1': 1.0},
            artifacts={'artifact1': np.array([1, 2, 3])},
            execution_time=5.0,
            success=True
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "test_result.json"
            
            # Save result
            result.save(temp_path)
            assert temp_path.exists()
            
            # Check that artifacts directory was created
            artifacts_dir = Path(temp_dir) / "test_result_artifacts"
            assert artifacts_dir.exists()
            
            # Load result
            loaded_result = ExperimentResult.load(temp_path)
            
            assert loaded_result.experiment_id == result.experiment_id
            assert loaded_result.success == result.success
            assert loaded_result.execution_time == result.execution_time
            assert loaded_result.metrics == result.metrics
            
            # Check artifacts were loaded
            assert 'artifact1' in loaded_result.artifacts
            np.testing.assert_array_equal(
                loaded_result.artifacts['artifact1'],
                result.artifacts['artifact1']
            )


class TestExperimentalFramework:
    """Test ExperimentalFramework."""
    
    def test_framework_creation(self):
        """Test creating experimental framework."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results_dir = Path(temp_dir)
            framework = ExperimentalFramework(results_dir)
            
            assert framework.results_dir == results_dir
            assert results_dir.exists()
            assert isinstance(framework.reproducibility_manager, ReproducibilityManager)
            assert isinstance(framework.dataset_generator, DatasetGenerator)
    
    def test_experiment_registration(self):
        """Test experiment registration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = ExperimentalFramework(Path(temp_dir))
            
            config = ExperimentConfig("Test Experiment", "Description")
            exp_id = framework.register_experiment(config)
            
            assert exp_id in framework.experiment_registry
            assert framework.experiment_registry[exp_id] == config
            
            # Check config file was saved
            config_file = Path(temp_dir) / f"{exp_id}_config.json"
            assert config_file.exists()
    
    def test_run_experiment_success(self):
        """Test running successful experiment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = ExperimentalFramework(Path(temp_dir))
            
            config = ExperimentConfig("Test", "Description", random_seed=42)
            
            # Mock experiment function
            def mock_experiment(cfg):
                return {
                    'accuracy': 0.95,
                    'loss': 0.05,
                    'artifacts': {'model': 'mock_model_data'}
                }
            
            result = framework.run_experiment(config, mock_experiment)
            
            assert result.success == True
            assert result.experiment_id == config.metadata['experiment_id']
            assert result.metrics['accuracy'] == 0.95
            assert result.execution_time > 0
            
            # Check result file was saved
            result_file = Path(temp_dir) / f"{result.experiment_id}_results.json"
            assert result_file.exists()
    
    def test_run_experiment_failure(self):
        """Test running failed experiment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = ExperimentalFramework(Path(temp_dir))
            
            config = ExperimentConfig("Failing Test", "Description")
            
            # Mock failing experiment function
            def failing_experiment(cfg):
                raise ValueError("Experiment failed!")
            
            result = framework.run_experiment(config, failing_experiment)
            
            assert result.success == False
            assert result.error_message == "Experiment failed!"
            assert result.execution_time > 0
    
    def test_run_experiment_with_reproducibility_check(self):
        """Test experiment with reproducibility verification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = ExperimentalFramework(Path(temp_dir))
            
            config = ExperimentConfig("Reproducible Test", "Description", random_seed=42)
            
            # Mock reproducible experiment
            def reproducible_experiment(cfg):
                np.random.seed(cfg.random_seed)
                value = np.random.random()
                return {'value': value}
            
            with patch.object(framework.reproducibility_manager, 'verify_reproducibility') as mock_verify:
                mock_verify.return_value = {'reproducible': True, 'differences': {}}
                
                result = framework.run_experiment(
                    config, reproducible_experiment, verify_reproducibility=True
                )
                
                assert result.success == True
                mock_verify.assert_called_once()
    
    def test_extract_metrics(self):
        """Test metric extraction from results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = ExperimentalFramework(Path(temp_dir))
            
            results = {
                'accuracy': 0.95,
                'nested': {'precision': 0.92, 'recall': 0.88},
                'array_metric': [0.8, 0.9, 0.85],
                'string_value': 'ignored',
                'boolean_value': True  # Should be ignored
            }
            
            metrics = framework._extract_metrics(results)
            
            assert 'accuracy' in metrics
            assert metrics['accuracy'] == 0.95
            assert 'nested.precision' in metrics
            assert metrics['nested.precision'] == 0.92
            assert 'array_metric.mean' in metrics
            assert abs(metrics['array_metric.mean'] - 0.85) < 1e-6
            assert 'array_metric.std' in metrics
            
            # String and boolean values should not be in metrics
            assert 'string_value' not in metrics
            assert 'boolean_value' not in metrics
    
    def test_comparative_study(self):
        """Test running comparative study."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = ExperimentalFramework(Path(temp_dir))
            
            # Create multiple configurations
            configs = [
                ExperimentConfig("Exp1", "First experiment", random_seed=42),
                ExperimentConfig("Exp2", "Second experiment", random_seed=43)
            ]
            
            # Mock experiment function
            def mock_experiment(cfg):
                # Different results based on seed
                np.random.seed(cfg.random_seed)
                return {'accuracy': 0.8 + np.random.random() * 0.1}
            
            study_results = framework.run_comparative_study(
                configs, mock_experiment, "Test Study"
            )
            
            assert study_results['study_name'] == "Test Study"
            assert 'experiments' in study_results
            assert len(study_results['experiments']) == 2
            assert 'comparisons' in study_results
            assert 'summary' in study_results
            
            # Check that study file was saved
            study_files = list(Path(temp_dir).glob("study_*.json"))
            assert len(study_files) == 1
    
    def test_compare_experiments(self):
        """Test experiment comparison."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = ExperimentalFramework(Path(temp_dir))
            
            # Create mock experiment results
            config1 = ExperimentConfig("Exp1", "Description")
            config2 = ExperimentConfig("Exp2", "Description")
            system_info = SystemInfo(
                platform="Test", architecture="64bit", cpu_model="Test",
                cpu_cores=4, memory_gb=8.0, python_version="3.8", numpy_version="1.21"
            )
            
            result1 = ExperimentResult(
                experiment_id="exp1",
                config=config1,
                system_info=system_info,
                results={},
                metrics={'accuracy': 0.95, 'latency': 10.0},
                success=True
            )
            
            result2 = ExperimentResult(
                experiment_id="exp2",
                config=config2,
                system_info=system_info,
                results={},
                metrics={'accuracy': 0.88, 'latency': 8.0},
                success=True
            )
            
            comparisons = framework._compare_experiments([result1, result2])
            
            assert 'accuracy' in comparisons
            assert 'latency' in comparisons
            
            # Check accuracy comparison
            acc_comp = comparisons['accuracy']
            assert acc_comp['mean'] == (0.95 + 0.88) / 2
            assert acc_comp['min'] == 0.88
            assert acc_comp['max'] == 0.95
            assert 'best_experiment' in acc_comp
            assert 'worst_experiment' in acc_comp
    
    def test_generate_experiment_report(self):
        """Test experiment report generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = ExperimentalFramework(Path(temp_dir))
            
            # Create and save a mock result
            config = ExperimentConfig("Report Test", "Description")
            system_info = SystemInfo(
                platform="Linux", architecture="64bit", cpu_model="Intel",
                cpu_cores=8, memory_gb=16.0, python_version="3.8", numpy_version="1.21"
            )
            
            result = ExperimentResult(
                experiment_id="report_test",
                config=config,
                system_info=system_info,
                results={},
                metrics={'accuracy': 0.92, 'power_consumption_mw': 1.5},
                execution_time=25.0,
                success=True
            )
            
            # Save result
            result_path = Path(temp_dir) / "report_test_results.json"
            result.save(result_path)
            
            # Generate report
            report = framework.generate_experiment_report(["report_test"])
            
            assert "Experimental Results Report" in report
            assert "Report Test" in report
            assert "1/1" in report  # Success rate
            assert "System Information" in report
            assert "Linux" in report
    
    def test_load_experiment_results(self):
        """Test loading experiment results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = ExperimentalFramework(Path(temp_dir))
            
            # Create and save results
            config = ExperimentConfig("Load Test", "Description")
            system_info = SystemInfo(
                platform="Test", architecture="64bit", cpu_model="Test",
                cpu_cores=4, memory_gb=8.0, python_version="3.8", numpy_version="1.21"
            )
            
            result1 = ExperimentResult(
                experiment_id="load_test_1",
                config=config,
                system_info=system_info,
                results={},
                metrics={'accuracy': 0.9},
                success=True
            )
            
            result2 = ExperimentResult(
                experiment_id="load_test_2",
                config=config,
                system_info=system_info,
                results={},
                metrics={'accuracy': 0.85},
                success=True
            )
            
            # Save results
            result1.save(Path(temp_dir) / "load_test_1_results.json")
            result2.save(Path(temp_dir) / "load_test_2_results.json")
            
            # Load results
            loaded_results = framework.load_experiment_results(["load_test_1", "load_test_2"])
            
            assert len(loaded_results) == 2
            experiment_ids = [r.experiment_id for r in loaded_results]
            assert "load_test_1" in experiment_ids
            assert "load_test_2" in experiment_ids


class TestIntegration:
    """Integration tests for the experimental framework."""
    
    def test_complete_experiment_workflow(self):
        """Test complete experiment workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            framework = ExperimentalFramework(Path(temp_dir))
            
            # Create experiment configuration
            config = ExperimentConfig(
                experiment_name="Complete Workflow Test",
                description="Testing complete workflow",
                random_seed=42,
                data_config={'dataset_size': 100},
                model_config={'hidden_dim': 64}
            )
            
            # Register experiment
            exp_id = framework.register_experiment(config)
            
            # Define experiment function
            def test_experiment(cfg):
                # Use framework's dataset generator
                generator = framework.dataset_generator
                X, y = generator.generate_audio_classification_dataset(
                    n_samples=cfg.data_config['dataset_size'],
                    n_features=40,
                    n_classes=8
                )
                
                # Mock model training and evaluation
                np.random.seed(cfg.random_seed)
                accuracy = 0.8 + np.random.random() * 0.1
                power = 1.0 + np.random.random() * 0.5
                
                return {
                    'training_accuracy': accuracy,
                    'validation_accuracy': accuracy - 0.05,
                    'power_consumption_mw': power,
                    'dataset_shape': X.shape,
                    'num_classes': len(np.unique(y))
                }
            
            # Run experiment
            result = framework.run_experiment(config, test_experiment)
            
            # Verify successful execution
            assert result.success == True
            assert result.experiment_id == exp_id
            assert 'training_accuracy' in result.metrics
            assert 'power_consumption_mw' in result.metrics
            
            # Generate report
            report = framework.generate_experiment_report([exp_id])
            assert "Complete Workflow Test" in report
            
            # Load and verify result persistence
            loaded_results = framework.load_experiment_results([exp_id])
            assert len(loaded_results) == 1
            assert loaded_results[0].experiment_id == exp_id


if __name__ == "__main__":
    pytest.main([__file__])