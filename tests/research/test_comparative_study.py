"""
Comprehensive tests for the comparative study framework.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from liquid_audio_nets.research.comparative_study import (
    ComparativeStudyFramework,
    CNNBaseline,
    LSTMBaseline,
    TinyMLBaseline,
    PowerEfficiencyAnalysis,
    ModelComparison,
    PerformanceMetrics,
    StatisticalTest
)


class MockLNNModel:
    """Mock LNN model for testing."""
    
    def __init__(self):
        self.power_mw = 1.2
        self.model_size = 65536  # 64KB
    
    def process(self, audio_buffer):
        """Mock processing with consistent results."""
        confidence = 0.85 + np.random.normal(0, 0.02)  # Small variation
        return {
            'confidence': confidence,
            'keyword_detected': confidence > 0.5,
            'keyword': 'test' if confidence > 0.8 else None
        }
    
    def measure_inference_time(self, X):
        """Mock inference time measurement."""
        return 8.5 + np.random.normal(0, 0.5)  # ms
    
    def estimate_power_consumption(self, X):
        """Mock power consumption."""
        return self.power_mw + np.random.normal(0, 0.1)
    
    def get_model_size(self):
        """Mock model size."""
        return self.model_size
    
    def current_power_mw(self):
        """Mock current power."""
        return self.power_mw


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating performance metrics."""
        metrics = PerformanceMetrics(
            accuracy=0.95,
            precision=0.93,
            recall=0.97,
            f1_score=0.95,
            inference_time_ms=10.5,
            power_consumption_mw=2.1,
            model_size_bytes=1024000,
            memory_usage_mb=4.5,
            throughput_samples_per_sec=0,  # Will be calculated
            latency_p95_ms=12.0,
            energy_per_inference_uj=0  # Will be calculated
        )
        
        # Check derived metrics
        assert metrics.energy_per_inference_uj == pytest.approx(2.1 * 10.5 / 1000.0, rel=1e-6)
        assert metrics.throughput_samples_per_sec == pytest.approx(1000.0 / 10.5, rel=1e-3)
    
    def test_metrics_zero_inference_time(self):
        """Test handling of zero inference time."""
        metrics = PerformanceMetrics(
            accuracy=0.95, precision=0.93, recall=0.97, f1_score=0.95,
            inference_time_ms=0.0, power_consumption_mw=2.1,
            model_size_bytes=1024000, memory_usage_mb=4.5,
            throughput_samples_per_sec=0, latency_p95_ms=0.0,
            energy_per_inference_uj=0
        )
        
        assert metrics.energy_per_inference_uj == 0.0
        assert metrics.throughput_samples_per_sec == float('inf')


class TestBaselineModels:
    """Test baseline model implementations."""
    
    def test_cnn_baseline(self):
        """Test CNN baseline model."""
        model = CNNBaseline(input_dim=40, hidden_dim=32, output_dim=8)
        
        # Test model structure
        assert model.input_dim == 40
        assert model.hidden_dim == 32
        assert model.output_dim == 8
        
        # Test forward pass
        import torch
        x = torch.randn(5, 40)  # Batch of 5 samples
        output = model(x)
        
        assert output.shape == (5, 8)
        assert torch.allclose(output.sum(dim=1), torch.ones(5), atol=1e-6)  # Softmax check
        
        # Test training
        y = torch.randint(0, 8, (5,))
        model.train_model(x.numpy(), y.numpy(), epochs=2)
        
        # Test prediction
        predictions = model.predict(x.numpy())
        assert predictions.shape == (5,)
        assert all(0 <= p < 8 for p in predictions)
        
        # Test inference time measurement
        inference_time = model.measure_inference_time(x.numpy())
        assert inference_time > 0
        
        # Test power estimation
        power = model.estimate_power_consumption(x.numpy())
        assert power > 0
        
        # Test model size
        size = model.get_model_size()
        assert size > 0
    
    def test_lstm_baseline(self):
        """Test LSTM baseline model."""
        model = LSTMBaseline(input_dim=40, hidden_dim=32, output_dim=8)
        
        # Test model structure
        assert model.input_dim == 40
        assert model.hidden_dim == 32
        assert model.output_dim == 8
        
        # Test forward pass
        import torch
        x = torch.randn(5, 10, 40)  # Batch of 5 sequences, length 10, 40 features
        output = model(x)
        
        assert output.shape == (5, 8)
        assert torch.allclose(output.sum(dim=1), torch.ones(5), atol=1e-6)
        
        # Test with 2D input (should add batch dimension)
        x_2d = torch.randn(10, 40)
        output_2d = model(x_2d)
        assert output_2d.shape == (1, 8)
    
    def test_tinyml_baseline(self):
        """Test TinyML baseline model."""
        model = TinyMLBaseline(input_dim=40, hidden_dim=16, output_dim=8)
        
        # Test model structure
        assert model.input_dim == 40
        assert model.hidden_dim == 16
        assert model.output_dim == 8
        
        # Check weight shapes
        assert model.w1.shape == (40, 16)
        assert model.w2.shape == (16, 8)
        assert model.b1.shape == (16,)
        assert model.b2.shape == (8,)
        
        # Test training
        X = np.random.randn(50, 40)
        y = np.random.randint(0, 8, 50)
        model.train_model(X, y, epochs=5)
        
        # Test prediction
        predictions = model.predict(X)
        assert predictions.shape == (50,)
        assert all(0 <= p < 8 for p in predictions)
        
        # Test power estimation (should be lowest)
        power = model.estimate_power_consumption(X[:1])
        assert power > 0
        assert power < 10  # Should be very low power


class TestPowerEfficiencyAnalysis:
    """Test power efficiency analysis."""
    
    def test_power_analysis_creation(self):
        """Test creating power efficiency analyzer."""
        analyzer = PowerEfficiencyAnalysis(confidence_level=0.95)
        assert analyzer.confidence_level == 0.95
        assert analyzer.alpha == 0.05
    
    def test_validate_power_claims(self):
        """Test power claims validation."""
        analyzer = PowerEfficiencyAnalysis()
        
        # Generate test data
        lnn_power = [1.0, 1.1, 0.9, 1.05, 0.95]  # Low power
        baseline_power = {
            'CNN': [10.0, 10.5, 9.8, 10.2, 9.9],  # ~10x higher
            'LSTM': [6.0, 6.2, 5.8, 6.1, 5.9],   # ~6x higher
        }
        
        results = analyzer.validate_power_claims(
            lnn_power, baseline_power, claimed_improvement=10.0
        )
        
        assert 'CNN' in results
        assert 'LSTM' in results
        
        # Check CNN results (should support 10x claim)
        cnn_result = results['CNN']
        assert isinstance(cnn_result, StatisticalTest)
        assert cnn_result.p_value < 0.05  # Should be significant
        assert cnn_result.is_significant
        
        # Check LSTM results (should not fully support 10x claim)
        lstm_result = results['LSTM']
        assert isinstance(lstm_result, StatisticalTest)
        assert lstm_result.p_value < 0.05  # Should be significant
    
    def test_bootstrap_confidence_interval(self):
        """Test bootstrap confidence interval calculation."""
        analyzer = PowerEfficiencyAnalysis()
        
        sample1 = [1.0, 1.1, 0.9, 1.05, 0.95]
        sample2 = [10.0, 10.5, 9.8, 10.2, 9.9]
        
        ci = analyzer._bootstrap_ratio_ci(sample1, sample2, n_bootstrap=1000)
        
        assert len(ci) == 2
        assert ci[0] < ci[1]  # Lower bound < upper bound
        assert ci[0] > 8.0   # Should be around 10x
        assert ci[1] < 12.0
    
    def test_cohen_d_calculation(self):
        """Test Cohen's d effect size calculation."""
        analyzer = PowerEfficiencyAnalysis()
        
        sample1 = [1.0, 1.0, 1.0, 1.0, 1.0]  # No variance
        sample2 = [2.0, 2.0, 2.0, 2.0, 2.0]  # No variance, different mean
        
        # Should handle zero variance case
        d = analyzer._cohen_d(sample1, sample2)
        assert abs(d) > 0  # Should detect large effect


class TestModelComparison:
    """Test model comparison framework."""
    
    def test_comparison_creation(self):
        """Test creating model comparison."""
        comparison = ModelComparison(significance_level=0.05)
        assert comparison.significance_level == 0.05
        assert isinstance(comparison.power_analyzer, PowerEfficiencyAnalysis)
    
    @patch('time.perf_counter')
    def test_compare_models(self, mock_time):
        """Test model comparison functionality."""
        # Mock time for consistent timing
        mock_time.side_effect = [0, 0.01, 0.02, 0.03, 0.04, 0.05]  # Incremental times
        
        comparison = ModelComparison()
        
        # Create mock models
        lnn_model = MockLNNModel()
        baseline_models = {
            'CNN': CNNBaseline(40, 32, 8),
            'TinyML': TinyMLBaseline(40, 16, 8)
        }
        
        # Create test data
        X_test = np.random.randn(10, 40)
        y_test = np.random.randint(0, 8, 10)
        test_data = (X_test, y_test)
        
        # Run comparison (with fewer trials for testing)
        results = comparison.compare_models(
            lnn_model, baseline_models, test_data, n_trials=3
        )
        
        # Check structure
        assert 'lnn' in results
        assert 'baselines' in results
        assert 'statistical_tests' in results
        assert 'power_analysis' in results
        assert 'summary' in results
        
        # Check LNN results
        assert len(results['lnn']) == 3  # n_trials
        assert all(isinstance(m, PerformanceMetrics) for m in results['lnn'])
        
        # Check baseline results
        assert 'CNN' in results['baselines']
        assert 'TinyML' in results['baselines']
        
        # Check statistical tests
        assert 'CNN' in results['statistical_tests']
        assert 'accuracy' in results['statistical_tests']['CNN']
        assert 'power' in results['statistical_tests']['CNN']
        assert 'latency' in results['statistical_tests']['CNN']
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        comparison = ModelComparison()
        
        model = CNNBaseline(40, 32, 8)
        X = np.random.randn(5, 40)
        
        memory_usage = comparison._estimate_memory_usage(model, X)
        assert memory_usage > 0
        assert isinstance(memory_usage, float)


class TestComparativeStudyFramework:
    """Test the main comparative study framework."""
    
    def test_framework_creation(self):
        """Test creating the framework."""
        framework = ComparativeStudyFramework(random_seed=42)
        
        assert len(framework.baseline_models) == 0
        assert len(framework.results_history) == 0
        assert isinstance(framework.model_comparison, ModelComparison)
    
    def test_register_baseline_model(self):
        """Test registering baseline models."""
        framework = ComparativeStudyFramework()
        
        model = CNNBaseline(40, 32, 8)
        framework.register_baseline_model('test_cnn', model)
        
        assert 'test_cnn' in framework.baseline_models
        assert framework.baseline_models['test_cnn'] is model
    
    def test_create_standard_baselines(self):
        """Test creating standard baseline models."""
        framework = ComparativeStudyFramework()
        framework.create_standard_baselines(input_dim=40, hidden_dim=32, output_dim=8)
        
        assert 'CNN' in framework.baseline_models
        assert 'LSTM' in framework.baseline_models
        assert 'TinyML' in framework.baseline_models
        
        assert isinstance(framework.baseline_models['CNN'], CNNBaseline)
        assert isinstance(framework.baseline_models['LSTM'], LSTMBaseline)
        assert isinstance(framework.baseline_models['TinyML'], TinyMLBaseline)
    
    @patch('time.perf_counter')
    def test_run_comparative_study(self, mock_time):
        """Test running a complete comparative study."""
        # Mock time for consistent timing
        mock_time.side_effect = [i * 0.01 for i in range(100)]
        
        framework = ComparativeStudyFramework(random_seed=42)
        framework.create_standard_baselines(input_dim=40, hidden_dim=16, output_dim=4)
        
        # Create mock LNN
        lnn_model = MockLNNModel()
        
        # Create smaller test data for faster testing
        X_train = np.random.randn(20, 40)
        y_train = np.random.randint(0, 4, 20)
        X_test = np.random.randn(10, 40)
        y_test = np.random.randint(0, 4, 10)
        
        train_data = (X_train, y_train)
        test_data = (X_test, y_test)
        
        # Run study with reduced parameters for testing
        with patch.object(framework.model_comparison, 'compare_models') as mock_compare:
            # Mock the comparison results
            mock_compare.return_value = {
                'lnn': [PerformanceMetrics(
                    accuracy=0.9, precision=0.9, recall=0.9, f1_score=0.9,
                    inference_time_ms=8.0, power_consumption_mw=1.2,
                    model_size_bytes=65536, memory_usage_mb=2.0,
                    throughput_samples_per_sec=125, latency_p95_ms=9.0,
                    energy_per_inference_uj=9.6
                )],
                'baselines': {
                    'CNN': [PerformanceMetrics(
                        accuracy=0.85, precision=0.85, recall=0.85, f1_score=0.85,
                        inference_time_ms=15.0, power_consumption_mw=8.5,
                        model_size_bytes=1048576, memory_usage_mb=8.0,
                        throughput_samples_per_sec=66.7, latency_p95_ms=18.0,
                        energy_per_inference_uj=127.5
                    )]
                },
                'statistical_tests': {
                    'CNN': {
                        'power': StatisticalTest(
                            test_name="Power test",
                            statistic=-5.2,
                            p_value=0.001,
                            effect_size=2.1,
                            confidence_interval=(5.0, 9.0),
                            is_significant=True,
                            interpretation="Significant power improvement"
                        )
                    }
                },
                'power_analysis': {},
                'summary': {
                    'overall_recommendation': 'LNN recommended for power-constrained deployments',
                    'key_findings': ['LNN shows significant power improvement'],
                    'performance_ranking': [('LNN', {'accuracy': 0.9})],
                    'power_efficiency_ranking': [('LNN', {'power': 1.2})]
                }
            }
            
            results = framework.run_comparative_study(
                lnn_model, train_data, test_data, study_name="Test Study"
            )
        
        # Check results structure
        assert 'study_metadata' in results
        assert results['study_metadata']['study_name'] == "Test Study"
        assert 'lnn' in results
        assert 'baselines' in results
        assert 'summary' in results
        
        # Check that study was added to history
        assert len(framework.results_history) == 1
    
    def test_generate_research_report(self):
        """Test research report generation."""
        framework = ComparativeStudyFramework()
        
        # Create mock results
        mock_results = {
            'study_metadata': {
                'study_name': 'Test Study',
                'timestamp': '2024-01-01T12:00:00',
                'train_samples': 100,
                'test_samples': 50,
                'feature_dim': 40,
                'num_baselines': 2
            },
            'summary': {
                'overall_recommendation': 'LNN recommended',
                'key_findings': ['Finding 1', 'Finding 2'],
                'performance_ranking': [('LNN', {'accuracy': 0.9}), ('CNN', {'accuracy': 0.85})],
                'power_efficiency_ranking': [('LNN', {'power': 1.2}), ('CNN', {'power': 8.5})],
                'statistical_confidence': {'power_claims': '2/2 comparisons show improvement'},
                'deployment_recommendations': ['Suitable for IoT devices']
            },
            'statistical_tests': {
                'CNN': {
                    'accuracy': StatisticalTest(
                        test_name="Accuracy test",
                        statistic=2.1,
                        p_value=0.03,
                        effect_size=0.8,
                        confidence_interval=(0.02, 0.08),
                        is_significant=True,
                        interpretation="LNN shows higher accuracy"
                    )
                }
            },
            'power_analysis': {
                'CNN': StatisticalTest(
                    test_name="Power analysis",
                    statistic=-5.2,
                    p_value=0.001,
                    effect_size=2.1,
                    confidence_interval=(5.0, 9.0),
                    is_significant=True,
                    interpretation="Strong evidence supports power improvement"
                )
            }
        }
        
        report = framework.generate_research_report(mock_results)
        
        # Check report structure
        assert '# Test Study' in report
        assert 'Executive Summary' in report
        assert 'Performance Rankings' in report
        assert 'Statistical Analysis' in report
        assert 'Power Efficiency Claims Validation' in report
        assert 'LNN recommended' in report
    
    def test_export_results(self):
        """Test exporting results to file."""
        framework = ComparativeStudyFramework()
        
        # Create simple mock results
        mock_results = {
            'study_metadata': {'study_name': 'Test'},
            'summary': {'overall_recommendation': 'Test recommendation'},
            'statistical_tests': {
                'CNN': {
                    'accuracy': StatisticalTest(
                        test_name="Test",
                        statistic=1.0,
                        p_value=0.05,
                        effect_size=0.5,
                        confidence_interval=(0.1, 0.9),
                        is_significant=True,
                        interpretation="Test interpretation"
                    )
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            framework.export_results(mock_results, temp_path)
            
            # Check that file was created and contains expected content
            import json
            with open(temp_path, 'r') as f:
                exported_data = json.load(f)
            
            assert 'study_metadata' in exported_data
            assert 'summary' in exported_data
            assert 'statistical_tests' in exported_data
            
        finally:
            # Clean up
            Path(temp_path).unlink(missing_ok=True)


class TestIntegration:
    """Integration tests for the complete framework."""
    
    @patch('time.perf_counter')
    def test_end_to_end_study(self, mock_time):
        """Test complete end-to-end comparative study."""
        # Mock time for consistent results
        mock_time.side_effect = [i * 0.001 for i in range(1000)]
        
        # Create framework
        framework = ComparativeStudyFramework(random_seed=42)
        framework.create_standard_baselines(input_dim=20, hidden_dim=8, output_dim=4)
        
        # Create simple test data
        X_train = np.random.randn(10, 20)
        y_train = np.random.randint(0, 4, 10)
        X_test = np.random.randn(5, 20)
        y_test = np.random.randint(0, 4, 5)
        
        # Create mock LNN
        lnn_model = MockLNNModel()
        
        # Patch the comparison to run quickly
        with patch.object(framework.model_comparison, 'compare_models') as mock_compare:
            mock_compare.return_value = {
                'lnn': [PerformanceMetrics(
                    accuracy=0.8, precision=0.8, recall=0.8, f1_score=0.8,
                    inference_time_ms=5.0, power_consumption_mw=1.0,
                    model_size_bytes=32768, memory_usage_mb=1.0,
                    throughput_samples_per_sec=200, latency_p95_ms=6.0,
                    energy_per_inference_uj=5.0
                )],
                'baselines': {},
                'statistical_tests': {},
                'power_analysis': {},
                'summary': {
                    'overall_recommendation': 'Test recommendation',
                    'key_findings': [],
                    'performance_ranking': [],
                    'power_efficiency_ranking': []
                }
            }
            
            # Run the study
            results = framework.run_comparative_study(
                lnn_model, 
                (X_train, y_train), 
                (X_test, y_test),
                study_name="Integration Test"
            )
        
        # Verify results
        assert results is not None
        assert 'study_metadata' in results
        assert results['study_metadata']['study_name'] == "Integration Test"
        
        # Generate report
        report = framework.generate_research_report(results)
        assert len(report) > 0
        assert "Integration Test" in report


if __name__ == "__main__":
    pytest.main([__file__])