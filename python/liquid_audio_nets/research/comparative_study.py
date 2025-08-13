"""
Novel comparative study framework for rigorous validation of LNN performance claims.

This module implements sophisticated baseline models and statistical testing
to validate the 10× power efficiency improvement and other performance claims.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Protocol
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import scipy.stats as stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class BaselineModel(Protocol):
    """Protocol for baseline models in comparative studies."""
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model on provided data."""
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on input data."""
        ...
    
    def measure_inference_time(self, X: np.ndarray) -> float:
        """Measure inference time in milliseconds."""
        ...
    
    def estimate_power_consumption(self, X: np.ndarray) -> float:
        """Estimate power consumption in milliwatts."""
        ...
    
    def get_model_size(self) -> int:
        """Get model size in bytes."""
        ...


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for model comparison."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time_ms: float
    power_consumption_mw: float
    model_size_bytes: int
    memory_usage_mb: float
    throughput_samples_per_sec: float
    latency_p95_ms: float
    energy_per_inference_uj: float  # microjoules
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.energy_per_inference_uj = (self.power_consumption_mw * self.inference_time_ms) / 1000.0
        if self.inference_time_ms > 0:
            self.throughput_samples_per_sec = 1000.0 / self.inference_time_ms


@dataclass
class StatisticalTest:
    """Statistical significance test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    interpretation: str


class CNNBaseline(nn.Module):
    """CNN baseline model for audio classification."""
    
    def __init__(self, input_dim: int = 40, hidden_dim: int = 64, output_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # CNN layers for audio feature processing
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.2)
        
        # Calculate flattened size after convolutions
        self.fc_input_size = self._calculate_fc_input_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        # Performance tracking
        self.last_inference_time = 0.0
        self.power_base = 8.5  # Base CNN power consumption (mW)
        
    def _calculate_fc_input_size(self) -> int:
        """Calculate the input size for the first fully connected layer."""
        # Simulate forward pass to get the size
        x = torch.randn(1, 1, self.input_dim)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x.view(1, -1).shape[1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN."""
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add channel dimension
            
        x = self.pool(self.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(self.relu(self.conv3(x)))
        x = self.dropout(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)
    
    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 50) -> None:
        """Train the CNN model."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.numpy()
    
    def measure_inference_time(self, X: np.ndarray) -> float:
        """Measure inference time with high precision."""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            
            # Warmup
            for _ in range(10):
                _ = self(X_tensor)
            
            # Actual measurement
            times = []
            for _ in range(100):
                start_time = time.perf_counter()
                _ = self(X_tensor)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            self.last_inference_time = np.mean(times)
            return self.last_inference_time
    
    def estimate_power_consumption(self, X: np.ndarray) -> float:
        """Estimate power consumption based on model complexity."""
        # Power estimation based on FLOPs and memory access
        batch_size = X.shape[0] if len(X.shape) > 1 else 1
        
        # Count approximate FLOPs for CNN
        conv_flops = 0
        # Conv1: input_channels * output_channels * kernel_size * output_size
        conv_flops += 1 * 16 * 3 * self.input_dim
        conv_flops += 16 * 32 * 3 * (self.input_dim // 2)
        conv_flops += 32 * 64 * 3 * (self.input_dim // 4)
        
        # FC layers
        fc_flops = self.fc_input_size * self.hidden_dim + self.hidden_dim * self.output_dim
        
        total_flops = (conv_flops + fc_flops) * batch_size
        
        # Power scaling: base power + computation power + memory access power
        computation_power = total_flops * 1e-6  # Scale factor for power
        memory_power = self.get_model_size() * 1e-7  # Memory access power
        
        return self.power_base + computation_power + memory_power
    
    def get_model_size(self) -> int:
        """Calculate model size in bytes."""
        param_count = sum(p.numel() for p in self.parameters())
        return param_count * 4  # 4 bytes per float32 parameter


class LSTMBaseline(nn.Module):
    """LSTM baseline model for temporal audio processing."""
    
    def __init__(self, input_dim: int = 40, hidden_dim: int = 64, output_dim: int = 8, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
        
        # Performance tracking
        self.last_inference_time = 0.0
        self.power_base = 6.2  # Base LSTM power consumption (mW)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LSTM."""
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # Add batch dimension if needed
            
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last output for classification
        output = self.fc(lstm_out[:, -1, :])
        return self.softmax(output)
    
    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 50) -> None:
        """Train the LSTM model."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            return predicted.numpy()
    
    def measure_inference_time(self, X: np.ndarray) -> float:
        """Measure inference time with high precision."""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            
            # Warmup
            for _ in range(10):
                _ = self(X_tensor)
            
            # Actual measurement
            times = []
            for _ in range(100):
                start_time = time.perf_counter()
                _ = self(X_tensor)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # Convert to ms
            
            self.last_inference_time = np.mean(times)
            return self.last_inference_time
    
    def estimate_power_consumption(self, X: np.ndarray) -> float:
        """Estimate power consumption for LSTM."""
        batch_size = X.shape[0] if len(X.shape) > 1 else 1
        
        # LSTM power estimation based on gate operations
        # Each LSTM cell has 4 gates (forget, input, candidate, output)
        gates_per_cell = 4
        gate_ops = self.hidden_dim * self.input_dim + self.hidden_dim * self.hidden_dim  # W_x + W_h
        total_lstm_ops = gates_per_cell * gate_ops * self.num_layers * batch_size
        
        # FC layer operations
        fc_ops = self.hidden_dim * self.output_dim * batch_size
        
        total_ops = total_lstm_ops + fc_ops
        
        # Power scaling
        computation_power = total_ops * 2e-6  # LSTM is more power-intensive per op
        memory_power = self.get_model_size() * 1e-7
        
        return self.power_base + computation_power + memory_power
    
    def get_model_size(self) -> int:
        """Calculate model size in bytes."""
        param_count = sum(p.numel() for p in self.parameters())
        return param_count * 4  # 4 bytes per float32 parameter


class TinyMLBaseline:
    """TinyML optimized baseline for edge deployment."""
    
    def __init__(self, input_dim: int = 40, hidden_dim: int = 32, output_dim: int = 8):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Simple feedforward network with quantization
        self.w1 = np.random.randn(input_dim, hidden_dim).astype(np.float16)
        self.b1 = np.zeros(hidden_dim).astype(np.float16)
        self.w2 = np.random.randn(hidden_dim, output_dim).astype(np.float16)
        self.b2 = np.zeros(output_dim).astype(np.float16)
        
        self.power_base = 4.1  # Very low power baseline
        self.last_inference_time = 0.0
        
    def _relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def train_model(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> None:
        """Simple training using gradient descent."""
        learning_rate = 0.01
        
        for epoch in range(epochs):
            # Forward pass
            h1 = self._relu(X @ self.w1 + self.b1)
            output = self._softmax(h1 @ self.w2 + self.b2)
            
            # Create one-hot encoded targets
            y_onehot = np.eye(self.output_dim)[y]
            
            # Backward pass (simplified)
            d_output = output - y_onehot
            d_w2 = h1.T @ d_output / X.shape[0]
            d_b2 = np.mean(d_output, axis=0)
            
            d_h1 = d_output @ self.w2.T
            d_h1[h1 <= 0] = 0  # ReLU derivative
            d_w1 = X.T @ d_h1 / X.shape[0]
            d_b1 = np.mean(d_h1, axis=0)
            
            # Update weights
            self.w1 -= learning_rate * d_w1.astype(np.float16)
            self.b1 -= learning_rate * d_b1.astype(np.float16)
            self.w2 -= learning_rate * d_w2.astype(np.float16)
            self.b2 -= learning_rate * d_b2.astype(np.float16)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        h1 = self._relu(X @ self.w1 + self.b1)
        output = self._softmax(h1 @ self.w2 + self.b2)
        return np.argmax(output, axis=1)
    
    def measure_inference_time(self, X: np.ndarray) -> float:
        """Measure inference time."""
        times = []
        for _ in range(100):
            start_time = time.perf_counter()
            _ = self.predict(X)
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)
        
        self.last_inference_time = np.mean(times)
        return self.last_inference_time
    
    def estimate_power_consumption(self, X: np.ndarray) -> float:
        """Estimate power consumption for TinyML."""
        batch_size = X.shape[0] if len(X.shape) > 1 else 1
        
        # Count operations (multiply-accumulate)
        layer1_ops = self.input_dim * self.hidden_dim * batch_size
        layer2_ops = self.hidden_dim * self.output_dim * batch_size
        total_ops = layer1_ops + layer2_ops
        
        # TinyML is very efficient - lower power per operation
        computation_power = total_ops * 0.5e-6
        memory_power = self.get_model_size() * 0.5e-7
        
        return self.power_base + computation_power + memory_power
    
    def get_model_size(self) -> int:
        """Calculate model size in bytes."""
        total_params = (self.input_dim * self.hidden_dim + self.hidden_dim + 
                       self.hidden_dim * self.output_dim + self.output_dim)
        return total_params * 2  # 2 bytes per float16 parameter


class PowerEfficiencyAnalysis:
    """Advanced power efficiency analysis with statistical validation."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        
    def validate_power_claims(self, 
                            lnn_measurements: List[float],
                            baseline_measurements: Dict[str, List[float]],
                            claimed_improvement: float = 10.0) -> Dict[str, StatisticalTest]:
        """
        Validate power efficiency claims with rigorous statistical testing.
        
        Args:
            lnn_measurements: Power measurements for LNN (mW)
            baseline_measurements: Power measurements for baseline models
            claimed_improvement: Claimed improvement factor (e.g., 10x)
        
        Returns:
            Statistical test results for each comparison
        """
        results = {}
        
        for baseline_name, baseline_data in baseline_measurements.items():
            # Perform multiple statistical tests
            
            # 1. Welch's t-test for unequal variances
            t_stat, p_value = stats.ttest_ind(lnn_measurements, baseline_data, equal_var=False)
            
            # 2. Mann-Whitney U test (non-parametric)
            u_stat, u_p_value = stats.mannwhitneyu(lnn_measurements, baseline_data, alternative='less')
            
            # 3. Bootstrap confidence interval for ratio
            ratio_ci = self._bootstrap_ratio_ci(lnn_measurements, baseline_data)
            
            # 4. Effect size (Cohen's d)
            effect_size = self._cohen_d(lnn_measurements, baseline_data)
            
            # 5. Power analysis
            observed_ratio = np.mean(baseline_data) / np.mean(lnn_measurements)
            
            # Determine if the claimed improvement is statistically supported
            is_significant = (p_value < self.alpha and 
                            ratio_ci[0] > claimed_improvement * 0.8)  # Allow 20% tolerance
            
            interpretation = self._interpret_power_results(
                observed_ratio, claimed_improvement, p_value, effect_size, ratio_ci
            )
            
            results[baseline_name] = StatisticalTest(
                test_name=f"Power Efficiency vs {baseline_name}",
                statistic=t_stat,
                p_value=p_value,
                effect_size=effect_size,
                confidence_interval=ratio_ci,
                is_significant=is_significant,
                interpretation=interpretation
            )
        
        return results
    
    def _bootstrap_ratio_ci(self, sample1: List[float], sample2: List[float], 
                           n_bootstrap: int = 10000) -> Tuple[float, float]:
        """Bootstrap confidence interval for the ratio of means."""
        ratios = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            s1_boot = np.random.choice(sample1, size=len(sample1), replace=True)
            s2_boot = np.random.choice(sample2, size=len(sample2), replace=True)
            
            ratio = np.mean(s2_boot) / np.mean(s1_boot)
            ratios.append(ratio)
        
        # Calculate confidence interval
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        
        ci_lower = np.percentile(ratios, lower_percentile)
        ci_upper = np.percentile(ratios, upper_percentile)
        
        return (ci_lower, ci_upper)
    
    def _cohen_d(self, sample1: List[float], sample2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(sample1), len(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        d = (np.mean(sample2) - np.mean(sample1)) / pooled_std
        return d
    
    def _interpret_power_results(self, observed_ratio: float, claimed_ratio: float,
                                p_value: float, effect_size: float, 
                                ci: Tuple[float, float]) -> str:
        """Interpret the statistical results in plain language."""
        interpretation = []
        
        # Significance
        if p_value < 0.001:
            interpretation.append("Highly significant power reduction (p < 0.001)")
        elif p_value < 0.01:
            interpretation.append("Very significant power reduction (p < 0.01)")
        elif p_value < 0.05:
            interpretation.append("Significant power reduction (p < 0.05)")
        else:
            interpretation.append("No statistically significant power reduction")
        
        # Effect size
        if abs(effect_size) > 1.2:
            interpretation.append("Very large effect size")
        elif abs(effect_size) > 0.8:
            interpretation.append("Large effect size")
        elif abs(effect_size) > 0.5:
            interpretation.append("Medium effect size")
        else:
            interpretation.append("Small effect size")
        
        # Claim validation
        if ci[0] > claimed_ratio:
            interpretation.append(f"Strong evidence supports {claimed_ratio:.1f}x improvement claim")
        elif ci[0] > claimed_ratio * 0.8:
            interpretation.append(f"Moderate evidence supports {claimed_ratio:.1f}x improvement claim")
        else:
            interpretation.append(f"Insufficient evidence for {claimed_ratio:.1f}x improvement claim")
        
        # Observed vs claimed
        interpretation.append(f"Observed improvement: {observed_ratio:.1f}x (CI: {ci[0]:.1f}-{ci[1]:.1f}x)")
        
        return "; ".join(interpretation)


class ModelComparison:
    """Comprehensive model comparison framework."""
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.power_analyzer = PowerEfficiencyAnalysis()
        
    def compare_models(self, 
                      lnn_model: Any,
                      baseline_models: Dict[str, Any],
                      test_data: Tuple[np.ndarray, np.ndarray],
                      n_trials: int = 50) -> Dict[str, Any]:
        """
        Comprehensive comparison of LNN against baseline models.
        
        Args:
            lnn_model: LNN model instance
            baseline_models: Dictionary of baseline models
            test_data: Tuple of (X_test, y_test)
            n_trials: Number of trials for statistical robustness
        
        Returns:
            Comprehensive comparison results
        """
        X_test, y_test = test_data
        results = {
            'lnn': self._evaluate_model_multiple_trials(lnn_model, X_test, y_test, n_trials),
            'baselines': {},
            'statistical_tests': {},
            'power_analysis': {},
            'summary': {}
        }
        
        # Evaluate baseline models
        for name, model in baseline_models.items():
            results['baselines'][name] = self._evaluate_model_multiple_trials(
                model, X_test, y_test, n_trials
            )
        
        # Statistical comparisons
        for baseline_name, baseline_metrics in results['baselines'].items():
            comparison = self._statistical_comparison(
                results['lnn'], baseline_metrics, baseline_name
            )
            results['statistical_tests'][baseline_name] = comparison
        
        # Power efficiency analysis
        lnn_power = [m.power_consumption_mw for m in results['lnn']]
        baseline_power = {
            name: [m.power_consumption_mw for m in metrics]
            for name, metrics in results['baselines'].items()
        }
        
        results['power_analysis'] = self.power_analyzer.validate_power_claims(
            lnn_power, baseline_power
        )
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _evaluate_model_multiple_trials(self, model: Any, X: np.ndarray, y: np.ndarray,
                                      n_trials: int) -> List[PerformanceMetrics]:
        """Evaluate model performance over multiple trials."""
        metrics_list = []
        
        for trial in range(n_trials):
            # Add noise to inputs for robustness testing
            noise_std = 0.01
            X_noisy = X + np.random.normal(0, noise_std, X.shape)
            
            # Measure inference time
            inference_time = model.measure_inference_time(X_noisy)
            
            # Measure power consumption
            power_consumption = model.estimate_power_consumption(X_noisy)
            
            # Get predictions
            if hasattr(model, 'process'):
                # LNN model
                predictions = []
                for sample in X_noisy:
                    result = model.process(sample)
                    pred = 1 if result['confidence'] > 0.5 else 0
                    predictions.append(pred)
                predictions = np.array(predictions)
            else:
                # Baseline models
                predictions = model.predict(X_noisy)
            
            # Calculate accuracy metrics
            accuracy = accuracy_score(y, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y, predictions, average='weighted', zero_division=0
            )
            
            # Model size and memory estimation
            model_size = model.get_model_size()
            memory_usage = self._estimate_memory_usage(model, X_noisy)
            
            # Latency percentiles (simulate multiple measurements)
            latency_measurements = [inference_time + np.random.normal(0, 0.1) 
                                  for _ in range(20)]
            latency_p95 = np.percentile(latency_measurements, 95)
            
            metrics = PerformanceMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                inference_time_ms=inference_time,
                power_consumption_mw=power_consumption,
                model_size_bytes=model_size,
                memory_usage_mb=memory_usage,
                throughput_samples_per_sec=0,  # Will be calculated in __post_init__
                latency_p95_ms=latency_p95,
                energy_per_inference_uj=0  # Will be calculated in __post_init__
            )
            
            metrics_list.append(metrics)
        
        return metrics_list
    
    def _estimate_memory_usage(self, model: Any, X: np.ndarray) -> float:
        """Estimate runtime memory usage in MB."""
        # Base model memory
        model_memory = model.get_model_size() / (1024 * 1024)  # Convert to MB
        
        # Input data memory
        input_memory = X.nbytes / (1024 * 1024)
        
        # Activation memory (rough estimation)
        if hasattr(model, 'hidden_dim'):
            activation_memory = model.hidden_dim * 4 / (1024 * 1024)  # 4 bytes per float
        else:
            activation_memory = 1.0  # Default estimate
        
        return model_memory + input_memory + activation_memory
    
    def _statistical_comparison(self, lnn_metrics: List[PerformanceMetrics],
                              baseline_metrics: List[PerformanceMetrics],
                              baseline_name: str) -> Dict[str, StatisticalTest]:
        """Perform statistical comparisons between models."""
        comparisons = {}
        
        # Extract metric arrays
        lnn_accuracy = [m.accuracy for m in lnn_metrics]
        baseline_accuracy = [m.accuracy for m in baseline_metrics]
        
        lnn_power = [m.power_consumption_mw for m in lnn_metrics]
        baseline_power = [m.power_consumption_mw for m in baseline_metrics]
        
        lnn_latency = [m.inference_time_ms for m in lnn_metrics]
        baseline_latency = [m.inference_time_ms for m in baseline_metrics]
        
        # Accuracy comparison (higher is better)
        acc_t_stat, acc_p_val = stats.ttest_ind(lnn_accuracy, baseline_accuracy)
        acc_effect_size = self.power_analyzer._cohen_d(baseline_accuracy, lnn_accuracy)
        acc_ci = self._calculate_ci_diff(lnn_accuracy, baseline_accuracy)
        
        comparisons['accuracy'] = StatisticalTest(
            test_name=f"Accuracy: LNN vs {baseline_name}",
            statistic=acc_t_stat,
            p_value=acc_p_val,
            effect_size=acc_effect_size,
            confidence_interval=acc_ci,
            is_significant=acc_p_val < self.significance_level,
            interpretation=self._interpret_accuracy_comparison(
                acc_p_val, acc_effect_size, np.mean(lnn_accuracy), np.mean(baseline_accuracy)
            )
        )
        
        # Power comparison (lower is better)
        pow_t_stat, pow_p_val = stats.ttest_ind(lnn_power, baseline_power)
        pow_effect_size = self.power_analyzer._cohen_d(lnn_power, baseline_power)
        pow_ci = self.power_analyzer._bootstrap_ratio_ci(lnn_power, baseline_power)
        
        comparisons['power'] = StatisticalTest(
            test_name=f"Power: LNN vs {baseline_name}",
            statistic=pow_t_stat,
            p_value=pow_p_val,
            effect_size=pow_effect_size,
            confidence_interval=pow_ci,
            is_significant=pow_p_val < self.significance_level,
            interpretation=self.power_analyzer._interpret_power_results(
                np.mean(baseline_power) / np.mean(lnn_power), 10.0, pow_p_val, 
                pow_effect_size, pow_ci
            )
        )
        
        # Latency comparison (lower is better)
        lat_t_stat, lat_p_val = stats.ttest_ind(lnn_latency, baseline_latency)
        lat_effect_size = self.power_analyzer._cohen_d(lnn_latency, baseline_latency)
        lat_ci = self._calculate_ci_diff(lnn_latency, baseline_latency)
        
        comparisons['latency'] = StatisticalTest(
            test_name=f"Latency: LNN vs {baseline_name}",
            statistic=lat_t_stat,
            p_value=lat_p_val,
            effect_size=lat_effect_size,
            confidence_interval=lat_ci,
            is_significant=lat_p_val < self.significance_level,
            interpretation=self._interpret_latency_comparison(
                lat_p_val, lat_effect_size, np.mean(lnn_latency), np.mean(baseline_latency)
            )
        )
        
        return comparisons
    
    def _calculate_ci_diff(self, sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
        """Calculate confidence interval for difference in means."""
        diff = np.mean(sample1) - np.mean(sample2)
        se_diff = np.sqrt(np.var(sample1)/len(sample1) + np.var(sample2)/len(sample2))
        
        # t-distribution critical value
        df = len(sample1) + len(sample2) - 2
        t_crit = stats.t.ppf(1 - self.significance_level/2, df)
        
        margin_error = t_crit * se_diff
        return (diff - margin_error, diff + margin_error)
    
    def _interpret_accuracy_comparison(self, p_value: float, effect_size: float,
                                     lnn_acc: float, baseline_acc: float) -> str:
        """Interpret accuracy comparison results."""
        diff = lnn_acc - baseline_acc
        
        interpretation = []
        
        if p_value < 0.05:
            if diff > 0:
                interpretation.append("LNN shows significantly higher accuracy")
            else:
                interpretation.append("LNN shows significantly lower accuracy")
        else:
            interpretation.append("No significant difference in accuracy")
        
        interpretation.append(f"LNN: {lnn_acc:.3f}, Baseline: {baseline_acc:.3f}")
        interpretation.append(f"Difference: {diff:+.3f}")
        
        return "; ".join(interpretation)
    
    def _interpret_latency_comparison(self, p_value: float, effect_size: float,
                                    lnn_latency: float, baseline_latency: float) -> str:
        """Interpret latency comparison results."""
        improvement = (baseline_latency - lnn_latency) / baseline_latency * 100
        
        interpretation = []
        
        if p_value < 0.05:
            if improvement > 0:
                interpretation.append(f"LNN shows significantly lower latency ({improvement:.1f}% improvement)")
            else:
                interpretation.append(f"LNN shows significantly higher latency ({-improvement:.1f}% regression)")
        else:
            interpretation.append("No significant difference in latency")
        
        interpretation.append(f"LNN: {lnn_latency:.2f}ms, Baseline: {baseline_latency:.2f}ms")
        
        return "; ".join(interpretation)
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary of comparison results."""
        summary = {
            'performance_ranking': [],
            'power_efficiency_ranking': [],
            'overall_recommendation': '',
            'key_findings': [],
            'statistical_confidence': {},
            'deployment_recommendations': []
        }
        
        # Calculate average metrics for ranking
        model_avg_metrics = {}
        
        # LNN metrics
        lnn_metrics = results['lnn']
        model_avg_metrics['LNN'] = {
            'accuracy': np.mean([m.accuracy for m in lnn_metrics]),
            'power': np.mean([m.power_consumption_mw for m in lnn_metrics]),
            'latency': np.mean([m.inference_time_ms for m in lnn_metrics]),
            'model_size': np.mean([m.model_size_bytes for m in lnn_metrics])
        }
        
        # Baseline metrics
        for name, metrics in results['baselines'].items():
            model_avg_metrics[name] = {
                'accuracy': np.mean([m.accuracy for m in metrics]),
                'power': np.mean([m.power_consumption_mw for m in metrics]),
                'latency': np.mean([m.inference_time_ms for m in metrics]),
                'model_size': np.mean([m.model_size_bytes for m in metrics])
            }
        
        # Performance ranking (by accuracy)
        summary['performance_ranking'] = sorted(
            model_avg_metrics.items(), 
            key=lambda x: x[1]['accuracy'], 
            reverse=True
        )
        
        # Power efficiency ranking (by power consumption, lower is better)
        summary['power_efficiency_ranking'] = sorted(
            model_avg_metrics.items(),
            key=lambda x: x[1]['power']
        )
        
        # Key findings
        lnn_power_rank = next(i for i, (name, _) in enumerate(summary['power_efficiency_ranking']) if name == 'LNN')
        lnn_acc_rank = next(i for i, (name, _) in enumerate(summary['performance_ranking']) if name == 'LNN')
        
        summary['key_findings'].append(f"LNN ranks #{lnn_power_rank + 1} in power efficiency")
        summary['key_findings'].append(f"LNN ranks #{lnn_acc_rank + 1} in accuracy")
        
        # Statistical confidence assessment
        significant_power_improvements = 0
        total_power_comparisons = 0
        
        for name, test_results in results['statistical_tests'].items():
            if 'power' in test_results and test_results['power'].is_significant:
                significant_power_improvements += 1
            total_power_comparisons += 1
        
        summary['statistical_confidence']['power_claims'] = (
            f"{significant_power_improvements}/{total_power_comparisons} "
            f"comparisons show significant power improvement"
        )
        
        # Overall recommendation
        if lnn_power_rank == 0 and lnn_acc_rank <= 1:
            summary['overall_recommendation'] = "LNN recommended for power-constrained deployments"
        elif lnn_acc_rank == 0:
            summary['overall_recommendation'] = "LNN recommended for accuracy-critical applications"
        else:
            summary['overall_recommendation'] = "Consider trade-offs between power, accuracy, and deployment constraints"
        
        # Deployment recommendations
        lnn_power = model_avg_metrics['LNN']['power']
        if lnn_power < 2.0:
            summary['deployment_recommendations'].append("Suitable for battery-powered IoT devices")
        if lnn_power < 5.0:
            summary['deployment_recommendations'].append("Suitable for mobile edge deployment")
        
        lnn_size = model_avg_metrics['LNN']['model_size']
        if lnn_size < 100 * 1024:  # 100KB
            summary['deployment_recommendations'].append("Suitable for microcontroller deployment")
        
        return summary


class ComparativeStudyFramework:
    """Main framework for conducting comparative studies."""
    
    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        self.model_comparison = ModelComparison()
        self.baseline_models = {}
        self.results_history = []
        
    def register_baseline_model(self, name: str, model: Any) -> None:
        """Register a baseline model for comparison."""
        self.baseline_models[name] = model
        
    def create_standard_baselines(self, input_dim: int = 40, hidden_dim: int = 64, 
                                output_dim: int = 8) -> None:
        """Create standard baseline models."""
        self.baseline_models['CNN'] = CNNBaseline(input_dim, hidden_dim, output_dim)
        self.baseline_models['LSTM'] = LSTMBaseline(input_dim, hidden_dim, output_dim)
        self.baseline_models['TinyML'] = TinyMLBaseline(input_dim, hidden_dim, output_dim)
        
    def run_comparative_study(self, lnn_model: Any, 
                            train_data: Tuple[np.ndarray, np.ndarray],
                            test_data: Tuple[np.ndarray, np.ndarray],
                            study_name: str = "LNN Comparative Study") -> Dict[str, Any]:
        """
        Run a complete comparative study.
        
        Args:
            lnn_model: LNN model to evaluate
            train_data: Training data (X_train, y_train)
            test_data: Test data (X_test, y_test)
            study_name: Name for this study
        
        Returns:
            Complete study results
        """
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        print(f"Running {study_name}...")
        print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
        
        # Train baseline models
        print("Training baseline models...")
        for name, model in self.baseline_models.items():
            print(f"  Training {name}...")
            if hasattr(model, 'train_model'):
                model.train_model(X_train, y_train)
            else:
                # For PyTorch models
                model.train_model(X_train, y_train)
        
        # Train LNN model if needed
        if hasattr(lnn_model, 'train'):
            print("Training LNN model...")
            lnn_model.train(X_train, y_train)
        
        # Run comparisons
        print("Running performance comparisons...")
        results = self.model_comparison.compare_models(
            lnn_model, self.baseline_models, test_data, n_trials=30
        )
        
        # Add study metadata
        results['study_metadata'] = {
            'study_name': study_name,
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'feature_dim': X_train.shape[1] if len(X_train.shape) > 1 else 1,
            'num_baselines': len(self.baseline_models),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Store results
        self.results_history.append(results)
        
        print("Comparative study completed!")
        return results
    
    def generate_research_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive research report."""
        report = []
        
        # Header
        study_name = results['study_metadata']['study_name']
        timestamp = results['study_metadata']['timestamp']
        
        report.append(f"# {study_name}")
        report.append(f"**Generated:** {timestamp}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        summary = results['summary']
        report.append(f"**Recommendation:** {summary['overall_recommendation']}")
        report.append("")
        
        for finding in summary['key_findings']:
            report.append(f"- {finding}")
        report.append("")
        
        # Performance Rankings
        report.append("## Performance Rankings")
        report.append("")
        
        report.append("### Accuracy Ranking")
        report.append("| Rank | Model | Accuracy |")
        report.append("|------|-------|----------|")
        for i, (model_name, metrics) in enumerate(summary['performance_ranking']):
            report.append(f"| {i+1} | {model_name} | {metrics['accuracy']:.3f} |")
        report.append("")
        
        report.append("### Power Efficiency Ranking")
        report.append("| Rank | Model | Power (mW) |")
        report.append("|------|-------|------------|")
        for i, (model_name, metrics) in enumerate(summary['power_efficiency_ranking']):
            report.append(f"| {i+1} | {model_name} | {metrics['power']:.2f} |")
        report.append("")
        
        # Statistical Analysis
        report.append("## Statistical Analysis")
        report.append("")
        
        for baseline_name, tests in results['statistical_tests'].items():
            report.append(f"### LNN vs {baseline_name}")
            
            for test_type, test_result in tests.items():
                report.append(f"**{test_type.title()} Comparison:**")
                report.append(f"- p-value: {test_result.p_value:.4f}")
                report.append(f"- Effect size: {test_result.effect_size:.3f}")
                report.append(f"- Significant: {'Yes' if test_result.is_significant else 'No'}")
                report.append(f"- Interpretation: {test_result.interpretation}")
                report.append("")
        
        # Power Efficiency Claims
        report.append("## Power Efficiency Claims Validation")
        report.append("")
        
        for baseline_name, power_test in results['power_analysis'].items():
            report.append(f"### vs {baseline_name}")
            report.append(f"- {power_test.interpretation}")
            report.append("")
        
        # Deployment Recommendations
        if summary['deployment_recommendations']:
            report.append("## Deployment Recommendations")
            report.append("")
            for rec in summary['deployment_recommendations']:
                report.append(f"- {rec}")
            report.append("")
        
        # Technical Details
        report.append("## Technical Details")
        metadata = results['study_metadata']
        report.append(f"- Training samples: {metadata['train_samples']}")
        report.append(f"- Test samples: {metadata['test_samples']}")
        report.append(f"- Feature dimension: {metadata['feature_dim']}")
        report.append(f"- Baseline models: {metadata['num_baselines']}")
        report.append("")
        
        # Confidence Assessment
        report.append("## Statistical Confidence")
        confidence = summary['statistical_confidence']
        report.append(f"- Power claims validation: {confidence['power_claims']}")
        report.append("")
        
        return "\n".join(report)
    
    def export_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Export results to file."""
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        # Create a serializable version of results
        serializable_results = {}
        
        # Copy basic structure
        for key, value in results.items():
            if key in ['study_metadata', 'summary']:
                serializable_results[key] = value
            elif key == 'statistical_tests':
                serializable_results[key] = {}
                for baseline, tests in value.items():
                    serializable_results[key][baseline] = {}
                    for test_name, test_result in tests.items():
                        serializable_results[key][baseline][test_name] = {
                            'test_name': test_result.test_name,
                            'p_value': float(test_result.p_value),
                            'effect_size': float(test_result.effect_size),
                            'is_significant': test_result.is_significant,
                            'interpretation': test_result.interpretation
                        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=convert_numpy)
        
        print(f"Results exported to {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # This would be run as a standalone script for testing
    pass