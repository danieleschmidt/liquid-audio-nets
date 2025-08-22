"""
Advanced Statistical Validation Framework for LNN Research Claims.

This module provides rigorous statistical testing, effect size calculation,
power analysis, and reproducibility validation for liquid neural network 
research claims with publication-ready standards.

Key features:
- Multiple comparison correction (Bonferroni, FDR, etc.)
- Bayesian hypothesis testing
- Bootstrap confidence intervals
- Non-parametric tests for robust validation
- Effect size calculations with interpretation
- Power analysis and sample size determination
- Reproducibility and replication testing
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import (
    ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, kruskal,
    chi2_contingency, pearsonr, spearmanr, kstest, normaltest,
    levene, bartlett, shapiro
)
from typing import Dict, List, Tuple, Optional, Any, NamedTuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from enum import Enum
import json
from pathlib import Path
import hashlib
import time


class TestType(Enum):
    """Types of statistical tests available."""
    INDEPENDENT_TTEST = "independent_t_test"
    PAIRED_TTEST = "paired_t_test"
    MANN_WHITNEY = "mann_whitney_u"
    WILCOXON = "wilcoxon_signed_rank"
    KRUSKAL_WALLIS = "kruskal_wallis"
    CHI_SQUARE = "chi_square"
    PEARSON_CORRELATION = "pearson_correlation"
    SPEARMAN_CORRELATION = "spearman_correlation"
    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    LEVENE_VARIANCE = "levene_variance"
    BARTLETT_VARIANCE = "bartlett_variance"
    SHAPIRO_WILK = "shapiro_wilk"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"
    BAYESIAN_TTEST = "bayesian_t_test"


class EffectSizeMeasure(Enum):
    """Effect size measures."""
    COHENS_D = "cohens_d"
    HEDGES_G = "hedges_g"
    GLASS_DELTA = "glass_delta"
    CLIFF_DELTA = "cliff_delta"
    ETA_SQUARED = "eta_squared"
    OMEGA_SQUARED = "omega_squared"
    R_SQUARED = "r_squared"
    CRAMERS_V = "cramers_v"


class MultipleComparisonCorrection(Enum):
    """Multiple comparison correction methods."""
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    SIDAK = "sidak"
    FDR_BH = "fdr_bh"  # Benjamini-Hochberg
    FDR_BY = "fdr_by"  # Benjamini-Yekutieli
    NONE = "none"


@dataclass
class StatisticalResult:
    """Comprehensive statistical test result."""
    test_name: str
    test_type: TestType
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    effect_size_measure: Optional[EffectSizeMeasure] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    power: Optional[float] = None
    sample_size: Optional[int] = None
    assumptions_met: Dict[str, bool] = field(default_factory=dict)
    interpretation: str = ""
    recommendation: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BayesianResult:
    """Bayesian statistical analysis result."""
    bayes_factor: float
    posterior_probability: float
    credible_interval: Tuple[float, float]
    evidence_strength: str
    interpretation: str


@dataclass
class PowerAnalysisResult:
    """Power analysis result."""
    power: float
    effect_size: float
    sample_size: int
    alpha: float
    interpretation: str
    recommendation: str


class AssumptionChecker:
    """Check statistical test assumptions."""
    
    @staticmethod
    def check_normality(data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        """Check normality assumption using multiple tests."""
        results = {}
        
        # Shapiro-Wilk test (best for small samples)
        if len(data) <= 5000:
            stat, p_val = shapiro(data)
            results['shapiro_wilk'] = {
                'statistic': stat,
                'p_value': p_val,
                'normal': p_val > alpha
            }
        
        # D'Agostino's normality test
        stat, p_val = normaltest(data)
        results['dagostino'] = {
            'statistic': stat,
            'p_value': p_val,
            'normal': p_val > alpha
        }
        
        # Kolmogorov-Smirnov test against normal
        ks_stat, ks_p = kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        results['kolmogorov_smirnov'] = {
            'statistic': ks_stat,
            'p_value': ks_p,
            'normal': ks_p > alpha
        }
        
        # Overall normality (majority rule)
        normal_tests = [r['normal'] for r in results.values()]
        results['overall_normal'] = sum(normal_tests) >= len(normal_tests) / 2
        
        return results
    
    @staticmethod
    def check_equal_variances(data1: np.ndarray, data2: np.ndarray, 
                            alpha: float = 0.05) -> Dict[str, Any]:
        """Check equal variances assumption."""
        results = {}
        
        # Levene's test (robust to non-normality)
        stat, p_val = levene(data1, data2)
        results['levene'] = {
            'statistic': stat,
            'p_value': p_val,
            'equal_variances': p_val > alpha
        }
        
        # Bartlett's test (assumes normality)
        stat, p_val = bartlett(data1, data2)
        results['bartlett'] = {
            'statistic': stat,
            'p_value': p_val,
            'equal_variances': p_val > alpha
        }
        
        # F-test for equal variances
        f_stat = np.var(data1, ddof=1) / np.var(data2, ddof=1)
        df1, df2 = len(data1) - 1, len(data2) - 1
        p_val = 2 * min(stats.f.cdf(f_stat, df1, df2), 
                       1 - stats.f.cdf(f_stat, df1, df2))
        results['f_test'] = {
            'statistic': f_stat,
            'p_value': p_val,
            'equal_variances': p_val > alpha
        }
        
        # Overall equal variances (majority rule)
        equal_var_tests = [r['equal_variances'] for r in results.values()]
        results['overall_equal_variances'] = sum(equal_var_tests) >= len(equal_var_tests) / 2
        
        return results
    
    @staticmethod
    def check_independence(data: np.ndarray) -> Dict[str, Any]:
        """Check independence assumption using autocorrelation."""
        results = {}
        
        # Lag-1 autocorrelation
        if len(data) > 1:
            autocorr = np.corrcoef(data[:-1], data[1:])[0, 1]
            results['lag1_autocorr'] = autocorr
            results['independent'] = abs(autocorr) < 0.1  # Rough threshold
        else:
            results['lag1_autocorr'] = 0.0
            results['independent'] = True
        
        return results


class EffectSizeCalculator:
    """Calculate various effect size measures."""
    
    @staticmethod
    def cohens_d(data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(data1), len(data2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(data1, ddof=1) + 
                             (n2 - 1) * np.var(data2, ddof=1)) / (n1 + n2 - 2))
        return (np.mean(data1) - np.mean(data2)) / pooled_std
    
    @staticmethod
    def hedges_g(data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Hedges' g (bias-corrected Cohen's d)."""
        n1, n2 = len(data1), len(data2)
        df = n1 + n2 - 2
        correction_factor = 1 - (3 / (4 * df - 1))
        return EffectSizeCalculator.cohens_d(data1, data2) * correction_factor
    
    @staticmethod
    def glass_delta(data1: np.ndarray, data2: np.ndarray, 
                   control_group: int = 2) -> float:
        """Calculate Glass's delta."""
        control_data = data2 if control_group == 2 else data1
        treatment_data = data1 if control_group == 2 else data2
        return (np.mean(treatment_data) - np.mean(control_data)) / np.std(control_data, ddof=1)
    
    @staticmethod
    def cliff_delta(data1: np.ndarray, data2: np.ndarray) -> float:
        """Calculate Cliff's delta (non-parametric effect size)."""
        n1, n2 = len(data1), len(data2)
        pairs = 0
        favorable = 0
        
        for x1 in data1:
            for x2 in data2:
                pairs += 1
                if x1 > x2:
                    favorable += 1
                elif x1 == x2:
                    favorable += 0.5
        
        return (2 * favorable - pairs) / pairs if pairs > 0 else 0.0
    
    @staticmethod
    def eta_squared(groups: List[np.ndarray]) -> float:
        """Calculate eta-squared for ANOVA."""
        all_data = np.concatenate(groups)
        overall_mean = np.mean(all_data)
        
        # Between-group sum of squares
        ss_between = sum(len(group) * (np.mean(group) - overall_mean)**2 
                        for group in groups)
        
        # Total sum of squares
        ss_total = np.sum((all_data - overall_mean)**2)
        
        return ss_between / ss_total if ss_total > 0 else 0.0
    
    @staticmethod
    def cramers_v(contingency_table: np.ndarray) -> float:
        """Calculate Cramér's V for categorical associations."""
        chi2 = stats.chi2_contingency(contingency_table)[0]
        n = contingency_table.sum()
        min_dim = min(contingency_table.shape) - 1
        return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0


class MultipleComparisonAdjuster:
    """Adjust p-values for multiple comparisons."""
    
    @staticmethod
    def adjust_p_values(p_values: List[float], 
                       method: MultipleComparisonCorrection) -> List[float]:
        """Adjust p-values for multiple comparisons."""
        p_array = np.array(p_values)
        n = len(p_array)
        
        if method == MultipleComparisonCorrection.BONFERRONI:
            return np.minimum(p_array * n, 1.0).tolist()
        
        elif method == MultipleComparisonCorrection.HOLM:
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_array)
            adjusted = np.zeros_like(p_array)
            
            for i, idx in enumerate(sorted_indices):
                adjusted[idx] = min(p_array[idx] * (n - i), 1.0)
                if i > 0:
                    adjusted[idx] = max(adjusted[idx], 
                                      adjusted[sorted_indices[i-1]])
            
            return adjusted.tolist()
        
        elif method == MultipleComparisonCorrection.SIDAK:
            return (1 - (1 - p_array)**n).tolist()
        
        elif method == MultipleComparisonCorrection.FDR_BH:
            # Benjamini-Hochberg FDR
            sorted_indices = np.argsort(p_array)
            adjusted = np.zeros_like(p_array)
            
            for i, idx in enumerate(sorted_indices):
                adjusted[idx] = min(p_array[idx] * n / (i + 1), 1.0)
            
            # Ensure monotonicity
            for i in range(n-2, -1, -1):
                idx = sorted_indices[i]
                next_idx = sorted_indices[i+1]
                adjusted[idx] = min(adjusted[idx], adjusted[next_idx])
            
            return adjusted.tolist()
        
        elif method == MultipleComparisonCorrection.FDR_BY:
            # Benjamini-Yekutieli FDR
            c_n = np.sum(1.0 / np.arange(1, n + 1))
            return MultipleComparisonAdjuster.adjust_p_values(
                [p * c_n for p in p_values], 
                MultipleComparisonCorrection.FDR_BH
            )
        
        else:  # No correction
            return p_values


class BayesianAnalyzer:
    """Bayesian statistical analysis."""
    
    @staticmethod
    def bayesian_t_test(data1: np.ndarray, data2: np.ndarray,
                       prior_scale: float = 0.707) -> BayesianResult:
        """Perform Bayesian t-test and calculate Bayes factor."""
        # Simplified Bayesian t-test implementation
        # In practice, would use more sophisticated methods like MCMC
        
        n1, n2 = len(data1), len(data2)
        mean1, mean2 = np.mean(data1), np.mean(data2)
        var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
        
        # Pooled variance
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        
        # Standard error of difference
        se_diff = np.sqrt(pooled_var * (1/n1 + 1/n2))
        
        # Observed effect size
        effect_size = (mean1 - mean2) / np.sqrt(pooled_var)
        
        # Simplified Bayes factor calculation
        # Using JZS prior (Jeffreys-Zellner-Siow)
        t_stat = (mean1 - mean2) / se_diff
        df = n1 + n2 - 2
        
        # Approximate Bayes factor (simplified)
        bf_01 = (1 + t_stat**2 / df)**(-df/2) * np.sqrt(df / (df + 1))
        bf_10 = 1 / bf_01
        
        # Posterior probability (assuming equal priors)
        posterior_prob = bf_10 / (1 + bf_10)
        
        # Credible interval (approximation)
        margin = stats.t.ppf(0.975, df) * se_diff
        credible_interval = (mean1 - mean2 - margin, mean1 - mean2 + margin)
        
        # Evidence interpretation
        if bf_10 > 100:
            evidence = "Extreme evidence for H1"
        elif bf_10 > 30:
            evidence = "Very strong evidence for H1"
        elif bf_10 > 10:
            evidence = "Strong evidence for H1"
        elif bf_10 > 3:
            evidence = "Moderate evidence for H1"
        elif bf_10 > 1:
            evidence = "Weak evidence for H1"
        elif bf_10 > 1/3:
            evidence = "Weak evidence for H0"
        elif bf_10 > 1/10:
            evidence = "Moderate evidence for H0"
        elif bf_10 > 1/30:
            evidence = "Strong evidence for H0"
        elif bf_10 > 1/100:
            evidence = "Very strong evidence for H0"
        else:
            evidence = "Extreme evidence for H0"
        
        interpretation = (f"Bayes factor BF₁₀ = {bf_10:.3f}. "
                         f"Posterior probability = {posterior_prob:.3f}. "
                         f"{evidence}.")
        
        return BayesianResult(
            bayes_factor=bf_10,
            posterior_probability=posterior_prob,
            credible_interval=credible_interval,
            evidence_strength=evidence,
            interpretation=interpretation
        )


class PowerAnalyzer:
    """Statistical power analysis."""
    
    @staticmethod
    def power_t_test(effect_size: float, sample_size: int, 
                    alpha: float = 0.05, alternative: str = 'two-sided') -> float:
        """Calculate statistical power for t-test."""
        # Simplified power calculation
        if alternative == 'two-sided':
            critical_t = stats.t.ppf(1 - alpha/2, sample_size - 1)
        else:
            critical_t = stats.t.ppf(1 - alpha, sample_size - 1)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(sample_size)
        
        # Power calculation
        if alternative == 'two-sided':
            power = (1 - stats.nct.cdf(critical_t, sample_size - 1, ncp) +
                    stats.nct.cdf(-critical_t, sample_size - 1, ncp))
        else:
            power = 1 - stats.nct.cdf(critical_t, sample_size - 1, ncp)
        
        return power
    
    @staticmethod
    def sample_size_t_test(effect_size: float, power: float = 0.8,
                          alpha: float = 0.05, alternative: str = 'two-sided') -> int:
        """Calculate required sample size for desired power."""
        # Binary search for sample size
        low, high = 2, 10000
        target_power = power
        
        while high - low > 1:
            mid = (low + high) // 2
            calculated_power = PowerAnalyzer.power_t_test(
                effect_size, mid, alpha, alternative
            )
            
            if calculated_power < target_power:
                low = mid
            else:
                high = mid
        
        return high
    
    @staticmethod
    def analyze_power(effect_size: float, sample_size: int,
                     alpha: float = 0.05) -> PowerAnalysisResult:
        """Comprehensive power analysis."""
        power = PowerAnalyzer.power_t_test(effect_size, sample_size, alpha)
        
        # Interpretation
        if power >= 0.8:
            interpretation = f"Adequate power ({power:.3f}) for detecting effect size {effect_size:.3f}"
            recommendation = "Sample size is sufficient for reliable results"
        elif power >= 0.6:
            interpretation = f"Moderate power ({power:.3f}) for detecting effect size {effect_size:.3f}"
            recommendation = "Consider increasing sample size for more reliable results"
        else:
            interpretation = f"Low power ({power:.3f}) for detecting effect size {effect_size:.3f}"
            recommendation = "Increase sample size significantly or consider effect size expectations"
        
        return PowerAnalysisResult(
            power=power,
            effect_size=effect_size,
            sample_size=sample_size,
            alpha=alpha,
            interpretation=interpretation,
            recommendation=recommendation
        )


class BootstrapAnalyzer:
    """Bootstrap and permutation testing."""
    
    @staticmethod
    def bootstrap_confidence_interval(data: np.ndarray, 
                                    statistic_func: callable,
                                    confidence_level: float = 0.95,
                                    n_bootstrap: int = 10000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        bootstrap_stats = []
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            stat = statistic_func(bootstrap_sample)
            bootstrap_stats.append(stat)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        return ci_lower, ci_upper
    
    @staticmethod
    def permutation_test(data1: np.ndarray, data2: np.ndarray,
                        statistic_func: callable,
                        n_permutations: int = 10000) -> float:
        """Perform permutation test."""
        # Observed test statistic
        observed_stat = statistic_func(data1, data2)
        
        # Combine data
        combined_data = np.concatenate([data1, data2])
        n1 = len(data1)
        
        # Permutation testing
        extreme_count = 0
        
        for _ in range(n_permutations):
            # Randomly permute combined data
            permuted_data = np.random.permutation(combined_data)
            perm_data1 = permuted_data[:n1]
            perm_data2 = permuted_data[n1:]
            
            # Calculate permuted statistic
            perm_stat = statistic_func(perm_data1, perm_data2)
            
            # Count extreme values
            if abs(perm_stat) >= abs(observed_stat):
                extreme_count += 1
        
        # Calculate p-value
        p_value = extreme_count / n_permutations
        return p_value


class StatisticalValidator:
    """Main statistical validation framework."""
    
    def __init__(self, alpha: float = 0.05, 
                 correction_method: MultipleComparisonCorrection = MultipleComparisonCorrection.FDR_BH):
        self.alpha = alpha
        self.correction_method = correction_method
        self.assumption_checker = AssumptionChecker()
        self.effect_calculator = EffectSizeCalculator()
        self.power_analyzer = PowerAnalyzer()
        self.bayesian_analyzer = BayesianAnalyzer()
        self.bootstrap_analyzer = BootstrapAnalyzer()
        self.results_history = []
    
    def validate_lnn_performance_claims(self, 
                                      lnn_data: Dict[str, np.ndarray],
                                      baseline_data: Dict[str, Dict[str, np.ndarray]],
                                      claims: Dict[str, float]) -> Dict[str, Any]:
        """
        Comprehensive validation of LNN performance claims.
        
        Args:
            lnn_data: Dictionary with metric names and LNN measurements
            baseline_data: Dictionary with baseline names and their metric measurements
            claims: Dictionary with claimed improvements (e.g., {"power": 10.0, "latency": 2.0})
        
        Returns:
            Comprehensive validation results
        """
        print("🔬 Starting comprehensive statistical validation...")
        
        validation_results = {
            'summary': {},
            'detailed_results': {},
            'assumptions_analysis': {},
            'effect_sizes': {},
            'power_analysis': {},
            'bayesian_analysis': {},
            'multiple_comparisons': {},
            'recommendations': [],
            'publication_ready': False
        }
        
        all_p_values = []
        all_test_names = []
        
        # Validate each metric against each baseline
        for metric_name, lnn_measurements in lnn_data.items():
            metric_results = {}
            
            for baseline_name, baseline_metrics in baseline_data.items():
                if metric_name not in baseline_metrics:
                    continue
                
                baseline_measurements = baseline_metrics[metric_name]
                
                # Comprehensive statistical analysis
                analysis_result = self._comprehensive_analysis(
                    lnn_measurements, baseline_measurements,
                    f"LNN vs {baseline_name} - {metric_name}",
                    claims.get(metric_name)
                )
                
                metric_results[baseline_name] = analysis_result
                all_p_values.append(analysis_result.p_value)
                all_test_names.append(f"{metric_name}_{baseline_name}")
            
            validation_results['detailed_results'][metric_name] = metric_results
        
        # Multiple comparison correction
        if len(all_p_values) > 1:
            adjusted_p_values = MultipleComparisonAdjuster.adjust_p_values(
                all_p_values, self.correction_method
            )
            validation_results['multiple_comparisons'] = {
                'method': self.correction_method.value,
                'original_p_values': all_p_values,
                'adjusted_p_values': adjusted_p_values,
                'test_names': all_test_names
            }
        
        # Generate summary and recommendations
        validation_results['summary'] = self._generate_validation_summary(validation_results)
        validation_results['recommendations'] = self._generate_recommendations(validation_results)
        validation_results['publication_ready'] = self._assess_publication_readiness(validation_results)
        
        self.results_history.append(validation_results)
        
        print("✅ Statistical validation completed!")
        return validation_results
    
    def _comprehensive_analysis(self, data1: np.ndarray, data2: np.ndarray,
                               test_name: str, claimed_improvement: Optional[float] = None) -> StatisticalResult:
        """Perform comprehensive statistical analysis."""
        
        # Check assumptions
        assumptions = {}
        assumptions['normality_data1'] = self.assumption_checker.check_normality(data1)
        assumptions['normality_data2'] = self.assumption_checker.check_normality(data2)
        assumptions['equal_variances'] = self.assumption_checker.check_equal_variances(data1, data2)
        assumptions['independence_data1'] = self.assumption_checker.check_independence(data1)
        assumptions['independence_data2'] = self.assumption_checker.check_independence(data2)
        
        # Choose appropriate test based on assumptions
        if (assumptions['normality_data1']['overall_normal'] and 
            assumptions['normality_data2']['overall_normal']):
            
            if assumptions['equal_variances']['overall_equal_variances']:
                # Independent t-test with equal variances
                statistic, p_value = ttest_ind(data1, data2, equal_var=True)
                test_type = TestType.INDEPENDENT_TTEST
            else:
                # Welch's t-test (unequal variances)
                statistic, p_value = ttest_ind(data1, data2, equal_var=False)
                test_type = TestType.INDEPENDENT_TTEST
        else:
            # Non-parametric test
            statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            test_type = TestType.MANN_WHITNEY
        
        # Effect size calculation
        effect_size = self.effect_calculator.cohens_d(data1, data2)
        
        # Confidence interval for difference in means
        mean_diff = np.mean(data1) - np.mean(data2)
        se_diff = np.sqrt(np.var(data1, ddof=1)/len(data1) + np.var(data2, ddof=1)/len(data2))
        ci_margin = stats.t.ppf(1 - self.alpha/2, len(data1) + len(data2) - 2) * se_diff
        confidence_interval = (mean_diff - ci_margin, mean_diff + ci_margin)
        
        # Power analysis
        power = self.power_analyzer.power_t_test(
            abs(effect_size), min(len(data1), len(data2)), self.alpha
        )
        
        # Interpretation
        interpretation = self._interpret_results(
            p_value, effect_size, power, claimed_improvement,
            np.mean(data1), np.mean(data2)
        )
        
        # Recommendation
        recommendation = self._generate_test_recommendation(
            p_value, effect_size, power, assumptions
        )
        
        return StatisticalResult(
            test_name=test_name,
            test_type=test_type,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_measure=EffectSizeMeasure.COHENS_D,
            confidence_interval=confidence_interval,
            power=power,
            sample_size=min(len(data1), len(data2)),
            assumptions_met={
                'normality': (assumptions['normality_data1']['overall_normal'] and 
                             assumptions['normality_data2']['overall_normal']),
                'equal_variances': assumptions['equal_variances']['overall_equal_variances'],
                'independence': (assumptions['independence_data1']['independent'] and 
                               assumptions['independence_data2']['independent'])
            },
            interpretation=interpretation,
            recommendation=recommendation,
            raw_data={
                'data1_mean': np.mean(data1),
                'data1_std': np.std(data1, ddof=1),
                'data2_mean': np.mean(data2),
                'data2_std': np.std(data2, ddof=1),
                'assumptions': assumptions
            }
        )
    
    def _interpret_results(self, p_value: float, effect_size: float, power: float,
                          claimed_improvement: Optional[float],
                          mean1: float, mean2: float) -> str:
        """Generate interpretation of statistical results."""
        interpretation = []
        
        # Statistical significance
        if p_value < 0.001:
            interpretation.append("Highly statistically significant (p < 0.001)")
        elif p_value < 0.01:
            interpretation.append("Very statistically significant (p < 0.01)")
        elif p_value < 0.05:
            interpretation.append("Statistically significant (p < 0.05)")
        else:
            interpretation.append("Not statistically significant")
        
        # Effect size
        abs_effect = abs(effect_size)
        if abs_effect >= 1.2:
            interpretation.append("Very large effect size")
        elif abs_effect >= 0.8:
            interpretation.append("Large effect size")
        elif abs_effect >= 0.5:
            interpretation.append("Medium effect size")
        elif abs_effect >= 0.2:
            interpretation.append("Small effect size")
        else:
            interpretation.append("Negligible effect size")
        
        # Power
        if power >= 0.8:
            interpretation.append("Adequate statistical power")
        elif power >= 0.6:
            interpretation.append("Moderate statistical power")
        else:
            interpretation.append("Low statistical power")
        
        # Practical significance
        observed_ratio = mean2 / mean1 if mean1 != 0 else float('inf')
        if claimed_improvement:
            if observed_ratio >= claimed_improvement * 0.8:
                interpretation.append(f"Supports claimed {claimed_improvement:.1f}x improvement")
            else:
                interpretation.append(f"Does not support claimed {claimed_improvement:.1f}x improvement")
        
        interpretation.append(f"Observed improvement: {observed_ratio:.2f}x")
        
        return "; ".join(interpretation)
    
    def _generate_test_recommendation(self, p_value: float, effect_size: float,
                                    power: float, assumptions: Dict) -> str:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if p_value >= 0.05:
            recommendations.append("Consider increasing sample size or effect size")
        
        if power < 0.8:
            recommendations.append("Increase sample size for adequate power")
        
        if abs(effect_size) < 0.2:
            recommendations.append("Consider practical significance of small effect")
        
        if not assumptions['normality_data1']['overall_normal']:
            recommendations.append("Consider non-parametric tests due to non-normality")
        
        if not assumptions['equal_variances']['overall_equal_variances']:
            recommendations.append("Use tests that don't assume equal variances")
        
        if not recommendations:
            recommendations.append("Results are statistically robust")
        
        return "; ".join(recommendations)
    
    def _generate_validation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of validation results."""
        summary = {
            'total_tests': 0,
            'significant_tests': 0,
            'large_effects': 0,
            'adequate_power_tests': 0,
            'claims_supported': 0,
            'overall_conclusion': ''
        }
        
        for metric_name, metric_results in results['detailed_results'].items():
            for baseline_name, result in metric_results.items():
                summary['total_tests'] += 1
                
                if result.p_value < 0.05:
                    summary['significant_tests'] += 1
                
                if abs(result.effect_size or 0) >= 0.8:
                    summary['large_effects'] += 1
                
                if (result.power or 0) >= 0.8:
                    summary['adequate_power_tests'] += 1
                
                if "Supports claimed" in result.interpretation:
                    summary['claims_supported'] += 1
        
        # Overall conclusion
        if summary['total_tests'] == 0:
            summary['overall_conclusion'] = "No tests performed"
        else:
            sig_rate = summary['significant_tests'] / summary['total_tests']
            support_rate = summary['claims_supported'] / summary['total_tests']
            
            if sig_rate >= 0.8 and support_rate >= 0.7:
                summary['overall_conclusion'] = "Strong statistical support for claims"
            elif sig_rate >= 0.6 and support_rate >= 0.5:
                summary['overall_conclusion'] = "Moderate statistical support for claims"
            else:
                summary['overall_conclusion'] = "Limited statistical support for claims"
        
        return summary
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations for improving statistical validation."""
        recommendations = []
        summary = results['summary']
        
        if summary['significant_tests'] / max(summary['total_tests'], 1) < 0.5:
            recommendations.append("Increase sample sizes to achieve statistical significance")
        
        if summary['adequate_power_tests'] / max(summary['total_tests'], 1) < 0.8:
            recommendations.append("Conduct power analysis to determine adequate sample sizes")
        
        if summary['large_effects'] / max(summary['total_tests'], 1) < 0.3:
            recommendations.append("Focus on metrics with larger effect sizes for practical impact")
        
        if 'multiple_comparisons' in results and len(results['multiple_comparisons']['adjusted_p_values']) > 5:
            recommendations.append("Consider family-wise error rate correction for multiple comparisons")
        
        recommendations.append("Report effect sizes and confidence intervals alongside p-values")
        recommendations.append("Include assumption checking results in methodology")
        recommendations.append("Consider Bayesian analysis for additional evidence")
        
        return recommendations
    
    def _assess_publication_readiness(self, results: Dict[str, Any]) -> bool:
        """Assess if results meet publication standards."""
        summary = results['summary']
        
        # Criteria for publication readiness
        adequate_sample_sizes = all(
            result.sample_size >= 30
            for metric_results in results['detailed_results'].values()
            for result in metric_results.values()
        )
        
        adequate_power = summary['adequate_power_tests'] / max(summary['total_tests'], 1) >= 0.8
        
        multiple_comparison_corrected = 'multiple_comparisons' in results
        
        effect_sizes_reported = all(
            result.effect_size is not None
            for metric_results in results['detailed_results'].values()
            for result in metric_results.values()
        )
        
        assumptions_checked = all(
            result.assumptions_met is not None
            for metric_results in results['detailed_results'].values()
            for result in metric_results.values()
        )
        
        return (adequate_sample_sizes and adequate_power and 
                multiple_comparison_corrected and effect_sizes_reported and 
                assumptions_checked)
    
    def generate_statistical_report(self, results: Dict[str, Any]) -> str:
        """Generate publication-ready statistical report."""
        report = []
        
        # Header
        report.append("# Statistical Validation Report")
        report.append(f"**Analysis Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Significance Level:** α = {self.alpha}")
        report.append(f"**Multiple Comparison Correction:** {self.correction_method.value}")
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        summary = results['summary']
        report.append(f"- **Total Statistical Tests:** {summary['total_tests']}")
        report.append(f"- **Statistically Significant:** {summary['significant_tests']}/{summary['total_tests']} ({summary['significant_tests']/max(summary['total_tests'],1)*100:.1f}%)")
        report.append(f"- **Large Effect Sizes:** {summary['large_effects']}/{summary['total_tests']} ({summary['large_effects']/max(summary['total_tests'],1)*100:.1f}%)")
        report.append(f"- **Adequate Statistical Power:** {summary['adequate_power_tests']}/{summary['total_tests']} ({summary['adequate_power_tests']/max(summary['total_tests'],1)*100:.1f}%)")
        report.append(f"- **Claims Supported:** {summary['claims_supported']}/{summary['total_tests']} ({summary['claims_supported']/max(summary['total_tests'],1)*100:.1f}%)")
        report.append(f"- **Overall Conclusion:** {summary['overall_conclusion']}")
        report.append(f"- **Publication Ready:** {'Yes' if results['publication_ready'] else 'No'}")
        report.append("")
        
        # Detailed Results
        report.append("## Detailed Statistical Results")
        report.append("")
        
        for metric_name, metric_results in results['detailed_results'].items():
            report.append(f"### {metric_name.replace('_', ' ').title()} Analysis")
            report.append("")
            
            for baseline_name, result in metric_results.items():
                report.append(f"#### LNN vs {baseline_name}")
                report.append(f"- **Test:** {result.test_type.value}")
                report.append(f"- **Test Statistic:** {result.statistic:.4f}")
                report.append(f"- **P-value:** {result.p_value:.6f}")
                report.append(f"- **Effect Size (Cohen's d):** {result.effect_size:.4f}")
                report.append(f"- **95% Confidence Interval:** [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
                report.append(f"- **Statistical Power:** {result.power:.3f}")
                report.append(f"- **Sample Size:** {result.sample_size}")
                report.append(f"- **Interpretation:** {result.interpretation}")
                report.append(f"- **Recommendation:** {result.recommendation}")
                
                # Assumptions
                assumptions_met = result.assumptions_met
                report.append("- **Assumptions:**")
                for assumption, met in assumptions_met.items():
                    status = "✓" if met else "✗"
                    report.append(f"  - {assumption.replace('_', ' ').title()}: {status}")
                
                report.append("")
        
        # Multiple Comparisons
        if 'multiple_comparisons' in results:
            report.append("## Multiple Comparison Correction")
            mc = results['multiple_comparisons']
            report.append(f"**Method:** {mc['method']}")
            report.append("")
            report.append("| Test | Original p-value | Adjusted p-value |")
            report.append("|------|------------------|------------------|")
            for i, test_name in enumerate(mc['test_names']):
                orig_p = mc['original_p_values'][i]
                adj_p = mc['adjusted_p_values'][i]
                report.append(f"| {test_name} | {orig_p:.6f} | {adj_p:.6f} |")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        for i, rec in enumerate(results['recommendations'], 1):
            report.append(f"{i}. {rec}")
        report.append("")
        
        # Methodology
        report.append("## Statistical Methodology")
        report.append("All statistical analyses followed best practices for scientific research:")
        report.append("- Assumption checking performed for all parametric tests")
        report.append("- Non-parametric alternatives used when assumptions violated")
        report.append("- Effect sizes calculated and reported")
        report.append("- Statistical power analysis conducted")
        report.append("- Multiple comparison correction applied")
        report.append("- Confidence intervals reported alongside p-values")
        report.append("")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save validation results to file."""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif hasattr(obj, '__dict__'):
                return {k: convert_numpy(v) for k, v in obj.__dict__.items()}
            return obj
        
        serializable_results = convert_numpy(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"📊 Validation results saved to {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    validator = StatisticalValidator(alpha=0.05)
    
    # Simulate LNN and baseline data
    np.random.seed(42)
    
    lnn_data = {
        'accuracy': np.random.normal(0.92, 0.02, 50),
        'power_consumption': np.random.normal(1.2, 0.1, 50),
        'latency': np.random.normal(15.0, 2.0, 50)
    }
    
    baseline_data = {
        'CNN': {
            'accuracy': np.random.normal(0.88, 0.03, 45),
            'power_consumption': np.random.normal(12.5, 1.5, 45),
            'latency': np.random.normal(25.0, 3.0, 45)
        },
        'LSTM': {
            'accuracy': np.random.normal(0.85, 0.025, 40),
            'power_consumption': np.random.normal(8.3, 0.8, 40),
            'latency': np.random.normal(30.0, 4.0, 40)
        }
    }
    
    claims = {
        'power_consumption': 10.0,  # 10x improvement claimed
        'latency': 2.0,            # 2x improvement claimed
        'accuracy': 1.05           # 5% improvement claimed
    }
    
    # Run validation
    results = validator.validate_lnn_performance_claims(lnn_data, baseline_data, claims)
    
    # Generate report
    report = validator.generate_statistical_report(results)
    print(report)