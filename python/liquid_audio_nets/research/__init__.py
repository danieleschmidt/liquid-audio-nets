"""
Research module for advanced Liquid Neural Network analysis.

This module provides novel research tools for comparative studies,
statistical analysis, and multi-objective optimization of LNNs.
"""

from .comparative_study import (
    ComparativeStudyFramework,
    BaselineModel,
    ModelComparison,
    StatisticalTest,
    PowerEfficiencyAnalysis
)

from .multi_objective import (
    MultiObjectiveOptimizer,
    ParetoFrontierAnalysis,
    ObjectiveFunction,
    OptimizationResult
)

from .experimental_framework import (
    ExperimentalFramework,
    ExperimentConfig,
    ReproducibilityManager,
    DatasetGenerator
)

__all__ = [
    'ComparativeStudyFramework',
    'BaselineModel', 
    'ModelComparison',
    'StatisticalTest',
    'PowerEfficiencyAnalysis',
    'MultiObjectiveOptimizer',
    'ParetoFrontierAnalysis',
    'ObjectiveFunction',
    'OptimizationResult',
    'ExperimentalFramework',
    'ExperimentConfig',
    'ReproducibilityManager',
    'DatasetGenerator'
]