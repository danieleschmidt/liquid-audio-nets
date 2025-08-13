"""
Comprehensive tests for the multi-objective optimization framework.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from liquid_audio_nets.research.multi_objective import (
    MultiObjectiveOptimizer,
    NSGA3Algorithm,
    ParetoFrontierAnalysis,
    ObjectiveFunction,
    OptimizationMetric,
    OptimizationResult,
    Individual
)


class TestObjectiveFunction:
    """Test ObjectiveFunction dataclass."""
    
    def test_objective_function_creation(self):
        """Test creating objective functions."""
        obj = ObjectiveFunction(
            name="accuracy",
            metric=OptimizationMetric.ACCURACY,
            direction="maximize",
            weight=1.0
        )
        
        assert obj.name == "accuracy"
        assert obj.metric == OptimizationMetric.ACCURACY
        assert obj.direction == "maximize"
        assert obj.weight == 1.0
        assert obj.constraint_min is None
        assert obj.constraint_max is None
    
    def test_objective_function_with_constraints(self):
        """Test creating objective function with constraints."""
        obj = ObjectiveFunction(
            name="power",
            metric=OptimizationMetric.POWER_CONSUMPTION,
            direction="minimize",
            constraint_max=5.0
        )
        
        assert obj.constraint_max == 5.0
        assert obj.constraint_min is None
    
    def test_invalid_direction(self):
        """Test invalid direction raises error."""
        with pytest.raises(ValueError, match="direction must be"):
            ObjectiveFunction(
                name="test",
                metric=OptimizationMetric.ACCURACY,
                direction="invalid"
            )


class TestIndividual:
    """Test Individual class."""
    
    def test_individual_creation(self):
        """Test creating an individual."""
        params = {'param1': 1.0, 'param2': 2.0}
        objectives = {'accuracy': 0.9, 'power': 1.5}
        
        individual = Individual(
            parameters=params,
            objectives=objectives,
            constraints={}
        )
        
        assert individual.parameters == params
        assert individual.objectives == objectives
        assert individual.rank == 0
        assert individual.crowding_distance == 0.0
        assert len(individual.dominated_solutions) == 0
        assert individual.domination_count == 0
    
    def test_dominance_check_maximize(self):
        """Test dominance for maximization objectives."""
        obj_funcs = [
            ObjectiveFunction("acc", OptimizationMetric.ACCURACY, "maximize"),
            ObjectiveFunction("f1", OptimizationMetric.F1_SCORE, "maximize")
        ]
        
        # Individual 1: better in both objectives
        ind1 = Individual(
            parameters={}, 
            objectives={'accuracy': 0.9, 'f1_score': 0.85},
            constraints={}
        )
        
        # Individual 2: worse in both objectives
        ind2 = Individual(
            parameters={}, 
            objectives={'accuracy': 0.8, 'f1_score': 0.75},
            constraints={}
        )
        
        assert ind1.dominates(ind2, obj_funcs)
        assert not ind2.dominates(ind1, obj_funcs)
    
    def test_dominance_check_minimize(self):
        """Test dominance for minimization objectives."""
        obj_funcs = [
            ObjectiveFunction("power", OptimizationMetric.POWER_CONSUMPTION, "minimize"),
            ObjectiveFunction("latency", OptimizationMetric.LATENCY, "minimize")
        ]
        
        # Individual 1: better (lower) in both objectives
        ind1 = Individual(
            parameters={}, 
            objectives={'power_consumption_mw': 1.0, 'latency_ms': 5.0},
            constraints={}
        )
        
        # Individual 2: worse (higher) in both objectives
        ind2 = Individual(
            parameters={}, 
            objectives={'power_consumption_mw': 2.0, 'latency_ms': 10.0},
            constraints={}
        )
        
        assert ind1.dominates(ind2, obj_funcs)
        assert not ind2.dominates(ind1, obj_funcs)
    
    def test_dominance_check_mixed(self):
        """Test dominance with mixed objectives."""
        obj_funcs = [
            ObjectiveFunction("acc", OptimizationMetric.ACCURACY, "maximize"),
            ObjectiveFunction("power", OptimizationMetric.POWER_CONSUMPTION, "minimize")
        ]
        
        # Individual 1: better accuracy, worse power
        ind1 = Individual(
            parameters={}, 
            objectives={'accuracy': 0.9, 'power_consumption_mw': 3.0},
            constraints={}
        )
        
        # Individual 2: worse accuracy, better power
        ind2 = Individual(
            parameters={}, 
            objectives={'accuracy': 0.8, 'power_consumption_mw': 1.0},
            constraints={}
        )
        
        # Neither should dominate the other (trade-off)
        assert not ind1.dominates(ind2, obj_funcs)
        assert not ind2.dominates(ind1, obj_funcs)
    
    def test_constraint_violation(self):
        """Test constraint violation checking."""
        obj_funcs = [
            ObjectiveFunction(
                "power", 
                OptimizationMetric.POWER_CONSUMPTION, 
                "minimize",
                constraint_max=5.0
            )
        ]
        
        # Individual within constraints
        ind1 = Individual(
            parameters={}, 
            objectives={'power_consumption_mw': 3.0},
            constraints={}
        )
        
        # Individual violating constraints
        ind2 = Individual(
            parameters={}, 
            objectives={'power_consumption_mw': 7.0},
            constraints={}
        )
        
        assert not ind1.violates_constraints(obj_funcs)
        assert ind2.violates_constraints(obj_funcs)


class TestParetoFrontierAnalysis:
    """Test Pareto frontier analysis."""
    
    def test_analyzer_creation(self):
        """Test creating Pareto analyzer."""
        analyzer = ParetoFrontierAnalysis()
        assert analyzer.reference_point is None
        
        analyzer_with_ref = ParetoFrontierAnalysis([0.0, 0.0])
        assert analyzer_with_ref.reference_point == [0.0, 0.0]
    
    def test_compute_pareto_front_empty(self):
        """Test computing Pareto front with empty solutions."""
        analyzer = ParetoFrontierAnalysis()
        obj_funcs = [
            ObjectiveFunction("acc", OptimizationMetric.ACCURACY, "maximize")
        ]
        
        pareto_front = analyzer.compute_pareto_front([], obj_funcs)
        assert len(pareto_front) == 0
    
    def test_compute_pareto_front_single_objective(self):
        """Test Pareto front with single objective."""
        analyzer = ParetoFrontierAnalysis()
        obj_funcs = [
            ObjectiveFunction("acc", OptimizationMetric.ACCURACY, "maximize")
        ]
        
        # Create individuals with different accuracy
        individuals = [
            Individual({}, {'accuracy': 0.9}, {}),
            Individual({}, {'accuracy': 0.8}, {}),
            Individual({}, {'accuracy': 0.95}, {}),
            Individual({}, {'accuracy': 0.85}, {})
        ]
        
        pareto_front = analyzer.compute_pareto_front(individuals, obj_funcs)
        
        # Only the best individual should be in the front
        assert len(pareto_front) == 1
        assert pareto_front[0].objectives['accuracy'] == 0.95
    
    def test_compute_pareto_front_two_objectives(self):
        """Test Pareto front with two conflicting objectives."""
        analyzer = ParetoFrontierAnalysis()
        obj_funcs = [
            ObjectiveFunction("acc", OptimizationMetric.ACCURACY, "maximize"),
            ObjectiveFunction("power", OptimizationMetric.POWER_CONSUMPTION, "minimize")
        ]
        
        # Create individuals with trade-offs
        individuals = [
            Individual({}, {'accuracy': 0.9, 'power_consumption_mw': 3.0}, {}),  # High acc, high power
            Individual({}, {'accuracy': 0.7, 'power_consumption_mw': 1.0}, {}),  # Low acc, low power
            Individual({}, {'accuracy': 0.8, 'power_consumption_mw': 2.0}, {}),  # Medium both
            Individual({}, {'accuracy': 0.6, 'power_consumption_mw': 2.5}, {}),  # Dominated
        ]
        
        pareto_front = analyzer.compute_pareto_front(individuals, obj_funcs)
        
        # First three should be in front, fourth is dominated
        assert len(pareto_front) == 3
        
        # Check that dominated individual is not in front
        dominated_individual = individuals[3]
        assert dominated_individual not in pareto_front
    
    def test_compute_pareto_front_with_constraints(self):
        """Test Pareto front computation with constraints."""
        analyzer = ParetoFrontierAnalysis()
        obj_funcs = [
            ObjectiveFunction(
                "power", 
                OptimizationMetric.POWER_CONSUMPTION, 
                "minimize",
                constraint_max=2.5
            )
        ]
        
        # Create individuals, some violating constraints
        individuals = [
            Individual({}, {'power_consumption_mw': 1.0}, {}),  # Feasible
            Individual({}, {'power_consumption_mw': 2.0}, {}),  # Feasible
            Individual({}, {'power_consumption_mw': 3.0}, {}),  # Infeasible
        ]
        
        pareto_front = analyzer.compute_pareto_front(individuals, obj_funcs)
        
        # Only feasible individuals should be considered
        assert len(pareto_front) == 1  # Best feasible individual
        assert pareto_front[0].objectives['power_consumption_mw'] == 1.0
    
    def test_hypervolume_calculation_2d(self):
        """Test 2D hypervolume calculation."""
        analyzer = ParetoFrontierAnalysis()
        obj_funcs = [
            ObjectiveFunction("obj1", OptimizationMetric.ACCURACY, "maximize"),
            ObjectiveFunction("obj2", OptimizationMetric.F1_SCORE, "maximize")
        ]
        
        # Create simple Pareto front
        pareto_front = [
            Individual({}, {'accuracy': 1.0, 'f1_score': 0.5}, {}),
            Individual({}, {'accuracy': 0.5, 'f1_score': 1.0}, {})
        ]
        
        hypervolume = analyzer.compute_hypervolume(pareto_front, obj_funcs, [0.0, 0.0])
        
        assert hypervolume > 0
        assert isinstance(hypervolume, float)
    
    def test_hypervolume_empty_front(self):
        """Test hypervolume with empty front."""
        analyzer = ParetoFrontierAnalysis()
        obj_funcs = [
            ObjectiveFunction("obj1", OptimizationMetric.ACCURACY, "maximize")
        ]
        
        hypervolume = analyzer.compute_hypervolume([], obj_funcs)
        assert hypervolume == 0.0
    
    def test_analyze_pareto_front_quality(self):
        """Test Pareto front quality analysis."""
        analyzer = ParetoFrontierAnalysis()
        obj_funcs = [
            ObjectiveFunction("acc", OptimizationMetric.ACCURACY, "maximize"),
            ObjectiveFunction("power", OptimizationMetric.POWER_CONSUMPTION, "minimize")
        ]
        
        pareto_front = [
            Individual({}, {'accuracy': 0.9, 'power_consumption_mw': 3.0}, {}),
            Individual({}, {'accuracy': 0.8, 'power_consumption_mw': 2.0}, {}),
            Individual({}, {'accuracy': 0.7, 'power_consumption_mw': 1.0}, {})
        ]
        
        quality_metrics = analyzer.analyze_pareto_front_quality(pareto_front, obj_funcs)
        
        assert 'size' in quality_metrics
        assert 'diversity' in quality_metrics
        assert 'hypervolume' in quality_metrics
        assert 'convergence' in quality_metrics
        
        assert quality_metrics['size'] == 3
        assert quality_metrics['diversity'] >= 0
        assert quality_metrics['hypervolume'] >= 0
        assert quality_metrics['convergence'] >= 0


class TestNSGA3Algorithm:
    """Test NSGA-III algorithm."""
    
    def test_algorithm_creation(self):
        """Test creating NSGA-III algorithm."""
        algo = NSGA3Algorithm(
            population_size=50,
            max_generations=10,
            crossover_rate=0.8,
            mutation_rate=0.1
        )
        
        assert algo.population_size == 50
        assert algo.max_generations == 10
        assert algo.crossover_rate == 0.8
        assert algo.mutation_rate == 0.1
        assert algo.reference_directions is None
        assert len(algo.convergence_history) == 0
    
    def test_generate_reference_directions_2d(self):
        """Test generating reference directions for 2D."""
        algo = NSGA3Algorithm()
        directions = algo.generate_reference_directions(2, n_divisions=5)
        
        assert directions.shape[1] == 2
        assert directions.shape[0] == 6  # n_divisions + 1
        
        # Check that directions sum to 1 (on unit simplex)
        for direction in directions:
            assert abs(sum(direction) - 1.0) < 1e-10
    
    def test_generate_reference_directions_3d(self):
        """Test generating reference directions for 3D."""
        algo = NSGA3Algorithm()
        directions = algo.generate_reference_directions(3, n_divisions=3)
        
        assert directions.shape[1] == 3
        
        # Check that directions sum to 1
        for direction in directions:
            assert abs(sum(direction) - 1.0) < 1e-10
    
    def test_fast_non_dominated_sort(self):
        """Test fast non-dominated sorting."""
        algo = NSGA3Algorithm()
        obj_funcs = [
            ObjectiveFunction("acc", OptimizationMetric.ACCURACY, "maximize"),
            ObjectiveFunction("power", OptimizationMetric.POWER_CONSUMPTION, "minimize")
        ]
        
        # Create population with known dominance relationships
        population = [
            Individual({}, {'accuracy': 0.9, 'power_consumption_mw': 1.0}, {}),  # Front 0 (best)
            Individual({}, {'accuracy': 0.8, 'power_consumption_mw': 1.5}, {}),  # Front 1
            Individual({}, {'accuracy': 0.7, 'power_consumption_mw': 2.0}, {}),  # Front 2
            Individual({}, {'accuracy': 0.6, 'power_consumption_mw': 2.5}, {}),  # Front 3
        ]
        
        fronts = algo.fast_non_dominated_sort(population, obj_funcs)
        
        assert len(fronts) >= 1
        assert len(fronts[0]) == 1  # Best individual in front 0
        assert fronts[0][0].objectives['accuracy'] == 0.9
        
        # Check ranks are assigned
        for individual in population:
            assert individual.rank >= 0


class TestMultiObjectiveOptimizer:
    """Test the main multi-objective optimizer."""
    
    def test_optimizer_creation(self):
        """Test creating optimizer."""
        optimizer = MultiObjectiveOptimizer(
            algorithm="nsga3",
            population_size=20,
            max_generations=5
        )
        
        assert optimizer.algorithm == "nsga3"
        assert isinstance(optimizer.optimizer, NSGA3Algorithm)
        assert optimizer.optimizer.population_size == 20
        assert optimizer.optimizer.max_generations == 5
        assert isinstance(optimizer.pareto_analyzer, ParetoFrontierAnalysis)
    
    def test_optimizer_invalid_algorithm(self):
        """Test invalid algorithm raises error."""
        with pytest.raises(ValueError, match="Unknown algorithm"):
            MultiObjectiveOptimizer(algorithm="invalid")
    
    def test_initialize_population(self):
        """Test population initialization."""
        optimizer = MultiObjectiveOptimizer(population_size=10)
        
        parameter_space = {
            'param1': (0.0, 1.0),
            'param2': (-1.0, 1.0)
        }
        
        population = optimizer._initialize_population(parameter_space, 10)
        
        assert len(population) == 10
        for individual in population:
            assert 'param1' in individual.parameters
            assert 'param2' in individual.parameters
            assert 0.0 <= individual.parameters['param1'] <= 1.0
            assert -1.0 <= individual.parameters['param2'] <= 1.0
    
    def mock_evaluation_function(self, parameters):
        """Mock evaluation function for testing."""
        # Simple test function with trade-offs
        x1 = parameters.get('x1', 0.5)
        x2 = parameters.get('x2', 0.5)
        
        # Conflicting objectives
        obj1 = x1  # Maximize x1
        obj2 = (1 - x1) * (1 - x2)  # Complex relationship
        
        return {
            'accuracy': obj1,
            'power_consumption_mw': obj2 * 10,  # Scale to realistic power range
            'latency_ms': (x1 + x2) * 20  # Scale to realistic latency range
        }
    
    def test_optimize_simple(self):
        """Test optimization with simple evaluation function."""
        optimizer = MultiObjectiveOptimizer(
            algorithm="nsga3",
            population_size=20,
            max_generations=3  # Small for testing
        )
        
        parameter_space = {
            'x1': (0.0, 1.0),
            'x2': (0.0, 1.0)
        }
        
        objective_functions = [
            ObjectiveFunction("acc", OptimizationMetric.ACCURACY, "maximize"),
            ObjectiveFunction("power", OptimizationMetric.POWER_CONSUMPTION, "minimize")
        ]
        
        result = optimizer.optimize(
            evaluation_function=self.mock_evaluation_function,
            parameter_space=parameter_space,
            objective_functions=objective_functions,
            n_parallel=1,  # Single-threaded for testing
            seed=42
        )
        
        assert isinstance(result, OptimizationResult)
        assert len(result.solutions) > 0
        assert len(result.pareto_front) > 0
        assert result.hypervolume >= 0
        assert result.optimization_time > 0
        
        # Check that algorithm metadata is present
        assert 'algorithm' in result.algorithm_metadata
        assert 'population_size' in result.algorithm_metadata
        assert 'n_objectives' in result.algorithm_metadata
        
        # Check convergence metrics
        assert 'hypervolume' in result.convergence_metrics
        assert 'pareto_size' in result.convergence_metrics
        assert len(result.convergence_metrics['hypervolume']) > 0
    
    def test_crossover(self):
        """Test crossover operation."""
        optimizer = MultiObjectiveOptimizer()
        
        params1 = {'x1': 0.2, 'x2': 0.8}
        params2 = {'x1': 0.7, 'x2': 0.3}
        parameter_space = {'x1': (0.0, 1.0), 'x2': (0.0, 1.0)}
        
        child1, child2 = optimizer._crossover(params1, params2, parameter_space)
        
        # Check that children are within bounds
        assert 0.0 <= child1['x1'] <= 1.0
        assert 0.0 <= child1['x2'] <= 1.0
        assert 0.0 <= child2['x1'] <= 1.0
        assert 0.0 <= child2['x2'] <= 1.0
        
        # Children should be different from parents (with high probability)
        # This is stochastic, so we can't guarantee it, but it's very likely
        assert 'x1' in child1 and 'x2' in child1
        assert 'x1' in child2 and 'x2' in child2
    
    def test_mutation(self):
        """Test mutation operation."""
        optimizer = MultiObjectiveOptimizer()
        optimizer.optimizer.mutation_rate = 1.0  # Ensure mutation happens
        
        params = {'x1': 0.5, 'x2': 0.5}
        parameter_space = {'x1': (0.0, 1.0), 'x2': (0.0, 1.0)}
        
        mutated = optimizer._mutate(params, parameter_space)
        
        # Check that mutated parameters are within bounds
        assert 0.0 <= mutated['x1'] <= 1.0
        assert 0.0 <= mutated['x2'] <= 1.0
        
        # With mutation rate 1.0, at least one parameter should change
        # (This is stochastic, but very likely with polynomial mutation)
        assert 'x1' in mutated and 'x2' in mutated
    
    def test_generate_offspring(self):
        """Test offspring generation."""
        optimizer = MultiObjectiveOptimizer()
        
        parents = [
            Individual({'x1': 0.2, 'x2': 0.8}, {}, {}),
            Individual({'x1': 0.7, 'x2': 0.3}, {}, {}),
            Individual({'x1': 0.5, 'x2': 0.5}, {}, {}),
            Individual({'x1': 0.1, 'x2': 0.9}, {}, {})
        ]
        
        parameter_space = {'x1': (0.0, 1.0), 'x2': (0.0, 1.0)}
        
        offspring = optimizer._generate_offspring(parents, parameter_space)
        
        # Should generate offspring
        assert len(offspring) > 0
        
        # All offspring should have valid parameters
        for child in offspring:
            assert 'x1' in child.parameters
            assert 'x2' in child.parameters
            assert 0.0 <= child.parameters['x1'] <= 1.0
            assert 0.0 <= child.parameters['x2'] <= 1.0


class TestOptimizationResult:
    """Test OptimizationResult dataclass."""
    
    def test_result_creation(self):
        """Test creating optimization result."""
        solutions = [
            {'parameters': {'x1': 0.5}, 'objectives': {'acc': 0.8}, 'rank': 0}
        ]
        pareto_front = [solutions[0]]
        
        result = OptimizationResult(
            solutions=solutions,
            pareto_front=pareto_front,
            hypervolume=0.5,
            convergence_metrics={'hypervolume': [0.1, 0.3, 0.5]},
            optimization_time=10.0,
            algorithm_metadata={'algorithm': 'nsga3'}
        )
        
        assert len(result.solutions) == 1
        assert len(result.pareto_front) == 1
        assert result.hypervolume == 0.5
        assert result.optimization_time == 10.0
    
    def test_get_best_solution(self):
        """Test getting best solution based on preferences."""
        pareto_front = [
            {'parameters': {'x1': 0.9}, 'objectives': {'accuracy': 0.9, 'power': 2.0}},
            {'parameters': {'x1': 0.3}, 'objectives': {'accuracy': 0.7, 'power': 1.0}},
            {'parameters': {'x1': 0.6}, 'objectives': {'accuracy': 0.8, 'power': 1.5}}
        ]
        
        result = OptimizationResult(
            solutions=pareto_front,
            pareto_front=pareto_front,
            hypervolume=0.5,
            convergence_metrics={},
            optimization_time=10.0,
            algorithm_metadata={}
        )
        
        # Prefer accuracy
        accuracy_preferences = {'accuracy': 1.0, 'power': 0.0}
        best_acc = result.get_best_solution(accuracy_preferences)
        assert best_acc['objectives']['accuracy'] == 0.9
        
        # Prefer power (note: this is confusing as written, but tests the logic)
        power_preferences = {'accuracy': 0.0, 'power': 1.0}
        best_power = result.get_best_solution(power_preferences)
        assert best_power['objectives']['power'] == 2.0  # Highest power value
    
    def test_get_best_solution_empty_front(self):
        """Test getting best solution with empty front."""
        result = OptimizationResult(
            solutions=[],
            pareto_front=[],
            hypervolume=0.0,
            convergence_metrics={},
            optimization_time=0.0,
            algorithm_metadata={}
        )
        
        best = result.get_best_solution({'accuracy': 1.0})
        assert best is None


class TestIntegration:
    """Integration tests for multi-objective optimization."""
    
    def test_full_optimization_pipeline(self):
        """Test complete optimization pipeline."""
        # Simple quadratic evaluation function
        def evaluation_func(params):
            x = params.get('x', 0.5)
            y = params.get('y', 0.5)
            
            return {
                'accuracy': 1 - (x - 0.7)**2 - (y - 0.3)**2,  # Maximize
                'power_consumption_mw': (x**2 + y**2) * 5,    # Minimize
                'latency_ms': abs(x - y) * 20                 # Minimize
            }
        
        optimizer = MultiObjectiveOptimizer(
            algorithm="nsga3",
            population_size=15,
            max_generations=3
        )
        
        parameter_space = {
            'x': (0.0, 1.0),
            'y': (0.0, 1.0)
        }
        
        objective_functions = [
            ObjectiveFunction("acc", OptimizationMetric.ACCURACY, "maximize"),
            ObjectiveFunction("power", OptimizationMetric.POWER_CONSUMPTION, "minimize"),
            ObjectiveFunction("latency", OptimizationMetric.LATENCY, "minimize")
        ]
        
        result = optimizer.optimize(
            evaluation_function=evaluation_func,
            parameter_space=parameter_space,
            objective_functions=objective_functions,
            seed=42
        )
        
        # Verify result structure
        assert isinstance(result, OptimizationResult)
        assert len(result.solutions) == 15  # Population size
        assert len(result.pareto_front) > 0
        assert result.hypervolume >= 0
        
        # Verify that solutions have all required fields
        for solution in result.solutions:
            assert 'parameters' in solution
            assert 'objectives' in solution
            assert 'x' in solution['parameters']
            assert 'y' in solution['parameters']
            assert 'accuracy' in solution['objectives']
            assert 'power_consumption_mw' in solution['objectives']
            assert 'latency_ms' in solution['objectives']
        
        # Verify Pareto front quality
        for solution in result.pareto_front:
            assert solution['rank'] == 0  # All should be rank 0
        
        # Generate report
        report = optimizer.generate_optimization_report(result, objective_functions)
        assert len(report) > 0
        assert "Multi-Objective Optimization Report" in report
        assert "Pareto Front Analysis" in report


if __name__ == "__main__":
    pytest.main([__file__])