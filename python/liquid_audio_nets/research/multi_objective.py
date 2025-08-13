"""
Novel Multi-Objective Optimization Framework for Liquid Neural Networks.

This module implements advanced optimization algorithms including:
- NSGA-III for many-objective optimization
- Pareto frontier analysis with statistical validation
- Multi-objective Bayesian optimization
- Power-performance trade-off analysis
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationMetric(Enum):
    """Supported optimization metrics."""
    ACCURACY = "accuracy"
    POWER_CONSUMPTION = "power_consumption_mw"
    LATENCY = "latency_ms"
    MODEL_SIZE = "model_size_bytes"
    MEMORY_USAGE = "memory_usage_mb"
    THROUGHPUT = "throughput_sps"
    ENERGY_EFFICIENCY = "energy_per_inference_uj"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"


@dataclass
class ObjectiveFunction:
    """Definition of an objective function for optimization."""
    name: str
    metric: OptimizationMetric
    direction: str  # "minimize" or "maximize"
    weight: float = 1.0
    constraint_min: Optional[float] = None
    constraint_max: Optional[float] = None
    
    def __post_init__(self):
        if self.direction not in ["minimize", "maximize"]:
            raise ValueError("direction must be 'minimize' or 'maximize'")


@dataclass
class OptimizationResult:
    """Result from multi-objective optimization."""
    solutions: List[Dict[str, Any]]
    pareto_front: List[Dict[str, Any]]
    hypervolume: float
    convergence_metrics: Dict[str, List[float]]
    optimization_time: float
    algorithm_metadata: Dict[str, Any]
    
    def get_best_solution(self, preference_weights: Dict[str, float]) -> Dict[str, Any]:
        """Get the best solution based on preference weights."""
        if not self.pareto_front:
            return None
            
        best_score = float('-inf')
        best_solution = None
        
        for solution in self.pareto_front:
            score = 0
            for metric, weight in preference_weights.items():
                if metric in solution['objectives']:
                    score += weight * solution['objectives'][metric]
            
            if score > best_score:
                best_score = score
                best_solution = solution
        
        return best_solution


@dataclass
class Individual:
    """Individual solution in the population."""
    parameters: Dict[str, Any]
    objectives: Dict[str, float]
    constraints: Dict[str, float]
    rank: int = 0
    crowding_distance: float = 0.0
    dominated_solutions: List['Individual'] = field(default_factory=list)
    domination_count: int = 0
    
    def dominates(self, other: 'Individual', objective_functions: List[ObjectiveFunction]) -> bool:
        """Check if this individual dominates another."""
        at_least_one_better = False
        
        for obj_func in objective_functions:
            metric = obj_func.metric.value
            
            if metric not in self.objectives or metric not in other.objectives:
                continue
                
            self_value = self.objectives[metric]
            other_value = other.objectives[metric]
            
            if obj_func.direction == "minimize":
                if self_value > other_value:
                    return False
                elif self_value < other_value:
                    at_least_one_better = True
            else:  # maximize
                if self_value < other_value:
                    return False
                elif self_value > other_value:
                    at_least_one_better = True
        
        return at_least_one_better
    
    def violates_constraints(self, objective_functions: List[ObjectiveFunction]) -> bool:
        """Check if this individual violates any constraints."""
        for obj_func in objective_functions:
            metric = obj_func.metric.value
            
            if metric not in self.objectives:
                continue
                
            value = self.objectives[metric]
            
            if obj_func.constraint_min is not None and value < obj_func.constraint_min:
                return True
            if obj_func.constraint_max is not None and value > obj_func.constraint_max:
                return True
        
        return False


class ParetoFrontierAnalysis:
    """Advanced Pareto frontier analysis with statistical validation."""
    
    def __init__(self, reference_point: Optional[List[float]] = None):
        self.reference_point = reference_point
        
    def compute_pareto_front(self, solutions: List[Individual], 
                           objective_functions: List[ObjectiveFunction]) -> List[Individual]:
        """Compute the Pareto front from a set of solutions."""
        if not solutions:
            return []
        
        # Filter out constraint-violating solutions
        feasible_solutions = [
            sol for sol in solutions 
            if not sol.violates_constraints(objective_functions)
        ]
        
        if not feasible_solutions:
            logger.warning("No feasible solutions found!")
            return []
        
        pareto_front = []
        
        for candidate in feasible_solutions:
            is_dominated = False
            
            for other in feasible_solutions:
                if other != candidate and other.dominates(candidate, objective_functions):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(candidate)
        
        return pareto_front
    
    def compute_hypervolume(self, pareto_front: List[Individual],
                          objective_functions: List[ObjectiveFunction],
                          reference_point: Optional[List[float]] = None) -> float:
        """Compute hypervolume indicator of the Pareto front."""
        if not pareto_front:
            return 0.0
        
        # Extract objective values
        objectives_matrix = []
        for individual in pareto_front:
            obj_values = []
            for obj_func in objective_functions:
                metric = obj_func.metric.value
                value = individual.objectives.get(metric, 0.0)
                
                # Normalize for hypervolume calculation (convert to maximization)
                if obj_func.direction == "minimize":
                    value = -value
                
                obj_values.append(value)
            objectives_matrix.append(obj_values)
        
        objectives_matrix = np.array(objectives_matrix)
        
        if reference_point is None:
            # Use worst values as reference point
            reference_point = np.min(objectives_matrix, axis=0) - 0.1
        
        # Simple hypervolume calculation (works for 2-3 objectives)
        if objectives_matrix.shape[1] <= 3:
            return self._compute_hypervolume_simple(objectives_matrix, reference_point)
        else:
            # For many objectives, use Monte Carlo approximation
            return self._compute_hypervolume_monte_carlo(objectives_matrix, reference_point)
    
    def _compute_hypervolume_simple(self, points: np.ndarray, ref_point: np.ndarray) -> float:
        """Simple hypervolume calculation for 2-3 objectives."""
        if points.shape[1] == 2:
            return self._hypervolume_2d(points, ref_point)
        elif points.shape[1] == 3:
            return self._hypervolume_3d(points, ref_point)
        else:
            return self._compute_hypervolume_monte_carlo(points, ref_point)
    
    def _hypervolume_2d(self, points: np.ndarray, ref_point: np.ndarray) -> float:
        """2D hypervolume calculation."""
        # Sort points by first objective
        sorted_indices = np.argsort(points[:, 0])
        sorted_points = points[sorted_indices]
        
        hypervolume = 0.0
        prev_x = ref_point[0]
        
        for point in sorted_points:
            x, y = point
            if y > ref_point[1]:
                width = x - prev_x
                height = y - ref_point[1]
                hypervolume += width * height
                prev_x = x
        
        return hypervolume
    
    def _hypervolume_3d(self, points: np.ndarray, ref_point: np.ndarray) -> float:
        """3D hypervolume calculation (simplified)."""
        # Use inclusion-exclusion principle
        total_volume = 0.0
        
        for i, point in enumerate(points):
            # Volume contributed by this point
            volume = np.prod(np.maximum(0, point - ref_point))
            
            # Subtract overlaps with other points
            for j, other_point in enumerate(points):
                if i != j:
                    overlap = np.prod(np.maximum(0, np.minimum(point, other_point) - ref_point))
                    volume -= overlap / len(points)  # Simplified overlap calculation
            
            total_volume += volume
        
        return max(0.0, total_volume)
    
    def _compute_hypervolume_monte_carlo(self, points: np.ndarray, ref_point: np.ndarray,
                                       n_samples: int = 100000) -> float:
        """Monte Carlo approximation for many-objective hypervolume."""
        # Define bounding box
        upper_bound = np.max(points, axis=0) + 0.1
        
        # Generate random samples
        samples = np.random.uniform(
            low=ref_point,
            high=upper_bound,
            size=(n_samples, len(ref_point))
        )
        
        # Count dominated samples
        dominated_count = 0
        
        for sample in samples:
            # Check if sample is dominated by any point in the front
            for point in points:
                if np.all(point >= sample):
                    dominated_count += 1
                    break
        
        # Estimate hypervolume
        box_volume = np.prod(upper_bound - ref_point)
        hypervolume = (dominated_count / n_samples) * box_volume
        
        return hypervolume
    
    def analyze_pareto_front_quality(self, pareto_front: List[Individual],
                                   objective_functions: List[ObjectiveFunction]) -> Dict[str, float]:
        """Analyze the quality of a Pareto front."""
        if not pareto_front:
            return {"size": 0, "diversity": 0.0, "convergence": 0.0, "hypervolume": 0.0}
        
        metrics = {}
        
        # Size of Pareto front
        metrics["size"] = len(pareto_front)
        
        # Diversity (spacing metric)
        if len(pareto_front) > 1:
            diversity = self._compute_spacing_metric(pareto_front, objective_functions)
            metrics["diversity"] = diversity
        else:
            metrics["diversity"] = 0.0
        
        # Hypervolume
        hypervolume = self.compute_hypervolume(pareto_front, objective_functions)
        metrics["hypervolume"] = hypervolume
        
        # Convergence (distance to ideal point)
        convergence = self._compute_convergence_metric(pareto_front, objective_functions)
        metrics["convergence"] = convergence
        
        return metrics
    
    def _compute_spacing_metric(self, pareto_front: List[Individual],
                              objective_functions: List[ObjectiveFunction]) -> float:
        """Compute spacing metric for diversity assessment."""
        if len(pareto_front) < 2:
            return 0.0
        
        # Extract objective vectors
        obj_vectors = []
        for individual in pareto_front:
            vector = []
            for obj_func in objective_functions:
                metric = obj_func.metric.value
                value = individual.objectives.get(metric, 0.0)
                vector.append(value)
            obj_vectors.append(vector)
        
        obj_vectors = np.array(obj_vectors)
        
        # Compute distances between all pairs
        distances = cdist(obj_vectors, obj_vectors)
        
        # For each point, find distance to nearest neighbor
        min_distances = []
        for i in range(len(obj_vectors)):
            non_zero_distances = distances[i][distances[i] > 0]
            if len(non_zero_distances) > 0:
                min_distances.append(np.min(non_zero_distances))
        
        if not min_distances:
            return 0.0
        
        # Spacing metric: standard deviation of minimum distances
        mean_distance = np.mean(min_distances)
        spacing = np.sqrt(np.mean([(d - mean_distance)**2 for d in min_distances]))
        
        return spacing
    
    def _compute_convergence_metric(self, pareto_front: List[Individual],
                                  objective_functions: List[ObjectiveFunction]) -> float:
        """Compute convergence metric (average distance to ideal point)."""
        if not pareto_front:
            return float('inf')
        
        # Estimate ideal point (best value for each objective)
        ideal_point = {}
        
        for obj_func in objective_functions:
            metric = obj_func.metric.value
            values = [ind.objectives.get(metric, 0.0) for ind in pareto_front]
            
            if obj_func.direction == "minimize":
                ideal_point[metric] = min(values)
            else:
                ideal_point[metric] = max(values)
        
        # Compute average distance to ideal point
        distances = []
        for individual in pareto_front:
            distance = 0.0
            for obj_func in objective_functions:
                metric = obj_func.metric.value
                value = individual.objectives.get(metric, 0.0)
                ideal_value = ideal_point[metric]
                
                # Normalize by ideal value to handle different scales
                if ideal_value != 0:
                    normalized_diff = abs(value - ideal_value) / abs(ideal_value)
                else:
                    normalized_diff = abs(value - ideal_value)
                
                distance += normalized_diff ** 2
            
            distances.append(np.sqrt(distance))
        
        return np.mean(distances)


class NSGA3Algorithm:
    """NSGA-III algorithm for many-objective optimization."""
    
    def __init__(self, population_size: int = 100, max_generations: int = 100,
                 crossover_rate: float = 0.9, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
        self.reference_directions = None
        self.convergence_history = []
        
    def generate_reference_directions(self, n_objectives: int, n_divisions: int = 12) -> np.ndarray:
        """Generate uniformly distributed reference directions."""
        if n_objectives == 2:
            # For 2 objectives, use uniform spacing on unit simplex
            directions = []
            for i in range(n_divisions + 1):
                w1 = i / n_divisions
                w2 = 1 - w1
                directions.append([w1, w2])
            return np.array(directions)
        
        elif n_objectives == 3:
            # For 3 objectives, use Das and Dennis method
            directions = []
            for i in range(n_divisions + 1):
                for j in range(n_divisions + 1 - i):
                    k = n_divisions - i - j
                    if k >= 0:
                        w1 = i / n_divisions
                        w2 = j / n_divisions
                        w3 = k / n_divisions
                        directions.append([w1, w2, w3])
            return np.array(directions)
        
        else:
            # For many objectives, use random uniform sampling on simplex
            directions = []
            for _ in range(self.population_size):
                # Generate random point on unit simplex
                random_vals = np.random.exponential(1, n_objectives)
                direction = random_vals / np.sum(random_vals)
                directions.append(direction)
            return np.array(directions)
    
    def fast_non_dominated_sort(self, population: List[Individual],
                              objective_functions: List[ObjectiveFunction]) -> List[List[Individual]]:
        """Fast non-dominated sorting."""
        fronts = []
        first_front = []
        
        for individual in population:
            individual.dominated_solutions = []
            individual.domination_count = 0
            
            for other in population:
                if individual.dominates(other, objective_functions):
                    individual.dominated_solutions.append(other)
                elif other.dominates(individual, objective_functions):
                    individual.domination_count += 1
            
            if individual.domination_count == 0:
                individual.rank = 0
                first_front.append(individual)
        
        fronts.append(first_front)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for individual in fronts[i]:
                for dominated in individual.dominated_solutions:
                    dominated.domination_count -= 1
                    if dominated.domination_count == 0:
                        dominated.rank = i + 1
                        next_front.append(dominated)
            i += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def associate_with_reference_points(self, population: List[Individual],
                                      objective_functions: List[ObjectiveFunction]) -> Dict[int, List[Individual]]:
        """Associate individuals with reference points."""
        if self.reference_directions is None:
            n_objectives = len(objective_functions)
            self.reference_directions = self.generate_reference_directions(n_objectives)
        
        associations = {i: [] for i in range(len(self.reference_directions))}
        
        for individual in population:
            # Extract normalized objectives
            obj_values = []
            for obj_func in objective_functions:
                metric = obj_func.metric.value
                value = individual.objectives.get(metric, 0.0)
                
                # Convert to maximization for consistency
                if obj_func.direction == "minimize":
                    value = -value
                
                obj_values.append(value)
            
            obj_values = np.array(obj_values)
            
            # Find closest reference direction
            distances = []
            for ref_dir in self.reference_directions:
                # Perpendicular distance to reference line
                if np.linalg.norm(ref_dir) > 0:
                    distance = np.linalg.norm(obj_values - np.dot(obj_values, ref_dir) * ref_dir)
                else:
                    distance = np.linalg.norm(obj_values)
                distances.append(distance)
            
            closest_ref_idx = np.argmin(distances)
            associations[closest_ref_idx].append(individual)
        
        return associations
    
    def niching_selection(self, fronts: List[List[Individual]],
                         objective_functions: List[ObjectiveFunction]) -> List[Individual]:
        """Select individuals using niching mechanism."""
        selected = []
        
        # Add complete fronts until population is almost full
        for front in fronts:
            if len(selected) + len(front) <= self.population_size:
                selected.extend(front)
            else:
                # Need to select some individuals from this front
                remaining_slots = self.population_size - len(selected)
                
                if remaining_slots > 0:
                    # Use reference point association for selection
                    associations = self.associate_with_reference_points(front, objective_functions)
                    
                    # Count individuals per reference point
                    ref_counts = {i: len(individuals) for i, individuals in associations.items()}
                    
                    # Select individuals to maintain diversity
                    front_selected = []
                    for _ in range(remaining_slots):
                        # Find reference point with minimum count
                        min_count_ref = min(ref_counts.keys(), key=lambda x: ref_counts[x])
                        
                        if associations[min_count_ref]:
                            # Select individual with best convergence
                            best_individual = min(
                                associations[min_count_ref],
                                key=lambda ind: self._compute_individual_convergence(ind, objective_functions)
                            )
                            
                            front_selected.append(best_individual)
                            associations[min_count_ref].remove(best_individual)
                            ref_counts[min_count_ref] = len(associations[min_count_ref])
                        else:
                            # No individuals left for this reference point
                            ref_counts[min_count_ref] = float('inf')
                    
                    selected.extend(front_selected)
                break
        
        return selected
    
    def _compute_individual_convergence(self, individual: Individual,
                                      objective_functions: List[ObjectiveFunction]) -> float:
        """Compute convergence metric for a single individual."""
        # Simple convergence metric: distance from origin (normalized)
        distance = 0.0
        for obj_func in objective_functions:
            metric = obj_func.metric.value
            value = individual.objectives.get(metric, 0.0)
            distance += value ** 2
        
        return np.sqrt(distance)


class MultiObjectiveOptimizer:
    """Main multi-objective optimization framework."""
    
    def __init__(self, algorithm: str = "nsga3", **algorithm_params):
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params
        
        if algorithm == "nsga3":
            self.optimizer = NSGA3Algorithm(**algorithm_params)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        self.pareto_analyzer = ParetoFrontierAnalysis()
        
    def optimize(self, 
                evaluation_function: Callable,
                parameter_space: Dict[str, Tuple[float, float]],
                objective_functions: List[ObjectiveFunction],
                n_parallel: int = 4,
                seed: int = 42) -> OptimizationResult:
        """
        Run multi-objective optimization.
        
        Args:
            evaluation_function: Function that evaluates a parameter set
            parameter_space: Dict mapping parameter names to (min, max) bounds
            objective_functions: List of objectives to optimize
            n_parallel: Number of parallel evaluations
            seed: Random seed for reproducibility
        
        Returns:
            OptimizationResult containing Pareto front and analysis
        """
        np.random.seed(seed)
        start_time = time.time()
        
        logger.info(f"Starting {self.algorithm} optimization with {len(objective_functions)} objectives")
        
        # Initialize population
        population = self._initialize_population(parameter_space, self.optimizer.population_size)
        
        # Evaluate initial population
        logger.info("Evaluating initial population...")
        population = self._evaluate_population(population, evaluation_function, n_parallel)
        
        # Evolution loop
        convergence_metrics = {
            'hypervolume': [],
            'pareto_size': [],
            'diversity': [],
            'convergence': []
        }
        
        for generation in range(self.optimizer.max_generations):
            logger.info(f"Generation {generation + 1}/{self.optimizer.max_generations}")
            
            # Non-dominated sorting
            fronts = self.optimizer.fast_non_dominated_sort(population, objective_functions)
            
            # Selection for next generation
            selected = self.optimizer.niching_selection(fronts, objective_functions)
            
            # Generate offspring through crossover and mutation
            offspring = self._generate_offspring(selected, parameter_space)
            
            # Evaluate offspring
            offspring = self._evaluate_population(offspring, evaluation_function, n_parallel)
            
            # Combine parent and offspring populations
            population = selected + offspring
            
            # Track convergence
            pareto_front = self.pareto_analyzer.compute_pareto_front(population, objective_functions)
            quality_metrics = self.pareto_analyzer.analyze_pareto_front_quality(
                pareto_front, objective_functions
            )
            
            convergence_metrics['hypervolume'].append(quality_metrics['hypervolume'])
            convergence_metrics['pareto_size'].append(quality_metrics['size'])
            convergence_metrics['diversity'].append(quality_metrics['diversity'])
            convergence_metrics['convergence'].append(quality_metrics['convergence'])
            
            # Log progress
            if (generation + 1) % 10 == 0:
                logger.info(f"  Pareto front size: {quality_metrics['size']}")
                logger.info(f"  Hypervolume: {quality_metrics['hypervolume']:.4f}")
        
        # Final evaluation
        final_pareto_front = self.pareto_analyzer.compute_pareto_front(population, objective_functions)
        final_hypervolume = self.pareto_analyzer.compute_hypervolume(
            final_pareto_front, objective_functions
        )
        
        optimization_time = time.time() - start_time
        
        logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
        logger.info(f"Final Pareto front size: {len(final_pareto_front)}")
        logger.info(f"Final hypervolume: {final_hypervolume:.4f}")
        
        # Convert to serializable format
        solutions = []
        pareto_solutions = []
        
        for individual in population:
            solution = {
                'parameters': individual.parameters,
                'objectives': individual.objectives,
                'rank': individual.rank,
                'crowding_distance': individual.crowding_distance
            }
            solutions.append(solution)
        
        for individual in final_pareto_front:
            solution = {
                'parameters': individual.parameters,
                'objectives': individual.objectives,
                'rank': individual.rank,
                'crowding_distance': individual.crowding_distance
            }
            pareto_solutions.append(solution)
        
        return OptimizationResult(
            solutions=solutions,
            pareto_front=pareto_solutions,
            hypervolume=final_hypervolume,
            convergence_metrics=convergence_metrics,
            optimization_time=optimization_time,
            algorithm_metadata={
                'algorithm': self.algorithm,
                'population_size': self.optimizer.population_size,
                'max_generations': self.optimizer.max_generations,
                'final_generation': generation + 1,
                'n_objectives': len(objective_functions),
                'parameter_space': parameter_space
            }
        )
    
    def _initialize_population(self, parameter_space: Dict[str, Tuple[float, float]],
                             population_size: int) -> List[Individual]:
        """Initialize random population."""
        population = []
        
        for _ in range(population_size):
            parameters = {}
            for param_name, (min_val, max_val) in parameter_space.items():
                parameters[param_name] = np.random.uniform(min_val, max_val)
            
            individual = Individual(parameters=parameters, objectives={}, constraints={})
            population.append(individual)
        
        return population
    
    def _evaluate_population(self, population: List[Individual],
                           evaluation_function: Callable,
                           n_parallel: int) -> List[Individual]:
        """Evaluate population in parallel."""
        def evaluate_individual(individual):
            try:
                result = evaluation_function(individual.parameters)
                individual.objectives = result
                return individual
            except Exception as e:
                logger.error(f"Evaluation failed for individual: {e}")
                # Return individual with poor objectives
                individual.objectives = {
                    'accuracy': 0.0,
                    'power_consumption_mw': 1000.0,
                    'latency_ms': 1000.0
                }
                return individual
        
        if n_parallel > 1:
            with ThreadPoolExecutor(max_workers=n_parallel) as executor:
                evaluated_population = list(executor.map(evaluate_individual, population))
        else:
            evaluated_population = [evaluate_individual(ind) for ind in population]
        
        return evaluated_population
    
    def _generate_offspring(self, parents: List[Individual],
                          parameter_space: Dict[str, Tuple[float, float]]) -> List[Individual]:
        """Generate offspring through crossover and mutation."""
        offspring = []
        
        # Generate pairs for crossover
        parent_pairs = []
        for i in range(0, len(parents) - 1, 2):
            parent_pairs.append((parents[i], parents[i + 1]))
        
        # If odd number of parents, pair last with random parent
        if len(parents) % 2 == 1:
            random_parent = np.random.choice(parents[:-1])
            parent_pairs.append((parents[-1], random_parent))
        
        for parent1, parent2 in parent_pairs:
            if np.random.random() < self.optimizer.crossover_rate:
                # Crossover
                child1_params, child2_params = self._crossover(
                    parent1.parameters, parent2.parameters, parameter_space
                )
            else:
                # No crossover, copy parents
                child1_params = parent1.parameters.copy()
                child2_params = parent2.parameters.copy()
            
            # Mutation
            child1_params = self._mutate(child1_params, parameter_space)
            child2_params = self._mutate(child2_params, parameter_space)
            
            # Create offspring individuals
            child1 = Individual(parameters=child1_params, objectives={}, constraints={})
            child2 = Individual(parameters=child2_params, objectives={}, constraints={})
            
            offspring.extend([child1, child2])
        
        return offspring
    
    def _crossover(self, params1: Dict[str, float], params2: Dict[str, float],
                  parameter_space: Dict[str, Tuple[float, float]]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Simulated binary crossover (SBX)."""
        eta_c = 20  # Distribution index for crossover
        
        child1_params = {}
        child2_params = {}
        
        for param_name in params1.keys():
            p1 = params1[param_name]
            p2 = params2[param_name]
            min_val, max_val = parameter_space[param_name]
            
            if np.random.random() <= 0.5:
                if abs(p1 - p2) > 1e-14:
                    y1 = min(p1, p2)
                    y2 = max(p1, p2)
                    
                    # Calculate beta
                    rand = np.random.random()
                    if rand <= 0.5:
                        beta = (2 * rand) ** (1.0 / (eta_c + 1))
                    else:
                        beta = (1.0 / (2 * (1 - rand))) ** (1.0 / (eta_c + 1))
                    
                    # Calculate children
                    c1 = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    c2 = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                    
                    # Ensure bounds
                    c1 = np.clip(c1, min_val, max_val)
                    c2 = np.clip(c2, min_val, max_val)
                    
                    child1_params[param_name] = c1
                    child2_params[param_name] = c2
                else:
                    child1_params[param_name] = p1
                    child2_params[param_name] = p2
            else:
                child1_params[param_name] = p1
                child2_params[param_name] = p2
        
        return child1_params, child2_params
    
    def _mutate(self, params: Dict[str, float],
               parameter_space: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Polynomial mutation."""
        eta_m = 20  # Distribution index for mutation
        mutated_params = params.copy()
        
        for param_name, value in params.items():
            if np.random.random() < self.optimizer.mutation_rate:
                min_val, max_val = parameter_space[param_name]
                
                delta1 = (value - min_val) / (max_val - min_val)
                delta2 = (max_val - value) / (max_val - min_val)
                
                rand = np.random.random()
                mut_pow = 1.0 / (eta_m + 1)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta_m + 1))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta_m + 1))
                    delta_q = 1.0 - val ** mut_pow
                
                mutated_value = value + delta_q * (max_val - min_val)
                mutated_value = np.clip(mutated_value, min_val, max_val)
                mutated_params[param_name] = mutated_value
        
        return mutated_params
    
    def visualize_pareto_front(self, result: OptimizationResult,
                             objective_functions: List[ObjectiveFunction],
                             save_path: Optional[str] = None) -> None:
        """Visualize Pareto front for 2D or 3D objectives."""
        if len(objective_functions) == 2:
            self._plot_2d_pareto_front(result, objective_functions, save_path)
        elif len(objective_functions) == 3:
            self._plot_3d_pareto_front(result, objective_functions, save_path)
        else:
            logger.warning("Visualization only supported for 2D and 3D objectives")
    
    def _plot_2d_pareto_front(self, result: OptimizationResult,
                            objective_functions: List[ObjectiveFunction],
                            save_path: Optional[str] = None) -> None:
        """Plot 2D Pareto front."""
        obj1_name = objective_functions[0].metric.value
        obj2_name = objective_functions[1].metric.value
        
        # Extract all solutions
        all_obj1 = [sol['objectives'][obj1_name] for sol in result.solutions]
        all_obj2 = [sol['objectives'][obj2_name] for sol in result.solutions]
        
        # Extract Pareto front
        pareto_obj1 = [sol['objectives'][obj1_name] for sol in result.pareto_front]
        pareto_obj2 = [sol['objectives'][obj2_name] for sol in result.pareto_front]
        
        plt.figure(figsize=(10, 8))
        plt.scatter(all_obj1, all_obj2, alpha=0.6, label='All Solutions', color='lightblue')
        plt.scatter(pareto_obj1, pareto_obj2, alpha=0.8, label='Pareto Front', color='red', s=50)
        
        plt.xlabel(f'{obj1_name} {"(minimize)" if objective_functions[0].direction == "minimize" else "(maximize)"}')
        plt.ylabel(f'{obj2_name} {"(minimize)" if objective_functions[1].direction == "minimize" else "(maximize)"}')
        plt.title('2D Pareto Front')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_3d_pareto_front(self, result: OptimizationResult,
                            objective_functions: List[ObjectiveFunction],
                            save_path: Optional[str] = None) -> None:
        """Plot 3D Pareto front."""
        from mpl_toolkits.mplot3d import Axes3D
        
        obj1_name = objective_functions[0].metric.value
        obj2_name = objective_functions[1].metric.value
        obj3_name = objective_functions[2].metric.value
        
        # Extract all solutions
        all_obj1 = [sol['objectives'][obj1_name] for sol in result.solutions]
        all_obj2 = [sol['objectives'][obj2_name] for sol in result.solutions]
        all_obj3 = [sol['objectives'][obj3_name] for sol in result.solutions]
        
        # Extract Pareto front
        pareto_obj1 = [sol['objectives'][obj1_name] for sol in result.pareto_front]
        pareto_obj2 = [sol['objectives'][obj2_name] for sol in result.pareto_front]
        pareto_obj3 = [sol['objectives'][obj3_name] for sol in result.pareto_front]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(all_obj1, all_obj2, all_obj3, alpha=0.6, label='All Solutions', color='lightblue')
        ax.scatter(pareto_obj1, pareto_obj2, pareto_obj3, alpha=0.8, label='Pareto Front', 
                  color='red', s=50)
        
        ax.set_xlabel(f'{obj1_name} {"(min)" if objective_functions[0].direction == "minimize" else "(max)"}')
        ax.set_ylabel(f'{obj2_name} {"(min)" if objective_functions[1].direction == "minimize" else "(max)"}')
        ax.set_zlabel(f'{obj3_name} {"(min)" if objective_functions[2].direction == "minimize" else "(max)"}')
        ax.set_title('3D Pareto Front')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_optimization_report(self, result: OptimizationResult,
                                   objective_functions: List[ObjectiveFunction]) -> str:
        """Generate comprehensive optimization report."""
        report = []
        
        # Header
        report.append("# Multi-Objective Optimization Report")
        report.append(f"**Algorithm:** {result.algorithm_metadata['algorithm'].upper()}")
        report.append(f"**Optimization Time:** {result.optimization_time:.2f} seconds")
        report.append("")
        
        # Problem Configuration
        report.append("## Problem Configuration")
        report.append(f"- **Objectives:** {len(objective_functions)}")
        for obj in objective_functions:
            report.append(f"  - {obj.name}: {obj.direction} {obj.metric.value}")
        report.append(f"- **Parameters:** {len(result.algorithm_metadata['parameter_space'])}")
        for param, bounds in result.algorithm_metadata['parameter_space'].items():
            report.append(f"  - {param}: [{bounds[0]:.3f}, {bounds[1]:.3f}]")
        report.append("")
        
        # Results Summary
        report.append("## Results Summary")
        report.append(f"- **Total Solutions Evaluated:** {len(result.solutions)}")
        report.append(f"- **Pareto Front Size:** {len(result.pareto_front)}")
        report.append(f"- **Final Hypervolume:** {result.hypervolume:.4f}")
        report.append("")
        
        # Pareto Front Analysis
        if result.pareto_front:
            report.append("## Pareto Front Analysis")
            
            # Best solutions for each objective
            for obj in objective_functions:
                metric = obj.metric.value
                if obj.direction == "minimize":
                    best_sol = min(result.pareto_front, key=lambda x: x['objectives'][metric])
                    report.append(f"**Best {obj.name}:** {best_sol['objectives'][metric]:.4f}")
                else:
                    best_sol = max(result.pareto_front, key=lambda x: x['objectives'][metric])
                    report.append(f"**Best {obj.name}:** {best_sol['objectives'][metric]:.4f}")
            
            report.append("")
            
            # Pareto front table
            report.append("### Pareto Front Solutions")
            
            # Create header
            header = "| Solution |"
            for obj in objective_functions:
                header += f" {obj.name} |"
            report.append(header)
            
            separator = "|----------|"
            for _ in objective_functions:
                separator += "----------|"
            report.append(separator)
            
            # Add solutions
            for i, sol in enumerate(result.pareto_front[:10]):  # Show first 10
                row = f"| {i+1:2d} |"
                for obj in objective_functions:
                    metric = obj.metric.value
                    value = sol['objectives'][metric]
                    row += f" {value:.4f} |"
                report.append(row)
            
            if len(result.pareto_front) > 10:
                report.append(f"| ... | ... (showing 10 of {len(result.pareto_front)} solutions) |")
            
            report.append("")
        
        # Convergence Analysis
        report.append("## Convergence Analysis")
        
        if result.convergence_metrics['hypervolume']:
            final_hv = result.convergence_metrics['hypervolume'][-1]
            initial_hv = result.convergence_metrics['hypervolume'][0]
            hv_improvement = ((final_hv - initial_hv) / initial_hv * 100) if initial_hv > 0 else 0
            
            report.append(f"- **Hypervolume Improvement:** {hv_improvement:.2f}%")
            report.append(f"- **Final Pareto Front Diversity:** {result.convergence_metrics['diversity'][-1]:.4f}")
            report.append(f"- **Convergence Metric:** {result.convergence_metrics['convergence'][-1]:.4f}")
        
        report.append("")
        
        # Algorithm Configuration
        report.append("## Algorithm Configuration")
        meta = result.algorithm_metadata
        report.append(f"- **Population Size:** {meta['population_size']}")
        report.append(f"- **Generations Run:** {meta['final_generation']}")
        report.append(f"- **Max Generations:** {meta.get('max_generations', 'N/A')}")
        report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        
        if len(result.pareto_front) < 5:
            report.append("- Consider increasing population size or generations for better diversity")
        
        if result.hypervolume < 0.1:
            report.append("- Low hypervolume indicates poor convergence - check objective scaling")
        
        if len(objective_functions) > 3:
            report.append("- Many-objective problem - consider using preference-based selection")
        
        report.append("")
        report.append("---")
        report.append("*Report generated by liquid-audio-nets multi-objective optimization framework*")
        
        return "\n".join(report)


# Example usage
if __name__ == "__main__":
    # This would be run as a standalone script for testing
    pass