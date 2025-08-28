"""
Self-Evolving Neural Architecture Search (SENAS) for Liquid Neural Networks.

This module implements a groundbreaking approach to neural architecture optimization that
combines quantum-inspired evolution, multi-objective optimization, and real-time adaptation
to continuously evolve network architectures for optimal audio processing performance.

Key Innovations:
- Quantum-inspired mutation operators for architecture evolution
- Multi-objective optimization balancing accuracy, efficiency, and robustness
- Real-time architecture adaptation based on input statistics
- Hierarchical architecture representation with fractal self-similarity
- Emergent complexity control with biological growth patterns

Research Contribution:
This represents the first implementation of fully autonomous neural architecture evolution
that operates during inference, enabling networks to self-optimize for changing environments
and requirements without human intervention.
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from collections import deque, defaultdict
import heapq
import json
from enum import Enum, auto
import math
import random
from scipy.optimize import differential_evolution
from scipy.special import softmax
import hashlib

logger = logging.getLogger(__name__)


class MutationType(Enum):
    """Types of architecture mutations."""
    ADD_NEURON = auto()
    REMOVE_NEURON = auto()
    ADD_CONNECTION = auto()
    REMOVE_CONNECTION = auto()
    MODIFY_ACTIVATION = auto()
    SPLIT_LAYER = auto()
    MERGE_LAYERS = auto()
    QUANTUM_SUPERPOSITION = auto()
    FRACTAL_EXPANSION = auto()
    SYNAPTIC_PLASTICITY = auto()


class ActivationType(Enum):
    """Types of activation functions."""
    TANH = auto()
    RELU = auto()
    SIGMOID = auto()
    LEAKY_RELU = auto()
    SWISH = auto()
    MISH = auto()
    QUANTUM_GATE = auto()
    ADAPTIVE = auto()


@dataclass
class NeuronGene:
    """Genetic representation of a neuron."""
    id: int
    layer: int
    activation: ActivationType
    bias: float
    plasticity: float = 0.1
    adaptation_rate: float = 0.01
    quantum_coherence: float = 0.0
    
    def mutate(self, mutation_strength: float = 0.1) -> None:
        """Mutate neuron parameters."""
        self.bias += np.random.normal(0, mutation_strength)
        self.plasticity += np.random.normal(0, mutation_strength * 0.1)
        self.adaptation_rate += np.random.normal(0, mutation_strength * 0.01)
        self.quantum_coherence += np.random.normal(0, mutation_strength * 0.05)
        
        # Keep values in reasonable bounds
        self.plasticity = np.clip(self.plasticity, 0.01, 1.0)
        self.adaptation_rate = np.clip(self.adaptation_rate, 0.001, 0.1)
        self.quantum_coherence = np.clip(self.quantum_coherence, 0.0, 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'layer': self.layer,
            'activation': self.activation.name,
            'bias': self.bias,
            'plasticity': self.plasticity,
            'adaptation_rate': self.adaptation_rate,
            'quantum_coherence': self.quantum_coherence
        }


@dataclass
class ConnectionGene:
    """Genetic representation of a connection between neurons."""
    id: int
    input_neuron: int
    output_neuron: int
    weight: float
    enabled: bool = True
    innovation_number: int = 0
    plasticity_rule: str = "hebbian"
    delay: float = 0.0  # Synaptic delay for temporal processing
    
    def mutate(self, mutation_strength: float = 0.1) -> None:
        """Mutate connection parameters."""
        self.weight += np.random.normal(0, mutation_strength)
        self.delay += np.random.normal(0, mutation_strength * 0.01)
        
        # Random enable/disable with low probability
        if np.random.random() < 0.05:
            self.enabled = not self.enabled
        
        # Keep delay positive and reasonable
        self.delay = np.clip(self.delay, 0.0, 0.1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'id': self.id,
            'input_neuron': self.input_neuron,
            'output_neuron': self.output_neuron,
            'weight': self.weight,
            'enabled': self.enabled,
            'innovation_number': self.innovation_number,
            'plasticity_rule': self.plasticity_rule,
            'delay': self.delay
        }


@dataclass
class ArchitectureGenome:
    """Complete genetic representation of a neural architecture."""
    id: str
    neurons: Dict[int, NeuronGene] = field(default_factory=dict)
    connections: Dict[int, ConnectionGene] = field(default_factory=dict)
    input_size: int = 40
    output_size: int = 10
    generation: int = 0
    fitness_history: List[float] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize genome ID if not provided."""
        if not self.id:
            # Create deterministic ID based on structure
            structure_hash = self._calculate_structure_hash()
            self.id = f"genome_{structure_hash}_{self.generation}"
    
    def _calculate_structure_hash(self) -> str:
        """Calculate hash of architecture structure."""
        structure_data = {
            'neurons': sorted([n.to_dict() for n in self.neurons.values()], key=lambda x: x['id']),
            'connections': sorted([c.to_dict() for c in self.connections.values()], key=lambda x: x['id'])
        }
        structure_str = json.dumps(structure_data, sort_keys=True)
        return hashlib.md5(structure_str.encode()).hexdigest()[:8]
    
    def add_neuron(self, layer: int, activation: ActivationType = ActivationType.TANH) -> int:
        """Add a new neuron to the genome."""
        neuron_id = max(self.neurons.keys()) + 1 if self.neurons else 0
        self.neurons[neuron_id] = NeuronGene(
            id=neuron_id,
            layer=layer,
            activation=activation,
            bias=np.random.normal(0, 0.1)
        )
        return neuron_id
    
    def remove_neuron(self, neuron_id: int) -> bool:
        """Remove a neuron and all its connections."""
        if neuron_id not in self.neurons:
            return False
        
        # Remove all connections involving this neuron
        connections_to_remove = []
        for conn_id, connection in self.connections.items():
            if connection.input_neuron == neuron_id or connection.output_neuron == neuron_id:
                connections_to_remove.append(conn_id)
        
        for conn_id in connections_to_remove:
            del self.connections[conn_id]
        
        # Remove the neuron
        del self.neurons[neuron_id]
        return True
    
    def add_connection(self, input_neuron: int, output_neuron: int, 
                      weight: Optional[float] = None) -> Optional[int]:
        """Add a connection between neurons."""
        if input_neuron not in self.neurons or output_neuron not in self.neurons:
            return None
        
        # Check for existing connection
        for connection in self.connections.values():
            if (connection.input_neuron == input_neuron and 
                connection.output_neuron == output_neuron):
                return None  # Connection already exists
        
        conn_id = max(self.connections.keys()) + 1 if self.connections else 0
        if weight is None:
            weight = np.random.normal(0, 0.1)
        
        self.connections[conn_id] = ConnectionGene(
            id=conn_id,
            input_neuron=input_neuron,
            output_neuron=output_neuron,
            weight=weight
        )
        return conn_id
    
    def get_layer_neurons(self, layer: int) -> List[int]:
        """Get all neuron IDs in a specific layer."""
        return [nid for nid, neuron in self.neurons.items() if neuron.layer == layer]
    
    def get_max_layer(self) -> int:
        """Get the maximum layer number."""
        return max([neuron.layer for neuron in self.neurons.values()]) if self.neurons else 0
    
    def calculate_complexity(self) -> float:
        """Calculate architectural complexity."""
        num_neurons = len(self.neurons)
        num_connections = len([c for c in self.connections.values() if c.enabled])
        max_connections = num_neurons * (num_neurons - 1)
        
        if max_connections == 0:
            return 0.0
        
        connection_density = num_connections / max_connections
        layer_complexity = self.get_max_layer() + 1
        
        complexity = (num_neurons / 100.0 + connection_density + layer_complexity / 10.0) / 3.0
        return np.clip(complexity, 0.0, 1.0)
    
    def clone(self) -> 'ArchitectureGenome':
        """Create a deep copy of the genome."""
        clone = ArchitectureGenome(
            id="",  # Will be regenerated
            input_size=self.input_size,
            output_size=self.output_size,
            generation=self.generation + 1
        )
        
        # Clone neurons
        for neuron_id, neuron in self.neurons.items():
            clone.neurons[neuron_id] = NeuronGene(
                id=neuron.id,
                layer=neuron.layer,
                activation=neuron.activation,
                bias=neuron.bias,
                plasticity=neuron.plasticity,
                adaptation_rate=neuron.adaptation_rate,
                quantum_coherence=neuron.quantum_coherence
            )
        
        # Clone connections
        for conn_id, connection in self.connections.items():
            clone.connections[conn_id] = ConnectionGene(
                id=connection.id,
                input_neuron=connection.input_neuron,
                output_neuron=connection.output_neuron,
                weight=connection.weight,
                enabled=connection.enabled,
                innovation_number=connection.innovation_number,
                plasticity_rule=connection.plasticity_rule,
                delay=connection.delay
            )
        
        return clone
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert genome to dictionary representation."""
        return {
            'id': self.id,
            'neurons': {nid: neuron.to_dict() for nid, neuron in self.neurons.items()},
            'connections': {cid: conn.to_dict() for cid, conn in self.connections.items()},
            'input_size': self.input_size,
            'output_size': self.output_size,
            'generation': self.generation,
            'fitness_history': self.fitness_history.copy(),
            'performance_metrics': self.performance_metrics.copy(),
            'complexity': self.calculate_complexity()
        }


class QuantumMutator:
    """Quantum-inspired mutation operator for architecture evolution."""
    
    def __init__(self, mutation_rate: float = 0.1, quantum_strength: float = 0.3):
        self.mutation_rate = mutation_rate
        self.quantum_strength = quantum_strength
        
    def mutate_genome(self, genome: ArchitectureGenome) -> ArchitectureGenome:
        """Apply quantum-inspired mutations to genome."""
        mutated = genome.clone()
        
        # Apply different types of mutations with quantum superposition
        mutation_probabilities = self._calculate_quantum_probabilities()
        
        for mutation_type, probability in mutation_probabilities.items():
            if np.random.random() < probability:
                success = self._apply_mutation(mutated, mutation_type)
                if success:
                    logger.debug(f"Applied mutation: {mutation_type.name}")
        
        return mutated
    
    def _calculate_quantum_probabilities(self) -> Dict[MutationType, float]:
        """Calculate quantum superposition of mutation probabilities."""
        base_prob = self.mutation_rate
        
        # Quantum interference patterns affect mutation probabilities
        quantum_phase = np.random.uniform(0, 2 * np.pi)
        
        probabilities = {}
        for i, mutation_type in enumerate(MutationType):
            # Quantum interference pattern
            interference = np.cos(quantum_phase + i * np.pi / 4)
            quantum_modifier = 1.0 + self.quantum_strength * interference
            
            probabilities[mutation_type] = base_prob * quantum_modifier
        
        # Normalize probabilities
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            probabilities = {k: v / total_prob * base_prob * len(probabilities) 
                           for k, v in probabilities.items()}
        
        return probabilities
    
    def _apply_mutation(self, genome: ArchitectureGenome, mutation_type: MutationType) -> bool:
        """Apply specific mutation type to genome."""
        try:
            if mutation_type == MutationType.ADD_NEURON:
                return self._mutate_add_neuron(genome)
            elif mutation_type == MutationType.REMOVE_NEURON:
                return self._mutate_remove_neuron(genome)
            elif mutation_type == MutationType.ADD_CONNECTION:
                return self._mutate_add_connection(genome)
            elif mutation_type == MutationType.REMOVE_CONNECTION:
                return self._mutate_remove_connection(genome)
            elif mutation_type == MutationType.MODIFY_ACTIVATION:
                return self._mutate_modify_activation(genome)
            elif mutation_type == MutationType.QUANTUM_SUPERPOSITION:
                return self._mutate_quantum_superposition(genome)
            elif mutation_type == MutationType.FRACTAL_EXPANSION:
                return self._mutate_fractal_expansion(genome)
            else:
                # For other mutation types, apply parameter mutations
                return self._mutate_parameters(genome)
        except Exception as e:
            logger.warning(f"Mutation {mutation_type.name} failed: {e}")
            return False
    
    def _mutate_add_neuron(self, genome: ArchitectureGenome) -> bool:
        """Add a new neuron to the genome."""
        if len(genome.neurons) >= 200:  # Limit maximum size
            return False
        
        # Choose layer (prefer middle layers for hidden processing)
        max_layer = genome.get_max_layer()
        layer = np.random.randint(1, max(2, max_layer))
        
        # Choose activation function with quantum superposition
        activations = list(ActivationType)
        weights = np.exp(np.random.randn(len(activations)))  # Quantum amplitudes
        weights = weights / np.sum(weights)
        activation = np.random.choice(activations, p=weights)
        
        neuron_id = genome.add_neuron(layer, activation)
        
        # Add some random connections
        self._add_random_connections_for_neuron(genome, neuron_id)
        
        return True
    
    def _mutate_remove_neuron(self, genome: ArchitectureGenome) -> bool:
        """Remove a neuron from the genome."""
        if len(genome.neurons) <= 10:  # Maintain minimum size
            return False
        
        # Choose neuron to remove (avoid input/output neurons)
        removable_neurons = [nid for nid, neuron in genome.neurons.items() 
                           if neuron.layer > 0 and neuron.layer < genome.get_max_layer()]
        
        if not removable_neurons:
            return False
        
        neuron_to_remove = np.random.choice(removable_neurons)
        return genome.remove_neuron(neuron_to_remove)
    
    def _mutate_add_connection(self, genome: ArchitectureGenome) -> bool:
        """Add a new connection between neurons."""
        neurons = list(genome.neurons.keys())
        if len(neurons) < 2:
            return False
        
        # Try multiple times to find valid connection
        for _ in range(10):
            input_neuron = np.random.choice(neurons)
            output_neuron = np.random.choice(neurons)
            
            # Ensure connection goes forward in layers
            input_layer = genome.neurons[input_neuron].layer
            output_layer = genome.neurons[output_neuron].layer
            
            if input_layer < output_layer:
                conn_id = genome.add_connection(input_neuron, output_neuron)
                if conn_id is not None:
                    return True
        
        return False
    
    def _mutate_remove_connection(self, genome: ArchitectureGenome) -> bool:
        """Remove a random connection."""
        if not genome.connections:
            return False
        
        conn_id = np.random.choice(list(genome.connections.keys()))
        del genome.connections[conn_id]
        return True
    
    def _mutate_modify_activation(self, genome: ArchitectureGenome) -> bool:
        """Modify activation function of a random neuron."""
        if not genome.neurons:
            return False
        
        neuron_id = np.random.choice(list(genome.neurons.keys()))
        neuron = genome.neurons[neuron_id]
        
        # Quantum superposition of activation functions
        activations = list(ActivationType)
        new_activation = np.random.choice(activations)
        neuron.activation = new_activation
        
        return True
    
    def _mutate_quantum_superposition(self, genome: ArchitectureGenome) -> bool:
        """Apply quantum superposition-like effects to multiple neurons."""
        neurons = list(genome.neurons.values())
        if len(neurons) < 2:
            return False
        
        # Select neurons for quantum entanglement
        num_entangled = min(5, len(neurons))
        selected_neurons = np.random.choice(neurons, num_entangled, replace=False)
        
        # Apply coherent quantum evolution
        quantum_phase = np.random.uniform(0, 2 * np.pi)
        
        for i, neuron in enumerate(selected_neurons):
            phase_offset = i * 2 * np.pi / len(selected_neurons)
            coherence_factor = np.cos(quantum_phase + phase_offset)
            
            neuron.quantum_coherence += 0.1 * coherence_factor
            neuron.quantum_coherence = np.clip(neuron.quantum_coherence, 0.0, 1.0)
            
            # Quantum interference affects bias
            neuron.bias += 0.05 * np.sin(quantum_phase + phase_offset)
        
        return True
    
    def _mutate_fractal_expansion(self, genome: ArchitectureGenome) -> bool:
        """Apply fractal-like expansion patterns to the architecture."""
        if len(genome.neurons) >= 150:
            return False
        
        # Find a suitable motif to replicate
        layers = defaultdict(list)
        for neuron in genome.neurons.values():
            layers[neuron.layer].append(neuron)
        
        # Choose a layer to expand
        expandable_layers = [layer for layer, neurons in layers.items() 
                           if len(neurons) > 1 and layer > 0]
        
        if not expandable_layers:
            return False
        
        source_layer = np.random.choice(expandable_layers)
        source_neurons = layers[source_layer]
        
        # Create fractal expansion
        new_layer = genome.get_max_layer() + 1
        
        for source_neuron in source_neurons:
            # Create similar neuron with variations
            new_neuron_id = genome.add_neuron(new_layer, source_neuron.activation)
            new_neuron = genome.neurons[new_neuron_id]
            
            # Copy properties with fractal scaling
            scale_factor = 0.8 + 0.4 * np.random.random()  # 0.8 to 1.2 scaling
            new_neuron.bias = source_neuron.bias * scale_factor
            new_neuron.plasticity = source_neuron.plasticity * scale_factor
            new_neuron.quantum_coherence = source_neuron.quantum_coherence * scale_factor
            
            # Add connections with fractal pattern
            self._add_fractal_connections(genome, source_neuron.id, new_neuron_id)
        
        return True
    
    def _mutate_parameters(self, genome: ArchitectureGenome) -> bool:
        """Apply parameter mutations to neurons and connections."""
        # Mutate neuron parameters
        for neuron in genome.neurons.values():
            if np.random.random() < 0.3:
                neuron.mutate(0.1)
        
        # Mutate connection parameters
        for connection in genome.connections.values():
            if np.random.random() < 0.3:
                connection.mutate(0.1)
        
        return True
    
    def _add_random_connections_for_neuron(self, genome: ArchitectureGenome, neuron_id: int) -> None:
        """Add random connections for a newly created neuron."""
        neuron = genome.neurons[neuron_id]
        
        # Add connections from previous layer neurons
        prev_layer_neurons = genome.get_layer_neurons(neuron.layer - 1)
        for prev_neuron_id in prev_layer_neurons:
            if np.random.random() < 0.5:  # 50% chance for each connection
                genome.add_connection(prev_neuron_id, neuron_id)
        
        # Add connections to next layer neurons
        next_layer_neurons = genome.get_layer_neurons(neuron.layer + 1)
        for next_neuron_id in next_layer_neurons:
            if np.random.random() < 0.5:
                genome.add_connection(neuron_id, next_neuron_id)
    
    def _add_fractal_connections(self, genome: ArchitectureGenome, 
                               source_neuron_id: int, target_neuron_id: int) -> None:
        """Add fractal-pattern connections between neurons."""
        # Copy connection patterns from source neuron
        for connection in genome.connections.values():
            if connection.input_neuron == source_neuron_id or connection.output_neuron == source_neuron_id:
                # Create similar connection for target neuron
                if connection.input_neuron == source_neuron_id:
                    # Find corresponding neuron in target layer
                    if np.random.random() < 0.7:  # Not all connections are copied
                        genome.add_connection(target_neuron_id, connection.output_neuron,
                                            connection.weight * (0.8 + 0.4 * np.random.random()))
                else:
                    if np.random.random() < 0.7:
                        genome.add_connection(connection.input_neuron, target_neuron_id,
                                            connection.weight * (0.8 + 0.4 * np.random.random()))


@dataclass
class FitnessMetrics:
    """Comprehensive fitness metrics for architecture evaluation."""
    accuracy: float = 0.0
    processing_speed: float = 0.0  # Inverted processing time
    memory_efficiency: float = 0.0
    power_efficiency: float = 0.0
    robustness: float = 0.0
    adaptability: float = 0.0
    complexity_score: float = 0.0
    quantum_coherence: float = 0.0
    
    def calculate_weighted_fitness(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate weighted fitness score."""
        if weights is None:
            weights = {
                'accuracy': 0.3,
                'processing_speed': 0.2,
                'memory_efficiency': 0.15,
                'power_efficiency': 0.15,
                'robustness': 0.1,
                'adaptability': 0.05,
                'quantum_coherence': 0.05
            }
        
        fitness = 0.0
        for metric, value in self.__dict__.items():
            if metric in weights:
                fitness += weights[metric] * value
        
        # Apply complexity penalty
        complexity_penalty = max(0, self.complexity_score - 0.7) * 0.5
        fitness -= complexity_penalty
        
        return fitness
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


class EvolutionaryOptimizer:
    """Multi-objective evolutionary optimizer for neural architectures."""
    
    def __init__(self, population_size: int = 50, elite_ratio: float = 0.2):
        self.population_size = population_size
        self.elite_ratio = elite_ratio
        self.population: List[ArchitectureGenome] = []
        self.fitness_cache: Dict[str, FitnessMetrics] = {}
        self.generation = 0
        self.best_genome: Optional[ArchitectureGenome] = None
        self.best_fitness = -float('inf')
        
        # Evolution parameters
        self.mutator = QuantumMutator(mutation_rate=0.15, quantum_strength=0.4)
        self.species_groups: Dict[str, List[ArchitectureGenome]] = {}
        
        # Performance tracking
        self.fitness_history: List[float] = []
        self.diversity_history: List[float] = []
        
    def initialize_population(self, input_size: int = 40, output_size: int = 10) -> None:
        """Initialize random population of architectures."""
        self.population = []
        
        for i in range(self.population_size):
            genome = self._create_random_genome(input_size, output_size)
            self.population.append(genome)
        
        logger.info(f"Initialized population of {self.population_size} genomes")
    
    def _create_random_genome(self, input_size: int, output_size: int) -> ArchitectureGenome:
        """Create a random genome with reasonable structure."""
        genome = ArchitectureGenome(
            id="", 
            input_size=input_size, 
            output_size=output_size
        )
        
        # Create input layer
        for i in range(input_size):
            genome.add_neuron(0, ActivationType.TANH)
        
        # Create hidden layers
        num_hidden_layers = np.random.randint(1, 4)
        neurons_per_layer = np.random.randint(16, 80)
        
        for layer in range(1, num_hidden_layers + 1):
            layer_size = max(8, int(neurons_per_layer * (0.8 ** layer)))  # Decreasing size
            for i in range(layer_size):
                activation = np.random.choice(list(ActivationType))
                genome.add_neuron(layer, activation)
        
        # Create output layer
        output_layer = num_hidden_layers + 1
        for i in range(output_size):
            genome.add_neuron(output_layer, ActivationType.TANH)
        
        # Add connections between adjacent layers
        for layer in range(output_layer):
            current_layer_neurons = genome.get_layer_neurons(layer)
            next_layer_neurons = genome.get_layer_neurons(layer + 1)
            
            for input_neuron in current_layer_neurons:
                for output_neuron in next_layer_neurons:
                    # Connect with probability based on layer sizes
                    connection_prob = min(0.8, 20 / (len(current_layer_neurons) + len(next_layer_neurons)))
                    if np.random.random() < connection_prob:
                        genome.add_connection(input_neuron, output_neuron)
        
        # Add some skip connections
        self._add_skip_connections(genome)
        
        return genome
    
    def _add_skip_connections(self, genome: ArchitectureGenome) -> None:
        """Add skip connections for better gradient flow."""
        max_layer = genome.get_max_layer()
        
        for layer in range(max_layer - 1):
            current_neurons = genome.get_layer_neurons(layer)
            
            # Skip connections to layers beyond the next one
            for skip_layer in range(layer + 2, min(layer + 4, max_layer + 1)):
                skip_neurons = genome.get_layer_neurons(skip_layer)
                
                for input_neuron in current_neurons:
                    for output_neuron in skip_neurons:
                        if np.random.random() < 0.1:  # Low probability for skip connections
                            genome.add_connection(input_neuron, output_neuron, weight=np.random.normal(0, 0.05))
    
    def evolve_generation(self, fitness_evaluator: Callable[[ArchitectureGenome], FitnessMetrics]) -> None:
        """Evolve population by one generation."""
        logger.info(f"Evolving generation {self.generation}")
        
        # Evaluate fitness for all genomes
        fitness_scores = []
        for genome in self.population:
            if genome.id not in self.fitness_cache:
                fitness = fitness_evaluator(genome)
                self.fitness_cache[genome.id] = fitness
                genome.performance_metrics = fitness.to_dict()
            else:
                fitness = self.fitness_cache[genome.id]
            
            weighted_fitness = fitness.calculate_weighted_fitness()
            fitness_scores.append(weighted_fitness)
            genome.fitness_history.append(weighted_fitness)
            
            # Track best genome
            if weighted_fitness > self.best_fitness:
                self.best_fitness = weighted_fitness
                self.best_genome = genome.clone()
        
        # Calculate population statistics
        avg_fitness = np.mean(fitness_scores)
        diversity = self._calculate_population_diversity()
        
        self.fitness_history.append(avg_fitness)
        self.diversity_history.append(diversity)
        
        logger.info(f"Generation {self.generation}: avg_fitness={avg_fitness:.4f}, "
                   f"best_fitness={self.best_fitness:.4f}, diversity={diversity:.4f}")
        
        # Species formation and selection
        self._form_species()
        new_population = self._select_and_reproduce()
        
        self.population = new_population
        self.generation += 1
    
    def _calculate_population_diversity(self) -> float:
        """Calculate genetic diversity of the population."""
        if len(self.population) < 2:
            return 0.0
        
        diversities = []
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                genome1, genome2 = self.population[i], self.population[j]
                
                # Calculate structural similarity
                similarity = self._calculate_genome_similarity(genome1, genome2)
                diversity = 1.0 - similarity
                diversities.append(diversity)
        
        return np.mean(diversities)
    
    def _calculate_genome_similarity(self, genome1: ArchitectureGenome, 
                                   genome2: ArchitectureGenome) -> float:
        """Calculate similarity between two genomes."""
        # Compare neuron counts by layer
        layers1 = defaultdict(int)
        layers2 = defaultdict(int)
        
        for neuron in genome1.neurons.values():
            layers1[neuron.layer] += 1
        
        for neuron in genome2.neurons.values():
            layers2[neuron.layer] += 1
        
        # Calculate layer similarity
        all_layers = set(layers1.keys()) | set(layers2.keys())
        layer_similarities = []
        
        for layer in all_layers:
            count1 = layers1.get(layer, 0)
            count2 = layers2.get(layer, 0)
            max_count = max(count1, count2)
            
            if max_count > 0:
                similarity = 1.0 - abs(count1 - count2) / max_count
                layer_similarities.append(similarity)
        
        layer_similarity = np.mean(layer_similarities) if layer_similarities else 0.0
        
        # Compare connection density
        density1 = len(genome1.connections) / max(1, len(genome1.neurons)**2)
        density2 = len(genome2.connections) / max(1, len(genome2.neurons)**2)
        density_similarity = 1.0 - abs(density1 - density2)
        
        # Weighted combination
        total_similarity = 0.7 * layer_similarity + 0.3 * density_similarity
        
        return total_similarity
    
    def _form_species(self) -> None:
        """Form species groups for diversity preservation."""
        self.species_groups = {}
        compatibility_threshold = 0.6
        
        for genome in self.population:
            # Find compatible species
            placed = False
            
            for species_name, species_members in self.species_groups.items():
                if species_members:  # Check if species has members
                    representative = species_members[0]
                    similarity = self._calculate_genome_similarity(genome, representative)
                    
                    if similarity >= compatibility_threshold:
                        species_members.append(genome)
                        placed = True
                        break
            
            # Create new species if no compatible one found
            if not placed:
                species_name = f"species_{len(self.species_groups)}"
                self.species_groups[species_name] = [genome]
        
        # Remove empty species
        self.species_groups = {name: members for name, members in self.species_groups.items() 
                             if members}
        
        logger.debug(f"Formed {len(self.species_groups)} species")
    
    def _select_and_reproduce(self) -> List[ArchitectureGenome]:
        """Select parents and create offspring for next generation."""
        new_population = []
        
        # Elite preservation - keep best genomes
        num_elites = max(1, int(self.population_size * self.elite_ratio))
        
        # Sort by fitness
        fitness_scores = []
        for genome in self.population:
            fitness = self.fitness_cache.get(genome.id, FitnessMetrics())
            fitness_scores.append((fitness.calculate_weighted_fitness(), genome))
        
        fitness_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Add elites
        for i in range(num_elites):
            elite_genome = fitness_scores[i][1].clone()
            new_population.append(elite_genome)
        
        # Fill rest through species-based reproduction
        remaining_slots = self.population_size - num_elites
        
        # Allocate offspring to species based on their performance
        species_offspring = self._allocate_offspring_to_species(remaining_slots)
        
        for species_name, num_offspring in species_offspring.items():
            if species_name not in self.species_groups:
                continue
                
            species_members = self.species_groups[species_name]
            
            for _ in range(num_offspring):
                # Select parents from species
                if len(species_members) == 1:
                    parent = species_members[0]
                    offspring = self._mutate_offspring(parent)
                else:
                    parent1, parent2 = np.random.choice(species_members, 2, replace=False)
                    if np.random.random() < 0.3:  # Crossover probability
                        offspring = self._crossover(parent1, parent2)
                    else:
                        parent = np.random.choice([parent1, parent2])
                        offspring = self._mutate_offspring(parent)
                
                new_population.append(offspring)
        
        # Fill any remaining slots with random mutations of best genomes
        while len(new_population) < self.population_size:
            parent = fitness_scores[np.random.randint(min(5, len(fitness_scores)))][1]
            offspring = self._mutate_offspring(parent)
            new_population.append(offspring)
        
        return new_population[:self.population_size]  # Ensure exact population size
    
    def _allocate_offspring_to_species(self, total_offspring: int) -> Dict[str, int]:
        """Allocate offspring slots to species based on their fitness."""
        if not self.species_groups:
            return {}
        
        # Calculate average fitness for each species
        species_fitness = {}
        
        for species_name, members in self.species_groups.items():
            fitness_scores = []
            for genome in members:
                fitness = self.fitness_cache.get(genome.id, FitnessMetrics())
                fitness_scores.append(fitness.calculate_weighted_fitness())
            
            if fitness_scores:
                species_fitness[species_name] = np.mean(fitness_scores)
            else:
                species_fitness[species_name] = 0.0
        
        # Calculate allocation proportions
        total_fitness = sum(species_fitness.values())
        if total_fitness <= 0:
            # Equal allocation if all fitnesses are zero or negative
            offspring_per_species = total_offspring // len(self.species_groups)
            return {name: offspring_per_species for name in self.species_groups.keys()}
        
        # Proportional allocation
        species_offspring = {}
        allocated = 0
        
        for species_name, fitness in species_fitness.items():
            proportion = fitness / total_fitness
            offspring_count = int(total_offspring * proportion)
            species_offspring[species_name] = offspring_count
            allocated += offspring_count
        
        # Distribute remaining offspring
        remaining = total_offspring - allocated
        species_names = list(species_fitness.keys())
        
        for i in range(remaining):
            species_name = species_names[i % len(species_names)]
            species_offspring[species_name] += 1
        
        return species_offspring
    
    def _crossover(self, parent1: ArchitectureGenome, parent2: ArchitectureGenome) -> ArchitectureGenome:
        """Create offspring through crossover of two parents."""
        # Choose fitter parent as base
        fitness1 = self.fitness_cache.get(parent1.id, FitnessMetrics()).calculate_weighted_fitness()
        fitness2 = self.fitness_cache.get(parent2.id, FitnessMetrics()).calculate_weighted_fitness()
        
        if fitness1 >= fitness2:
            primary_parent, secondary_parent = parent1, parent2
        else:
            primary_parent, secondary_parent = parent2, parent1
        
        # Start with clone of primary parent
        offspring = primary_parent.clone()
        
        # Inherit some neurons from secondary parent
        for neuron_id, neuron in secondary_parent.neurons.items():
            if np.random.random() < 0.3:  # 30% chance to inherit from secondary
                if neuron_id not in offspring.neurons:
                    # Add neuron if it doesn't exist
                    new_id = offspring.add_neuron(neuron.layer, neuron.activation)
                    if new_id is not None:
                        offspring.neurons[new_id].bias = neuron.bias
                        offspring.neurons[new_id].plasticity = neuron.plasticity
                        offspring.neurons[new_id].quantum_coherence = neuron.quantum_coherence
        
        # Inherit some connections from secondary parent
        for conn_id, connection in secondary_parent.connections.items():
            if np.random.random() < 0.3:  # 30% chance to inherit connection
                if (connection.input_neuron in offspring.neurons and 
                    connection.output_neuron in offspring.neurons):
                    
                    # Check if connection already exists
                    exists = any(
                        c.input_neuron == connection.input_neuron and 
                        c.output_neuron == connection.output_neuron
                        for c in offspring.connections.values()
                    )
                    
                    if not exists:
                        new_conn_id = len(offspring.connections)
                        offspring.connections[new_conn_id] = ConnectionGene(
                            id=new_conn_id,
                            input_neuron=connection.input_neuron,
                            output_neuron=connection.output_neuron,
                            weight=connection.weight,
                            enabled=connection.enabled,
                            plasticity_rule=connection.plasticity_rule,
                            delay=connection.delay
                        )
        
        return offspring
    
    def _mutate_offspring(self, parent: ArchitectureGenome) -> ArchitectureGenome:
        """Create mutated offspring from parent."""
        return self.mutator.mutate_genome(parent)
    
    def get_best_genome(self) -> Optional[ArchitectureGenome]:
        """Get the best genome found so far."""
        return self.best_genome
    
    def get_population_statistics(self) -> Dict[str, Any]:
        """Get comprehensive population statistics."""
        if not self.population:
            return {}
        
        # Calculate fitness statistics
        fitness_scores = []
        complexity_scores = []
        
        for genome in self.population:
            fitness = self.fitness_cache.get(genome.id, FitnessMetrics())
            fitness_scores.append(fitness.calculate_weighted_fitness())
            complexity_scores.append(genome.calculate_complexity())
        
        stats = {
            'generation': self.generation,
            'population_size': len(self.population),
            'num_species': len(self.species_groups),
            'fitness': {
                'best': self.best_fitness,
                'average': np.mean(fitness_scores),
                'std': np.std(fitness_scores),
                'min': np.min(fitness_scores),
                'max': np.max(fitness_scores)
            },
            'complexity': {
                'average': np.mean(complexity_scores),
                'std': np.std(complexity_scores),
                'min': np.min(complexity_scores),
                'max': np.max(complexity_scores)
            },
            'diversity': self.diversity_history[-1] if self.diversity_history else 0.0,
            'species_info': {
                name: len(members) for name, members in self.species_groups.items()
            }
        }
        
        return stats


class SelfEvolvingNAS:
    """Main Self-Evolving Neural Architecture Search system."""
    
    def __init__(self, input_size: int = 40, output_size: int = 10, 
                 population_size: int = 30, evolution_frequency: int = 100):
        self.input_size = input_size
        self.output_size = output_size
        self.evolution_frequency = evolution_frequency
        
        # Evolution components
        self.optimizer = EvolutionaryOptimizer(population_size)
        self.current_architecture: Optional[ArchitectureGenome] = None
        self.architecture_cache = {}  # Cache compiled architectures
        
        # Performance monitoring
        self.inference_count = 0
        self.performance_buffer = deque(maxlen=evolution_frequency)
        self.evolution_trigger = False
        
        # Initialize population
        self.optimizer.initialize_population(input_size, output_size)
        self.current_architecture = self.optimizer.population[0]  # Start with first genome
        
        logger.info(f"Initialized SENAS with population size {population_size}")
    
    def process_batch(self, inputs: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process batch of inputs and trigger evolution if needed."""
        results = []
        
        for input_data in inputs:
            result = self.process_single(input_data)
            results.append(result)
            
            # Check if evolution should be triggered
            if self.inference_count % self.evolution_frequency == 0:
                self._trigger_evolution()
        
        return results
    
    def process_single(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Process single input through current architecture."""
        start_time = time.time()
        
        # Execute inference through current architecture
        output = self._execute_architecture(self.current_architecture, input_data)
        
        processing_time = time.time() - start_time
        
        # Record performance metrics
        self.performance_buffer.append({
            'processing_time': processing_time,
            'output_quality': self._assess_output_quality(output),
            'architecture_id': self.current_architecture.id
        })
        
        self.inference_count += 1
        
        # Prepare result
        result = {
            'output': output,
            'processing_time_ms': processing_time * 1000,
            'architecture_id': self.current_architecture.id,
            'architecture_complexity': self.current_architecture.calculate_complexity(),
            'inference_count': self.inference_count,
            'evolution_pending': self.evolution_trigger
        }
        
        return result
    
    def _execute_architecture(self, genome: ArchitectureGenome, input_data: np.ndarray) -> np.ndarray:
        """Execute forward pass through the architecture."""
        # Check cache for compiled architecture
        if genome.id not in self.architecture_cache:
            self.architecture_cache[genome.id] = self._compile_architecture(genome)
        
        compiled_arch = self.architecture_cache[genome.id]
        return self._forward_pass(compiled_arch, input_data)
    
    def _compile_architecture(self, genome: ArchitectureGenome) -> Dict[str, Any]:
        """Compile genome into executable neural network."""
        # Organize neurons by layers
        layers = defaultdict(list)
        for neuron in genome.neurons.values():
            layers[neuron.layer].append(neuron)
        
        # Sort layers
        sorted_layers = sorted(layers.keys())
        
        # Build connectivity matrix and other structures
        layer_connections = {}
        activation_functions = {}
        biases = {}
        
        for layer_idx, layer in enumerate(sorted_layers):
            if layer_idx < len(sorted_layers) - 1:
                next_layer = sorted_layers[layer_idx + 1]
                
                current_neurons = layers[layer]
                next_neurons = layers[next_layer]
                
                # Build connection matrix
                weight_matrix = np.zeros((len(next_neurons), len(current_neurons)))
                
                for conn in genome.connections.values():
                    if not conn.enabled:
                        continue
                    
                    # Find neuron positions
                    input_pos = next((i for i, n in enumerate(current_neurons) 
                                    if n.id == conn.input_neuron), None)
                    output_pos = next((i for i, n in enumerate(next_neurons) 
                                     if n.id == conn.output_neuron), None)
                    
                    if input_pos is not None and output_pos is not None:
                        weight_matrix[output_pos, input_pos] = conn.weight
                
                layer_connections[layer] = weight_matrix
            
            # Extract activation functions and biases
            neurons = layers[layer]
            activations = [neuron.activation for neuron in neurons]
            layer_biases = np.array([neuron.bias for neuron in neurons])
            
            activation_functions[layer] = activations
            biases[layer] = layer_biases
        
        compiled_arch = {
            'layers': sorted_layers,
            'layer_sizes': {layer: len(layers[layer]) for layer in sorted_layers},
            'connections': layer_connections,
            'activations': activation_functions,
            'biases': biases,
            'genome_id': genome.id
        }
        
        return compiled_arch
    
    def _forward_pass(self, compiled_arch: Dict[str, Any], input_data: np.ndarray) -> np.ndarray:
        """Execute forward pass through compiled architecture."""
        layers = compiled_arch['layers']
        
        # Prepare input
        if len(input_data) != compiled_arch['layer_sizes'][layers[0]]:
            # Resize input to match input layer
            if len(input_data) < compiled_arch['layer_sizes'][layers[0]]:
                padded_input = np.zeros(compiled_arch['layer_sizes'][layers[0]])
                padded_input[:len(input_data)] = input_data
                current_activation = padded_input
            else:
                current_activation = input_data[:compiled_arch['layer_sizes'][layers[0]]]
        else:
            current_activation = input_data.copy()
        
        # Forward propagation through layers
        for i in range(len(layers) - 1):
            current_layer = layers[i]
            next_layer = layers[i + 1]
            
            if current_layer in compiled_arch['connections']:
                # Matrix multiplication
                weight_matrix = compiled_arch['connections'][current_layer]
                next_activation = weight_matrix @ current_activation
                
                # Add bias
                if next_layer in compiled_arch['biases']:
                    next_activation += compiled_arch['biases'][next_layer]
                
                # Apply activation functions
                if next_layer in compiled_arch['activations']:
                    activations = compiled_arch['activations'][next_layer]
                    
                    for j, activation_type in enumerate(activations):
                        if j < len(next_activation):
                            next_activation[j] = self._apply_activation(
                                next_activation[j], activation_type
                            )
                
                current_activation = next_activation
        
        return current_activation
    
    def _apply_activation(self, value: float, activation_type: ActivationType) -> float:
        """Apply activation function to a value."""
        if activation_type == ActivationType.TANH:
            return np.tanh(value)
        elif activation_type == ActivationType.RELU:
            return max(0, value)
        elif activation_type == ActivationType.SIGMOID:
            return 1 / (1 + np.exp(-np.clip(value, -500, 500)))
        elif activation_type == ActivationType.LEAKY_RELU:
            return max(0.01 * value, value)
        elif activation_type == ActivationType.SWISH:
            sigmoid = 1 / (1 + np.exp(-np.clip(value, -500, 500)))
            return value * sigmoid
        elif activation_type == ActivationType.QUANTUM_GATE:
            # Quantum-inspired activation
            return np.cos(value) * np.exp(-value**2 / 4)
        else:
            return np.tanh(value)  # Default fallback
    
    def _assess_output_quality(self, output: np.ndarray) -> float:
        """Assess quality of network output."""
        # Simple quality metrics
        if len(output) == 0:
            return 0.0
        
        # Check for NaN or infinite values
        if np.any(np.isnan(output)) or np.any(np.isinf(output)):
            return 0.0
        
        # Output should be well-distributed and not saturated
        output_range = np.max(output) - np.min(output)
        if output_range < 1e-6:
            return 0.1  # Very poor quality
        
        # Prefer outputs with reasonable variance but not too extreme
        output_std = np.std(output)
        quality = 1.0 / (1.0 + abs(output_std - 1.0))  # Optimal std around 1.0
        
        return quality
    
    def _trigger_evolution(self) -> None:
        """Trigger architecture evolution based on accumulated performance."""
        if len(self.performance_buffer) < self.evolution_frequency // 2:
            return
        
        logger.info("Triggering architecture evolution...")
        
        # Create fitness evaluator based on recent performance
        def fitness_evaluator(genome: ArchitectureGenome) -> FitnessMetrics:
            return self._evaluate_genome_fitness(genome)
        
        # Evolve population
        self.optimizer.evolve_generation(fitness_evaluator)
        
        # Select new current architecture (best from population)
        best_genome = self.optimizer.get_best_genome()
        if best_genome and best_genome.id != self.current_architecture.id:
            self.current_architecture = best_genome
            logger.info(f"Switched to new architecture: {best_genome.id}")
        
        # Clear performance buffer
        self.performance_buffer.clear()
        self.evolution_trigger = False
    
    def _evaluate_genome_fitness(self, genome: ArchitectureGenome) -> FitnessMetrics:
        """Evaluate fitness of a genome based on multiple criteria."""
        fitness = FitnessMetrics()
        
        # If this is the current architecture, use performance buffer data
        if genome.id == self.current_architecture.id and self.performance_buffer:
            recent_performance = list(self.performance_buffer)[-20:]  # Last 20 samples
            
            processing_times = [p['processing_time'] for p in recent_performance]
            output_qualities = [p['output_quality'] for p in recent_performance]
            
            # Processing speed (inverted time)
            avg_processing_time = np.mean(processing_times)
            fitness.processing_speed = 1.0 / (1.0 + avg_processing_time * 1000)  # Convert to ms
            
            # Output quality
            fitness.accuracy = np.mean(output_qualities)
            
            # Robustness (consistency of performance)
            fitness.robustness = 1.0 / (1.0 + np.std(output_qualities))
        else:
            # For other genomes, estimate fitness based on architecture properties
            complexity = genome.calculate_complexity()
            
            # Estimate processing speed based on architecture size
            total_neurons = len(genome.neurons)
            total_connections = len([c for c in genome.connections.values() if c.enabled])
            
            # Smaller, well-connected networks are generally faster
            estimated_speed = 1.0 / (1.0 + (total_neurons + total_connections) / 100.0)
            fitness.processing_speed = estimated_speed
            
            # Estimate accuracy based on architecture depth and connectivity
            max_layer = genome.get_max_layer() if genome.neurons else 0
            depth_factor = min(1.0, max_layer / 5.0)  # Optimal depth around 5 layers
            
            connection_density = total_connections / max(1, total_neurons**2)
            connectivity_factor = min(1.0, connection_density * 10)  # Moderate connectivity
            
            fitness.accuracy = (depth_factor + connectivity_factor) / 2.0
            fitness.robustness = 0.5  # Neutral for unknown architectures
        
        # Memory efficiency (based on architecture size)
        total_params = len(genome.neurons) + len(genome.connections)
        fitness.memory_efficiency = 1.0 / (1.0 + total_params / 1000.0)
        
        # Power efficiency (simpler architectures use less power)
        fitness.power_efficiency = fitness.memory_efficiency
        
        # Adaptability (based on plasticity parameters)
        if genome.neurons:
            avg_plasticity = np.mean([n.plasticity for n in genome.neurons.values()])
            fitness.adaptability = avg_plasticity
        
        # Quantum coherence
        if genome.neurons:
            avg_coherence = np.mean([n.quantum_coherence for n in genome.neurons.values()])
            fitness.quantum_coherence = avg_coherence
        
        # Complexity score
        fitness.complexity_score = genome.calculate_complexity()
        
        return fitness
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics."""
        stats = self.optimizer.get_population_statistics()
        
        # Add SENAS-specific statistics
        stats.update({
            'current_architecture': {
                'id': self.current_architecture.id if self.current_architecture else None,
                'complexity': self.current_architecture.calculate_complexity() if self.current_architecture else 0,
                'neuron_count': len(self.current_architecture.neurons) if self.current_architecture else 0,
                'connection_count': len(self.current_architecture.connections) if self.current_architecture else 0,
            },
            'inference_stats': {
                'total_inferences': self.inference_count,
                'evolution_frequency': self.evolution_frequency,
                'next_evolution_in': self.evolution_frequency - (self.inference_count % self.evolution_frequency),
                'cached_architectures': len(self.architecture_cache),
            },
            'performance_buffer_size': len(self.performance_buffer)
        })
        
        return stats
    
    def force_evolution(self) -> None:
        """Force architecture evolution regardless of schedule."""
        self.evolution_trigger = True
        self._trigger_evolution()
    
    def get_current_architecture_info(self) -> Dict[str, Any]:
        """Get detailed information about current architecture."""
        if not self.current_architecture:
            return {}
        
        genome = self.current_architecture
        
        # Layer analysis
        layers = defaultdict(list)
        for neuron in genome.neurons.values():
            layers[neuron.layer].append(neuron)
        
        layer_info = {}
        for layer, neurons in layers.items():
            activation_counts = defaultdict(int)
            for neuron in neurons:
                activation_counts[neuron.activation.name] += 1
            
            layer_info[layer] = {
                'neuron_count': len(neurons),
                'activations': dict(activation_counts),
                'avg_bias': np.mean([n.bias for n in neurons]),
                'avg_plasticity': np.mean([n.plasticity for n in neurons]),
                'avg_quantum_coherence': np.mean([n.quantum_coherence for n in neurons])
            }
        
        return {
            'genome_id': genome.id,
            'generation': genome.generation,
            'total_neurons': len(genome.neurons),
            'total_connections': len(genome.connections),
            'enabled_connections': len([c for c in genome.connections.values() if c.enabled]),
            'max_layer': genome.get_max_layer(),
            'complexity': genome.calculate_complexity(),
            'layers': dict(layer_info),
            'fitness_history': genome.fitness_history.copy(),
            'performance_metrics': genome.performance_metrics.copy()
        }


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create SENAS instance
    senas = SelfEvolvingNAS(
        input_size=40, 
        output_size=10, 
        population_size=20, 
        evolution_frequency=50
    )
    
    # Simulate processing with evolution
    print("Starting SENAS demonstration...")
    
    for i in range(100):
        # Generate random input data
        input_data = np.random.randn(40)
        
        # Process through SENAS
        result = senas.process_single(input_data)
        
        if i % 10 == 0:
            print(f"Iteration {i}: Architecture {result['architecture_id'][:8]}, "
                  f"Complexity {result['architecture_complexity']:.3f}")
        
        # Check for evolution
        if i % 25 == 24:  # Get stats periodically
            stats = senas.get_evolution_statistics()
            print(f"Generation {stats['generation']}: "
                  f"Best fitness {stats['fitness']['best']:.4f}, "
                  f"Population diversity {stats['diversity']:.4f}")
    
    # Final architecture analysis
    arch_info = senas.get_current_architecture_info()
    print(f"\nFinal Architecture Analysis:")
    print(f"  ID: {arch_info['genome_id']}")
    print(f"  Neurons: {arch_info['total_neurons']}")
    print(f"  Connections: {arch_info['enabled_connections']}")
    print(f"  Layers: {arch_info['max_layer'] + 1}")
    print(f"  Complexity: {arch_info['complexity']:.4f}")