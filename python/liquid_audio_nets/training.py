"""Training framework for Liquid Neural Networks.

Implements power-aware training with PyTorch Lightning for edge deployment.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from dataclasses import dataclass, asdict
import math

from .lnn import AdaptiveConfig


@dataclass
class TrainingConfig:
    """Configuration for LNN training."""
    
    # Model architecture
    input_dim: int = 40
    hidden_dim: int = 64
    output_dim: int = 10
    
    # Training parameters
    learning_rate: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 100
    
    # Loss function weights
    lambda_accuracy: float = 1.0      # Classification accuracy
    lambda_power: float = 0.1         # Power consumption penalty
    lambda_sparse: float = 0.05       # Sparsity regularization
    lambda_temporal: float = 0.02     # Temporal consistency
    
    # Adaptive timestep learning
    enable_adaptive: bool = True
    timestep_range: Tuple[float, float] = (0.001, 0.05)  # 1ms to 50ms
    
    # Optimization
    optimizer: str = "adam"
    scheduler: str = "cosine"
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    
    # Power constraints for training
    target_power_mw: float = 1.0
    power_penalty_start: float = 0.5  # Start penalty at 0.5mW
    
    # Quantization for edge deployment
    quantization: str = "dynamic_int8"  # "none", "dynamic_int8", "static_int8"
    calibration_samples: int = 1000


class LiquidNeuralNetworkPyTorch(nn.Module):
    """PyTorch implementation of Liquid Neural Network for training."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Network layers
        self.w_input = nn.Linear(input_dim, hidden_dim, bias=True)
        self.w_recurrent = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_output = nn.Linear(hidden_dim, output_dim, bias=True)
        
        # Time constants (learnable)
        self.tau = nn.Parameter(torch.ones(hidden_dim) * 0.1)
        
        # Initialize weights
        self._init_weights()
        
        # State tracking
        self.hidden_state = None
        self.reset_state()
        
    def _init_weights(self):
        """Initialize network weights for liquid dynamics."""
        # Input weights: Xavier initialization
        nn.init.xavier_uniform_(self.w_input.weight)
        nn.init.zeros_(self.w_input.bias)
        
        # Recurrent weights: scaled random for stability  
        nn.init.uniform_(self.w_recurrent.weight, -0.1, 0.1)
        
        # Output weights: Xavier initialization
        nn.init.xavier_uniform_(self.w_output.weight)
        nn.init.zeros_(self.w_output.bias)
        
        # Time constants: reasonable defaults
        nn.init.uniform_(self.tau, 0.05, 0.2)
        
    def reset_state(self, batch_size: int = 1):
        """Reset liquid state."""
        device = next(self.parameters()).device
        self.hidden_state = torch.zeros(batch_size, self.hidden_dim, device=device)
        
    def forward(self, x: torch.Tensor, timestep: float = 0.01) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with liquid dynamics integration."""
        batch_size = x.size(0)
        
        if self.hidden_state is None or self.hidden_state.size(0) != batch_size:
            self.reset_state(batch_size)
            
        # Compute currents
        input_current = self.w_input(x)
        recurrent_current = self.w_recurrent(self.hidden_state)
        decay_current = -self.hidden_state / self.tau.unsqueeze(0)
        
        # Total derivative
        dhdt = input_current + recurrent_current + decay_current
        
        # Euler integration (can be upgraded to higher-order methods)
        self.hidden_state = self.hidden_state + timestep * dhdt
        
        # Apply activation (tanh for liquid dynamics)
        self.hidden_state = torch.tanh(self.hidden_state)
        
        # Output computation
        output = self.w_output(self.hidden_state)
        
        # Compute metrics for loss functions
        metrics = {
            'liquid_energy': torch.sum(self.hidden_state ** 2, dim=1),
            'sparsity': torch.sum(torch.abs(self.hidden_state) < 0.1, dim=1).float(),
            'tau_diversity': torch.std(self.tau),
            'hidden_state': self.hidden_state.clone(),
        }
        
        return output, metrics


class TimestepController(nn.Module):
    """Learnable timestep controller for adaptive computation."""
    
    def __init__(self, input_features: int = 1):
        super().__init__()
        
        self.controller = nn.Sequential(
            nn.Linear(input_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8), 
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, complexity: torch.Tensor, 
                min_timestep: float = 0.001, 
                max_timestep: float = 0.05) -> torch.Tensor:
        """Predict optimal timestep based on signal complexity."""
        if complexity.dim() == 0:
            complexity = complexity.unsqueeze(0)
        if complexity.dim() == 1:
            complexity = complexity.unsqueeze(1)
            
        # Controller output [0, 1] 
        controller_out = self.controller(complexity)
        
        # Map to timestep range
        timestep = min_timestep + (max_timestep - min_timestep) * controller_out
        
        return timestep.squeeze()


class LNNTrainer(pl.LightningModule):
    """PyTorch Lightning trainer for Liquid Neural Networks."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(asdict(config) if hasattr(config, '__dict__') else config.__dict__)
        
        # Build model
        self.lnn = LiquidNeuralNetworkPyTorch(
            config.input_dim, 
            config.hidden_dim, 
            config.output_dim
        )
        
        # Timestep controller (if adaptive)
        if config.enable_adaptive:
            self.timestep_controller = TimestepController()
        else:
            self.timestep_controller = None
            
        # Loss functions
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.training_metrics = []
        self.validation_metrics = []
        
    def forward(self, x: torch.Tensor, complexity: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through LNN."""
        batch_size = x.size(0)
        
        # Determine timestep
        if self.timestep_controller is not None and complexity is not None:
            timestep = self.timestep_controller(complexity, 
                                              self.config.timestep_range[0],
                                              self.config.timestep_range[1])
            if timestep.dim() == 0:
                timestep = timestep.item()
            else:
                timestep = timestep.mean().item()  # Average for batch
        else:
            timestep = (self.config.timestep_range[0] + self.config.timestep_range[1]) / 2
            
        # Process through LNN
        output, metrics = self.lnn(x, timestep)
        
        # Add timestep to metrics
        metrics['timestep'] = torch.tensor(timestep, device=x.device)
        
        return {'logits': output, 'metrics': metrics}
        
    def compute_power_estimate(self, metrics: Dict[str, torch.Tensor], timestep: float) -> torch.Tensor:
        """Estimate power consumption from network metrics."""
        batch_size = metrics['liquid_energy'].size(0)
        
        # Base power consumption
        base_power = 0.08  # mW
        
        # Signal-dependent power
        energy = metrics['liquid_energy'].mean()
        signal_power = energy * 1.2
        
        # Computation power (inversely related to timestep)
        computation_power = (1.0 / timestep) * 0.1
        
        # Network complexity power
        network_power = (self.config.hidden_dim / 64.0) * 0.4
        
        total_power = base_power + signal_power + computation_power + network_power
        
        return torch.tensor(total_power, device=metrics['liquid_energy'].device)
        
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        x, y = batch
        
        # Estimate input complexity for adaptive timestep
        complexity = None
        if self.config.enable_adaptive:
            # Simple complexity measure: signal energy + spectral change
            energy = torch.mean(x ** 2, dim=-1, keepdim=True)
            if x.size(-1) > 1:
                spectral_change = torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]), dim=-1, keepdim=True)
                complexity = (torch.sqrt(energy) + spectral_change) * 0.5
            else:
                complexity = torch.sqrt(energy)
                
        # Forward pass
        outputs = self.forward(x, complexity)
        logits = outputs['logits']
        metrics = outputs['metrics']
        
        # Classification loss
        loss_classification = self.classification_loss(logits, y)
        
        # Power loss
        timestep = metrics['timestep'].item()
        power_estimate = self.compute_power_estimate(metrics, timestep)
        power_penalty = torch.relu(power_estimate - self.config.power_penalty_start)
        loss_power = self.config.lambda_power * power_penalty
        
        # Sparsity loss (encourage sparse activations)
        sparsity = metrics['sparsity'].mean() / self.config.hidden_dim
        loss_sparse = self.config.lambda_sparse * (1.0 - sparsity)
        
        # Temporal consistency loss (minimize rapid state changes)
        loss_temporal = self.config.lambda_temporal * torch.var(metrics['hidden_state'], dim=0).mean()
        
        # Total loss
        total_loss = (self.config.lambda_accuracy * loss_classification + 
                     loss_power + loss_sparse + loss_temporal)
        
        # Logging
        self.log_dict({
            'train/loss_total': total_loss,
            'train/loss_classification': loss_classification,
            'train/loss_power': loss_power,
            'train/loss_sparse': loss_sparse,
            'train/loss_temporal': loss_temporal,
            'train/power_mw': power_estimate,
            'train/timestep_ms': timestep * 1000,
            'train/sparsity': sparsity,
            'train/accuracy': (torch.argmax(logits, dim=1) == y).float().mean(),
        }, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss
        
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Validation step."""
        x, y = batch
        
        # Estimate complexity
        complexity = None
        if self.config.enable_adaptive:
            energy = torch.mean(x ** 2, dim=-1, keepdim=True)
            if x.size(-1) > 1:
                spectral_change = torch.mean(torch.abs(x[:, :, 1:] - x[:, :, :-1]), dim=-1, keepdim=True)
                complexity = (torch.sqrt(energy) + spectral_change) * 0.5
            else:
                complexity = torch.sqrt(energy)
                
        # Forward pass
        outputs = self.forward(x, complexity)
        logits = outputs['logits']
        metrics = outputs['metrics']
        
        # Compute losses
        loss_classification = self.classification_loss(logits, y)
        timestep = metrics['timestep'].item()
        power_estimate = self.compute_power_estimate(metrics, timestep)
        
        # Compute accuracy
        accuracy = (torch.argmax(logits, dim=1) == y).float().mean()
        
        # Log validation metrics
        self.log_dict({
            'val/loss': loss_classification,
            'val/accuracy': accuracy,
            'val/power_mw': power_estimate,
            'val/timestep_ms': timestep * 1000,
        }, on_epoch=True, prog_bar=True)
        
        return {'val_loss': loss_classification, 'val_accuracy': accuracy}
        
    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        
        # Optimizer
        if self.config.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                self.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                self.parameters(),
                lr=self.config.learning_rate, 
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
            
        # Scheduler
        if self.config.scheduler.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.config.max_epochs
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        elif self.config.scheduler.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config.max_epochs // 3,
                gamma=0.1
            )
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            return optimizer
            
    def export_embedded(self, 
                       output_path: str,
                       quantization: Optional[str] = None,
                       target: str = 'cortex-m4') -> Dict[str, Any]:
        """Export trained model for embedded deployment."""
        
        if quantization is None:
            quantization = self.config.quantization
            
        self.eval()
        
        # Prepare dummy input for tracing
        dummy_input = torch.randn(1, self.config.input_dim)
        
        # Trace the model
        traced_model = torch.jit.trace(self.lnn, dummy_input)
        
        # Apply quantization
        if quantization == "dynamic_int8":
            quantized_model = torch.quantization.quantize_dynamic(
                traced_model,
                {nn.Linear},
                dtype=torch.qint8
            )
        elif quantization == "static_int8":
            # Static quantization requires calibration data
            # This is a simplified implementation
            quantized_model = traced_model  # Placeholder
        else:
            quantized_model = traced_model
            
        # Extract weights for custom format
        state_dict = self.lnn.state_dict()
        
        # Create binary model file (.lnn format)
        model_data = self._create_lnn_binary(state_dict, target)
        
        # Save binary model
        with open(output_path, 'wb') as f:
            f.write(model_data)
            
        # Return export info
        export_info = {
            'model_path': output_path,
            'quantization': quantization,
            'target': target,
            'input_dim': self.config.input_dim,
            'hidden_dim': self.config.hidden_dim,
            'output_dim': self.config.output_dim,
            'estimated_memory_kb': self._estimate_memory_usage(),
            'estimated_power_mw': self.config.target_power_mw,
        }
        
        return export_info
        
    def _create_lnn_binary(self, state_dict: Dict[str, torch.Tensor], target: str) -> bytes:
        """Create binary .lnn format for embedded deployment."""
        import struct
        
        # Header: magic (4) + version (4) + input_dim (4) + hidden_dim (4) + output_dim (4) + reserved (12)
        header = struct.pack('<4sIIII', b'LNN\x01', 1, 
                           self.config.input_dim, 
                           self.config.hidden_dim,
                           self.config.output_dim)
        header += b'\x00' * 12  # Reserved bytes
        
        # Extract and serialize weights
        weights_data = b''
        
        # Input weights [hidden_dim x input_dim]
        w_input = state_dict['w_input.weight'].detach().numpy().astype(np.float32)
        weights_data += w_input.tobytes()
        
        # Recurrent weights [hidden_dim x hidden_dim]  
        w_recurrent = state_dict['w_recurrent.weight'].detach().numpy().astype(np.float32)
        weights_data += w_recurrent.tobytes()
        
        # Output weights [output_dim x hidden_dim]
        w_output = state_dict['w_output.weight'].detach().numpy().astype(np.float32)
        weights_data += w_output.tobytes()
        
        # Biases
        b_input = state_dict['w_input.bias'].detach().numpy().astype(np.float32)
        weights_data += b_input.tobytes()
        
        b_output = state_dict['w_output.bias'].detach().numpy().astype(np.float32)
        weights_data += b_output.tobytes()
        
        # Time constants
        tau = state_dict['tau'].detach().numpy().astype(np.float32)
        weights_data += tau.tobytes()
        
        return header + weights_data
        
    def _estimate_memory_usage(self) -> int:
        """Estimate memory usage in KB."""
        # Simplified calculation
        weights_size = (self.config.input_dim * self.config.hidden_dim + 
                       self.config.hidden_dim * self.config.hidden_dim +
                       self.config.hidden_dim * self.config.output_dim) * 4
        
        bias_size = (self.config.hidden_dim + self.config.output_dim) * 4
        state_size = self.config.hidden_dim * 4
        buffer_size = self.config.input_dim * 4
        
        total_bytes = weights_size + bias_size + state_size + buffer_size + 1024
        return total_bytes // 1024


def create_synthetic_dataset(num_samples: int = 1000, 
                           input_dim: int = 40,
                           output_dim: int = 10,
                           noise_level: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create synthetic dataset for testing."""
    
    # Generate synthetic audio features (MFCC-like)
    X = torch.randn(num_samples, input_dim)
    
    # Add some structure (temporal patterns)
    for i in range(input_dim // 4):
        freq = 0.1 + 0.2 * i
        phase = torch.rand(num_samples) * 2 * math.pi
        X[:, i*4:(i+1)*4] += 0.3 * torch.sin(
            freq * torch.arange(4).float().unsqueeze(0) + phase.unsqueeze(1)
        )
    
    # Generate labels based on features
    feature_sums = torch.sum(X[:, :output_dim], dim=1)
    y = torch.argmax(torch.stack([
        feature_sums + torch.randn(num_samples) * noise_level
        for _ in range(output_dim)
    ], dim=1), dim=1)
    
    return X, y


# Utility functions for training
def train_model(config: TrainingConfig, 
                train_loader: DataLoader,
                val_loader: Optional[DataLoader] = None,
                callbacks: Optional[List] = None) -> LNNTrainer:
    """Train LNN model with given configuration."""
    
    # Create trainer
    model = LNNTrainer(config)
    
    # Create PyTorch Lightning trainer
    trainer_kwargs = {
        'max_epochs': config.max_epochs,
        'gradient_clip_val': config.gradient_clip,
        'enable_progress_bar': True,
        'enable_model_summary': True,
    }
    
    if callbacks:
        trainer_kwargs['callbacks'] = callbacks
        
    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    return model


if __name__ == "__main__":
    # Demo training script
    config = TrainingConfig(
        input_dim=40,
        hidden_dim=64,
        output_dim=10,
        max_epochs=50,
        batch_size=32,
    )
    
    # Create synthetic dataset
    X_train, y_train = create_synthetic_dataset(2000, config.input_dim, config.output_dim)
    X_val, y_val = create_synthetic_dataset(500, config.input_dim, config.output_dim)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Train model
    model = train_model(config, train_loader, val_loader)
    
    # Export for embedded deployment
    export_info = model.export_embedded("demo_model.lnn")
    print(f"Model exported: {export_info}")