//! ODE solvers for liquid neural dynamics integration

use crate::{Result, LiquidAudioError};
use crate::core::LiquidState;
use nalgebra::DVector;
use core::fmt::Debug;

/// Trait for ODE solvers used in liquid neural integration
pub trait ODESolver: Debug + Send + Sync {
    /// Perform one integration step
    /// 
    /// # Arguments
    /// * `state` - Current liquid state
    /// * `derivatives` - State derivatives (dx/dt)
    /// * `timestep` - Integration timestep (dt)
    /// 
    /// # Returns
    /// New liquid state after integration step
    fn step(&self, state: &LiquidState, derivatives: &DVector<f32>, timestep: f32) -> Result<LiquidState>;
    
    /// Get solver name for identification
    fn name(&self) -> &'static str;
    
    /// Get solver order (accuracy)
    fn order(&self) -> u8;
}

/// Forward Euler solver (1st order)
/// Simple but fast integration method suitable for embedded systems
#[derive(Debug, Clone)]
pub struct EulerSolver {
    /// Optional step size limiter for stability
    max_step_size: Option<f32>,
}

impl EulerSolver {
    /// Create new Euler solver
    pub fn new() -> Self {
        Self {
            max_step_size: None,
        }
    }
    
    /// Create new Euler solver with maximum step size limit
    pub fn with_max_step(max_step: f32) -> Self {
        Self {
            max_step_size: Some(max_step),
        }
    }
}

impl ODESolver for EulerSolver {
    fn step(&self, state: &LiquidState, derivatives: &DVector<f32>, timestep: f32) -> Result<LiquidState> {
        if derivatives.len() != state.dim() {
            return Err(LiquidAudioError::ComputationError(
                format!("Dimension mismatch: state {} vs derivatives {}", 
                       state.dim(), derivatives.len())
            ));
        }
        
        // Limit timestep if configured
        let dt = if let Some(max_dt) = self.max_step_size {
            timestep.min(max_dt)
        } else {
            timestep
        };
        
        // Forward Euler: x(t+dt) = x(t) + dt * dx/dt
        let current_state = state.hidden_state();
        let new_state = current_state + dt * derivatives;
        
        // Apply activation function (tanh for liquid networks)
        let activated_state = new_state.map(|x| x.tanh());
        
        Ok(LiquidState::from_vector(activated_state))
    }
    
    fn name(&self) -> &'static str {
        "Forward Euler"
    }
    
    fn order(&self) -> u8 {
        1
    }
}

/// Heun's method solver (2nd order)
/// More accurate than Euler, still computationally efficient
#[derive(Debug, Clone)]
pub struct HeunSolver {
    /// Maximum step size for stability
    max_step_size: Option<f32>,
    /// Temporary storage for intermediate calculations
    temp_buffer: Option<DVector<f32>>,
}

impl HeunSolver {
    /// Create new Heun solver
    pub fn new() -> Self {
        Self {
            max_step_size: None,
            temp_buffer: None,
        }
    }
    
    /// Create new Heun solver with step size limit
    pub fn with_max_step(max_step: f32) -> Self {
        Self {
            max_step_size: Some(max_step),
            temp_buffer: None,
        }
    }
    
    /// Initialize temporary buffer for given dimension
    fn ensure_buffer(&mut self, dim: usize) {
        if self.temp_buffer.as_ref().map_or(true, |buf| buf.len() != dim) {
            self.temp_buffer = Some(DVector::zeros(dim));
        }
    }
}

impl ODESolver for HeunSolver {
    fn step(&self, state: &LiquidState, derivatives: &DVector<f32>, timestep: f32) -> Result<LiquidState> {
        if derivatives.len() != state.dim() {
            return Err(LiquidAudioError::ComputationError(
                format!("Dimension mismatch: state {} vs derivatives {}", 
                       state.dim(), derivatives.len())
            ));
        }
        
        let dt = if let Some(max_dt) = self.max_step_size {
            timestep.min(max_dt)
        } else {
            timestep
        };
        
        let current_state = state.hidden_state();
        
        // Heun's method (predictor-corrector)
        // Step 1: Euler predictor step
        let predictor = current_state + dt * derivatives;
        let predictor_activated = predictor.map(|x| x.tanh());
        
        // Step 2: Compute derivatives at predictor point
        // Note: In practice, this would need the full liquid dynamics function
        // For now, we'll use a simplified approximation
        let predictor_derivatives = self.approximate_derivatives(&predictor_activated, derivatives);
        
        // Step 3: Corrector step using average of derivatives
        let avg_derivatives = (derivatives + &predictor_derivatives) * 0.5;
        let corrected_state = current_state + dt * avg_derivatives;
        
        // Apply activation
        let final_state = corrected_state.map(|x| x.tanh());
        
        Ok(LiquidState::from_vector(final_state))
    }
    
    fn name(&self) -> &'static str {
        "Heun's Method"
    }
    
    fn order(&self) -> u8 {
        2
    }
}

impl HeunSolver {
    /// Approximate derivatives at predictor point
    /// This is a simplified version - real implementation would evaluate full dynamics
    fn approximate_derivatives(&self, predictor_state: &DVector<f32>, original_derivatives: &DVector<f32>) -> DVector<f32> {
        // Simple approximation: scale derivatives by state magnitude
        let state_magnitude = predictor_state.norm();
        let scale_factor = if state_magnitude > 1e-6 {
            (1.0 + state_magnitude).recip()
        } else {
            1.0
        };
        
        original_derivatives * scale_factor
    }
}

/// Runge-Kutta 4th order solver
/// High accuracy but more computationally expensive
#[derive(Debug, Clone)]
pub struct RK4Solver {
    max_step_size: Option<f32>,
}

impl RK4Solver {
    /// Create new RK4 solver
    pub fn new() -> Self {
        Self {
            max_step_size: None,
        }
    }
    
    /// Create RK4 solver with step size limit
    pub fn with_max_step(max_step: f32) -> Self {
        Self {
            max_step_size: Some(max_step),
        }
    }
}

impl ODESolver for RK4Solver {
    fn step(&self, state: &LiquidState, derivatives: &DVector<f32>, timestep: f32) -> Result<LiquidState> {
        if derivatives.len() != state.dim() {
            return Err(LiquidAudioError::ComputationError(
                format!("Dimension mismatch: state {} vs derivatives {}", 
                       state.dim(), derivatives.len())
            ));
        }
        
        let dt = if let Some(max_dt) = self.max_step_size {
            timestep.min(max_dt)
        } else {
            timestep
        };
        
        let y = state.hidden_state();
        
        // RK4 coefficients
        let k1 = derivatives * dt;
        let k2 = self.eval_derivatives(&(y + &k1 * 0.5), derivatives) * dt;
        let k3 = self.eval_derivatives(&(y + &k2 * 0.5), derivatives) * dt;
        let k4 = self.eval_derivatives(&(y + &k3), derivatives) * dt;
        
        // Weighted average
        let new_state = y + (k1 + k2 * 2.0 + k3 * 2.0 + k4) / 6.0;
        
        // Apply activation
        let activated_state = new_state.map(|x| x.tanh());
        
        Ok(LiquidState::from_vector(activated_state))
    }
    
    fn name(&self) -> &'static str {
        "Runge-Kutta 4"
    }
    
    fn order(&self) -> u8 {
        4
    }
}

impl RK4Solver {
    /// Evaluate derivatives at given state (simplified approximation)
    fn eval_derivatives(&self, state: &DVector<f32>, base_derivatives: &DVector<f32>) -> DVector<f32> {
        // Simplified derivative evaluation
        // Real implementation would compute full liquid dynamics
        let state_factor = 1.0 / (1.0 + state.norm_squared().sqrt());
        base_derivatives * state_factor
    }
}

/// Adaptive step size solver wrapper
/// Automatically adjusts step size based on local error estimation
#[derive(Debug, Clone)]
pub struct AdaptiveStepSolver {
    /// Base solver to use
    inner_solver: Box<dyn ODESolver>,
    /// Error tolerance
    tolerance: f32,
    /// Minimum step size
    min_step: f32,
    /// Maximum step size
    max_step: f32,
    /// Step size adjustment factor
    safety_factor: f32,
}

impl AdaptiveStepSolver {
    /// Create adaptive solver wrapping another solver
    pub fn new(inner_solver: Box<dyn ODESolver>, tolerance: f32) -> Self {
        Self {
            inner_solver,
            tolerance,
            min_step: 1e-6,
            max_step: 0.1,
            safety_factor: 0.9,
        }
    }
    
    /// Set step size bounds
    pub fn with_step_bounds(mut self, min_step: f32, max_step: f32) -> Self {
        self.min_step = min_step;
        self.max_step = max_step;
        self
    }
}

impl ODESolver for AdaptiveStepSolver {
    fn step(&self, state: &LiquidState, derivatives: &DVector<f32>, timestep: f32) -> Result<LiquidState> {
        let mut current_step = timestep.clamp(self.min_step, self.max_step);
        let mut current_state = state.clone();
        let mut remaining_time = timestep;
        
        // Adaptive stepping loop
        while remaining_time > 1e-8 {
            // Take full step
            let full_step = self.inner_solver.step(&current_state, derivatives, current_step)?;
            
            // Take two half steps for error estimation
            let half_step = current_step * 0.5;
            let half_step1 = self.inner_solver.step(&current_state, derivatives, half_step)?;
            let half_step2 = self.inner_solver.step(&half_step1, derivatives, half_step)?;
            
            // Estimate local error
            let error = (full_step.hidden_state() - half_step2.hidden_state()).norm();
            
            if error <= self.tolerance {
                // Accept step
                current_state = full_step;
                remaining_time -= current_step;
                
                // Increase step size for next iteration
                if error < self.tolerance * 0.1 {
                    current_step = (current_step * 1.5).min(self.max_step);
                }
            } else {
                // Reject step and reduce step size
                let scale_factor = self.safety_factor * (self.tolerance / error).powf(0.25);
                current_step = (current_step * scale_factor).max(self.min_step);
            }
            
            // Adjust final step to not overshoot
            if current_step > remaining_time {
                current_step = remaining_time;
            }
        }
        
        Ok(current_state)
    }
    
    fn name(&self) -> &'static str {
        "Adaptive Step"
    }
    
    fn order(&self) -> u8 {
        self.inner_solver.order()
    }
}

/// Factory for creating solvers
pub struct SolverFactory;

impl SolverFactory {
    /// Create solver by name
    pub fn create(name: &str) -> Result<Box<dyn ODESolver>> {
        match name.to_lowercase().as_str() {
            "euler" => Ok(Box::new(EulerSolver::new())),
            "heun" => Ok(Box::new(HeunSolver::new())),
            "rk4" => Ok(Box::new(RK4Solver::new())),
            _ => Err(LiquidAudioError::ConfigError(
                format!("Unknown solver: {}", name)
            )),
        }
    }
    
    /// Create adaptive solver
    pub fn create_adaptive(base_solver: &str, tolerance: f32) -> Result<Box<dyn ODESolver>> {
        let inner = Self::create(base_solver)?;
        Ok(Box::new(AdaptiveStepSolver::new(inner, tolerance)))
    }
    
    /// List available solvers
    pub fn available_solvers() -> &'static [&'static str] {
        &["euler", "heun", "rk4"]
    }
}