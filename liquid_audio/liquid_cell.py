"""Liquid Neural Network audio cell using ODE-based dynamics."""
import numpy as np


class LiquidAudioCell:
    """Liquid time-constant (LTC) cell for audio filtering.
    
    Implements a first-order ODE: tau * dx/dt = -x + f(W*x + u)
    Discretized with Euler method.
    """

    def __init__(self, state_dim: int = 16, input_dim: int = 1, dt: float = 1/16000,
                 seed: int = 42):
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.dt = dt
        rng = np.random.default_rng(seed)
        scale = np.sqrt(1.0 / state_dim)
        self.W_in = rng.standard_normal((state_dim, input_dim)) * scale
        self.W_rec = rng.standard_normal((state_dim, state_dim)) * scale * 0.5
        self.tau = np.abs(rng.standard_normal(state_dim)) * 0.01 + 0.001  # >0
        self.bias = np.zeros(state_dim)
        self._state = np.zeros(state_dim)

    def reset_state(self) -> None:
        self._state = np.zeros(self.state_dim)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def step(self, u: np.ndarray) -> np.ndarray:
        """Single ODE step. u shape: (input_dim,). Returns state."""
        x = self._state
        net_input = self.W_in @ u + self.W_rec @ x + self.bias
        f = np.tanh(net_input)
        dx = (-x + f) / self.tau
        self._state = x + self.dt * dx
        return self._state.copy()

    def process(self, signal: np.ndarray) -> np.ndarray:
        """Process 1D audio signal. Returns state trajectory (T, state_dim)."""
        signal = signal.ravel()
        self.reset_state()
        states = []
        for s in signal:
            state = self.step(np.array([s]))
            states.append(state)
        return np.stack(states)

    def readout(self, states: np.ndarray, W_out: np.ndarray) -> np.ndarray:
        """Linear readout: y = states @ W_out."""
        return states @ W_out
