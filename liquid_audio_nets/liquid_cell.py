"""
LiquidAudioCell — ODE-based audio filter with a learned time constant (tau).

The continuous-time dynamics follow:
    dx/dt = (-x + tanh(W * input + U * x + b)) / tau

We integrate one discrete step using Euler method with step_size dt.
"""

import numpy as np


class LiquidAudioCell:
    """
    A single liquid (ODE-based) recurrent cell for audio processing.

    Parameters
    ----------
    input_size : int
        Dimensionality of the input frame (e.g. number of MFCC coefficients).
    hidden_size : int
        Number of hidden units.
    tau : float, optional
        Initial time constant (must be positive). Default 1.0.
    dt : float, optional
        Integration step size for Euler ODE solver. Default 0.1.
    seed : int, optional
        Random seed for reproducible weight initialisation.
    """

    def __init__(self, input_size: int, hidden_size: int, tau: float = 1.0,
                 dt: float = 0.1, seed: int = 42):
        if tau <= 0:
            raise ValueError(f"tau must be positive, got {tau}")
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt

        rng = np.random.default_rng(seed)
        scale_w = np.sqrt(2.0 / (input_size + hidden_size))

        # Input-to-hidden weight matrix  [hidden_size, input_size]
        self.W = rng.normal(0.0, scale_w, (hidden_size, input_size))
        # Hidden-to-hidden weight matrix [hidden_size, hidden_size]
        self.U = rng.normal(0.0, scale_w, (hidden_size, hidden_size))
        # Bias vector                    [hidden_size]
        self.b = np.zeros(hidden_size)
        # Log-parameterised tau so that exp(log_tau) is always positive
        self.log_tau = np.log(float(tau))

    @property
    def tau(self) -> float:
        """Current (positive) time constant."""
        return float(np.exp(self.log_tau))

    @tau.setter
    def tau(self, value: float):
        if value <= 0:
            raise ValueError(f"tau must be positive, got {value}")
        self.log_tau = np.log(float(value))

    def step(self, x: np.ndarray, state: np.ndarray) -> np.ndarray:
        """
        Advance the hidden state by one discrete step via Euler integration.

        Implements:
            dstate/dt = (-state + tanh(W @ x + U @ state + b)) / tau
            new_state = state + dt * dstate/dt

        Parameters
        ----------
        x : np.ndarray, shape (input_size,) or (batch, input_size)
            Input frame at the current time step.
        state : np.ndarray, shape (hidden_size,) or (batch, hidden_size)
            Previous hidden state.

        Returns
        -------
        np.ndarray
            Updated hidden state with the same shape as *state*.
        """
        x = np.asarray(x, dtype=float)
        state = np.asarray(state, dtype=float)

        # Linear pre-activation
        pre = x @ self.W.T + state @ self.U.T + self.b   # (..., hidden_size)

        # ODE derivative
        dstate = (-state + np.tanh(pre)) / self.tau       # (..., hidden_size)

        # Euler update
        new_state = state + self.dt * dstate
        return new_state

    def initial_state(self, batch_size: int = 1) -> np.ndarray:
        """
        Return a zero initial hidden state.

        Parameters
        ----------
        batch_size : int
            Number of parallel sequences. Default 1 returns shape (hidden_size,).

        Returns
        -------
        np.ndarray
        """
        if batch_size == 1:
            return np.zeros(self.hidden_size)
        return np.zeros((batch_size, self.hidden_size))

    def forward(self, sequence: np.ndarray,
                initial_state: np.ndarray | None = None) -> np.ndarray:
        """
        Process a full sequence of frames.

        Parameters
        ----------
        sequence : np.ndarray, shape (T, input_size)
            Time-series of input frames.
        initial_state : np.ndarray or None
            Starting hidden state. Defaults to zeros.

        Returns
        -------
        np.ndarray, shape (T, hidden_size)
            Hidden states for each time step.
        """
        sequence = np.asarray(sequence, dtype=float)
        T = sequence.shape[0]
        state = self.initial_state() if initial_state is None else initial_state
        outputs = np.empty((T, self.hidden_size))
        for t in range(T):
            state = self.step(sequence[t], state)
            outputs[t] = state
        return outputs

    def __repr__(self) -> str:
        return (f"LiquidAudioCell(input_size={self.input_size}, "
                f"hidden_size={self.hidden_size}, tau={self.tau:.4f}, "
                f"dt={self.dt})")
