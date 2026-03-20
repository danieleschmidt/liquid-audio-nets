"""
LiquidAudioClassifier — stacks multiple LiquidAudioCells and produces class logits.

Architecture:
  input (T, input_size)
  → LiquidAudioCell_0  (input_size → hidden_size)
  → LiquidAudioCell_1  (hidden_size → hidden_size)
  ...
  → mean pooling over time
  → linear projection → n_classes logits
"""

from __future__ import annotations

import numpy as np

from .liquid_cell import LiquidAudioCell


class LiquidAudioClassifier:
    """
    Multi-layer liquid neural network for audio classification.

    Parameters
    ----------
    input_size : int
        Dimensionality of input features (e.g. n_mfcc).
    hidden_size : int
        Number of hidden units in each LiquidAudioCell.
    n_cells : int
        Number of stacked LiquidAudioCell layers.
    n_classes : int
        Number of output classes.
    tau : float, optional
        Initial time constant for all cells. Default 1.0.
    dt : float, optional
        ODE integration step size. Default 0.1.
    seed : int, optional
        Base random seed. Each cell gets seed+layer_index.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 n_cells: int, n_classes: int,
                 tau: float = 1.0, dt: float = 0.1, seed: int = 0):
        if n_cells < 1:
            raise ValueError("n_cells must be ≥ 1")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_cells = n_cells
        self.n_classes = n_classes

        # Build stacked cells
        self.cells: list[LiquidAudioCell] = []
        for i in range(n_cells):
            in_sz = input_size if i == 0 else hidden_size
            cell = LiquidAudioCell(
                input_size=in_sz,
                hidden_size=hidden_size,
                tau=tau,
                dt=dt,
                seed=seed + i,
            )
            self.cells.append(cell)

        # Output projection: hidden_size → n_classes
        rng = np.random.default_rng(seed + n_cells)
        scale = np.sqrt(2.0 / (hidden_size + n_classes))
        self.W_out = rng.normal(0.0, scale, (n_classes, hidden_size))
        self.b_out = np.zeros(n_classes)

    def _forward_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Pass a (T, input_size) sequence through all stacked cells.

        Returns
        -------
        np.ndarray, shape (T, hidden_size)
            Output of the last cell layer.
        """
        x = sequence
        for cell in self.cells:
            x = cell.forward(x)   # (T, hidden_size)
        return x                  # (T, hidden_size)

    def classify(self, mfcc_features: np.ndarray) -> np.ndarray:
        """
        Classify a sequence of MFCC frames.

        Parameters
        ----------
        mfcc_features : np.ndarray, shape (T, input_size)
            Pre-computed MFCC feature matrix.

        Returns
        -------
        np.ndarray, shape (n_classes,)
            Raw class logits (not softmaxed).
        """
        mfcc_features = np.asarray(mfcc_features, dtype=float)
        if mfcc_features.ndim == 1:
            # Single frame — add time dimension
            mfcc_features = mfcc_features[np.newaxis, :]

        # Run through liquid layers
        hidden = self._forward_sequence(mfcc_features)  # (T, hidden_size)

        # Temporal mean pooling
        pooled = hidden.mean(axis=0)  # (hidden_size,)

        # Linear projection to class logits
        logits = self.W_out @ pooled + self.b_out  # (n_classes,)
        return logits

    def predict(self, mfcc_features: np.ndarray) -> int:
        """
        Return the predicted class index (argmax of logits).

        Parameters
        ----------
        mfcc_features : np.ndarray, shape (T, input_size)

        Returns
        -------
        int
        """
        logits = self.classify(mfcc_features)
        return int(np.argmax(logits))

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def predict_proba(self, mfcc_features: np.ndarray) -> np.ndarray:
        """
        Return class probabilities via softmax.

        Returns
        -------
        np.ndarray, shape (n_classes,)
        """
        return self.softmax(self.classify(mfcc_features))

    def __repr__(self) -> str:
        return (f"LiquidAudioClassifier(input_size={self.input_size}, "
                f"hidden_size={self.hidden_size}, n_cells={self.n_cells}, "
                f"n_classes={self.n_classes})")
