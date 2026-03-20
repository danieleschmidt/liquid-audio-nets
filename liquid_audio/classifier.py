"""Liquid Audio Classifier for keyword spotting."""
import numpy as np
from liquid_audio.liquid_cell import LiquidAudioCell
from liquid_audio.mfcc import MFCCEncoder


class LiquidAudioClassifier:
    """Liquid neural network classifier for audio keywords."""

    def __init__(self, n_classes: int = 2, state_dim: int = 32,
                 sample_rate: int = 16000, n_mfcc: int = 13, seed: int = 0):
        self.n_classes = n_classes
        self.state_dim = state_dim
        self.sample_rate = sample_rate
        self.encoder = MFCCEncoder(sample_rate=sample_rate, n_mfcc=n_mfcc)
        self.cell = LiquidAudioCell(state_dim=state_dim, input_dim=n_mfcc, seed=seed)
        rng = np.random.default_rng(seed)
        self.W_out = rng.standard_normal((state_dim, n_classes)) * 0.1

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def extract_features(self, signal: np.ndarray) -> np.ndarray:
        """Extract MFCC features and pool to single vector."""
        mfcc = self.encoder.encode(signal)
        return mfcc.mean(axis=0)

    def forward(self, signal: np.ndarray) -> np.ndarray:
        """Forward pass: audio → class probabilities."""
        mfcc = self.encoder.encode(signal)
        self.cell.reset_state()
        for frame in mfcc:
            self.cell.step(frame)
        # Use final state for classification
        logits = self.cell._state @ self.W_out
        return self._softmax(logits)

    def predict(self, signal: np.ndarray) -> int:
        """Return predicted class index."""
        probs = self.forward(signal)
        return int(np.argmax(probs))

    def fit(self, signals: list, labels: list, lr: float = 0.01, epochs: int = 50) -> list:
        """Simple gradient-free training via weight perturbation."""
        losses = []
        rng = np.random.default_rng(42)
        for epoch in range(epochs):
            epoch_loss = 0.0
            for sig, label in zip(signals, labels):
                probs = self.forward(sig)
                epoch_loss -= np.log(probs[label] + 1e-10)
            losses.append(epoch_loss / len(signals))
            # Perturb W_out
            noise = rng.standard_normal(self.W_out.shape) * 0.01
            self.W_out += noise
        return losses


def keyword_spotting_demo(n_samples: int = 20, duration_s: float = 0.5,
                           sample_rate: int = 16000) -> dict:
    """Demo: classify synthetic 'keyword' vs 'noise' audio."""
    rng = np.random.default_rng(7)
    n = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n)
    
    # Class 0: 440 Hz tone (keyword)
    # Class 1: noise
    signals, labels = [], []
    for i in range(n_samples):
        if i % 2 == 0:
            sig = np.sin(2 * np.pi * 440 * t) + rng.standard_normal(n) * 0.1
            labels.append(0)
        else:
            sig = rng.standard_normal(n) * 0.5
            labels.append(1)
        signals.append(sig)
    
    clf = LiquidAudioClassifier(n_classes=2, state_dim=16, sample_rate=sample_rate)
    # Eval (no training — just structural demo)
    preds = [clf.predict(s) for s in signals]
    acc = np.mean(np.array(preds) == np.array(labels))
    return {"n_samples": n_samples, "accuracy": float(acc), "predictions": preds[:5], "labels": labels[:5]}
