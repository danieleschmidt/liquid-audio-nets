"""
Keyword Spotting Demo — Liquid Audio Nets
==========================================

Generates three synthetic sinusoidal tones (440 Hz, 880 Hz, 1760 Hz),
encodes each as MFCC features, and classifies them with a
LiquidAudioClassifier.

Because the classifier uses random weights (no training), the
class predictions are arbitrary — the demo illustrates the full
pipeline and shows that different tones produce distinct feature
vectors, which a trained model could separate.

Usage
-----
    python -m liquid_audio_nets.demo
    # or
    python liquid_audio_nets/demo.py
"""

from __future__ import annotations

import numpy as np

from .mfcc import MFCCEncoder
from .classifier import LiquidAudioClassifier


# ── Configuration ─────────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000          # Hz
DURATION    = 1.0             # seconds
FREQUENCIES = [440, 880, 1760]  # Hz — "do", "do octave up", "do two octaves up"
LABELS      = ["440 Hz (A4)", "880 Hz (A5)", "1760 Hz (A6)"]

MFCC_PARAMS = dict(n_mfcc=13, n_fft=512, hop_length=160, n_mels=40)
N_CELLS     = 2
HIDDEN_SIZE = 32


# ── Tone generation ────────────────────────────────────────────────────────────

def generate_tone(frequency: float, duration: float, sr: int,
                  amplitude: float = 0.5) -> np.ndarray:
    """Return a pure sinusoidal tone as a 1-D float array."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return amplitude * np.sin(2 * np.pi * frequency * t)


# ── Main demo ─────────────────────────────────────────────────────────────────

def run_demo(verbose: bool = True) -> list[dict]:
    """
    Run the keyword spotting demo.

    Returns a list of result dicts with keys:
        label, frequency, mfcc_shape, logits, predicted_class, probabilities
    """
    encoder    = MFCCEncoder(**MFCC_PARAMS)
    classifier = LiquidAudioClassifier(
        input_size=MFCC_PARAMS["n_mfcc"],
        hidden_size=HIDDEN_SIZE,
        n_cells=N_CELLS,
        n_classes=len(FREQUENCIES),
    )

    if verbose:
        print("=" * 60)
        print("  Liquid Audio Nets — Keyword Spotting Demo")
        print("=" * 60)
        print(f"  Encoder  : {encoder}")
        print(f"  Classifier: {classifier}")
        print()

    results = []
    for freq, label in zip(FREQUENCIES, LABELS):
        # 1. Generate tone
        signal = generate_tone(freq, DURATION, SAMPLE_RATE)

        # 2. Encode MFCC features
        features = encoder.encode(signal, SAMPLE_RATE)  # (T, n_mfcc)

        # 3. Classify
        logits = classifier.classify(features)
        probs  = classifier.softmax(logits)
        pred   = int(np.argmax(logits))

        result = {
            "label":           label,
            "frequency":       freq,
            "mfcc_shape":      features.shape,
            "logits":          logits,
            "predicted_class": pred,
            "probabilities":   probs,
        }
        results.append(result)

        if verbose:
            print(f"  Tone : {label}")
            print(f"  MFCC : shape={features.shape}  "
                  f"mean={features.mean():.4f}  std={features.std():.4f}")
            prob_str = "  ".join(
                f"class{i}={p:.3f}" for i, p in enumerate(probs)
            )
            print(f"  Probs: {prob_str}")
            print(f"  → Predicted class: {pred} ({LABELS[pred]})")
            print()

    if verbose:
        print("Demo complete. (Weights are random — train for real classification.)")
        print("=" * 60)

    return results


if __name__ == "__main__":
    run_demo(verbose=True)
