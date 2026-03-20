# Liquid Audio Nets 🎵🧠

**ODE-based (Liquid) Neural Networks for audio classification — pure NumPy/SciPy, no deep learning frameworks required.**

Liquid Neural Networks (LNNs) are a class of continuous-time recurrent networks where the hidden state evolves according to an ordinary differential equation (ODE). This repository implements them for audio classification tasks.

---

## Architecture

```
Raw audio signal
       │
       ▼
  MFCCEncoder          ← scipy.signal.spectrogram → mel filterbank → DCT
       │  (T, n_mfcc)
       ▼
LiquidAudioCell #0     ← dx/dt = (-x + tanh(Wx + Ux + b)) / τ   [Euler]
       │  (T, hidden)
       ▼
LiquidAudioCell #1
       │  (T, hidden)
       ▼
  mean pooling          ← collapse time dimension
       │  (hidden,)
       ▼
 Linear projection
       │  (n_classes,)
       ▼
    logits / probs
```

---

## Components

| Module | Class | Description |
|--------|-------|-------------|
| `liquid_audio_nets.liquid_cell` | `LiquidAudioCell` | Single ODE-based recurrent cell with learned τ |
| `liquid_audio_nets.mfcc` | `MFCCEncoder` | Computes MFCC features using `scipy.signal` |
| `liquid_audio_nets.classifier` | `LiquidAudioClassifier` | Stacks cells + linear head for classification |
| `liquid_audio_nets.demo` | — | Keyword spotting demo on synthetic tones |

---

## Quick Start

```bash
pip install numpy scipy
```

```python
import numpy as np
from liquid_audio_nets import MFCCEncoder, LiquidAudioClassifier

sr = 16_000
# Generate a 440 Hz tone
t = np.linspace(0, 1, sr, endpoint=False)
signal = 0.5 * np.sin(2 * np.pi * 440 * t)

# Encode MFCC features
enc = MFCCEncoder(n_mfcc=13, n_fft=512, hop_length=160)
features = enc.encode(signal, sr)   # shape: (T, 13)

# Classify (random weights — needs training for real use)
clf = LiquidAudioClassifier(
    input_size=13, hidden_size=64,
    n_cells=2, n_classes=3
)
logits = clf.classify(features)     # shape: (3,)
probs  = clf.predict_proba(features)
print(f"Predicted class: {clf.predict(features)}")
```

---

## Keyword Spotting Demo

```bash
python -m liquid_audio_nets.demo
```

Generates three sinusoidal tones (440 Hz, 880 Hz, 1760 Hz), extracts MFCC features, and runs classification.

```
============================================================
  Liquid Audio Nets — Keyword Spotting Demo
============================================================
  Tone : 440 Hz (A4)
  MFCC : shape=(49, 13)  mean=-4.1234  std=3.5678
  Probs: class0=0.312  class1=0.345  class2=0.343
  → Predicted class: 1 (880 Hz (A5))
  ...
```

---

## The ODE Cell

The `LiquidAudioCell` implements the continuous-time dynamics:

$$\frac{d\mathbf{h}}{dt} = \frac{-\mathbf{h} + \tanh(\mathbf{W}\mathbf{x} + \mathbf{U}\mathbf{h} + \mathbf{b})}{\tau}$$

Integrated with a simple Euler step:

```
h[t+1] = h[t] + dt * dh/dt
```

Where `τ` (tau) is a learnable time constant controlling how fast the state responds to new input.

---

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Dependencies

- `numpy >= 1.24`
- `scipy >= 1.10`
- `pytest >= 7.0` (for tests only)

---

## References

- Hasani, R. et al. (2021). *Liquid Time-constant Networks*. AAAI 2021. [arXiv:2006.04439](https://arxiv.org/abs/2006.04439)
- Lechner, M. et al. (2020). *Neural circuit policies enabling auditable autonomy*. Nature Machine Intelligence.
