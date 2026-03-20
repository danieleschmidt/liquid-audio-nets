# liquid-audio-nets

Liquid Neural Networks for audio processing and keyword spotting, using ODE-based dynamics.

## Components

- **LiquidAudioCell** — Liquid time-constant (LTC) cell for audio ODE filtering
- **MFCCEncoder** — Mel-Frequency Cepstral Coefficients via custom filterbank + DCT
- **LiquidAudioClassifier** — End-to-end audio classifier using liquid cell + linear readout
- **keyword_spotting_demo** — Synthetic keyword vs noise classification demo

## Usage

```python
from liquid_audio.classifier import keyword_spotting_demo
result = keyword_spotting_demo(n_samples=20)
print(result)
```

## Install & Test

```bash
pip install -r requirements.txt
pytest tests/ -v
```
