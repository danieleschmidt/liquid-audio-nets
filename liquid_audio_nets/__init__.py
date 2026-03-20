"""
liquid_audio_nets
=================

ODE-based (Liquid) Neural Networks for audio classification.

Modules
-------
liquid_cell  : LiquidAudioCell  — single recurrent ODE cell
mfcc         : MFCCEncoder      — mel-frequency cepstral coefficient encoder
classifier   : LiquidAudioClassifier — full classifier stacking liquid cells
demo         : keyword spotting demo on synthetic tones

Example
-------
>>> import numpy as np
>>> from liquid_audio_nets import MFCCEncoder, LiquidAudioClassifier
>>> sr = 16_000
>>> signal = np.sin(2 * np.pi * 440 * np.linspace(0, 1, sr))
>>> enc = MFCCEncoder(n_mfcc=13)
>>> features = enc.encode(signal, sr)   # (T, 13)
>>> clf = LiquidAudioClassifier(input_size=13, hidden_size=32,
...                              n_cells=2, n_classes=3)
>>> logits = clf.classify(features)     # (3,)
"""

from .liquid_cell import LiquidAudioCell
from .mfcc import MFCCEncoder
from .classifier import LiquidAudioClassifier

__all__ = ["LiquidAudioCell", "MFCCEncoder", "LiquidAudioClassifier"]
__version__ = "0.1.0"
