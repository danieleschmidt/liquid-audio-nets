"""Tests for liquid-audio-nets."""
import numpy as np
import pytest
from liquid_audio.liquid_cell import LiquidAudioCell
from liquid_audio.mfcc import MFCCEncoder
from liquid_audio.classifier import LiquidAudioClassifier, keyword_spotting_demo


SR = 16000


class TestLiquidAudioCell:
    def test_step_shape(self):
        cell = LiquidAudioCell(state_dim=8, input_dim=1)
        state = cell.step(np.array([0.5]))
        assert state.shape == (8,)

    def test_process_output_shape(self):
        cell = LiquidAudioCell(state_dim=16, input_dim=1)
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.1, 1600))
        states = cell.process(signal)
        assert states.shape == (1600, 16)

    def test_reset_state(self):
        cell = LiquidAudioCell(state_dim=8, input_dim=1)
        cell.step(np.array([1.0]))
        cell.reset_state()
        np.testing.assert_array_equal(cell._state, np.zeros(8))

    def test_tau_positive(self):
        cell = LiquidAudioCell(state_dim=16, input_dim=1)
        assert np.all(cell.tau > 0)

    def test_readout_shape(self):
        cell = LiquidAudioCell(state_dim=8, input_dim=1)
        signal = np.random.randn(100)
        states = cell.process(signal)
        W_out = np.random.randn(8, 3)
        y = cell.readout(states, W_out)
        assert y.shape == (100, 3)

    def test_state_changes_with_input(self):
        cell = LiquidAudioCell(state_dim=8, input_dim=1, seed=1)
        s0 = cell.step(np.array([0.0]))
        s1 = cell.step(np.array([1.0]))
        # States should differ
        assert not np.allclose(s0, s1)


class TestMFCCEncoder:
    def setup_method(self):
        self.enc = MFCCEncoder(sample_rate=SR, n_mfcc=13, n_fft=256, hop_length=80)
        self.signal = np.sin(2 * np.pi * 440 * np.linspace(0, 0.5, SR // 2))

    def test_encode_shape(self):
        mfcc = self.enc.encode(self.signal)
        assert mfcc.shape[1] == 13

    def test_mel_spectrogram_shape(self):
        mel = self.enc.mel_spectrogram(self.signal)
        assert mel.shape[1] == self.enc.n_mels

    def test_stft_shape(self):
        S = self.enc.stft(self.signal)
        assert S.shape[1] == self.enc.n_fft // 2 + 1

    def test_filterbank_shape(self):
        fb = self.enc._mel_filterbank
        assert fb.shape == (self.enc.n_mels, self.enc.n_fft // 2 + 1)

    def test_hz_mel_roundtrip(self):
        hz = 1000.0
        recovered = self.enc._mel_to_hz(self.enc._hz_to_mel(hz))
        assert abs(recovered - hz) < 1e-6

    def test_mfcc_finite(self):
        mfcc = self.enc.encode(self.signal)
        assert np.all(np.isfinite(mfcc))


class TestLiquidAudioClassifier:
    def setup_method(self):
        self.clf = LiquidAudioClassifier(n_classes=2, state_dim=16, n_mfcc=13)
        rng = np.random.default_rng(0)
        self.signal = rng.standard_normal(SR // 2)

    def test_forward_shape(self):
        probs = self.clf.forward(self.signal)
        assert probs.shape == (2,)

    def test_forward_probabilities_sum_to_one(self):
        probs = self.clf.forward(self.signal)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_predict_valid_class(self):
        pred = self.clf.predict(self.signal)
        assert pred in [0, 1]

    def test_extract_features_shape(self):
        feat = self.clf.extract_features(self.signal)
        assert feat.shape == (13,)

    def test_demo_runs(self):
        result = keyword_spotting_demo(n_samples=6, duration_s=0.2)
        assert "accuracy" in result
        assert 0 <= result["accuracy"] <= 1
