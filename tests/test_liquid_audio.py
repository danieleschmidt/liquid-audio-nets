"""
Tests for liquid_audio_nets — 12+ tests covering all major components.
"""

import numpy as np
import pytest

from liquid_audio_nets.liquid_cell import LiquidAudioCell
from liquid_audio_nets.mfcc import MFCCEncoder
from liquid_audio_nets.classifier import LiquidAudioClassifier


# ── Helpers ────────────────────────────────────────────────────────────────────

SR = 16_000

def make_tone(freq: float, duration: float = 0.5, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * freq * t)


# ── LiquidAudioCell tests ──────────────────────────────────────────────────────

class TestLiquidAudioCell:

    def test_init_shapes(self):
        cell = LiquidAudioCell(input_size=8, hidden_size=16)
        assert cell.W.shape == (16, 8)
        assert cell.U.shape == (16, 16)
        assert cell.b.shape == (16,)

    def test_tau_positive(self):
        cell = LiquidAudioCell(input_size=4, hidden_size=8, tau=2.0)
        assert cell.tau > 0
        assert abs(cell.tau - 2.0) < 1e-9

    def test_tau_setter_rejects_non_positive(self):
        cell = LiquidAudioCell(input_size=4, hidden_size=8)
        with pytest.raises(ValueError):
            cell.tau = -1.0
        with pytest.raises(ValueError):
            cell.tau = 0.0

    def test_step_output_shape(self):
        cell = LiquidAudioCell(input_size=13, hidden_size=32)
        x = np.random.randn(13)
        state = cell.initial_state()
        new_state = cell.step(x, state)
        assert new_state.shape == (32,)

    def test_step_changes_state(self):
        cell = LiquidAudioCell(input_size=4, hidden_size=8)
        x = np.ones(4)
        state = cell.initial_state()
        new_state = cell.step(x, state)
        assert not np.allclose(state, new_state)

    def test_step_bounded(self):
        """Hidden state should remain finite after many steps."""
        cell = LiquidAudioCell(input_size=4, hidden_size=8)
        state = cell.initial_state()
        for _ in range(1000):
            x = np.random.randn(4)
            state = cell.step(x, state)
        assert np.all(np.isfinite(state))

    def test_forward_sequence_shape(self):
        cell = LiquidAudioCell(input_size=13, hidden_size=32)
        seq = np.random.randn(50, 13)
        out = cell.forward(seq)
        assert out.shape == (50, 32)

    def test_initial_state_zeros(self):
        cell = LiquidAudioCell(input_size=4, hidden_size=8)
        state = cell.initial_state()
        assert np.all(state == 0)
        assert state.shape == (8,)

    def test_invalid_tau_at_init(self):
        with pytest.raises(ValueError):
            LiquidAudioCell(input_size=4, hidden_size=8, tau=-1.0)

    def test_invalid_dt_at_init(self):
        with pytest.raises(ValueError):
            LiquidAudioCell(input_size=4, hidden_size=8, dt=0.0)


# ── MFCCEncoder tests ──────────────────────────────────────────────────────────

class TestMFCCEncoder:

    def test_encode_output_shape(self):
        enc = MFCCEncoder(n_mfcc=13, n_fft=512, hop_length=160)
        signal = make_tone(440)
        features = enc.encode(signal, SR)
        assert features.ndim == 2
        assert features.shape[1] == 13

    def test_encode_finite(self):
        enc = MFCCEncoder()
        signal = make_tone(880)
        features = enc.encode(signal, SR)
        assert np.all(np.isfinite(features))

    def test_different_tones_produce_different_features(self):
        enc = MFCCEncoder()
        f1 = enc.encode(make_tone(440), SR)
        f2 = enc.encode(make_tone(1760), SR)
        # Features should differ significantly
        assert not np.allclose(f1, f2, atol=1e-3)

    def test_encode_1d_only(self):
        enc = MFCCEncoder()
        with pytest.raises(ValueError):
            enc.encode(np.zeros((16000, 2)), SR)

    def test_n_mfcc_parameter(self):
        for n in [8, 13, 20]:
            enc = MFCCEncoder(n_mfcc=n)
            features = enc.encode(make_tone(440), SR)
            assert features.shape[1] == n


# ── LiquidAudioClassifier tests ────────────────────────────────────────────────

class TestLiquidAudioClassifier:

    def _make_clf(self, n_classes=3):
        return LiquidAudioClassifier(
            input_size=13, hidden_size=16, n_cells=2, n_classes=n_classes
        )

    def test_classify_output_shape(self):
        clf = self._make_clf(n_classes=3)
        features = np.random.randn(50, 13)
        logits = clf.classify(features)
        assert logits.shape == (3,)

    def test_classify_finite(self):
        clf = self._make_clf()
        logits = clf.classify(np.random.randn(30, 13))
        assert np.all(np.isfinite(logits))

    def test_predict_returns_valid_class(self):
        clf = self._make_clf(n_classes=4)
        pred = clf.predict(np.random.randn(40, 13))
        assert 0 <= pred < 4

    def test_predict_proba_sums_to_one(self):
        clf = self._make_clf(n_classes=3)
        probs = clf.predict_proba(np.random.randn(30, 13))
        assert abs(probs.sum() - 1.0) < 1e-9

    def test_predict_proba_non_negative(self):
        clf = self._make_clf()
        probs = clf.predict_proba(np.random.randn(30, 13))
        assert np.all(probs >= 0)

    def test_n_cells_parameter(self):
        for n_cells in [1, 3, 5]:
            clf = LiquidAudioClassifier(13, 16, n_cells=n_cells, n_classes=2)
            assert len(clf.cells) == n_cells
            logits = clf.classify(np.random.randn(20, 13))
            assert logits.shape == (2,)

    def test_invalid_n_cells(self):
        with pytest.raises(ValueError):
            LiquidAudioClassifier(13, 16, n_cells=0, n_classes=2)

    def test_single_frame_input(self):
        """Classifier should handle a single frame (T=1) gracefully."""
        clf = self._make_clf()
        logits = clf.classify(np.random.randn(1, 13))
        assert logits.shape == (3,)


# ── Demo integration test ──────────────────────────────────────────────────────

class TestDemo:

    def test_demo_runs(self):
        from liquid_audio_nets.demo import run_demo
        results = run_demo(verbose=False)
        assert len(results) == 3

    def test_demo_results_have_expected_keys(self):
        from liquid_audio_nets.demo import run_demo
        results = run_demo(verbose=False)
        required_keys = {"label", "frequency", "mfcc_shape",
                         "logits", "predicted_class", "probabilities"}
        for r in results:
            assert required_keys.issubset(r.keys())

    def test_demo_distinct_features(self):
        """Each tone should produce MFCC of the right shape."""
        from liquid_audio_nets.demo import run_demo
        results = run_demo(verbose=False)
        shapes = [r["mfcc_shape"] for r in results]
        # Each should have n_mfcc == 13
        for shape in shapes:
            assert shape[1] == 13
        # All three tones produce feature tensors
        assert len(shapes) == 3
