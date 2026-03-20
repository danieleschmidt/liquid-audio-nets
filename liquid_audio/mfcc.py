"""MFCC feature extraction using scipy."""
import numpy as np
from scipy.fft import fft
from scipy.signal import get_window


class MFCCEncoder:
    """Mel-Frequency Cepstral Coefficients encoder."""

    def __init__(self, sample_rate: int = 16000, n_mfcc: int = 13,
                 n_fft: int = 512, hop_length: int = 160,
                 n_mels: int = 40, f_min: float = 80.0, f_max: float = 7600.0):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self._mel_filterbank = self._build_mel_filterbank()

    def _hz_to_mel(self, hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _build_mel_filterbank(self) -> np.ndarray:
        mel_min = self._hz_to_mel(self.f_min)
        mel_max = self._hz_to_mel(self.f_max)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = np.array([self._mel_to_hz(m) for m in mel_points])
        bins = np.floor((self.n_fft // 2 + 1) * hz_points / self.sample_rate).astype(int)
        filterbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))
        for m in range(1, self.n_mels + 1):
            f_start, f_center, f_end = bins[m - 1], bins[m], bins[m + 1]
            for k in range(f_start, f_center):
                if f_center != f_start:
                    filterbank[m - 1, k] = (k - f_start) / (f_center - f_start)
            for k in range(f_center, f_end):
                if f_end != f_center:
                    filterbank[m - 1, k] = (f_end - k) / (f_end - f_center)
        return filterbank

    def stft(self, signal: np.ndarray) -> np.ndarray:
        """Short-time Fourier transform. Returns magnitude spectrogram."""
        window = get_window("hann", self.n_fft)
        frames = []
        for start in range(0, len(signal) - self.n_fft, self.hop_length):
            frame = signal[start:start + self.n_fft] * window
            frames.append(np.abs(fft(frame)[:self.n_fft // 2 + 1]))
        return np.stack(frames) if frames else np.zeros((1, self.n_fft // 2 + 1))

    def mel_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram. Returns (T, n_mels)."""
        S = self.stft(signal)
        mel = S @ self._mel_filterbank.T
        return np.log(mel + 1e-8)

    def encode(self, signal: np.ndarray) -> np.ndarray:
        """Compute MFCC features. Returns (T, n_mfcc)."""
        mel = self.mel_spectrogram(signal)
        # DCT
        n_frames, n_mels = mel.shape
        k = np.arange(self.n_mfcc)[None, :]
        m = np.arange(n_mels)[:, None]
        dct_matrix = np.cos(np.pi * k * (2 * m + 1) / (2 * n_mels))
        mfcc = mel @ dct_matrix
        return mfcc
