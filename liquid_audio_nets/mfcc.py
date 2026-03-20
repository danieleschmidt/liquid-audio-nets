"""
MFCCEncoder — Mel-Frequency Cepstral Coefficients using scipy.signal.

Pipeline:
  1. Compute STFT power spectrum via scipy.signal.spectrogram
  2. Apply triangular mel filterbank (manually computed)
  3. Log-compress filterbank energies
  4. Apply DCT (Type-II) to produce cepstral coefficients
"""

from __future__ import annotations

import numpy as np
from scipy.signal import spectrogram
from scipy.fft import dct


class MFCCEncoder:
    """
    Compute MFCC features from a raw audio signal.

    Parameters
    ----------
    n_mfcc : int
        Number of cepstral coefficients to keep. Default 13.
    n_fft : int
        FFT size. Default 512.
    hop_length : int
        Number of samples between successive frames. Default 160.
    n_mels : int
        Number of mel filterbank channels. Default 40.
    fmin : float
        Lowest mel filter frequency in Hz. Default 0.0.
    fmax : float or None
        Highest mel filter frequency in Hz.  None → sr / 2.
    """

    def __init__(self, n_mfcc: int = 13, n_fft: int = 512,
                 hop_length: int = 160, n_mels: int = 40,
                 fmin: float = 0.0, fmax: float | None = None):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    # ------------------------------------------------------------------
    # Mel conversion helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _hz_to_mel(f: float) -> float:
        return 2595.0 * np.log10(1.0 + f / 700.0)

    @staticmethod
    def _mel_to_hz(m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    # ------------------------------------------------------------------
    # Filterbank construction
    # ------------------------------------------------------------------
    def _build_mel_filterbank(self, sr: int, n_freqs: int) -> np.ndarray:
        """
        Build a triangular mel filterbank matrix.

        Returns
        -------
        np.ndarray, shape (n_mels, n_freqs)
        """
        fmax = self.fmax if self.fmax is not None else sr / 2.0

        mel_min = self._hz_to_mel(self.fmin)
        mel_max = self._hz_to_mel(fmax)

        # n_mels + 2 evenly spaced points in mel space (includes edges)
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = np.array([self._mel_to_hz(m) for m in mel_points])

        # Map to FFT bin indices
        bin_points = np.floor(hz_points / (sr / self.n_fft)).astype(int)
        bin_points = np.clip(bin_points, 0, n_freqs - 1)

        filterbank = np.zeros((self.n_mels, n_freqs))
        for m in range(1, self.n_mels + 1):
            left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
            # Rising slope
            for k in range(left, center + 1):
                denom = center - left
                filterbank[m - 1, k] = (k - left) / denom if denom > 0 else 1.0
            # Falling slope
            for k in range(center, right + 1):
                denom = right - center
                filterbank[m - 1, k] = (right - k) / denom if denom > 0 else 1.0

        return filterbank

    # ------------------------------------------------------------------
    # Main encode method
    # ------------------------------------------------------------------
    def encode(self, signal: np.ndarray, sr: int) -> np.ndarray:
        """
        Compute MFCCs for a 1-D audio signal.

        Parameters
        ----------
        signal : np.ndarray, shape (N,)
            Raw audio samples (float, any amplitude range).
        sr : int
            Sample rate in Hz.

        Returns
        -------
        np.ndarray, shape (T, n_mfcc)
            MFCC feature matrix where T is the number of frames.
        """
        signal = np.asarray(signal, dtype=float)
        if signal.ndim != 1:
            raise ValueError("signal must be 1-D")

        # 1. Power spectrogram via scipy.signal.spectrogram
        freqs, times, Sxx = spectrogram(
            signal,
            fs=sr,
            nperseg=self.n_fft,
            noverlap=self.n_fft - self.hop_length,
            window='hann',
            scaling='spectrum',
        )
        # Sxx shape: (n_freqs, T)  — power spectrum

        n_freqs = Sxx.shape[0]

        # 2. Mel filterbank
        mel_fb = self._build_mel_filterbank(sr, n_freqs)  # (n_mels, n_freqs)

        # 3. Apply filterbank
        mel_energies = mel_fb @ Sxx   # (n_mels, T)

        # 4. Log compression (add small epsilon for numerical stability)
        log_mel = np.log(mel_energies + 1e-10)  # (n_mels, T)

        # 5. DCT Type-II across mel channels → cepstral coefficients
        #    Result: (n_mels, T); keep first n_mfcc rows
        mfccs = dct(log_mel, type=2, axis=0, norm='ortho')  # (n_mels, T)
        mfccs = mfccs[:self.n_mfcc, :]                      # (n_mfcc, T)

        # Return shape (T, n_mfcc) for frame-major convention
        return mfccs.T

    def __repr__(self) -> str:
        return (f"MFCCEncoder(n_mfcc={self.n_mfcc}, n_fft={self.n_fft}, "
                f"hop_length={self.hop_length}, n_mels={self.n_mels})")
