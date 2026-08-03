"""Signal-processing utilities: FFT / PSD, Butterworth filtering, SNR and RMSE."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sp_signal


def compute_fft(series: pd.Series, fs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided (frequencies, magnitude spectrum) of a real signal.

    ``fs`` is the sampling frequency (1 sample per hour -> fs=1.0 cycles/hour).
    """
    x = np.asarray(pd.Series(series).dropna(), dtype=float)
    x = x - x.mean()  # remove DC component so the spectrum is not dominated by the mean
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(x)) * 2.0 / n
    return freqs, mag


def power_spectral_density(series: pd.Series, fs: float = 1.0,
                           nperseg: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density estimate (freqs, PSD)."""
    x = np.asarray(pd.Series(series).dropna(), dtype=float)
    nperseg = min(nperseg, len(x))
    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=nperseg)
    return freqs, psd


def snr_db(clean: pd.Series, noise: pd.Series) -> float:
    """Signal-to-noise ratio (dB) given the clean reference and the noisy signal.

    SNR = 10*log10( P_signal / P_noise ), where the noise is estimated as
    ``noise - clean`` (the injected perturbation).
    """
    c = np.asarray(pd.Series(clean).dropna(), dtype=float)
    n = np.asarray(pd.Series(noise).dropna(), dtype=float)
    m = min(len(c), len(n))
    c, n = c[:m], n[:m]
    residual = n - c
    p_signal = np.mean(c ** 2)
    p_noise = np.mean(residual ** 2)
    if p_noise == 0:
        return float("inf")
    return float(10.0 * np.log10(p_signal / p_noise))


def butter_lowpass(series: pd.Series, cutoff: float, fs: float = 1.0,
                   order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter (filtfilt) of a 1-D signal.

    ``cutoff`` and ``fs`` are in the same units (cycles/hour). ``filtfilt`` avoids
    phase distortion so the filtered series stays time-aligned with the reference.
    """
    x = np.asarray(pd.Series(series).dropna(), dtype=float)
    nyq = 0.5 * fs
    wn = cutoff / nyq
    b, a = sp_signal.butter(order, wn, btype="low", analog=False)
    return sp_signal.filtfilt(b, a, x)


def moving_average(series: pd.Series, window: int = 10) -> np.ndarray:
    """Centered moving-average smoother (fallback denoiser)."""
    return pd.Series(series).rolling(window, center=True, min_periods=1).mean().to_numpy()


def rmse(a, b) -> float:
    """Root mean squared error between two aligned arrays."""
    a = np.asarray(pd.Series(a).dropna(), dtype=float)
    b = np.asarray(pd.Series(b).dropna(), dtype=float)
    m = min(len(a), len(b))
    return float(np.sqrt(np.mean((a[:m] - b[:m]) ** 2)))
