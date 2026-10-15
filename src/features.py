"""
Unified 16-Feature Extraction Module
Single source of truth for offline model training, feature clustering, and live GUI inference.
Feature dimension: exactly 16.
"""

import numpy as np
from scipy.stats import skew, kurtosis

FEATURE_NAMES = [
    "RMS", "MAV", "WL", "VAR", "SSI",
    "ZC", "SSC", "WAMP", "ABS_SKEW", "KURT",
    "MNF", "MDF", "SE", "PKF", "MNP", "TTP"
]


def extract_features(window: np.ndarray, fs: float = 1000.0, zc_thresh: float = 0.005, wamp_thresh: float = 0.01) -> np.ndarray:
    """
    Extracts the unified 16-feature vector from a single sEMG analysis window.
    
    Parameters:
        window: 1D numpy array of filtered sEMG biopotentials.
        fs: Sampling frequency (default 1000.0 Hz).
        zc_thresh: Voltage noise deadband threshold for Zero Crossings / SSC.
        wamp_thresh: Voltage threshold for Willison Amplitude.
        
    Returns:
        1D numpy array of shape (16,) containing float64 feature values.
    """
    w = np.asarray(window, dtype=np.float64)
    N = len(w)
    if N < 10:
        return np.zeros(16, dtype=np.float64)
        
    # --- 1. Time-Domain Energy & Amplitude ---
    rms = np.sqrt(np.mean(w ** 2))
    mav = np.mean(np.abs(w))
    wl = np.sum(np.abs(np.diff(w)))
    var = np.var(w, ddof=1) if N > 1 else 0.0
    ssi = np.sum(w ** 2)
    
    # --- 2. Time-Domain Complexity & Counts ---
    # Zero Crossings (with deadband)
    diff_sign = (w[:-1] * w[1:] < 0) & (np.abs(w[:-1] - w[1:]) >= zc_thresh)
    zc = float(np.sum(diff_sign))
    
    # Slope Sign Changes
    if N >= 3:
        d1 = w[1:-1] - w[:-2]
        d2 = w[1:-1] - w[2:]
        ssc_mask = (d1 * d2 > 0) & ((np.abs(d1) >= zc_thresh) | (np.abs(d2) >= zc_thresh))
        ssc = float(np.sum(ssc_mask))
    else:
        ssc = 0.0
        
    # Willison Amplitude
    wamp = float(np.sum(np.abs(np.diff(w)) >= wamp_thresh))
    
    # --- 3. Statistical Distribution Shape ---
    # Absolute skewness (invariant to electrode +/- polarity snap inversion)
    abs_sk = float(np.abs(skew(w))) if var > 1e-12 else 0.0
    kurt = float(kurtosis(w)) if var > 1e-12 else 0.0
    
    # --- 4. Spectral / Frequency-Domain (FFT Power Spectrum) ---
    # Fast real FFT
    fft_vals = np.fft.rfft(w)
    psd = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(N, 1.0 / fs)
    
    ttp = float(np.sum(psd))  # Total Power
    mnp = float(np.mean(psd)) # Mean Power
    
    if ttp > 1e-12:
        # Mean Frequency (Spectral Centroid)
        mnf = float(np.sum(freqs * psd) / ttp)
        
        # Median Frequency (splits power in half)
        cumulative_power = np.cumsum(psd)
        mdf_idx = np.searchsorted(cumulative_power, ttp / 2.0)
        mdf = float(freqs[min(mdf_idx, len(freqs) - 1)])
        
        # Spectral Entropy
        p_norm = psd / ttp
        p_norm_safe = p_norm[p_norm > 1e-12]
        se = float(-np.sum(p_norm_safe * np.log2(p_norm_safe)))
        
        # Peak Frequency
        pkf = float(freqs[np.argmax(psd)])
    else:
        mnf, mdf, se, pkf = 0.0, 0.0, 0.0, 0.0
        
    return np.array([
        rms, mav, wl, var, ssi,
        zc, ssc, wamp, abs_sk, kurt,
        mnf, mdf, se, pkf, mnp, ttp
    ], dtype=np.float64)
