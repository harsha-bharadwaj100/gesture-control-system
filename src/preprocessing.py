"""
Signal Conditioning & Preprocessing Module
Implements trial-level zero-phase filtering (Notch + Bandpass) and data loading.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import iirnotch, butter, filtfilt
import src.config as cfg

# Precompute bandpass filter coefficients
B_BAND, A_BAND = butter(cfg.BANDPASS_ORDER, [cfg.BANDPASS_LOW, cfg.BANDPASS_HIGH], btype='bandpass', fs=cfg.SAMPLE_RATE)


def filter_signal(v: np.ndarray, demean: bool = True) -> np.ndarray:
    """
    Applies zero-phase signal conditioning:
    1. DC Offset Demeaning.
    2. Adaptive Harmonic Regression (AHR): Least-squares 50 Hz, 100 Hz, 150 Hz mains hum cancellation.
       (Completely eliminates the boundary derivative discontinuity that causes filter-ringing cone artifacts).
    3. 4th-order Butterworth Bandpass filter (20 - 450 Hz) with zero-phase filtfilt.
    """
    v = np.asarray(v, dtype=np.float64)
    N = len(v)
    if N < 20:
        return v
        
    # 1. DC Demean
    if demean:
        v = v - np.mean(v)
        
    # 2. Harmonic Powerline Subtraction (50 Hz and harmonics)
    t = np.arange(N) / float(cfg.SAMPLE_RATE)
    cols = []
    for f in [50.0, 100.0, 150.0]:
        cols.append(np.sin(2.0 * np.pi * f * t))
        cols.append(np.cos(2.0 * np.pi * f * t))
    A = np.column_stack(cols)
    coeffs, _, _, _ = np.linalg.lstsq(A, v, rcond=None)
    v_dehummed = v - (A @ coeffs)
    
    # 3. Bandpass Filtering (20 Hz - 450 Hz)
    padlen = min(cfg.FILTER_PADLEN, N // 4)
    try:
        v_clean = filtfilt(B_BAND, A_BAND, v_dehummed, padlen=padlen)
        return v_clean
    except Exception:
        return v_dehummed


def load_participant_data(participant_name: str, datasets_dir: str = "All_Datasets") -> pd.DataFrame:
    """
    Loads raw CSV data for a given participant.
    """
    csv_name = f"{participant_name}5gesture_dataset.csv"
    path = os.path.join(datasets_dir, csv_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Participant CSV not found: {path}")
    return pd.read_csv(path)


def load_participant_trials(participant_name: str, datasets_dir: str = "All_Datasets", apply_filter: bool = True):
    """
    Loads and structures all 4.0-second trials for a participant.
    Returns: dict with keys (gesture_name, rep_index) -> 1D numpy array of length 4000
    """
    df = load_participant_data(participant_name, datasets_dir)
    trials = {}
    
    for (gesture, rep), grp in df.groupby(['Gesture_Label', 'Repetition']):
        v = grp['Voltage'].values
        if len(v) < 4000:
            continue
        v_4k = v[:4000]
        if apply_filter:
            v_proc = filter_signal(v_4k)
        else:
            v_proc = v_4k - np.mean(v_4k)
        trials[(gesture, rep)] = v_proc
        
    return trials
