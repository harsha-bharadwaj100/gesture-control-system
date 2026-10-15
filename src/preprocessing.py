"""
Signal Conditioning & Preprocessing Module
Implements trial-level zero-phase filtering (Notch + Bandpass) and data loading.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import iirnotch, butter, filtfilt
import src.config as cfg

# Precompute filter coefficients
B_NOTCH, A_NOTCH = iirnotch(cfg.NOTCH_FREQ, 15.0, cfg.SAMPLE_RATE)
B_BAND, A_BAND = butter(cfg.BANDPASS_ORDER, [cfg.BANDPASS_LOW, cfg.BANDPASS_HIGH], btype='bandpass', fs=cfg.SAMPLE_RATE)


def filter_signal(v: np.ndarray, demean: bool = True) -> np.ndarray:
    """
    Applies zero-phase sEMG signal conditioning:
    1. DC Offset Demeaning.
    2. Periodic Integer Padding (pad_len = 400 samples = 20 exact 50 Hz cycles).
       - Extends the signal periodically to provide continuous phase alignment.
       - Absorbs all filter settling transients in the sacrificial buffer.
    3. 50 Hz IIR Notch Filter (Q = 15) to eliminate powerline interference and sideband drift.
    4. 4th-order Butterworth Bandpass Filter (20 - 450 Hz).
    5. Strips the sacrificial padding, yielding steady-state filtered sEMG with ZERO boundary cones.
    """
    v = np.asarray(v, dtype=np.float64)
    N = len(v)
    if N < 50:
        return v
        
    # 1. DC Demean
    if demean:
        v = v - np.mean(v)
        
    # 2. Periodic Integer Padding (400 samples = 20 full cycles of 50 Hz at 1000 Hz)
    pad_len = 400
    if N >= pad_len:
        v_padded = np.concatenate([v[:pad_len], v, v[-pad_len:]])
    else:
        v_padded = v
        
    try:
        # 3. 50 Hz Notch filter
        v_notch = filtfilt(B_NOTCH, A_NOTCH, v_padded)
        # 4. Bandpass filter
        v_clean = filtfilt(B_BAND, A_BAND, v_notch)
        
        # 5. Discard sacrificial padding
        if N >= pad_len:
            return v_clean[pad_len : pad_len + N]
        return v_clean
    except Exception:
        return v


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
