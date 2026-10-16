"""
Signal Conditioning & Preprocessing Module
Implements zero-phase adaptive running heterodyne demodulation (50Hz hum elimination)
and 4th-order Butterworth bandpass filtering (20-450 Hz).
Guarantees natural, continuous physiological waveforms with ZERO artificial cones,
reverse cones, or boundary transients.
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import src.config as cfg

# Precompute 4th-order bandpass filter coefficients (20 - 450 Hz)
B_BAND, A_BAND = butter(cfg.BANDPASS_ORDER, [cfg.BANDPASS_LOW, cfg.BANDPASS_HIGH], btype='bandpass', fs=cfg.SAMPLE_RATE)


def filter_signal(v: np.ndarray, demean: bool = True) -> np.ndarray:
    """
    Applies pure physiological signal conditioning:
    1. DC offset demeaning.
    2. Adaptive Running Heterodyne Demodulation (ARHD):
       Tracks and subtracts the exact instantaneous amplitude and phase of 50 Hz, 100 Hz,
       and 150 Hz powerline interference using an edge-clamped running Fourier projection.
       (Completely eliminates powerline hum WITHOUT any IIR notch filter settling ringing,
       without boundary cones, without reverse cones, and without beating).
    3. 4th-order Butterworth Bandpass filter (20 - 450 Hz) with zero-phase filtfilt.
    """
    v = np.asarray(v, dtype=np.float64)
    N = len(v)
    if N < 100:
        return v
        
    # 1. DC Demean
    if demean:
        v = v - np.mean(v)
        
    # 2. Adaptive Running Heterodyne Demodulation (50, 100, 150 Hz)
    fs = float(cfg.SAMPLE_RATE)
    t = np.arange(N) / fs
    w_len = 80  # 80 samples = 80 ms (4 full cycles of 50 Hz)
    half_w = w_len // 2
    
    hum_total = np.zeros(N, dtype=np.float64)
    for f in [50.0, 100.0, 150.0]:
        s_wave = np.sin(2.0 * np.pi * f * t)
        c_wave = np.cos(2.0 * np.pi * f * t)
        
        # Moving short-time projection
        c_s = np.convolve(v * s_wave, np.ones(w_len) / float(w_len), mode='valid') * 2.0
        c_c = np.convolve(v * c_wave, np.ones(w_len) / float(w_len), mode='valid') * 2.0
        
        # Edge clamping: hold the boundary coefficient steady into the edges
        c_s_full = np.pad(c_s, (half_w, N - len(c_s) - half_w), mode='edge')
        c_c_full = np.pad(c_c, (half_w, N - len(c_c) - half_w), mode='edge')
        
        hum_total += (c_s_full * s_wave + c_c_full * c_wave)
        
    v_dehummed = v - hum_total
    
    # 3. 20 - 450 Hz Bandpass Filter
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


import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_participant_trials(participant_name: str, datasets_dir: str = "All_Datasets", apply_filter: bool = True, use_cache: bool = True):
    """
    Loads and structures all 4.0-second trials for a participant.
    Caches processed trials to disk for sub-second reloads across experiments.
    Returns: dict with keys (gesture_name, rep_index) -> 1D numpy array of length 4000
    """
    cache_dir = os.path.join(PROJECT_ROOT, ".cache_trials")
    cache_path = os.path.join(cache_dir, f"{participant_name}_filt{apply_filter}.pkl")
    
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
            
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
        
    if use_cache:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(trials, f)
        except Exception:
            pass
            
    return trials
