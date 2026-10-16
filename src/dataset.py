"""
Dataset Generation and Window Extraction Module
Extracts sliding analysis windows (500 ms length, 100 ms stride) across the stable
contraction plateau (0.6s to 3.4s) and computes the unified 16-feature vector.
Implements strict zero-leakage repetition partitioning (Reps 1-24 Train vs Reps 25-30 Test)
and conservative physical outlier rejection (<1% target).
"""

import os
import sys

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
import src.config as cfg
import src.preprocessing as prep
import src.features as feat


def extract_windows_from_trial(
    signal: np.ndarray,
    window_size: int = 500,
    step_size: int = 100,
    start_sample: int = 600,
    end_sample: int = 3400,
    max_saturation_ratio: float = 0.05,
    min_rms_voltage: float = 1e-4
) -> Tuple[List[np.ndarray], int]:
    """
    Slices the active plateau of a 4-second trial into sliding windows and extracts 16 features.
    Applies conservative physical validity checks (discarding only severe sensor saturation or disconnected flatlines).
    
    Returns:
        features_list: List of 1D numpy arrays (each shape (16,))
        num_rejected: Count of corrupted windows dropped
    """
    if len(signal) < end_sample:
        return [], 0

    active_sig = signal[start_sample:end_sample]
    total_len = len(active_sig)
    
    features = []
    rejected = 0
    
    for i in range(0, total_len - window_size + 1, step_size):
        w = active_sig[i : i + window_size]
        
        # 1. Conservative Outlier Check: Hardware rail saturation (|V| >= 4.95V)
        sat_samples = np.sum(np.abs(w) >= 4.95)
        if sat_samples / float(window_size) > max_saturation_ratio:
            rejected += 1
            continue
            
        # 2. Conservative Outlier Check: Complete zero dead flatline (disconnected sensor)
        rms = np.sqrt(np.mean(w ** 2))
        if rms < min_rms_voltage:
            rejected += 1
            continue
            
        # Extract unified 16 features
        f = feat.extract_features(w, fs=float(cfg.SAMPLE_RATE))
        
        # Check for non-finite values (safety check)
        if not np.all(np.isfinite(f)):
            rejected += 1
            continue
            
        features.append(f)
        
    return features, rejected


import pickle

def build_subject_dataset(
    participant_name: str,
    split_rep: int = 24,
    datasets_dir: str = "All_Datasets",
    use_cache: bool = True
) -> Dict:
    """
    Builds the train/test feature sets for a single participant with strict repetition-level partitioning.
    
    Parameters:
        participant_name: Name of participant (e.g. 'Arundhati', 'chinmaya')
        split_rep: Repetition cutoff (1 to split_rep for Train, split_rep+1 to 30 for Test)
        datasets_dir: Directory containing participant CSVs
        use_cache: Whether to cache feature vectors to disk for fast reload
        
    Returns:
        Dictionary containing:
            'X_train': (N_train, 16) array
            'y_train': (N_train,) array of gesture labels
            'meta_train': DataFrame with [Gesture, Repetition, Window_Idx]
            'X_test': (N_test, 16) array
            'y_test': (N_test,) array of gesture labels
            'meta_test': DataFrame with [Gesture, Repetition, Window_Idx]
            'rejected_count': Total corrupted windows discarded
    """
    cache_dir = os.path.join(PROJECT_ROOT, ".cache_datasets")
    cache_path = os.path.join(cache_dir, f"{participant_name}_split{split_rep}.pkl")
    
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass  # Recompute if cache corrupted

    start_sample = int(cfg.ACTIVE_START_SEC * cfg.SAMPLE_RATE)
    end_sample = int(cfg.ACTIVE_END_SEC * cfg.SAMPLE_RATE)
    win_size = int((cfg.WINDOW_SIZE_MS / 1000.0) * cfg.SAMPLE_RATE)
    step_size = int((cfg.WINDOW_STEP_MS / 1000.0) * cfg.SAMPLE_RATE)
    
    trials = prep.load_participant_trials(participant_name, datasets_dir=datasets_dir, apply_filter=True)
    
    X_train_list, y_train_list, meta_train_list = [], [], []
    X_test_list, y_test_list, meta_test_list = [], [], []
    total_rejected = 0
    
    for (gesture, rep), sig in trials.items():
        if gesture not in cfg.GESTURES:
            continue
            
        feats, n_rej = extract_windows_from_trial(
            sig,
            window_size=win_size,
            step_size=step_size,
            start_sample=start_sample,
            end_sample=end_sample
        )
        total_rejected += n_rej
        
        is_train = (rep <= split_rep)
        for w_idx, f_vec in enumerate(feats):
            meta_record = {
                "Participant": participant_name,
                "Gesture": gesture,
                "Repetition": rep,
                "Window_Idx": w_idx
            }
            if is_train:
                X_train_list.append(f_vec)
                y_train_list.append(gesture)
                meta_train_list.append(meta_record)
            else:
                X_test_list.append(f_vec)
                y_test_list.append(gesture)
                meta_test_list.append(meta_record)
                
    result = {
        "participant": participant_name,
        "X_train": np.array(X_train_list, dtype=np.float64),
        "y_train": np.array(y_train_list),
        "meta_train": pd.DataFrame(meta_train_list),
        "X_test": np.array(X_test_list, dtype=np.float64),
        "y_test": np.array(y_test_list),
        "meta_test": pd.DataFrame(meta_test_list),
        "rejected_count": total_rejected
    }
    
    if use_cache:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
        except Exception:
            pass
            
    return result


def build_cohort_dataset(cohort: Optional[List[str]] = None, split_rep: int = 24) -> Dict:
    """
    Builds the aggregated dataset across all participants in the cohort.
    
    Returns:
        Dictionary with combined 'X_train', 'y_train', 'meta_train',
        'X_test', 'y_test', 'meta_test', and per-subject breakdown.
    """
    if cohort is None:
        cohort = cfg.CLEAN_COHORT
        
    all_X_train, all_y_train, all_meta_train = [], [], []
    all_X_test, all_y_test, all_meta_test = [], [], []
    subject_stats = {}
    
    for participant in cohort:
        ds = build_subject_dataset(participant, split_rep=split_rep)
        subject_stats[participant] = {
            "n_train_windows": len(ds["X_train"]),
            "n_test_windows": len(ds["X_test"]),
            "n_rejected_windows": ds["rejected_count"]
        }
        all_X_train.append(ds["X_train"])
        all_y_train.append(ds["y_train"])
        all_meta_train.append(ds["meta_train"])
        
        all_X_test.append(ds["X_test"])
        all_y_test.append(ds["y_test"])
        all_meta_test.append(ds["meta_test"])
        
    return {
        "cohort": cohort,
        "X_train": np.vstack(all_X_train),
        "y_train": np.concatenate(all_y_train),
        "meta_train": pd.concat(all_meta_train, ignore_index=True),
        "X_test": np.vstack(all_X_test),
        "y_test": np.concatenate(all_y_test),
        "meta_test": pd.concat(all_meta_test, ignore_index=True),
        "subject_stats": subject_stats
    }


if __name__ == "__main__":
    print(f"Testing dataset builder on first clean subject: {cfg.CLEAN_COHORT[0]}...")
    data = build_subject_dataset(cfg.CLEAN_COHORT[0])
    print(f"Participant: {data['participant']}")
    print(f"X_train shape: {data['X_train'].shape}, y_train shape: {data['y_train'].shape}")
    print(f"X_test shape:  {data['X_test'].shape},  y_test shape:  {data['y_test'].shape}")
    print(f"Corrupted windows rejected: {data['rejected_count']}")
