"""
Experiment 2: Window Length & Step Size Sensitivity Grid (Multi-Core Parallel Execution)
Evaluates the trade-off between temporal resolution (controller latency) and classification accuracy.
Parallelized across 6-7 CPU cores for maximum speed with ZERO compromise.

Full 8-Configuration Grid:
- Window Lengths (W): 200 ms, 350 ms, 500 ms, 750 ms
- Step Sizes (S): 50 ms, 100 ms, 150 ms

Overfitting Guardrail:
Trained strictly on Repetitions 1-24.
Evaluated strictly on Holdout Repetitions 25-30.
"""

import os
import sys
import time
import concurrent.futures
import numpy as np
import pandas as pd
from typing import Dict, List

# Prevent nested loky conflict
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# Project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import src.config as cfg
import src.preprocessing as prep
import src.dataset as ds_builder


# Full 8-Configuration Grid (Zero Compromise)
FULL_GRID_CONFIGS = [
    {"W": 200, "S": 50,  "label": "W=200ms, S=50ms (20 Hz)"},
    {"W": 200, "S": 100, "label": "W=200ms, S=100ms (10 Hz)"},
    {"W": 350, "S": 50,  "label": "W=350ms, S=50ms (20 Hz)"},
    {"W": 350, "S": 100, "label": "W=350ms, S=100ms (10 Hz)"},
    {"W": 500, "S": 50,  "label": "W=500ms, S=50ms (20 Hz)"},
    {"W": 500, "S": 100, "label": "W=500ms, S=100ms (10 Hz) [Default]"},
    {"W": 500, "S": 150, "label": "W=500ms, S=150ms (6.7 Hz)"},
    {"W": 750, "S": 100, "label": "W=750ms, S=100ms (10 Hz)"},
]


def process_subject_grid(sub: str) -> List[Dict]:
    """Evaluates all 8 grid configurations for a single subject on an independent CPU core."""
    p_t0 = time.time()
    trials = prep.load_participant_trials(sub, apply_filter=True, use_cache=True)
    
    start_sample = int(cfg.ACTIVE_START_SEC * cfg.SAMPLE_RATE)
    end_sample = int(cfg.ACTIVE_END_SEC * cfg.SAMPLE_RATE)
    
    sub_records = []
    
    for grid_item in FULL_GRID_CONFIGS:
        W_ms = grid_item["W"]
        S_ms = grid_item["S"]
        lbl = grid_item["label"]
        
        w_samples = int((W_ms / 1000.0) * cfg.SAMPLE_RATE)
        s_samples = int((S_ms / 1000.0) * cfg.SAMPLE_RATE)
        
        X_tr, y_tr, X_te, y_te = [], [], [], []
        for (gesture, rep), sig in trials.items():
            if gesture not in cfg.GESTURES:
                continue
            feats, _ = ds_builder.extract_windows_from_trial(
                sig,
                window_size=w_samples,
                step_size=s_samples,
                start_sample=start_sample,
                end_sample=end_sample
            )
            if rep <= 24:
                X_tr.extend(feats)
                y_tr.extend([gesture] * len(feats))
            else:
                X_te.extend(feats)
                y_te.extend([gesture] * len(feats))
                
        X_tr = np.array(X_tr, dtype=np.float64)
        X_te = np.array(X_te, dtype=np.float64)
        y_tr = np.array(y_tr)
        y_te = np.array(y_te)
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        
        # SVM evaluation
        svm = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)
        svm.fit(X_tr_sc, y_tr)
        acc_svm = accuracy_score(y_te, svm.predict(X_te_sc)) * 100.0
        
        # Random Forest evaluation
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=1)
        rf.fit(X_tr_sc, y_tr)
        acc_rf = accuracy_score(y_te, rf.predict(X_te_sc)) * 100.0
        
        sub_records.append({
            "Participant": sub,
            "Window_ms": W_ms,
            "Step_ms": S_ms,
            "Label": lbl,
            "Num_Train_Windows": len(X_tr),
            "Num_Test_Windows": len(X_te),
            "SVM_Accuracy": round(acc_svm, 2),
            "RF_Accuracy": round(acc_rf, 2)
        })
        
    print(f"[{sub:15s}] Completed all 8 grid configurations in {time.time() - p_t0:4.1f}s", flush=True)
    return sub_records


def run_experiment_2():
    max_workers = min(6, os.cpu_count() or 4)
    print("=" * 80, flush=True)
    print(f"EXPERIMENT 2: MULTI-CORE PARALLEL WINDOWING GRID SEARCH (WORKERS = {max_workers})", flush=True)
    print(f"Cohort: N = {len(cfg.CLEAN_COHORT)} clean subjects | Full 8 Configurations (Zero Compromise)", flush=True)
    print("Testing W in [200, 350, 500, 750] ms and S in [50, 100, 150] ms", flush=True)
    print("=" * 80, flush=True)
    
    t0 = time.time()
    os.makedirs(os.path.join(PROJECT_ROOT, "experiments", "logs"), exist_ok=True)
    
    all_records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_subject_grid, sub): sub for sub in cfg.CLEAN_COHORT}
        for future in concurrent.futures.as_completed(futures):
            sub_results = future.result()
            all_records.extend(sub_results)
            
    df_grid = pd.DataFrame(all_records)
    csv_path = os.path.join(PROJECT_ROOT, "experiments", "logs", "exp2_windowing_grid.csv")
    df_grid.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 80, flush=True)
    print("FULL WINDOWING GRID SUMMARY: HOLDOUT TEST ACCURACY ACROSS ALL 12 PARTICIPANTS", flush=True)
    print("=" * 80, flush=True)
    summary = df_grid.groupby(["Window_ms", "Step_ms", "Label"])[["SVM_Accuracy", "RF_Accuracy"]].agg(["mean", "std"])
    summary = summary.sort_values(by=("SVM_Accuracy", "mean"), ascending=False)
    print(summary.to_string(), flush=True)
    
    print(f"\nExperiment 2 completed in {time.time() - t0:.1f}s.", flush=True)
    print(f"Results saved to: {csv_path}", flush=True)


if __name__ == "__main__":
    run_experiment_2()
