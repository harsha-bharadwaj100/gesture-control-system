"""
Experiment 3: Real-Time Temporal Smoothing & Majority Voting Simulation
Simulates live inference debouncing across continuous sequential windows.
Answers:
1. How much does multi-window debouncing improve holdout test accuracy?
2. Does it reduce erratic frame-to-frame switching (jitter rate)?
3. What is the optimal filter order (K=3 vs K=5 vs K=7)?

Overfitting Guardrail:
Trained strictly on Repetitions 1-24.
Smoothing is evaluated strictly forward-in-time on unseen Repetitions 25-30.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List
from scipy.stats import mode

# Prevent Windows loky/wmic spawn hang
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# Project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

import src.config as cfg
import src.dataset as ds_builder


def apply_majority_vote(predictions: List[str], k: int) -> List[str]:
    """Applies a causal (forward-in-time) rolling majority vote filter of length k."""
    smoothed = []
    for i in range(len(predictions)):
        start_idx = max(0, i - k + 1)
        window_preds = predictions[start_idx : i + 1]
        # Compute mode
        vals, counts = np.unique(window_preds, return_counts=True)
        smoothed.append(vals[np.argmax(counts)])
    return smoothed


def compute_jitter(predictions: List[str]) -> float:
    """Computes the percentage of consecutive frames that switch prediction."""
    if len(predictions) <= 1:
        return 0.0
    switches = sum(1 for i in range(1, len(predictions)) if predictions[i] != predictions[i - 1])
    return (switches / float(len(predictions) - 1)) * 100.0


def run_experiment_3():
    print("=" * 80)
    print("EXPERIMENT 3: TEMPORAL SMOOTHING & MAJORITY VOTING SIMULATION")
    print(f"Cohort: N = {len(cfg.CLEAN_COHORT)} clean subjects")
    print("Evaluated forward-in-time strictly on Holdout Repetitions (25-30)")
    print("=" * 80)
    
    t0 = time.time()
    os.makedirs(os.path.join(PROJECT_ROOT, "experiments", "logs"), exist_ok=True)
    
    K_VALUES = [1, 3, 5, 7]
    records = []
    
    for sub in cfg.CLEAN_COHORT:
        data = ds_builder.build_subject_dataset(sub, split_rep=24, use_cache=True)
        X_tr, y_tr = data["X_train"], data["y_train"]
        X_te, y_te = data["X_test"], data["y_test"]
        meta_te = data["meta_test"]
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        
        # Train SVM (RBF) with probability support
        svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42)
        svm.fit(X_tr_scaled, y_tr)
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=1)
        rf.fit(X_tr_scaled, y_tr)
        
        # Group test predictions by individual (Gesture, Repetition) continuous sequence
        for model_name, clf in [("SVM (RBF)", svm), ("Random Forest", rf)]:
            raw_preds = clf.predict(X_te_scaled)
            
            # Organize by repetition trial sequence
            for k in K_VALUES:
                trial_smoothed_preds = []
                trial_true_labels = []
                trial_jitters = []
                
                for (gesture, rep), grp in meta_te.groupby(["Gesture", "Repetition"]):
                    indices = grp.index.values
                    trial_raw = [raw_preds[i] for i in indices]
                    trial_y = [y_te[i] for i in indices]
                    
                    if k == 1:
                        trial_smooth = trial_raw
                    else:
                        trial_smooth = apply_majority_vote(trial_raw, k=k)
                        
                    trial_smoothed_preds.extend(trial_smooth)
                    trial_true_labels.extend(trial_y)
                    trial_jitters.append(compute_jitter(trial_smooth))
                    
                acc = accuracy_score(trial_true_labels, trial_smoothed_preds) * 100.0
                macro_f1 = f1_score(trial_true_labels, trial_smoothed_preds, average="macro", zero_division=0) * 100.0
                mean_jitter = np.mean(trial_jitters)
                
                records.append({
                    "Participant": sub,
                    "Model": model_name,
                    "K_Window": k,
                    "Latency_Buffer_ms": (k - 1) * cfg.WINDOW_STEP_MS,
                    "Accuracy": acc,
                    "Macro_F1": macro_f1,
                    "Jitter_Rate": mean_jitter
                })
                
    df_results = pd.DataFrame(records)
    csv_path = os.path.join(PROJECT_ROOT, "experiments", "logs", "exp3_temporal_smoothing.csv")
    df_results.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 3 SUMMARY: IMPACT OF TEMPORAL SMOOTHING ON HOLDOUT TEST ACCURACY")
    print("=" * 80)
    summary = df_results.groupby(["Model", "K_Window", "Latency_Buffer_ms"])[["Accuracy", "Macro_F1", "Jitter_Rate"]].agg(["mean", "std"])
    print(summary.to_string())
    
    print(f"\nExperiment 3 completed in {time.time() - t0:.1f}s.")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    run_experiment_3()
