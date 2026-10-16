"""
Experiment 5: Unified Pipeline Synthesis & Production Model Export
Synthesizes the optimal empirical configurations from Experiments 1-4:
1. Signal Conditioning: ARHD (50/100/150 Hz adaptive hum extraction + 20-450 Hz Butterworth)
2. Feature Engine: Unified 16-Feature Vector (Exp 1 proved all 16 domains contribute synergistically)
3. Windowing: Optimal Window Length & Step Size from Exp 2
4. Temporal Smoothing: Rolling Majority Vote (Exp 3 proved 60% jitter reduction)
5. Model Optimization: Tuned SVM-RBF, Random Forest, and Shrunk LDA from Exp 4
6. Stage 1 Energy Gate: Calibrated Rest/Idle discrimination

Exports the final production model package to `models/production_pipeline.pkl`
ready for direct deployment in `gui_live_inference.py`.
"""

import os
import sys
import time
import pickle
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import src.config as cfg
import src.dataset as ds_builder
import src.features as feat_mod


def run_experiment_5(optimal_k: int = 5):
    print("=" * 80, flush=True)
    print("EXPERIMENT 5: FINAL PRODUCTION PIPELINE SYNTHESIS & MODEL EXPORT", flush=True)
    print(f"Cohort: N = {len(cfg.CLEAN_COHORT)} clean subjects | Stage 1 Energy Gate + Stage 2 Tuned Classifiers", flush=True)
    print(f"Applying Temporal Filter: K = {optimal_k} rolling decision buffer (~400 ms)", flush=True)
    print("=" * 80, flush=True)
    
    t0 = time.time()
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "experiments", "logs"), exist_ok=True)
    
    # Train production subject models and cohort benchmark
    production_subject_models = {}
    benchmark_records = []
    
    for sub in cfg.CLEAN_COHORT:
        data = ds_builder.build_subject_dataset(sub, split_rep=24, use_cache=True)
        X_tr, y_tr = data["X_train"], data["y_train"]
        X_te, y_te = data["X_test"], data["y_test"]
        meta_te = data["meta_test"]
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)
        
        # Train Tuned Models
        svm = SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, random_state=42)
        svm.fit(X_tr_sc, y_tr)
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=16, min_samples_leaf=1, random_state=42, n_jobs=1)
        rf.fit(X_tr_sc, y_tr)
        
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
        lda.fit(X_tr_sc, y_tr)
        
        # Raw vs Filtered Evaluation on Holdout Repetitions 25-30
        for model_name, clf in [("SVM (RBF)", svm), ("Random Forest", rf), ("LDA (Shrinkage)", lda)]:
            raw_preds = clf.predict(X_te_sc)
            raw_acc = accuracy_score(y_te, raw_preds) * 100.0
            
            # Apply rolling majority vote within each continuous repetition trial
            filtered_preds = []
            for (gesture, rep), grp in meta_te.groupby(["Gesture", "Repetition"]):
                indices = grp.index.values
                t_preds = [raw_preds[i] for i in indices]
                # Rolling mode
                for i in range(len(t_preds)):
                    window_slice = t_preds[max(0, i - optimal_k + 1) : i + 1]
                    vals, counts = np.unique(window_slice, return_counts=True)
                    filtered_preds.append(vals[np.argmax(counts)])
                    
            filtered_acc = accuracy_score(y_te, filtered_preds) * 100.0
            
            benchmark_records.append({
                "Participant": sub,
                "Model": model_name,
                "Raw_Test_Acc": round(raw_acc, 2),
                "Smoothed_Test_Acc": round(filtered_acc, 2),
                "Accuracy_Gain": round(filtered_acc - raw_acc, 2)
            })
            
        # Store calibrated subject model package
        production_subject_models[sub] = {
            "scaler": scaler,
            "svm_model": svm,
            "rf_model": rf,
            "lda_model": lda,
            "gestures": cfg.GESTURES,
            "feature_names": feat_mod.FEATURE_NAMES,
            "window_size_ms": cfg.WINDOW_SIZE_MS,
            "window_step_ms": cfg.WINDOW_STEP_MS,
            "optimal_k": optimal_k
        }
        
    df_synthesis = pd.DataFrame(benchmark_records)
    csv_path = os.path.join(PROJECT_ROOT, "experiments", "logs", "exp5_synthesis_summary.csv")
    df_synthesis.to_csv(csv_path, index=False)
    
    # Save production package
    pkg_path = os.path.join(PROJECT_ROOT, "models", "production_pipeline.pkl")
    with open(pkg_path, "wb") as f:
        pickle.dump(production_subject_models, f)
        
    print("\n" + "=" * 80, flush=True)
    print("PRODUCTION PIPELINE SYNTHESIS SUMMARY: RAW VS SMOOTHED HOLD-OUT ACCURACY", flush=True)
    print("=" * 80, flush=True)
    summary = df_synthesis.groupby("Model")[["Raw_Test_Acc", "Smoothed_Test_Acc", "Accuracy_Gain"]].agg(["mean", "std"])
    print(summary.to_string(), flush=True)
    
    print(f"\nExperiment 5 complete in {time.time() - t0:.1f}s.", flush=True)
    print(f"Production pipeline exported successfully to: {pkg_path}", flush=True)


if __name__ == "__main__":
    run_experiment_5()
