"""
Experiment 4: Hyperparameter Optimization via Multi-Core Parallel Nested Repetition CV
Parallelized across 6-7 CPU cores for maximum speed with zero compromise.
Answers:
1. What are the optimal hyperparameters for SVM, Random Forest, and LDA?
2. Does Ledoit-Wolf shrinkage improve LDA on high-dimensional sEMG features?
3. How much generalization gain is achieved on holdout test data?

Overfitting Guardrail:
Hyperparameter selection is performed STRICTLY inside Training Repetitions (1-24)
using 4-Fold Repetition GroupKFold cross-validation.
The selected configuration is evaluated ONCE on Holdout Repetitions (25-30).
"""

import os
import sys
import time
import concurrent.futures
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Prevent nested loky conflict
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# Project root path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

import src.config as cfg
import src.dataset as ds_builder


# Full Systematic Hyperparameter Grids (Zero Compromise)
SVM_PARAM_GRID = [
    {"C": 0.5, "gamma": "scale", "kernel": "rbf"},
    {"C": 1.0, "gamma": "scale", "kernel": "rbf"},
    {"C": 5.0, "gamma": "scale", "kernel": "rbf"},
    {"C": 10.0, "gamma": "scale", "kernel": "rbf"},
    {"C": 25.0, "gamma": "scale", "kernel": "rbf"},
    {"C": 1.0, "kernel": "linear"},
    {"C": 5.0, "kernel": "linear"}
]

RF_PARAM_GRID = [
    {"n_estimators": 100, "max_depth": 8,  "min_samples_leaf": 2},
    {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": 16, "min_samples_leaf": 1},
    {"n_estimators": 150, "max_depth": 16, "min_samples_leaf": 1},
    {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
]

LDA_PARAM_GRID = [
    {"solver": "svd", "shrinkage": None},
    {"solver": "lsqr", "shrinkage": "auto"}
]


def evaluate_grid_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model_cls, param_grid: List[Dict]) -> Tuple[Dict, float]:
    """Evaluates parameter grid using 4-fold Repetition GroupKFold on training data."""
    gkf = GroupKFold(n_splits=4)
    best_score = -1.0
    best_params = param_grid[0]
    
    for params in param_grid:
        fold_scores = []
        for tr_idx, val_idx in gkf.split(X, y, groups):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_val = scaler.transform(X[val_idx])
            
            p_clean = params.copy()
            if model_cls == RandomForestClassifier:
                p_clean["n_jobs"] = 1
                p_clean["random_state"] = 42
            elif model_cls == SVC:
                p_clean["random_state"] = 42
                
            clf = model_cls(**p_clean)
            clf.fit(X_tr, y[tr_idx])
            fold_scores.append(accuracy_score(y[val_idx], clf.predict(X_val)))
            
        mean_cv = float(np.mean(fold_scores))
        if mean_cv > best_score:
            best_score = mean_cv
            best_params = params
            
    return best_params, best_score


def process_subject_tuning(sub: str) -> Dict:
    """Worker function executed across parallel CPU worker processes."""
    p_t0 = time.time()
    data = ds_builder.build_subject_dataset(sub, split_rep=24, use_cache=True)
    
    X_tr, y_tr = data["X_train"], data["y_train"]
    groups_tr = data["meta_train"]["Repetition"].values
    X_te, y_te = data["X_test"], data["y_test"]
    
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    
    # 1. SVM Tuning
    best_svm_params, svm_cv = evaluate_grid_cv(X_tr, y_tr, groups_tr, SVC, SVM_PARAM_GRID)
    p_svm = best_svm_params.copy()
    p_svm["random_state"] = 42
    svm_tuned = SVC(**p_svm)
    svm_tuned.fit(X_tr_sc, y_tr)
    acc_svm_test = accuracy_score(y_te, svm_tuned.predict(X_te_sc)) * 100.0
    
    svm_def = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)
    svm_def.fit(X_tr_sc, y_tr)
    acc_svm_def = accuracy_score(y_te, svm_def.predict(X_te_sc)) * 100.0
    
    # 2. Random Forest Tuning
    best_rf_params, rf_cv = evaluate_grid_cv(X_tr, y_tr, groups_tr, RandomForestClassifier, RF_PARAM_GRID)
    p_rf = best_rf_params.copy()
    p_rf["n_jobs"] = 1
    p_rf["random_state"] = 42
    rf_tuned = RandomForestClassifier(**p_rf)
    rf_tuned.fit(X_tr_sc, y_tr)
    acc_rf_test = accuracy_score(y_te, rf_tuned.predict(X_te_sc)) * 100.0
    
    rf_def = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=1)
    rf_def.fit(X_tr_sc, y_tr)
    acc_rf_def = accuracy_score(y_te, rf_def.predict(X_te_sc)) * 100.0
    
    # 3. LDA Tuning
    best_lda_params, lda_cv = evaluate_grid_cv(X_tr, y_tr, groups_tr, LinearDiscriminantAnalysis, LDA_PARAM_GRID)
    lda_tuned = LinearDiscriminantAnalysis(**best_lda_params)
    lda_tuned.fit(X_tr_sc, y_tr)
    acc_lda_test = accuracy_score(y_te, lda_tuned.predict(X_te_sc)) * 100.0
    
    lda_def = LinearDiscriminantAnalysis()
    lda_def.fit(X_tr_sc, y_tr)
    acc_lda_def = accuracy_score(y_te, lda_def.predict(X_te_sc)) * 100.0
    
    elapsed = time.time() - p_t0
    print(f"[{sub:15s}] Finished in {elapsed:4.1f}s | SVM: {acc_svm_def:4.1f}% -> {acc_svm_test:4.1f}% | RF: {acc_rf_def:4.1f}% -> {acc_rf_test:4.1f}% | LDA: {acc_lda_def:4.1f}% -> {acc_lda_test:4.1f}%", flush=True)
    
    return {
        "Participant": sub,
        "SVM_Best_Params": str(best_svm_params),
        "SVM_CV_Train": round(svm_cv * 100, 2),
        "SVM_Default_Test": round(acc_svm_def, 2),
        "SVM_Tuned_Test": round(acc_svm_test, 2),
        "RF_Best_Params": str(best_rf_params),
        "RF_CV_Train": round(rf_cv * 100, 2),
        "RF_Default_Test": round(acc_rf_def, 2),
        "RF_Tuned_Test": round(acc_rf_test, 2),
        "LDA_Best_Params": str(best_lda_params),
        "LDA_CV_Train": round(lda_cv * 100, 2),
        "LDA_Default_Test": round(acc_lda_def, 2),
        "LDA_Tuned_Test": round(acc_lda_test, 2),
        "Execution_Time_s": round(elapsed, 2)
    }


def run_experiment_4():
    max_workers = min(6, os.cpu_count() or 4)
    print("=" * 80, flush=True)
    print(f"EXPERIMENT 4: MULTI-CORE PARALLEL HYPERPARAMETER OPTIMIZATION (WORKERS = {max_workers})", flush=True)
    print(f"Cohort: N = {len(cfg.CLEAN_COHORT)} clean subjects | Repetition GroupKFold on Train (1-24)", flush=True)
    print("Zero Compromise: Testing full 7 SVM configs, 5 RF configs, 2 LDA configs in parallel", flush=True)
    print("=" * 80, flush=True)
    
    t0 = time.time()
    os.makedirs(os.path.join(PROJECT_ROOT, "experiments", "logs"), exist_ok=True)
    
    # Run across parallel processes
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_subject_tuning, sub): sub for sub in cfg.CLEAN_COHORT}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            records.append(res)
            
    df_tuning = pd.DataFrame(records)
    # Sort by participant name for clean presentation
    df_tuning = df_tuning.sort_values(by="Participant").reset_index(drop=True)
    
    csv_path = os.path.join(PROJECT_ROOT, "experiments", "logs", "exp4_model_tuning.csv")
    df_tuning.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 80, flush=True)
    print("HYPERPARAMETER OPTIMIZATION SUMMARY (DEFAULT VS TUNED TEST ACCURACY)", flush=True)
    print("=" * 80, flush=True)
    print(df_tuning[["Participant", "SVM_Default_Test", "SVM_Tuned_Test", "RF_Default_Test", "RF_Tuned_Test", "LDA_Default_Test", "LDA_Tuned_Test"]].to_string(index=False), flush=True)
    
    print("\n" + "-" * 60, flush=True)
    print("COHORT AVERAGE IMPACT:", flush=True)
    print(f"SVM Mean: {df_tuning['SVM_Default_Test'].mean():.2f}% -> {df_tuning['SVM_Tuned_Test'].mean():.2f}%", flush=True)
    print(f"RF  Mean: {df_tuning['RF_Default_Test'].mean():.2f}% -> {df_tuning['RF_Tuned_Test'].mean():.2f}%", flush=True)
    print(f"LDA Mean: {df_tuning['LDA_Default_Test'].mean():.2f}% -> {df_tuning['LDA_Tuned_Test'].mean():.2f}%", flush=True)
    print("-" * 60, flush=True)
    
    print(f"\nExperiment 4 completed in {time.time() - t0:.1f}s.", flush=True)
    print(f"Results saved to: {csv_path}", flush=True)


if __name__ == "__main__":
    run_experiment_4()
