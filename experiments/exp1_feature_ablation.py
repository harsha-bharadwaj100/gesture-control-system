"""
Experiment 1: Feature Importance Ranking & Sub-Domain Ablation Study
Answers:
1. Which of the 16 features contribute the most genuine discriminative power?
2. How does each biomechanical sub-domain perform in isolation?
3. Can we reduce feature dimensionality without losing accuracy?

Overfitting Guardrail:
Feature importances and rankings are computed strictly on Training Repetitions (1-24).
Evaluation is performed on Holdout Repetitions (25-30).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from typing import Dict, List

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
import src.features as feat_mod

FEATURE_NAMES = feat_mod.FEATURE_NAMES

FEATURE_SUBDOMAINS = {
    "All 16 Features": list(range(16)),
    "Energy & Amplitude (5)": [0, 1, 2, 3, 4],            # RMS, MAV, WL, VAR, SSI
    "Complexity & Dynamics (3)": [5, 6, 7],               # ZC, SSC, WAMP
    "Statistical Shape (2)": [8, 9],                      # ABS_SKEW, KURT
    "Spectral & Frequency (6)": [10, 11, 12, 13, 14, 15],  # MNF, MDF, SE, PKF, MNP, TTP
    "Energy + Complexity (8)": [0, 1, 2, 3, 4, 5, 6, 7],
    "Energy + Spectral (11)": [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 15]
}


def run_experiment_1():
    print("=" * 80)
    print("EXPERIMENT 1: FEATURE IMPORTANCE & SUB-DOMAIN ABLATION STUDY")
    print(f"Cohort: N = {len(cfg.CLEAN_COHORT)} clean subjects")
    print("Train Reps: 1-24 | Holdout Test Reps: 25-30 | Theoretical Chance: 20.00%")
    print("=" * 80)
    
    t0 = time.time()
    os.makedirs(os.path.join(PROJECT_ROOT, "experiments", "logs"), exist_ok=True)
    
    # 1. Feature Importance Collection (MDI) across all subjects
    feature_importances = []
    subject_datasets = {}
    
    print("\nPhase 1: Computing Feature Importance on Training Repetitions (1-24)...")
    for sub in cfg.CLEAN_COHORT:
        data = ds_builder.build_subject_dataset(sub, split_rep=24, use_cache=True)
        subject_datasets[sub] = data
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(data["X_train"])
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=1)
        rf.fit(X_tr, data["y_train"])
        feature_importances.append(rf.feature_importances_)
        
    mean_importances = np.mean(feature_importances, axis=0)
    std_importances = np.std(feature_importances, axis=0)
    
    # Create Feature Ranking DataFrame
    df_ranking = pd.DataFrame({
        "Rank": np.argsort(-mean_importances) + 1,
        "Feature": [FEATURE_NAMES[i] for i in np.argsort(-mean_importances)],
        "Importance_Mean": np.sort(mean_importances)[::-1],
        "Importance_Std": std_importances[np.argsort(-mean_importances)],
        "Index": np.argsort(-mean_importances)
    })
    
    print("\n" + "-" * 60)
    print("COHORT-WIDE FEATURE IMPORTANCE RANKING (Random Forest MDI on Train):")
    print("-" * 60)
    for idx, row in df_ranking.iterrows():
        bar = "=" * int(row["Importance_Mean"] * 100)
        print(f"#{idx+1:02d} | {row['Feature']:10s} : {row['Importance_Mean']*100:5.2f}% +/- {row['Importance_Std']*100:4.2f}% | {bar}")
        
    top_8_indices = list(df_ranking["Index"].values[:8])
    FEATURE_SUBDOMAINS["Top 8 Ranked Features"] = top_8_indices
    
    # 2. Sub-Domain Ablation Study
    print("\nPhase 2: Evaluating Sub-Domain Ablation on Holdout Repetitions (25-30)...")
    ablation_records = []
    
    for sub in cfg.CLEAN_COHORT:
        data = subject_datasets[sub]
        X_tr_full = data["X_train"]
        y_tr = data["y_train"]
        X_te_full = data["X_test"]
        y_te = data["y_test"]
        
        for domain_name, indices in FEATURE_SUBDOMAINS.items():
            X_tr_sub = X_tr_full[:, indices]
            X_te_sub = X_te_full[:, indices]
            
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_sub)
            X_te_scaled = scaler.transform(X_te_sub)
            
            # Evaluate RF
            rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=1)
            rf.fit(X_tr_scaled, y_tr)
            acc_rf = accuracy_score(y_te, rf.predict(X_te_scaled)) * 100.0
            
            # Evaluate SVM
            svm = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42)
            svm.fit(X_tr_scaled, y_tr)
            acc_svm = accuracy_score(y_te, svm.predict(X_te_scaled)) * 100.0
            
            ablation_records.append({
                "Participant": sub,
                "Domain": domain_name,
                "Num_Features": len(indices),
                "RF_Accuracy": acc_rf,
                "SVM_Accuracy": acc_svm
            })
            
    df_ablation = pd.DataFrame(ablation_records)
    
    # Save results to log
    csv_path = os.path.join(PROJECT_ROOT, "experiments", "logs", "exp1_feature_ablation.csv")
    df_ablation.to_csv(csv_path, index=False)
    
    ranking_csv_path = os.path.join(PROJECT_ROOT, "experiments", "logs", "exp1_feature_ranking.csv")
    df_ranking.to_csv(ranking_csv_path, index=False)
    
    # Summary Table
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY: TEST ACCURACY ACROSS ALL 12 PARTICIPANTS")
    print("=" * 80)
    summary = df_ablation.groupby("Domain")[["Num_Features", "RF_Accuracy", "SVM_Accuracy"]].agg({
        "Num_Features": "first",
        "RF_Accuracy": ["mean", "std"],
        "SVM_Accuracy": ["mean", "std"]
    })
    # Sort by SVM Accuracy mean descending
    summary = summary.sort_values(by=("SVM_Accuracy", "mean"), ascending=False)
    print(summary.to_string())
    
    print(f"\nExperiment 1 completed in {time.time() - t0:.1f}s.")
    print(f"Results saved to:\n  - {csv_path}\n  - {ranking_csv_path}")


if __name__ == "__main__":
    run_experiment_1()
