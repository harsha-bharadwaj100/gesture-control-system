"""
Multi-Model Benchmark Evaluation Suite
Evaluates 5 standard biomedical pattern recognition classifiers across the 12-subject clean cohort:
1. Random Forest (RF)
2. Support Vector Machine (SVM with RBF kernel)
3. Histogram Gradient Boosting (HGB)
4. Linear Discriminant Analysis (LDA)
5. k-Nearest Neighbors (k-NN, k=5)

Strictly enforces zero-leakage repetition partitioning:
- Train: Repetitions 1 to 24 (80%)
- Test:  Repetitions 25 to 30 (20%, completely unseen hold-out)
Compares performance rigorously against the true 20.00% theoretical chance level.
"""

import os
import sys

# Prevent joblib/loky subprocess hanging on Windows
os.environ["LOKY_MAX_CPU_COUNT"] = "1"

import time
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import src.config as cfg
import src.dataset as ds_builder


def get_models() -> Dict:
    """Returns the dictionary of benchmark classifier architectures."""
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=1),
        "SVM (RBF)": SVC(kernel="rbf", C=10.0, gamma="scale", random_state=42),
        "Gradient Boosting": HistGradientBoostingClassifier(max_iter=100, random_state=42),
        "LDA": LinearDiscriminantAnalysis(),
        "k-NN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    }


def run_benchmark(cohort: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Runs intra-subject (personalized) benchmark evaluation across the cohort.
    """
    if cohort is None:
        cohort = cfg.CLEAN_COHORT
        
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "models"), exist_ok=True)
    
    records = []
    confusion_matrices = {m: np.zeros((len(cfg.GESTURES), len(cfg.GESTURES)), dtype=int) for m in get_models().keys()}
    
    print(f"=" * 80)
    print(f"STARTING INTRA-SUBJECT BENCHMARK EVALUATION (COHORT N = {len(cohort)})")
    print(f"5 Active Gestures: {', '.join(cfg.GESTURES)}")
    print(f"Theoretical Chance Level: 20.00% (1/5)")
    print(f"Train Repetitions: 1-24 | Holdout Test Repetitions: 25-30")
    print(f"=" * 80)
    
    total_start = time.time()
    
    for idx, participant in enumerate(cohort, 1):
        p_start = time.time()
        print(f"\n[{idx}/{len(cohort)}] Loading data & extracting features for: {participant}...")
        data = ds_builder.build_subject_dataset(participant, split_rep=24, use_cache=True)
        
        X_train, y_train = data["X_train"], data["y_train"]
        X_test, y_test = data["X_test"], data["y_test"]
        
        # Scaling (Strict isolation: fit on train, transform test)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = get_models()
        p_results = []
        for model_name, clf in models.items():
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            
            acc = accuracy_score(y_test, y_pred) * 100.0
            macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0) * 100.0
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0) * 100.0
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0) * 100.0
            
            # Accumulate confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=cfg.GESTURES)
            confusion_matrices[model_name] += cm
            
            records.append({
                "Participant": participant,
                "Model": model_name,
                "Train_Windows": len(X_train),
                "Test_Windows": len(X_test),
                "Accuracy (%)": round(acc, 2),
                "Macro_F1 (%)": round(macro_f1, 2),
                "Precision (%)": round(prec, 2),
                "Recall (%)": round(rec, 2)
            })
            p_results.append(f"{model_name}: {acc:.1f}%")
            
        print(f"    Done in {time.time() - p_start:.1f}s -> " + " | ".join(p_results))
        
    df_results = pd.DataFrame(records)
    
    # Save CSV report
    csv_path = os.path.join(PROJECT_ROOT, "results", "benchmark_intra_subject.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nSaved raw per-subject benchmark records to: {csv_path}")
    
    # Pivot summary table
    print("\n" + "=" * 80)
    print("SUMMARY: TEST ACCURACY (%) ACROSS 12 CLEAN PARTICIPANTS")
    print("=" * 80)
    pivot_acc = df_results.pivot(index="Participant", columns="Model", values="Accuracy (%)")
    print(pivot_acc.to_string())
    
    print("\n" + "=" * 80)
    print("COHORT AGGREGATE PERFORMANCE (MEAN +/- STD)")
    print("=" * 80)
    agg_df = df_results.groupby("Model")[["Accuracy (%)", "Macro_F1 (%)", "Precision (%)", "Recall (%)"]].agg(["mean", "std"])
    print(agg_df.to_string())
    
    # Save confusion matrices
    cm_path = os.path.join(PROJECT_ROOT, "results", "confusion_matrices.pkl")
    with open(cm_path, "wb") as f:
        pickle.dump(confusion_matrices, f)
        
    print(f"\nTotal benchmark completed in {time.time() - total_start:.1f}s.")
    return df_results


if __name__ == "__main__":
    run_benchmark()
