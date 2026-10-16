# Experiments Registry & Scientific Journal Log

This directory contains formal, controlled, reproducible experiments for the Elsevier *Biomedical Signal Processing and Control* journal submission.

## Directory Structure
- `exp1_feature_ablation.py`: Feature importance ranking (MDI + Permutation) and sub-domain ablation across the 12 clean subjects.
- `exp2_windowing_grid.py`: Systematic sweep of window lengths ($W \in [200, 300, 400, 500, 750]\text{ ms}$) and step sizes ($S \in [50, 100, 150]\text{ ms}$).
- `exp3_temporal_smoothing.py`: Real-time decision filtering and rolling majority voting ($K \in [1, 3, 5, 7]$).
- `exp4_model_tuning.py`: Zero-leakage hyperparameter optimization via inner-fold Repetition GroupKFold on Reps 1–24.
- `exp5_synthesis.py`: Synthesis of optimal pipeline parameters and model export.
- `logs/`: Machine-readable CSV files, metrics logs, and publication plots.

## Overfitting Guardrails
1. **Zero Test-Set Snooping**: Parameter searches and ablation ranking are guided strictly by cross-validation within training repetitions (Reps 1–24).
2. **Strict Temporal Holdout**: Final reported evaluation is performed strictly on unseen repetitions (Reps 25–30).
3. **No Window Leaks**: Grouping is enforced by `Repetition_Number`. Adjacent overlapping windows never cross train/test splits.
