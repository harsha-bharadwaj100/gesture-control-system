# Contributor Discipline & Scientific Engineering Guidelines

This document defines the strict scientific, mathematical, and architectural rules for the **Single-Channel sEMG Gesture Recognition System** and our publication in Elsevier *Biomedical Signal Processing and Control* (BSPC).

All current and future contributors, researchers, and automated coding agents must adhere strictly to these principles.

---

## 1. Zero Root-Clutter Policy

The repository root must remain pristine, clean, and professional at all times.
* **Prohibited in Root**: Test scripts, scratch files, loose `.pkl` models, benchmark runners, CSV dumps, or temporary plots.
* **Approved Code Placement**:
  * Core pipeline modules $\implies$ `src/` (`config.py`, `preprocessing.py`, `features.py`, `dataset.py`)
  * Controlled evaluations $\implies$ `experiments/` (`exp*.py`, `benchmark.py`, logs in `experiments/logs/`)
  * Interactive tools $\implies$ `visualizations/`
  * Model artifacts $\implies$ `models/`
  * Documentation and research notes $\implies$ `notes/` and `knowledge/`
  * Legacy/deprecated code $\implies$ `legacy/` (never left loose)

---

## 2. Zero-Leakage Validation Standards (Zero Fabrication)

Under no circumstances should accuracy be artificially inflated. Reviewers at Elsevier BSPC will detect and reject synthetic or leaked metrics.

1. **Strict Repetition Partitioning**:
   * **Never use random `train_test_split()` on sliding windows.** Adjacent overlapping windows share up to $80\%$ identical raw data. Random splitting causes massive temporal data leakage, creating the illusion of $99\%$ accuracy while failing in real-time.
   * Split strictly by **Repetition ID**: Repetitions 1–24 for Training ($80\%$), Repetitions 25–30 for Holdout Testing ($20\%$, completely unseen in time).
   * Cross-validation must strictly use `GroupKFold(groups=Repetition)`.
2. **Standardization Isolation**:
   * `StandardScaler` must be fitted **strictly on the training fold** and used only to transform the test fold. Never fit scalers on full datasets.
3. **True Theoretical Chance Level ($20.00\%$)**:
   * With 5 authentic active gestures, random guessing is strictly $1/5 = 20.00\%$.
   * Never fabricate or harvest a synthetic 6th "Rest" class. Rest must be handled via the Cascaded Two-Stage Energy Gate.

---

## 3. The Two-Stage Architecture Standard

Real-time gesture control must follow the dual-stage biomedical architecture:

```
Incoming 500 ms sEMG Window
            │
            ▼
┌─────────────────────────┐
│ STAGE 1: Energy Gate    │  Is RMS Voltage < Calibrated Rest Noise Threshold?
│ (Hardware Calibration)  │
└───────────┬─────────────┘
            ├──────────────► [YES] ──► Output: "REST / IDLE"
            │                          (Latency: < 0.01 ms | ML Bypassed | 0% Hallucinations)
            │
            ▼ [NO: Active Contraction]
┌─────────────────────────┐
│ STAGE 2: 16-Feature ML  │  Extracted from src/features.py
│ (Tuned 5-Class Model)   │  Smoothed with K=5 Rolling Majority Filter (Jitter < 13.5%)
└───────────┬─────────────┘
            ▼
Output: Fist / Open Palm / Pinch / Wrist Up / Wrist Down
```

---

## 4. Signal Conditioning & Preprocessing Standards

* **50 Hz Powerline Hum Elimination**: Must be performed using **Adaptive Running Heterodyne Demodulation (ARHD)** (`src/preprocessing.py`). 
  * *Never use standard IIR notch filters on segmented trials* (causes boundary ringing cones).
  * *Never use window tapers (e.g. Tukey/Hanning)* (causes artificial "reverse cone" pinching).
  * *Never use static 50.0 Hz fixed subtraction* (causes 0.25 Hz grid drift beating/bowties).
* **Bandpass Filtering**: 4th-order zero-phase Butterworth bandpass ($20-450\text{ Hz}$).

---

## 5. Pre-Analytical vs. Post-Hoc Data Quality Screening

Participant exclusion must **never** be based on post-hoc accuracy (cherry-picking):
* **Objective Hardware Failure Criteria Only**:
  * Mean active contraction $\text{RMS} < 8.0\text{ mV}$ or Dynamic Contrast Ratio $\text{DCR} < 1.30\times$ (indicates dried gel, air gaps, or electrode detachment, as in Keerthi and Vishal).
  * Protocol task inversion (subject rested during cue and flexed during rest countdown, as in Priyanka Ma'am).
* **Natural Physiological Variance Must Be Retained**:
  * Human force variations (e.g. higher-force contractions) are natural biological signals.
  * Pruning statistical outliers degrades holdout generalization ($39.14\% \to 37.93\%$). Retain all valid trials across the 12 clean subjects.

---

## 6. Engineering Rule: "Optimize, Don't Compromise"

When designing benchmarks or experiments:
* **Full Hardware Utilization**: Modern CPUs have multiple cores (e.g. 8 cores). Always leverage `concurrent.futures.ProcessPoolExecutor` across independent subjects or grid configurations.
* **Vectorization & Disk Caching**: Use disk caching (`.cache_datasets/`, `.cache_trials/`) so expensive filtering and FFTs run once.
* **Never Truncate Search Spaces**: Do not cut experimental grids in half to save time; optimize the code and run the complete parameter space.
