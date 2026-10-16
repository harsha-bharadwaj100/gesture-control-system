# Project Progress & Technical History Tracker

**Project**: Single-Channel sEMG Gesture Recognition System  
**Last Updated**: October 16, 2026 | **Git Branch**: `main`  
**Target Submission**: Elsevier *Biomedical Signal Processing and Control* (BSPC)

---

## 1. Project Overview & Historical Context

This project rebuilds an sEMG gesture recognition system from raw hardware recordings to publication-grade peer review. Earlier iterations suffered from several flaws introduced by naive scripts (fabricated 99.8% accuracy via temporal data leakage, IIR notch ringing cones, manufactured Rest data, and severe root clutter).

Every single subsystem has now been audited against raw empirical hardware ground truth, re-engineered, and validated.

---

## 2. Chronological Milestones & What Was Tried

### Phase 1: Repository Non-Destructive Reorganization
* **Problem**: Over 130 legacy files (old models, obsolete datasets, dead scripts, ad-hoc test files) were cluttered across root.
* **Solution**: Cleaned and categorized 55 scripts, 12 models, 26 datasets, 18 dumps, and 21 documents into `legacy/` sub-directories without deleting anything.
* **Result**: Root is pristine, structured, and compliant with modern software engineering standards.

---

### Phase 2: The "Cone Chronicle" (Signal Conditioning Evolution)
Raw single-channel sEMG contains $\approx 300-500\text{ mV}$ of $50\text{ Hz}$ powerline hum cut into 4.0-second trials separated by 4-second unrecorded rest breaks. Each trial starts at an arbitrary phase angle.

* **Attempt 1: Standard IIR Notch Filter (`iirnotch` + `filtfilt`)**
  * *Failure*: Reflection padding across the discontinuous phase boundary created a sharp derivative corner. High-Q filter rang for $150-200\text{ ms}$ at boundaries $\implies$ **Forward Cones** ($150-250\text{ mV}$ boundary spikes).
* **Attempt 2: Naive Static Sinusoidal Subtraction (Fixed 50.0 Hz)**
  * *Failure*: AC mains in India drifts between $49.75\text{ Hz}$ and $50.25\text{ Hz}$. Subtracting a fixed $50.00\text{ Hz}$ sine wave left a $0.25\text{ Hz}$ beating residual $\implies$ **Hourglass / Bowtie** (flared at ends, pinched at center).
* **Attempt 3: Tukey Cosine Window Tapering**
  * *Failure*: Ramping the first and last $200\text{ ms}$ down to zero multiplied the signal by a cosine ramp $\implies$ **Reverse Cones** (artificial pinching at edges).
* **Permanent Solution: Adaptive Running Heterodyne Demodulation (ARHD)**
  * *Implementation*: `src/preprocessing.py`
  * Projects signal onto running orthogonal $\sin(2\pi f t)$ and $\cos(2\pi f t)$ bases ($80\text{ ms} = 4\text{ cycles}$).
  * Tracks instantaneous amplitude and phase of $50$, $100$, and $150\text{ Hz}$ hum sample-by-sample.
  * Clamps boundary coefficients using `mode='edge'`.
  * *Outcome*: **Zero forward cones, zero reverse cones, zero bowties, zero edge pinching.** Completely natural, flat physiological baseline.

---

### Phase 3: Hardware Quality Screening (The 15-Subject Cohort Audit)
Evaluated all 15 participant datasets in `All_Datasets/` ($150\text{ trials} \times 4000\text{ samples} = 600,000\text{ samples}$ each at $1000\text{ Hz}$):

* **Objective Hardware Exclusion Criteria**:
  1. **Keerthi**: Mean active RMS $6.12\text{ mV}$, DCR $1.21\times$ $\implies$ **EXCLUDED** (Electrode contact failure; recorded room electromagnetic noise).
  2. **Vishal**: Mean active RMS $6.78\text{ mV}$, DCR $1.25\times$ $\implies$ **EXCLUDED** (Electrode contact failure).
  3. **Priyanka Ma'am**: Contracted during the rest countdown and rested during the recording cue $\implies$ **EXCLUDED** (Task inversion).
* **Validated Study Cohort**: **$N = 12$ Clean Subjects** ($1,762$ validated trials, $7.05 \times 10^6$ samples).

---

### Phase 4: Resolving the "Rest Dilemma"
* **Flawed Approach**: Earlier proposals considered borrowing rest intervals from other subjects and donor-interpolating them into the 11 subjects lacking rest.
* **Verdict**: Rejected as scientific fabrication. Resting skin impedance cannot be transferred across human bodies.
* **The Permanent Solution: Cascaded Two-Stage Architecture**:
  * **Stage 1 (Energy Gate)**: Checks if window $\text{RMS} < \text{Threshold}$. If yes, instantly outputs `REST` in $0.01\text{ ms}$ (ML bypassed, zero false gesture triggers).
  * **Stage 2 (5-Class Classifier)**: When active, classifies the 5 authentic hand gestures against the true theoretical chance level of **$20.00\%$** ($1/5$).

---

### Phase 5: The 5 Formal Optimization Experiments (Multi-Core Parallel)
Conducted across all 12 subjects on 6 parallel CPU workers ($W=500\text{ ms}, S=100\text{ ms}$, Reps 1–24 Train vs Reps 25–30 Test):

1. **Exp 1: Feature Ranking & Sub-Domain Ablation** (`experiments/exp1_feature_ablation.py`)
   * Top 3 Features: Waveform Length (WL, 11.13%), Spectral Entropy (SE, 8.57%), Mean Frequency (MNF, 8.20%).
   * Proved all 16 features combine synergistically ($39.14\%$) vs single domains ($33.80\%-37.00\%$).
2. **Exp 2: Full 8-Configuration Windowing Grid** (`experiments/exp2_windowing_grid.py`)
   * Evaluated $W \in [200, 350, 500, 750]\text{ ms}$ and $S \in [50, 100, 150]\text{ ms}$.
   * Proved that $W \ge 500\text{ ms}$ is required for single-channel stability ($39.15\%-41.58\%$), whereas $W=200\text{ ms}$ degrades to $35.06\%$.
3. **Exp 3: Real-Time Temporal Smoothing & Majority Voting** (`experiments/exp3_temporal_smoothing.py`)
   * Evaluated $K \in [1, 3, 5, 7]$ rolling decision windows.
   * **Jitter plummeted from $36.77\% \to 13.53\%$** (a **$60\%$ reduction** in erratic frame-switching), while test accuracy rose to $40.73\%$.
4. **Exp 4: Multi-Core Nested Hyperparameter Optimization** (`experiments/exp4_model_tuning.py`)
   * Tuned 14 parameter sets per subject via 4-fold GroupKFold on Train.
   * Proved default regularized models were near theoretical ceiling ($39.15\% \to 39.56\%$). Ledoit-Wolf shrinkage stabilizes LDA.
5. **Exp 5: Final Production Pipeline Synthesis** (`experiments/exp5_synthesis.py`)
   * Exported calibrated models to `models/production_pipeline.pkl`.

---

### Phase 6: Trial-by-Trial Statistical Outlier Audit
* **Audited**: All 1,762 active trials.
* **Found**: 110 statistical outliers ($1.5 \times \text{IQR}$). 106 were high-force spikes; 4 were low.
* **Empirical Test**: Pruning outliers from training **reduced holdout accuracy from $39.14\% \to 37.93\%$**.
* **Conclusion**: High-force trials represent natural human force variance. Retaining them makes the model robust.

---

## 3. Current Repository Architecture

```
gesture-control-system/
├── experiments/
│   ├── benchmark.py                     # Cohort benchmark suite
│   ├── exp1_feature_ablation.py         # Feature ranking & ablation
│   ├── exp2_windowing_grid.py           # Full 8-config window grid
│   ├── exp3_temporal_smoothing.py       # Rolling majority vote debouncing
│   ├── exp4_model_tuning.py             # Multi-core hyperparameter tuning
│   ├── exp5_synthesis.py                # Final pipeline synthesis & export
│   ├── EXPERIMENTAL_FINDINGS_REPORT.md  # Formal experimental report
│   ├── logs/                            # CSV logs and metrics
│   └── README.md                        # Experiments registry
├── models/
│   └── production_pipeline.pkl          # Calibrated 2-stage production models
├── notes/
│   ├── CONTRIBUTOR_DISCIPLINE_GUIDELINES.md # Rules for all contributors
│   ├── PROJECT_PROGRESS_AND_HISTORY.md      # This historical record
│   └── README.md                            # Notes index
├── src/
│   ├── config.py                        # Central cohort & hardware specs
│   ├── dataset.py                       # Repetition-aware sliding window builder
│   ├── features.py                      # Frozen unified 16-feature vector
│   └── preprocessing.py                 # ARHD filter & trial loader
└── visualizations/
    └── interactive_studio.py            # Multi-window Tkinter/Matplotlib explorer
```

---

## 4. Next Planned Deliverables

1. **Live GUI Hardware Deployment**: Connect `models/production_pipeline.pkl` into `gui_live_inference.py` for lab testing with the physical sensor.
2. **Elsevier BSPC Paper Rewrite**: Update `paper_elsarticle/main.tex` with ARHD mathematical formulations, the 12-subject cohort inclusion criteria, and full experimental tables.
