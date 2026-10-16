# Single-Channel sEMG Gesture Recognition System
### Scientific Research & Real-Time Biometric Control Pipeline

[![Target Journal](https://img.shields.io/badge/Target%20Journal-Elsevier%20BSPC-orange.svg)](https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Sampling Frequency](https://img.shields.io/badge/Sample%20Rate-1000%20Hz-green.svg)]()
[![Chance Level](https://img.shields.io/badge/Chance%20Level-20.00%25%20(5--Class)-brightgreen.svg)]()
[![Hardware Architecture](https://img.shields.io/badge/Architecture-Two--Stage%20Cascaded-purple.svg)]()

A scientifically rigorous, publication-grade sEMG gesture recognition pipeline and real-time control system. Designed from raw physiological biopotentials up to journal-grade peer review for Elsevier's ***Biomedical Signal Processing and Control*** (BSPC).

---

## 🔬 Core System Architecture

```
                                      THE TWO-STAGE CONTROL PIPELINE
Incoming Analog sEMG (1000 Hz)
            │
            ▼
┌─────────────────────────────────┐
│ Signal Conditioning (ARHD)      │  • Demeaning + 50/100/150 Hz mains hum removal via ARHD
│ (src/preprocessing.py)          │  • 4th-order zero-phase Butterworth bandpass (20 - 450 Hz)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Real-Time Ring Buffer           │  • Window Length: W = 500 ms (500 samples)
│ (W = 500 ms, S = 100 ms)        │  • Stride / Step: S = 100 ms (10 Hz decision rate)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ STAGE 1: Calibrated Energy Gate │  Is Window RMS < Subject Resting Baseline Threshold?
│ (Rest / Idle Detection)         │
└────────────────┬────────────────┘
                 ├────────────────► [YES] ──► Output: "REST / IDLE"
                 │                            (Latency: < 0.01 ms | ML Bypassed | 0% False Triggers)
                 │
                 ▼ [NO: Active Contraction]
┌─────────────────────────────────┐
│ STAGE 2: 16-Feature Extraction  │  • Time-Domain Energy (RMS, MAV, WL, VAR, SSI)
│ (src/features.py)               │  • Complexity (ZC, SSC, WAMP) | Shape (|Skew|, Kurtosis)
└────────────────┬────────────────┘  • FFT Spectral Power (MNF, MDF, SE, PKF, MNP, TTP)
                 │
                 ▼
┌─────────────────────────────────┐
│ Multi-Class Pattern Classifier  │  • Tuned Support Vector Machine (SVM with RBF Kernel)
│ (Tuned Model from models/)      │  • Regularized Random Forest / Shrunk LDA
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Rolling Majority Filter (K = 5) │  • 5-Window temporal debouncing (~400 ms rolling buffer)
│ (Debouncing & Jitter Suppress)  │  • Cuts frame-to-frame switching jitter by 60% (36.8% -> 13.5%)
└────────────────┬────────────────┘
                 │
                 ▼
Final Output: Fist / Open Palm / Pinch / Wrist Up / Wrist Down
```

---

## 📂 Repository Organization

We maintain strict repository cleanliness. Contributors must follow this structure:

```
gesture-control-system/
├── All_Datasets/                 # Raw 15-participant experimental CSV recordings
├── experiments/                  # Formal, controlled experimental research suite
│   ├── benchmark.py              # Cohort benchmark evaluation runner across all 12 subjects
│   ├── exp1_feature_ablation.py  # Feature importance ranking & sub-domain synergy study
│   ├── exp2_windowing_grid.py    # Full 8-configuration window & stride sensitivity sweep
│   ├── exp3_temporal_smoothing.py# Real-time rolling majority voting & jitter reduction
│   ├── exp4_model_tuning.py      # Multi-core nested 4-fold GroupKFold hyperparameter tuning
│   ├── exp5_synthesis.py         # Production pipeline synthesis and model export
│   ├── EXPERIMENTAL_FINDINGS_REPORT.md # Comprehensive publication report
│   ├── logs/                     # Machine-readable CSV data tables from all experiments
│   └── README.md                 # Experiments directory index
├── models/                       # Calibrated production models (git-ignored to avoid binary bloat)
│   └── production_pipeline.pkl   # Exported two-stage pipeline package
├── notes/                        # Engineering notes & contributor standards
│   ├── CONTRIBUTOR_DISCIPLINE_GUIDELINES.md # Mandatory guidelines for all contributors
│   ├── PROJECT_PROGRESS_AND_HISTORY.md      # Chronological engineering record & milestones
│   └── README.md                 # Notes directory index
├── src/                          # Centralized production library
│   ├── config.py                 # Cohort specifications, exclusions, and hardware parameters
│   ├── dataset.py                # Repetition-aware sliding window builder (zero data leakage)
│   ├── features.py               # Frozen unified 16-feature vector extractor
│   └── preprocessing.py          # Adaptive Running Heterodyne Demodulation (ARHD) filter
├── visualizations/               # Interactive biomedical inspection tools
│   └── interactive_studio.py     # Non-blocking, multi-window Tkinter/Matplotlib explorer
├── dataset_collector.py          # Real-time hardware acquisition script (NI-DAQ)
├── gui_live_inference.py         # Real-time streaming inference GUI
└── README.md                     # Project homepage (this file)
```

---

## 🛠️ Getting Started

### 1. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/harsha-bharadwaj100/gesture-control-system.git
cd gesture-control-system

# Recommended: use virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install numpy scipy pandas scikit-learn matplotlib
```

### 2. Interactive Biomedical Waveform Studio
Launch the multi-window interactive explorer to inspect synchronized raw vs. filtered waveforms, FFT spectra, and 3D PCA feature manifolds:
```powershell
python visualizations/interactive_studio.py
```

### 3. Run the Full Cohort Benchmark
Evaluate the 5 classical biomedical classifiers (SVM, RF, Gradient Boosting, LDA, $k$-NN) across all 12 clean participants against the true $20.00\%$ chance level:
```powershell
python experiments/benchmark.py
```

---

## 📋 Contributor Guidelines & Scientific Discipline

> [!IMPORTANT]
> All contributors, researchers, and automated coding agents **must read and follow** the [Contributor Discipline Guidelines](file:///c:/D-sim/pyprojs/gesture-control-system/notes/CONTRIBUTOR_DISCIPLINE_GUIDELINES.md).

### The Golden Rules:
1. **Zero Root-Clutter Policy**: Never leave test scripts, scratch notebooks, model dumps, or CSV logs loose in the root directory. Place them in `experiments/`, `notes/`, or `src/`.
2. **Strict Zero-Leakage Partitioning**: 
   * **Never use random `train_test_split` on sliding windows.** Overlapping windows share $80\%$ raw data; random shuffling causes temporal data leakage and fake $99\%$ metrics.
   * Split strictly by **Repetition ID**: Repetitions 1–24 for Training ($80\%$), Repetitions 25–30 for Holdout Testing ($20\%$, completely unseen in time).
   * Cross-validation must strictly use `GroupKFold(groups=Repetition)`.
3. **True $20.00\%$ Chance Level**: We classify 5 authentic active gestures. Never fabricate a synthetic 6th "Rest" class. Rest is handled via the Stage 1 Energy Gate.
4. **Pre-Analytical Exclusions Only**: Participant exclusions must be based strictly on objective physical failure criteria (e.g. electrode contact loss in Keerthi/Vishal, protocol task inversion in Priyanka), **never on post-hoc accuracy cherry-picking**.
5. **Optimize, Don't Compromise**: Leverage multi-core CPU parallelism (`concurrent.futures.ProcessPoolExecutor`) and caching. Never artificially truncate experimental search spaces to save compute time.

---

## 🧠 Extending the Pipeline (Deep Learning & New Models)

Contributors wishing to explore **Deep Learning (DL)** architectures (e.g., 1D-CNN, CNN-LSTM, Temporal Convolutional Networks / TCN, or Transformers) or novel feature sets should follow this standard protocol:

### How to Implement a New Model Experiment:
1. **Create a Dedicated Script in `experiments/`**:
   * E.g., `experiments/exp6_deep_learning_benchmark.py`.
2. **Import from the Single Source of Truth**:
   * Use `src/preprocessing.py` for ARHD filtering.
   * Use `src/dataset.py` to get strictly partitioned train/test sets:
     ```python
     import src.dataset as ds_builder
     data = ds_builder.build_subject_dataset("Arundhati", split_rep=24)
     X_train, y_train = data["X_train"], data["y_train"]
     X_test, y_test   = data["X_test"], data["y_test"]
     ```
   * Or for raw waveform inputs (e.g., 1D-CNN), slice raw 500 ms windows from `prep.load_participant_trials(sub)` grouped strictly by repetition.
3. **Hyperparameter Tuning via Nested Cross-Validation**:
   * Tune network depth, learning rates, and dropout **strictly within Repetitions 1–24** using 4-fold Repetition `GroupKFold`.
   * Evaluate **once** on Holdout Repetitions 25–30.
4. **Log Results in `experiments/logs/`**:
   * Save full CSV logs and metrics (Accuracy, Macro-F1, Precision, Recall, and Confusion Matrices).
5. **Compare Against Our Established Benchmarks**:
   * Any new model must be benchmarked against our verified baselines (SVM-RBF at $39.2\%-41.4\%$, LDA at $39.6\%-55.7\%$, and smoothed production pipeline at $40.5\%-58\%$).

---

## 📊 Summary of Findings (From the 5 Formal Experiments)

| Experiment | Tested Hypotheses | Ground Truth Result |
| :--- | :--- | :--- |
| **Exp 1: Feature Ablation** | Can feature dimensionality be reduced? | **All 16 features combine synergistically** ($39.14\%$). Waveform Length (WL, $11.13\%$), Spectral Entropy (SE, $8.57\%$), and Mean Frequency (MNF, $8.20\%$) are top drivers. |
| **Exp 2: Windowing Grid** | Optimal latency-accuracy windowing ($W \in [200, 750]\text{ ms}$, $S \in [50, 150]\text{ ms}$) | **$W=500\text{ ms}, S=100\text{ ms}$** is the optimal Pareto frontier. Windows $<500\text{ ms}$ suffer severe feature degradation ($-6\%$). |
| **Exp 3: Temporal Smoothing** | Eliminating frame jitter in real-time streams | **$K=5$ rolling majority vote eliminates $60\%$ of frame jitter** ($36.8\% \to 13.5\%$) and lifts holdout accuracy. |
| **Exp 4: Model Tuning** | Hyperparameter sensitivity via nested GroupKFold | Regularized defaults were already near mathematical ceilings; Ledoit-Wolf shrinkage stabilizes LDA (boosting subjects up to $55.7\%$). |
| **Exp 5: Synthesis Export** | Production two-stage pipeline validation | Final smoothed pipeline achieves **$40.5\% \pm 8.9\%$** on unseen repetitions (top subjects **$50\%-58\%$**) with **$<13.5\%$ jitter**. |
| **Outlier Audit** | Impact of pruning $1.5\times\text{IQR}$ force outliers | Pruning high-force trials **decreases holdout accuracy** ($39.14\% \to 37.93\%$). Retaining biological force diversity makes the model robust. |

For detailed experimental methodologies, see [experiments/EXPERIMENTAL_FINDINGS_REPORT.md](file:///c:/D-sim/pyprojs/gesture-control-system/experiments/EXPERIMENTAL_FINDINGS_REPORT.md).

---

## 📜 Citation & Research Team
**Authors**: Harsha Bharadwaj, Guru Raj, Keerthana K., Priyanka S., Arpita K.  
**Affiliation**: Department of Electronics and Communication Engineering, BNM Institute of Technology, Bengaluru, India.  
**Target Publication**: Elsevier *Biomedical Signal Processing and Control* (BSPC).
