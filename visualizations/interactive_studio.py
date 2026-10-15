"""
Interactive Gesture Studio (Non-Blocking Multi-Window Biomedical Explorer)
Provides interactive, pannable, zoomable, and 3D rotatable visual exploration
of sEMG gestures, waveforms, spectra, and feature cluster geometry.
"""

import os
import sys
import tempfile
# Ensure Matplotlib uses a writable temp cache directory to avoid permissions warnings
os.environ['MPLCONFIGDIR'] = tempfile.gettempdir()

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.config as cfg
import src.preprocessing as prep
import src.features as feat

# Consistent color palette for the 5 gestures
GESTURE_COLORS = {
    "Fist": "#e74c3c",       # Crimson Red
    "Open palm": "#3498db",  # Sky Blue
    "Pinch": "#2ecc71",      # Emerald Green
    "Wrist up": "#f39c12",   # Vibrant Amber
    "Wrist down": "#9b59b6"  # Royal Purple
}


class NonBlockingWindow:
    """Base helper for spawning fully independent, non-blocking interactive plotting windows."""
    def __init__(self, master, title, geometry="1000x650"):
        self.win = tk.Toplevel(master)
        self.win.title(title)
        self.win.geometry(geometry)
        self.fig = plt.Figure(tight_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.win)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


class InteractiveGestureStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("sEMG Interactive Gesture Studio (Multi-Window)")
        self.root.geometry("460x680")
        self.root.configure(bg="#1e1e24")
        
        # State
        self.current_participant = tk.StringVar(value=cfg.CLEAN_COHORT[0])
        self.current_gesture = tk.StringVar(value=cfg.GESTURES[0])
        self.current_rep = tk.IntVar(value=5)
        self.trials_cache = {}
        self.raw_trials_cache = {}
        
        self._build_ui()
        self._preload_participant()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        title_lbl = tk.Label(
            self.root, text="sEMG Gesture Studio", font=("Helvetica", 18, "bold"),
            fg="#00d2d3", bg="#1e1e24"
        )
        title_lbl.pack(pady=(12, 3))
        
        subtitle = tk.Label(
            self.root, text="Non-Blocking Interactive Biomedical Explorer", font=("Helvetica", 10),
            fg="#c8d6e5", bg="#1e1e24"
        )
        subtitle.pack(pady=(0, 10))
        
        # --- Participant Selection Frame ---
        part_frame = tk.LabelFrame(
            self.root, text=" 1. Select Participant ", font=("Helvetica", 11, "bold"),
            fg="#54a0ff", bg="#222f3e", padx=10, pady=8
        )
        part_frame.pack(fill=tk.X, padx=18, pady=4)
        
        cohort_display = []
        for p in cfg.ALL_PARTICIPANTS:
            if p in cfg.EXCLUDED_PARTICIPANTS:
                cohort_display.append(f"{p} [EXCLUDED]")
            else:
                cohort_display.append(p)
                
        self.combo_part = ttk.Combobox(
            part_frame, textvariable=self.current_participant,
            values=cohort_display, state="readonly", font=("Helvetica", 11)
        )
        self.combo_part.pack(fill=tk.X, pady=4)
        self.combo_part.bind("<<ComboboxSelected>>", self._on_participant_change)
        
        # --- Target Gesture & Repetition Frame ---
        param_frame = tk.LabelFrame(
            self.root, text=" 2. Gesture & Repetition Selection ", font=("Helvetica", 11, "bold"),
            fg="#54a0ff", bg="#222f3e", padx=10, pady=8
        )
        param_frame.pack(fill=tk.X, padx=18, pady=4)
        
        row1 = tk.Frame(param_frame, bg="#222f3e")
        row1.pack(fill=tk.X, pady=(0, 5))
        tk.Label(row1, text="Gesture:", font=("Helvetica", 10, "bold"), fg="#ffffff", bg="#222f3e").pack(side=tk.LEFT, padx=(0, 10))
        
        self.combo_gesture = ttk.Combobox(
            row1, textvariable=self.current_gesture,
            values=cfg.GESTURES, state="readonly", font=("Helvetica", 10), width=15
        )
        self.combo_gesture.pack(side=tk.LEFT)
        
        self.slider_rep = tk.Scale(
            param_frame, from_=1, to=30, orient=tk.HORIZONTAL, variable=self.current_rep,
            bg="#222f3e", fg="#ffffff", highlightthickness=0, font=("Helvetica", 9),
            label="Repetition Index (1 to 30)"
        )
        self.slider_rep.pack(fill=tk.X)
        
        # --- Action Buttons Frame ---
        act_frame = tk.LabelFrame(
            self.root, text=" 3. Launch Interactive Windows (Simultaneous) ", font=("Helvetica", 11, "bold"),
            fg="#54a0ff", bg="#222f3e", padx=10, pady=8
        )
        act_frame.pack(fill=tk.BOTH, expand=True, padx=18, pady=8)
        
        btn_opts = {"font": ("Helvetica", 10, "bold"), "pady": 5, "relief": tk.FLAT, "cursor": "hand2"}
        
        b_sync = tk.Button(
            act_frame, text="🔍 Raw vs. Filtered Inspector (Up & Down Zoom)", bg="#00cec9", fg="#2d3436",
            command=self.open_raw_vs_filtered_window, **btn_opts
        )
        b_sync.pack(fill=tk.X, pady=3)
        
        b1 = tk.Button(
            act_frame, text="📈 5-Gesture Time-Domain Stack (Pan/Zoom)", bg="#10ac84", fg="white",
            command=self.open_waveforms_window, **btn_opts
        )
        b1.pack(fill=tk.X, pady=3)
        
        b2 = tk.Button(
            act_frame, text="⚡ Frequency Spectra & Median Frequency (FFT)", bg="#0984e3", fg="white",
            command=self.open_spectrum_window, **btn_opts
        )
        b2.pack(fill=tk.X, pady=3)
        
        b3 = tk.Button(
            act_frame, text="🌐 3D Feature Cluster Manifold (Rotatable)", bg="#6c5ce7", fg="white",
            command=self.open_3d_clusters_window, **btn_opts
        )
        b3.pack(fill=tk.X, pady=3)
        
        b4 = tk.Button(
            act_frame, text="⏱️ 30-Trial Overlay & Contraction Envelopes", bg="#d63031", fg="white",
            command=self.open_overlay_envelope_window, **btn_opts
        )
        b4.pack(fill=tk.X, pady=3)

    def _get_clean_participant_name(self):
        raw = self.current_participant.get()
        return raw.replace(" [EXCLUDED]", "").strip()

    def _on_participant_change(self, event=None):
        self._preload_participant()

    def _preload_participant(self):
        p = self._get_clean_participant_name()
        if p not in self.trials_cache:
            try:
                self.trials_cache[p] = prep.load_participant_trials(p, datasets_dir="All_Datasets", apply_filter=True)
                self.raw_trials_cache[p] = prep.load_participant_trials(p, datasets_dir="All_Datasets", apply_filter=False)
            except Exception as e:
                messagebox.showerror("Error Loading Data", f"Could not load data for {p}:\n{e}")

    # =========================================================================
    # 0. RAW VS. FILTERED INSPECTOR WINDOW (SYNCHRONIZED UP & DOWN ZOOM)
    # =========================================================================
    def open_raw_vs_filtered_window(self):
        p = self._get_clean_participant_name()
        g = self.current_gesture.get()
        rep = self.current_rep.get()
        
        raw_sig = self.raw_trials_cache.get(p, {}).get((g, rep), np.zeros(4000))
        clean_sig = self.trials_cache.get(p, {}).get((g, rep), np.zeros(4000))
        
        win_wrapper = NonBlockingWindow(
            self.root,
            f"Raw vs Filtered Inspector — {p} | {g} | Repetition {rep}",
            "1150x700"
        )
        fig = win_wrapper.fig
        time_axis = np.linspace(0, cfg.TOTAL_TRIAL_DURATION, 4000)
        
        # Create 2 subplots vertically stacked with shared X-axis
        ax_raw, ax_clean = fig.subplots(2, 1, sharex=True)
        color = GESTURE_COLORS.get(g, "#2980b9")
        
        # 1. TOP SUBPLOT: Raw Signal (Volts)
        ax_raw.plot(time_axis, raw_sig, color="#7f8c8d", lw=1.0, label="Raw Input (Volts)")
        ax_raw.axvspan(cfg.ACTIVE_START_SEC, cfg.ACTIVE_END_SEC, color="#f1f2f6", alpha=0.5, label="Active Analysis Window [0.6s - 3.4s]")
        ax_raw.set_title(f"RAW sEMG INPUT: {p} — {g} (Repetition {rep})\n[Notice DC baseline offset and raw 50 Hz mains hum]", fontsize=11, fontweight="bold", pad=8)
        ax_raw.set_ylabel("Raw Amplitude (V)", fontdict={"weight": "bold", "size": 10})
        ax_raw.grid(True, linestyle=":", alpha=0.6)
        ax_raw.legend(loc="upper right", fontsize=9)
        
        # 2. BOTTOM SUBPLOT: Clean Filtered Signal (Millivolts)
        ax_clean.plot(time_axis, clean_sig * 1000.0, color=color, lw=1.2, label=f"Clean Filtered sEMG (mV)")
        ax_clean.axvspan(cfg.ACTIVE_START_SEC, cfg.ACTIVE_END_SEC, color="#f1f2f6", alpha=0.5)
        ax_clean.set_title(f"CLEAN FILTERED sEMG (50Hz Notch + Periodic Pad Buffer + 20-450Hz Bandpass)\n[Zero cone boundary artifact — steady-state isometric contraction]", fontsize=11, fontweight="bold", pad=8)
        ax_clean.set_xlabel("Time (seconds) — [Zoom in on either plot; both synchronize automatically]", fontsize=10, fontweight="bold")
        ax_clean.set_ylabel("Filtered (mV)", fontdict={"weight": "bold", "size": 10})
        ax_clean.grid(True, linestyle=":", alpha=0.6)
        ax_clean.legend(loc="upper right", fontsize=9)
        
        win_wrapper.canvas.draw()

    # =========================================================================
    # 1. TIME-DOMAIN WAVEFORMS WINDOW (5-GESTURE STACK)
    # =========================================================================
    def open_waveforms_window(self):
        p = self._get_clean_participant_name()
        rep = self.current_rep.get()
        trials = self.trials_cache.get(p, {})
        
        win_wrapper = NonBlockingWindow(self.root, f"Waveforms — Participant: {p} | Repetition: {rep}", "1100x750")
        fig = win_wrapper.fig
        
        time_axis = np.linspace(0, cfg.TOTAL_TRIAL_DURATION, 4000)
        axes = fig.subplots(5, 1, sharex=True)
        
        for idx, g in enumerate(cfg.GESTURES):
            ax = axes[idx]
            sig = trials.get((g, rep), np.zeros(4000))
            color = GESTURE_COLORS.get(g, "#333333")
            
            # Highlight active analysis plateau (0.6s to 3.4s)
            ax.axvspan(cfg.ACTIVE_START_SEC, cfg.ACTIVE_END_SEC, color="#f1f2f6", alpha=0.6, label="Active Analysis Plateau" if idx == 0 else "")
            
            # Plot filtered waveform (mV)
            ax.plot(time_axis, sig * 1000, color=color, lw=1.1, label=f"{g} (Rep {rep})")
            
            ax.set_ylabel(f"{g}\n(mV)", fontdict={"weight": "bold", "size": 9})
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="upper right", fontsize=8)
            
        axes[-1].set_xlabel("Time (seconds)", fontsize=11, fontweight="bold")
        fig.suptitle(f"Clean Time-Domain Contraction Waveforms: {p} (Repetition {rep})", fontsize=13, fontweight="bold")
        win_wrapper.canvas.draw()

    # =========================================================================
    # 2. FREQUENCY SPECTRA & PSD (FFT) WINDOW
    # =========================================================================
    def open_spectrum_window(self):
        p = self._get_clean_participant_name()
        rep = self.current_rep.get()
        trials = self.trials_cache.get(p, {})
        
        win_wrapper = NonBlockingWindow(self.root, f"Power Spectra (FFT) — Participant: {p} | Repetition: {rep}", "1000x650")
        fig = win_wrapper.fig
        ax = fig.add_subplot(111)
        
        for g in cfg.GESTURES:
            sig = trials.get((g, rep), np.zeros(4000))
            active_sig = sig[int(cfg.ACTIVE_START_SEC * cfg.SAMPLE_RATE):int(cfg.ACTIVE_END_SEC * cfg.SAMPLE_RATE)]
            
            fft_vals = np.fft.rfft(active_sig)
            psd = np.abs(fft_vals) ** 2
            freqs = np.fft.rfftfreq(len(active_sig), 1.0 / cfg.SAMPLE_RATE)
            
            color = GESTURE_COLORS.get(g, "#333333")
            total_power = np.sum(psd) + 1e-12
            cum_power = np.cumsum(psd)
            mdf = freqs[np.searchsorted(cum_power, total_power / 2.0)]
            
            norm_psd = psd / total_power
            ax.plot(freqs, norm_psd, color=color, lw=1.5, label=f"{g} (MDF = {mdf:.1f} Hz)")
            ax.axvline(mdf, color=color, linestyle="--", alpha=0.5, lw=1.0)
            
        ax.set_xlim(20, 450)
        ax.set_xlabel("Frequency (Hz)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalized Power Spectral Density", fontsize=11, fontweight="bold")
        ax.set_title(f"Frequency Distribution & Median Frequency (MDF) Shift: {p} (Rep {rep})", fontsize=13, fontweight="bold")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=10)
        win_wrapper.canvas.draw()

    # =========================================================================
    # 3. 3D INTERACTIVE FEATURE CLUSTER SCATTER
    # =========================================================================
    def open_3d_clusters_window(self):
        p = self._get_clean_participant_name()
        trials = self.trials_cache.get(p, {})
        
        win_wrapper = NonBlockingWindow(self.root, f"3D Feature Clusters (Rotatable) — Participant: {p}", "1050x700")
        fig = win_wrapper.fig
        ax = fig.add_subplot(111, projection="3d")
        
        feature_data = {g: [] for g in cfg.GESTURES}
        win_size = int(cfg.WINDOW_SIZE_MS * cfg.SAMPLE_RATE / 1000)
        win_step = int(cfg.WINDOW_STEP_MS * cfg.SAMPLE_RATE / 1000)
        
        for g in cfg.GESTURES:
            for rep in range(1, 31):
                sig = trials.get((g, rep))
                if sig is None or len(sig) < 4000:
                    continue
                active_sig = sig[int(cfg.ACTIVE_START_SEC * cfg.SAMPLE_RATE):int(cfg.ACTIVE_END_SEC * cfg.SAMPLE_RATE)]
                for start in range(0, len(active_sig) - win_size, win_step):
                    w = active_sig[start:start + win_size]
                    fvec = feat.extract_features(w, fs=cfg.SAMPLE_RATE)
                    feature_data[g].append(fvec)
                    
        all_X = []
        labels = []
        for g in cfg.GESTURES:
            for fv in feature_data[g]:
                all_X.append(fv)
                labels.append(g)
                
        if len(all_X) > 10:
            pca = PCA(n_components=3)
            X_3d = pca.fit_transform(all_X)
            
            idx = 0
            for g in cfg.GESTURES:
                count = len(feature_data[g])
                g_pts = X_3d[idx:idx + count]
                idx += count
                color = GESTURE_COLORS.get(g, "#333333")
                ax.scatter(
                    g_pts[:, 0], g_pts[:, 1], g_pts[:, 2],
                    color=color, label=g, s=25, alpha=0.7, edgecolors="none"
                )
                
            ax.set_xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontweight="bold")
            ax.set_ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontweight="bold")
            ax.set_zlabel(f"PCA 3 ({pca.explained_variance_ratio_[2]*100:.1f}%)", fontweight="bold")
            ax.set_title(f"3D Feature Manifold: Click & Drag to Rotate in 3D ({p})", fontsize=12, fontweight="bold")
            ax.legend(loc="upper left")
        else:
            ax.text(0.5, 0.5, 0.5, "Insufficient Data Extracted", ha="center")
            
        win_wrapper.canvas.draw()

    # =========================================================================
    # 4. REPETITION OVERLAY & ENVELOPE WINDOW
    # =========================================================================
    def open_overlay_envelope_window(self):
        p = self._get_clean_participant_name()
        trials = self.trials_cache.get(p, {})
        
        win_wrapper = NonBlockingWindow(self.root, f"Repetition Envelope Overlay — Participant: {p}", "1100x700")
        fig = win_wrapper.fig
        axes = fig.subplots(2, 3)
        axes_flat = axes.flatten()
        time_axis = np.linspace(0, cfg.TOTAL_TRIAL_DURATION, 4000)
        
        for idx, g in enumerate(cfg.GESTURES):
            ax = axes_flat[idx]
            color = GESTURE_COLORS.get(g, "#333333")
            rep_envelopes = []
            
            for rep in range(1, 31):
                sig = trials.get((g, rep))
                if sig is None or len(sig) < 4000:
                    continue
                env = np.abs(sig)
                rep_envelopes.append(env)
                ax.plot(time_axis, env * 1000, color=color, alpha=0.12, lw=0.8)
                
            if len(rep_envelopes) > 0:
                rep_envelopes = np.array(rep_envelopes)
                mean_env = np.mean(rep_envelopes, axis=0) * 1000
                std_env = np.std(rep_envelopes, axis=0) * 1000
                
                ax.plot(time_axis, mean_env, color=color, lw=2.2, label="Mean Contraction Envelope")
                ax.fill_between(time_axis, mean_env - std_env, mean_env + std_env, color=color, alpha=0.25)
                
            ax.axvspan(cfg.ACTIVE_START_SEC, cfg.ACTIVE_END_SEC, color="#f1f2f6", alpha=0.5)
            ax.set_title(f"{g} (30 Repetitions)", fontdict={"weight": "bold", "size": 11})
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_ylabel("Amplitude (mV)", fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.6)
            
        axes_flat[5].axis("off")
        fig.suptitle(f"Trial-to-Trial Reproducibility & Envelope Stability across 30 Repetitions ({p})", fontsize=13, fontweight="bold")
        win_wrapper.canvas.draw()


if __name__ == "__main__":
    app_root = tk.Tk()
    app = InteractiveGestureStudio(app_root)
    app_root.mainloop()
