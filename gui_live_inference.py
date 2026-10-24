"""
Live GUI Inference System (National Instruments DAQ & Single-Channel sEMG)
Implements the Production Cascaded Two-Stage Architecture:
  Stage 1: Hardware-Calibrated Energy Gate (< 0.01 ms Rest / Idle detection)
  Stage 2: 16-Feature ML Classifier with K=5 Rolling Majority Debouncing

Hardware: Anthriq / Nexstem amplifier -> NI-DAQ USB (Dev1/ai0) @ 1000 Hz
"""

import os
import sys
import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

import time
import queue
import threading
import collections
import pickle
import warnings
import numpy as np

# GUI & Plotting Imports
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Project Root Setup
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.config as cfg
import src.preprocessing as prep
import src.features as feat_mod

# Hardware DAQ Import
try:
    import nidaqmx
    from nidaqmx.constants import TerminalConfiguration, AcquisitionType
    NIDAQ_AVAILABLE = True
except (ImportError, Exception):
    NIDAQ_AVAILABLE = False

warnings.filterwarnings("ignore")

# --- Configuration Constants ---
CHANNEL = "Dev1/ai0"
FS = 1000              # 1000 Hz
WINDOW_SIZE = 500      # 500 ms analysis window (500 samples)
UPDATE_STEP = 100      # 100 ms step (10 Hz refresh rate)
MODEL_PKG_PATH = os.path.join(PROJECT_ROOT, "models", "production_pipeline.pkl")

GESTURE_COLORS = {
    "REST": "#7f8c8d",       # Neutral Gray
    "Fist": "#e74c3c",       # Crimson Red
    "Open palm": "#3498db",  # Sky Blue
    "Pinch": "#2ecc71",      # Emerald Green
    "Wrist up": "#f39c12",   # Vibrant Amber
    "Wrist down": "#9b59b6", # Royal Purple
    "Waiting...": "#34495e"  # Dark Slate
}


# --- Background DAQ & Inference Engine ---
class RealTimeInferenceWorker:
    def __init__(self, data_queue, status_queue):
        self.data_queue = data_queue
        self.status_queue = status_queue
        self.running = False
        self.simulated_mode = not NIDAQ_AVAILABLE
        
        # Two-stage parameters
        self.rest_threshold_mv = 3.5  # Lowered default to 3.5 mV for responsive contraction detection
        self.rolling_buffer = np.zeros(WINDOW_SIZE * 2, dtype=np.float64)
        self.prediction_history = collections.deque(maxlen=5)  # K=5 majority filter
        self.loop_count = 0
        self.daq_channel = CHANNEL
        
        # Model & Profiles
        self.pipeline_pkg = {}
        self.current_profile = "Universal (All Cohort)"
        self.current_model_type = "svm"  # "svm", "rf", "lda"
        self.available_profiles = []
        
        # Calibration state
        self.calibrating_baseline = False
        self.calib_samples = []
        
        self.load_models()

    def _send_status(self, msg_type, val):
        if self.status_queue is not None:
            try:
                self.status_queue.put((msg_type, val))
            except Exception:
                pass

    def load_models(self):
        if os.path.exists(MODEL_PKG_PATH):
            try:
                with open(MODEL_PKG_PATH, "rb") as f:
                    self.pipeline_pkg = pickle.load(f)
                profiles = list(self.pipeline_pkg.keys())
                # Ensure Universal is first if available
                if "Universal (All Cohort)" in profiles:
                    profiles.remove("Universal (All Cohort)")
                    profiles = ["Universal (All Cohort)"] + profiles
                self.available_profiles = profiles
                self.current_profile = profiles[0]
                self._send_status("STATUS", f"Loaded {len(profiles)} profiles (Default: {self.current_profile})")
            except Exception as e:
                self._send_status("ERROR", f"Failed to load models: {e}")
        else:
            self._send_status("ERROR", "production_pipeline.pkl missing in models/!")

    def start(self, simulated=False):
        if self.running:
            return
        self.simulated_mode = simulated
        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def trigger_baseline_calibration(self, duration_sec=3.0):
        """Triggers 3-second resting baseline calculation."""
        self.calib_samples = []
        self.calibrating_baseline = True
        self.calib_end_time = time.time() + duration_sec
        self._send_status("CALIB_START", duration_sec)

    def _worker_loop(self):
        task = None
        current_chan = getattr(self, "daq_channel", CHANNEL)
        if not self.simulated_mode and NIDAQ_AVAILABLE:
            try:
                task = nidaqmx.Task()
                task.ai_channels.add_ai_voltage_chan(
                    current_chan,
                    terminal_config=TerminalConfiguration.DIFF,
                    min_val=-5.0,
                    max_val=5.0
                )
                task.timing.cfg_samp_clk_timing(
                    rate=FS,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=UPDATE_STEP * 4
                )
                task.start()
                self._send_status("HARDWARE", f"🟢 HARDWARE CONNECTED: NI-DAQ ({current_chan}) @ 1000 Hz")
            except Exception as e:
                self._send_status("HARDWARE", f"⚠️ NI-DAQ ERROR ({e}) -> Running Simulation Mode")
                self.simulated_mode = True
                if task:
                    try: task.close()
                    except: pass
                task = None
        else:
            self.simulated_mode = True
            self._send_status("HARDWARE", "⚠️ SIMULATION MODE (Virtual Biopotential Generator)")

        while self.running:
            try:
                # 1. Acquire raw 100 ms chunk (100 samples)
                if task and not self.simulated_mode:
                    raw_chunk = task.read(number_of_samples_per_channel=UPDATE_STEP)
                else:
                    time.sleep(UPDATE_STEP / FS)
                    # Simulated physical arm: 300 mV 50 Hz hum + baseline noise
                    t_sub = np.linspace(0, UPDATE_STEP/FS, UPDATE_STEP, endpoint=False)
                    raw_chunk = np.random.normal(0, 0.005, UPDATE_STEP) + 0.25 * np.sin(2 * np.pi * 50 * t_sub)

                raw_chunk = np.array(raw_chunk, dtype=np.float64)
                
                # Roll FIFO buffer
                self.rolling_buffer = np.roll(self.rolling_buffer, -UPDATE_STEP)
                self.rolling_buffer[-UPDATE_STEP:] = raw_chunk
                
                # Extract most recent 500 ms window
                raw_window = self.rolling_buffer[-WINDOW_SIZE:]

                # 2. Real-Time ARHD Signal Conditioning (50Hz hum removal + 20-450Hz Bandpass)
                raw_rms_mv = float(np.sqrt(np.mean(raw_window ** 2)) * 1000.0)
                clean_window = prep.filter_signal(raw_window, demean=True)
                clean_rms_mv = float(np.sqrt(np.mean(clean_window ** 2)) * 1000.0)

                # Compute hum attenuation efficiency
                hum_reduction = max(0.0, min(100.0, (1.0 - (clean_rms_mv / max(clean_rms_mv, raw_rms_mv))) * 100.0))

                # 3. Baseline Calibration Handler
                if self.calibrating_baseline:
                    self.calib_samples.append(clean_rms_mv)
                    remaining = max(0.0, self.calib_end_time - time.time())
                    self._send_status("CALIB_TICK", remaining)
                    if time.time() >= self.calib_end_time:
                        self.calibrating_baseline = False
                        if len(self.calib_samples) > 0:
                            mu_rest = np.mean(self.calib_samples)
                            sigma_rest = np.std(self.calib_samples)
                            # Set threshold to mu + 2.0 sigma (min 1.5 mV)
                            new_thresh = max(1.5, round(mu_rest + 2.0 * sigma_rest, 2))
                            self.rest_threshold_mv = new_thresh
                            self._send_status("CALIB_DONE", new_thresh)
                    continue

                # 4. Stage 1: Energy Gate (Rest / Idle Detection)
                is_gate_open = clean_rms_mv >= self.rest_threshold_mv
                if not is_gate_open:
                    self.prediction_history.append("REST")
                    final_gesture = "REST"
                    raw_pred = "REST"
                    confidence = 1.0
                    probs = np.zeros(len(cfg.GESTURES))
                else:
                    # 5. Stage 2: Active 16-Feature Classification
                    f_vec = feat_mod.extract_features(clean_window, fs=float(FS))
                    
                    profile_data = self.pipeline_pkg.get(self.current_profile, None)
                    if profile_data is not None:
                        scaler = profile_data["scaler"]
                        f_scaled = scaler.transform(f_vec.reshape(1, -1))
                        
                        if self.current_model_type == "svm":
                            clf = profile_data["svm_model"]
                        elif self.current_model_type == "rf":
                            clf = profile_data["rf_model"]
                        else:
                            clf = profile_data["lda_model"]
                            
                        # Scikit-learn alphabetical classes ground truth
                        classes = list(clf.classes_) if hasattr(clf, "classes_") else cfg.GESTURES
                        if hasattr(clf, "predict_proba"):
                            raw_probs = clf.predict_proba(f_scaled)[0]
                            max_idx = int(np.argmax(raw_probs))
                            raw_pred = str(classes[max_idx])
                            confidence = float(raw_probs[max_idx])
                            
                            # Reorder probabilities to match standard cfg.GESTURES order for UI bars
                            probs_dict = {cls_name: p for cls_name, p in zip(classes, raw_probs)}
                            probs = np.array([probs_dict.get(g, 0.0) for g in cfg.GESTURES], dtype=np.float64)
                        else:
                            raw_pred = str(clf.predict(f_scaled)[0])
                            confidence = 0.85
                            probs = np.zeros(len(cfg.GESTURES), dtype=np.float64)
                            if raw_pred in cfg.GESTURES:
                                probs[cfg.GESTURES.index(raw_pred)] = 1.0
                    else:
                        raw_pred = "Waiting..."
                        confidence = 0.0
                        probs = np.zeros(len(cfg.GESTURES))

                    # 6. Debouncing: Rolling Majority Vote (K=5 Buffer)
                    self.prediction_history.append(raw_pred)
                    vals, counts = np.unique(self.prediction_history, return_counts=True)
                    final_gesture = vals[np.argmax(counts)]

                # Live Console Telemetry (every 1 second = 10 frames)
                self.loop_count += 1
                if self.loop_count % 10 == 0:
                    mode_str = "HARDWARE" if not self.simulated_mode else "SIMULATION"
                    if is_gate_open:
                        print(f"[{mode_str}] Clean RMS: {clean_rms_mv:5.1f} mV | Gate: OPEN | Raw: {raw_pred:10s} -> Filtered: {final_gesture:10s} ({confidence*100:4.1f}%) | Model: {self.current_model_type.upper()} ({self.current_profile})", flush=True)
                    else:
                        contraction_hint = " (Muscle flexing detected! Lower slider on left to open gate)" if clean_rms_mv > 1.8 else ""
                        print(f"[{mode_str}] Clean RMS: {clean_rms_mv:5.1f} mV < Thresh: {self.rest_threshold_mv:.1f} mV | Gate: REST{contraction_hint}", flush=True)

                # Push frame to GUI display queue
                if not self.data_queue.full():
                    self.data_queue.put({
                        "raw_waveform": raw_window * 1000.0,     # in mV
                        "clean_waveform": clean_window * 1000.0, # in mV
                        "raw_rms_mv": raw_rms_mv,
                        "rms_mv": clean_rms_mv,
                        "hum_reduction": hum_reduction,
                        "threshold_mv": self.rest_threshold_mv,
                        "gate_open": is_gate_open,
                        "gesture": final_gesture,
                        "confidence": confidence,
                        "probabilities": probs,
                        "gestures": cfg.GESTURES
                    })

            except Exception as e:
                time.sleep(0.05)

            except Exception as e:
                time.sleep(0.05)

        if task:
            try:
                task.stop()
                task.close()
            except:
                pass


# --- Live GUI Application ---
class LiveInferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("sEMG Live Inference System | Two-Stage Gesture Recognition")
        self.root.geometry("1260x820")
        self.root.configure(bg="#1e1e24")
        self.root.minsize(1050, 720)

        self.data_queue = queue.Queue(maxsize=3)
        self.status_queue = queue.Queue(maxsize=10)
        self.worker = RealTimeInferenceWorker(self.data_queue, self.status_queue)
        self.show_dual_view = False  # Toggle between single filtered vs raw+filtered

        self._build_ui()
        self.worker.start(simulated=not NIDAQ_AVAILABLE)
        self.root.after(50, self._gui_update_loop)

    def _build_ui(self):
        # Top Header Banner
        header = tk.Frame(self.root, bg="#2b2d42", height=60)
        header.pack(fill=tk.X, side=tk.TOP)
        
        lbl_title = tk.Label(header, text="SINGLE-CHANNEL sEMG LIVE INFERENCE ENGINE", font=("Segoe UI", 14, "bold"), fg="#ffffff", bg="#2b2d42")
        lbl_title.pack(side=tk.LEFT, padx=20, pady=12)

        self.lbl_hw_status = tk.Label(header, text="Checking Hardware...", font=("Segoe UI", 10, "bold"), fg="#48cae4", bg="#2b2d42")
        self.lbl_hw_status.pack(side=tk.RIGHT, padx=20)

        # Control Panel (Left / Sidebar)
        sidebar = tk.Frame(self.root, bg="#2b2d42", width=340)
        sidebar.pack(fill=tk.Y, side=tk.LEFT, padx=12, pady=12)

        # 1. Profile Selection
        tk.Label(sidebar, text="SUBJECT / COHORT PROFILE", font=("Segoe UI", 10, "bold"), fg="#edf2f4", bg="#2b2d42").pack(anchor=tk.W, padx=16, pady=(12, 4))
        self.combo_profile = ttk.Combobox(sidebar, values=self.worker.available_profiles, state="readonly", font=("Segoe UI", 10))
        self.combo_profile.set(self.worker.current_profile)
        self.combo_profile.pack(fill=tk.X, padx=16, pady=(0, 2))
        self.combo_profile.bind("<<ComboboxSelected>>", self._on_profile_change)
        
        lbl_prof_note = tk.Label(sidebar, text="• Pick YOUR calibrated name (e.g. harsha / guru_raj)\n• Or 'Universal' for non-calibrated subject", font=("Segoe UI", 8), fg="#48cae4", bg="#2b2d42", justify=tk.LEFT)
        lbl_prof_note.pack(anchor=tk.W, padx=16, pady=(0, 10))

        # 2. Classifier Architecture
        tk.Label(sidebar, text="CLASSIFIER ENGINE", font=("Segoe UI", 10, "bold"), fg="#edf2f4", bg="#2b2d42").pack(anchor=tk.W, padx=16, pady=(6, 4))
        self.combo_model = ttk.Combobox(sidebar, values=["Support Vector Machine (SVM-RBF)", "Random Forest (RF)", "Linear Discriminant (LDA)"], state="readonly", font=("Segoe UI", 10))
        self.combo_model.set("Support Vector Machine (SVM-RBF)")
        self.combo_model.pack(fill=tk.X, padx=16, pady=(0, 10))
        self.combo_model.bind("<<ComboboxSelected>>", self._on_model_change)

        # 3. Stage 1 Energy Gate Calibration
        tk.Label(sidebar, text="STAGE 1: REST ENERGY GATE", font=("Segoe UI", 10, "bold"), fg="#edf2f4", bg="#2b2d42").pack(anchor=tk.W, padx=16, pady=(6, 4))
        self.btn_calib = tk.Button(sidebar, text="⚡ Quick Rest Calibration (3s)", font=("Segoe UI", 10, "bold"), bg="#3a86ff", fg="#ffffff", activebackground="#2667ff", cursor="hand2", relief=tk.FLAT, pady=5, command=self._start_calibration)
        self.btn_calib.pack(fill=tk.X, padx=16, pady=(0, 6))

        self.lbl_calib_status = tk.Label(sidebar, text="Rest Threshold: 3.5 mV", font=("Segoe UI", 9), fg="#8d99ae", bg="#2b2d42")
        self.lbl_calib_status.pack(anchor=tk.W, padx=16, pady=(0, 2))

        self.slider_thresh = tk.Scale(sidebar, from_=1.0, to=30.0, resolution=0.2, orient=tk.HORIZONTAL, bg="#2b2d42", fg="#ffffff", highlightthickness=0, command=self._on_slider_change)
        self.slider_thresh.set(3.5)
        self.slider_thresh.pack(fill=tk.X, padx=16, pady=(0, 10))

        # 4. Hardware DAQ Channel Selector
        tk.Label(sidebar, text="NI-DAQ CHANNEL", font=("Segoe UI", 10, "bold"), fg="#edf2f4", bg="#2b2d42").pack(anchor=tk.W, padx=16, pady=(6, 4))
        f_daq = tk.Frame(sidebar, bg="#2b2d42")
        f_daq.pack(fill=tk.X, padx=16, pady=(0, 10))
        self.combo_chan = ttk.Combobox(f_daq, values=["Dev1/ai0", "Dev2/ai0", "Dev1/ai1", "Dev2/ai1"], width=10, font=("Segoe UI", 9))
        self.combo_chan.set("Dev1/ai0")
        self.combo_chan.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        btn_reconnect = tk.Button(f_daq, text="Connect", font=("Segoe UI", 9, "bold"), bg="#06d6a0", fg="#000000", relief=tk.FLAT, command=self._reconnect_daq)
        btn_reconnect.pack(side=tk.RIGHT)

        # 5. View Switch: Single Filtered vs Raw+Filtered
        tk.Label(sidebar, text="OSCILLOSCOPE VIEW", font=("Segoe UI", 10, "bold"), fg="#edf2f4", bg="#2b2d42").pack(anchor=tk.W, padx=16, pady=(6, 4))
        self.btn_view = tk.Button(sidebar, text="👁️ Toggle Raw vs Filtered View", font=("Segoe UI", 9, "bold"), bg="#4a4e69", fg="#ffffff", activebackground="#22223b", relief=tk.FLAT, pady=4, command=self._toggle_view)
        self.btn_view.pack(fill=tk.X, padx=16, pady=(0, 10))

        # 6. Mode Switch: Hardware vs Simulation
        tk.Label(sidebar, text="OPERATION MODE", font=("Segoe UI", 10, "bold"), fg="#edf2f4", bg="#2b2d42").pack(anchor=tk.W, padx=16, pady=(6, 4))
        self.btn_mode = tk.Button(sidebar, text="Switch to Simulation Mode" if NIDAQ_AVAILABLE else "Simulation Mode (DAQ Missing)", font=("Segoe UI", 9), bg="#5c677d", fg="#ffffff", activebackground="#33415c", relief=tk.FLAT, pady=4, command=self._toggle_mode)
        self.btn_mode.pack(fill=tk.X, padx=16, pady=(0, 10))

        # Main Dashboard Display (Right)
        main_panel = tk.Frame(self.root, bg="#1e1e24")
        main_panel.pack(fill=tk.BOTH, expand=True, side=tk.RIGHT, padx=12, pady=12)

        # Big Gesture Banner
        self.banner_frame = tk.Frame(main_panel, bg="#7f8c8d", height=95)
        self.banner_frame.pack(fill=tk.X, side=tk.TOP, pady=(0, 8))
        self.banner_frame.pack_propagate(False)

        self.lbl_gesture = tk.Label(self.banner_frame, text="REST / IDLE", font=("Segoe UI", 26, "bold"), fg="#ffffff", bg="#7f8c8d")
        self.lbl_gesture.pack(pady=(4, 0))
        
        self.lbl_subtext = tk.Label(self.banner_frame, text="Arm Relaxed | Ready for Gesture Contraction", font=("Segoe UI", 10), fg="#f1faee", bg="#7f8c8d")
        self.lbl_subtext.pack(pady=(0, 4))

        # Live Signal Diagnostics Header
        self.lbl_metrics = tk.Label(main_panel, text="Raw: 0.0 mV | Filtered: 0.0 mV | Gate: REST | Conf: 100%", font=("Segoe UI", 10, "bold"), fg="#00f5d4", bg="#1e1e24")
        self.lbl_metrics.pack(anchor=tk.W, pady=(0, 6))

        # Plots: Oscilloscope + Probabilities
        self.fig = Figure(figsize=(7, 5), dpi=100, facecolor="#1e1e24")
        self.ax_wave = self.fig.add_subplot(211)
        self.ax_bar = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=2.2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=main_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._init_plots()

    def _init_plots(self):
        # Oscilloscope
        self.ax_wave.set_facecolor("#121217")
        self.ax_wave.tick_params(colors="#8d99ae")
        self.ax_wave.grid(True, linestyle=":", alpha=0.3, color="#8d99ae")
        self.ax_wave.set_ylabel("Signal (mV)", fontdict={"weight": "bold", "color": "#edf2f4", "size": 9})
        self.ax_wave.set_ylim(-35, 35)
        self.ax_wave.set_xlim(0, 500)
        t_ms = np.linspace(0, 500, WINDOW_SIZE)
        
        # Two traces: Raw (red/orange, faint) and Filtered (cyan, bright)
        self.line_raw, = self.ax_wave.plot(t_ms, np.zeros(WINDOW_SIZE), color="#e63946", lw=0.9, alpha=0.4, label="Raw Stream (50Hz Mains Hum)")
        self.line_clean, = self.ax_wave.plot(t_ms, np.zeros(WINDOW_SIZE), color="#00f5d4", lw=1.3, label="Clean Filtered sEMG (Physiological)")
        self.line_gate = self.ax_wave.axhline(3.5, color="#f1faee", linestyle="--", lw=1.2, label="Rest Gate Threshold (3.5 mV)")
        self.line_raw.set_visible(self.show_dual_view)
        
        self.ax_wave.legend(loc="upper right", facecolor="#1e1e24", edgecolor="none", labelcolor="#edf2f4", fontsize=8)

        # Bar Chart
        self.ax_bar.set_facecolor("#121217")
        self.ax_bar.tick_params(colors="#8d99ae")
        self.ax_bar.set_ylabel("Probability", fontdict={"weight": "bold", "color": "#edf2f4", "size": 9})
        self.ax_bar.set_ylim(0, 1.0)
        colors = [GESTURE_COLORS.get(g, "#3498db") for g in cfg.GESTURES]
        self.bars = self.ax_bar.bar(cfg.GESTURES, np.zeros(len(cfg.GESTURES)), color=colors, width=0.55)

    def _start_calibration(self):
        self.btn_calib.config(state=tk.DISABLED, text="Calibrating... (Keep Arm Relaxed)")
        self.worker.trigger_baseline_calibration(duration_sec=3.0)

    def _on_slider_change(self, val):
        self.worker.rest_threshold_mv = float(val)
        self.line_gate.set_ydata([float(val), float(val)])
        self.lbl_calib_status.config(text=f"Rest Threshold: {float(val):.1f} mV")

    def _on_profile_change(self, event):
        selected = self.combo_profile.get()
        self.worker.current_profile = selected
        print(f"[UI] Switched profile to: {selected}")

    def _on_model_change(self, event):
        m = self.combo_model.get()
        if "SVM" in m:
            self.worker.current_model_type = "svm"
        elif "Random Forest" in m:
            self.worker.current_model_type = "rf"
        else:
            self.worker.current_model_type = "lda"
        print(f"[UI] Switched model to: {self.worker.current_model_type}")

    def _toggle_view(self):
        self.show_dual_view = not self.show_dual_view
        self.line_raw.set_visible(self.show_dual_view)
        mode_str = "Showing Raw + Filtered Overlay" if self.show_dual_view else "Showing Clean Filtered View Only"
        self.btn_view.config(text=f"👁️ View: {'Raw+Clean' if self.show_dual_view else 'Clean Only'}")
        print(f"[UI] {mode_str}")

    def _toggle_mode(self):
        new_mode = not self.worker.simulated_mode
        self.worker.stop()
        time.sleep(0.2)
        self.worker.start(simulated=new_mode)
        self.btn_mode.config(text="Switch to Hardware Mode" if new_mode else "Switch to Simulation Mode")

    def _reconnect_daq(self):
        selected_chan = self.combo_chan.get().strip()
        self.worker.daq_channel = selected_chan
        print(f"[UI] Reconnecting NI-DAQ on: {selected_chan}")
        self.worker.stop()
        time.sleep(0.3)
        self.worker.start(simulated=False)

    def _gui_update_loop(self):
        # 1. Process Status Events
        while not self.status_queue.empty():
            msg_type, val = self.status_queue.get_nowait()
            if msg_type == "HARDWARE":
                self.lbl_hw_status.config(text=val)
            elif msg_type == "CALIB_TICK":
                self.lbl_calib_status.config(text=f"Relax arm... {val:.1f}s remaining")
            elif msg_type == "CALIB_DONE":
                self.btn_calib.config(state=tk.NORMAL, text="⚡ Quick Rest Calibration (3s)")
                self.slider_thresh.set(val)
                self.lbl_calib_status.config(text=f"Rest Threshold: {val:.1f} mV (Saved)")

        # 2. Process Live Data Frames
        if not self.data_queue.empty():
            frame = self.data_queue.get_nowait()
            raw_wave = frame["raw_waveform"]
            clean_wave = frame["clean_waveform"]
            raw_rms = frame["raw_rms_mv"]
            clean_rms = frame["rms_mv"]
            hum_red = frame["hum_reduction"]
            thresh_mv = frame["threshold_mv"]
            gate_open = frame.get("gate_open", False)
            gesture = frame["gesture"]
            conf = frame["confidence"]
            probs = frame["probabilities"]

            # Update Banner Color and Text
            bg_color = GESTURE_COLORS.get(gesture, "#7f8c8d")
            self.banner_frame.config(bg=bg_color)
            self.lbl_gesture.config(text=gesture.upper(), bg=bg_color)
            self.lbl_subtext.config(bg=bg_color)
            
            if not gate_open:
                if clean_rms > 1.8:
                    self.lbl_subtext.config(text=f"⚠️ Flexing ({clean_rms:.1f} mV) below gate ({thresh_mv:.1f} mV) -> Lower slider on left to trigger!")
                else:
                    self.lbl_subtext.config(text=f"Arm Relaxed (Energy: {clean_rms:.1f} mV < Gate: {thresh_mv:.1f} mV)")
            else:
                self.lbl_subtext.config(text=f"Confidence: {conf*100:.1f}% | K=5 Rolling Filter Active")
            
            gate_status = "REST / IDLE" if not gate_open else "ACTIVE CONTRACTION"
            self.lbl_metrics.config(
                text=f"Raw RMS: {raw_rms:5.1f} mV | Clean RMS: {clean_rms:4.1f} mV ({hum_red:4.1f}% Hum Scrubbed) | Gate: {gate_status} (Thresh: {thresh_mv:.1f} mV) | Conf: {conf*100:4.1f}%"
            )

            # Update Waveform Plot
            self.line_clean.set_ydata(clean_wave)
            if self.show_dual_view:
                self.line_raw.set_ydata(raw_wave)
                y_max = max(50.0, np.max(np.abs(raw_wave)) * 1.1)
            else:
                y_max = max(25.0, np.max(np.abs(clean_wave)) * 1.3)
                
            self.ax_wave.set_ylim(-y_max, y_max)
            self.line_gate.set_ydata([thresh_mv, thresh_mv])

            # Update Bar Chart
            for bar, p in zip(self.bars, probs):
                bar.set_height(p)

            self.canvas.draw_idle()

        self.root.after(40, self._gui_update_loop)

    def on_closing(self):
        self.worker.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = LiveInferenceApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
