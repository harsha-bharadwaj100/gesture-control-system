"""
========================================================================================
Advanced Live GUI Inference & Guided Calibration System
Project: Single-Channel Surface EMG Gesture Recognition (15-Subject Generalized Pipeline)
Application: Real-Time Hand Gesture Recognition & Sign-to-Speech Assistive Interface
========================================================================================
Features:
  - Multi-stage guided calibration with customizable preparation countdowns (3..2..1)
  - Real-time sEMG oscilloscope (Filtered Signal & TKEO Energy Envelope)
  - Live 6-Class Probability Distribution Bar Chart
  - Sign-to-Speech translation engine (Text-to-Speech audio synthesizer)
  - Automatic fallback to Simulation Mode if NI-DAQ hardware is not detected
  - Temporal majority voting (5-window debounce) for rock-solid stability
========================================================================================
"""

import os
import time
import queue
import threading
import collections
import warnings
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import skew, kurtosis
import joblib

# GUI Imports
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Text-to-Speech Import
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# DAQ Hardware Import
try:
    import nidaqmx
    from nidaqmx.constants import TerminalConfiguration, AcquisitionType
    NIDAQ_AVAILABLE = True
except (ImportError, Exception):
    NIDAQ_AVAILABLE = False

warnings.filterwarnings('ignore')

# --- DSP & Acquisition Parameters ---
CHANNEL = "Dev1/ai0"
FS = 1000               # 1000 Hz sampling frequency
WINDOW_SIZE = 1500      # 1.5-second feature window (1500 samples)
UPDATE_STEP = 200       # 200ms slide step (5 Hz update rate)
BUFFER_SIZE = 100       # Read chunk from DAQ
MODEL_FILE = "generalized_ensemble_model.pkl"
CLASSES_FILE = "generalized_classes.npy"

# Sign-to-Speech Mapping
SIGN_TO_SPEECH = {
    "Fist": "Stop! Need assistance.",
    "Open palm": "Hello! Welcome.",
    "Pinch": "Select or Confirm.",
    "Wrist up": "Yes, I agree.",
    "Wrist down": "No, I disagree.",
    "Rest": ""
}

GESTURE_COLORS = {
    "Fist": "#E74C3C",        # Red
    "Open palm": "#2ECC71",   # Green
    "Pinch": "#F39C12",       # Orange
    "Wrist up": "#3498DB",     # Blue
    "Wrist down": "#9B59B6",   # Purple
    "Rest": "#7F8C8D",         # Gray
    "Uncertain": "#34495E"    # Dark Slate
}

# --- Signal Processing Core ---
def notch_filter(data, fs=FS, f0=50.0, Q=30.0):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, data)

def bandpass_filter(data, fs=FS, lowcut=20.0, highcut=450.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)

def tkeo(data):
    y = np.zeros_like(data)
    y[1:-1] = data[1:-1]**2 - data[:-2] * data[2:]
    return y

def envelope(data, window=50):
    return pd.Series(np.abs(data)).rolling(window, min_periods=1, center=True).mean().values

def hjorth_parameters(signal):
    if len(signal) <= 1:
        return 0.0, 0.0, 0.0
    activity = np.var(signal)
    if activity == 0:
        return 0.0, 0.0, 0.0
    diff1 = np.diff(signal)
    mobility = np.sqrt(np.var(diff1) / activity)
    if mobility == 0:
        return activity, mobility, 0.0
    diff2 = np.diff(diff1)
    complexity = np.sqrt(np.var(diff2) / (np.var(diff1) + 1e-10)) / (mobility + 1e-10)
    return activity, mobility, complexity

def spectral_entropy(signal, fs=FS):
    fft_vals = np.abs(np.fft.rfft(signal))
    psd = fft_vals ** 2
    psd_sum = np.sum(psd)
    if psd_sum == 0:
        return 0.0
    psd_norm = psd / psd_sum
    psd_norm = psd_norm[psd_norm > 0]
    return -np.sum(psd_norm * np.log2(psd_norm))

def extract_features(signal, fs=FS):
    if len(signal) == 0:
        return [0.0] * 16
    mav = np.mean(np.abs(signal))
    rms = np.sqrt(np.mean(signal**2))
    var = np.var(signal)
    wl = np.sum(np.abs(np.diff(signal)))
    zc = np.sum(np.diff(np.signbit(signal)))
    ssc = np.sum(np.diff(np.sign(np.diff(signal))) != 0)
    
    env = envelope(signal, window=50)
    env_mean = np.mean(env)
    env_max = np.max(env)
    
    sk = skew(signal) if len(signal) > 2 else 0.0
    ku = kurtosis(signal) if len(signal) > 2 else 0.0
    
    activity, mobility, complexity = hjorth_parameters(signal)
    
    freqs = np.fft.rfftfreq(len(signal), d=1/fs)
    fft_vals = np.abs(np.fft.rfft(signal))
    total_power = np.sum(fft_vals)
    if total_power == 0:
        mean_freq = 0.0
        peak_freq = 0.0
    else:
        mean_freq = np.sum(freqs * fft_vals) / total_power
        peak_freq = freqs[np.argmax(fft_vals)]
        
    se = spectral_entropy(signal, fs)
    return [mav, rms, var, wl, zc, ssc, env_mean, env_max, activity, mobility, complexity, mean_freq, peak_freq, se, sk, ku]


# --- Speech Output Worker ---
class SpeechEngine:
    def __init__(self):
        self.queue = queue.Queue()
        self.enabled = True
        self.last_spoken = ""
        self.last_time = 0
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        engine = None
        if TTS_AVAILABLE:
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 160)
            except Exception:
                engine = None
        
        while True:
            text = self.queue.get()
            if not self.enabled or not text:
                continue
            now = time.time()
            # Debounce: avoid repeating the same phrase within 3 seconds
            if text == self.last_spoken and (now - self.last_time) < 3.0:
                continue
            self.last_spoken = text
            self.last_time = now
            
            if engine:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception:
                    pass

    def speak(self, text):
        if self.enabled and text:
            self.queue.put(text)


# --- Background DAQ & Inference Engine ---
class DAQInferenceEngine:
    def __init__(self, data_queue, status_queue):
        self.data_queue = data_queue
        self.status_queue = status_queue
        self.running = False
        self.simulated_mode = False
        self.model = None
        self.classes = []
        
        # User Scaling Profile
        self.global_mean = None
        self.global_std = None
        
        # Calibration state management
        self.calibration_active = False
        self.current_calib_step = None
        self.calib_countdown = 0
        self.calib_buffer = []
        
        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(MODEL_FILE) and os.path.exists(CLASSES_FILE):
                self.model = joblib.load(MODEL_FILE)
                self.classes = np.load(CLASSES_FILE)
                self.status_queue.put(("STATUS", f"Loaded Generalized Model ({len(self.classes)} Classes)"))
            else:
                self.status_queue.put(("ERROR", "Model files missing! Run generalized_model.py first."))
        except Exception as e:
            self.status_queue.put(("ERROR", f"Error loading model: {e}"))

    def start(self, simulated=False):
        if self.running:
            return
        self.simulated_mode = simulated
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread and self.thread.is_alive():
            if threading.current_thread() != self.thread:
                self.thread.join(timeout=0.5)

    def _generate_synthetic_emg(self, active=False, gesture_type="Fist"):
        """Generates realistic sEMG for simulation mode when hardware is disconnected."""
        t = np.linspace(0, UPDATE_STEP/FS, UPDATE_STEP, endpoint=False)
        base_noise = np.random.normal(0, 0.005, UPDATE_STEP)
        hum_50hz = 0.02 * np.sin(2 * np.pi * 50 * t + np.random.uniform(0, 2*np.pi))
        
        if not active:
            return base_noise + hum_50hz
        
        # Simulate active contraction burst
        burst_freq = np.random.uniform(60, 180)
        burst_amp = np.random.uniform(0.08, 0.25)
        emg_burst = burst_amp * np.sin(2 * np.pi * burst_freq * t) * np.random.normal(1.0, 0.3, UPDATE_STEP)
        return emg_burst + base_noise + hum_50hz

    def _run_loop(self):
        rolling_window = np.zeros(WINDOW_SIZE)
        prediction_history = collections.deque(maxlen=5)
        task = None
        
        # Check DAQ hardware availability
        if not self.simulated_mode and NIDAQ_AVAILABLE:
            try:
                task = nidaqmx.Task()
                task.ai_channels.add_ai_voltage_chan(
                    CHANNEL,
                    terminal_config=TerminalConfiguration.DIFF,
                    min_val=-5.0,
                    max_val=5.0
                )
                task.timing.cfg_samp_clk_timing(
                    rate=FS,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=BUFFER_SIZE * 4
                )
                task.start()
                self.status_queue.put(("STATUS", f"Connected to NI-DAQ ({CHANNEL}) @ 1000 Hz"))
            except Exception as e:
                self.status_queue.put(("STATUS", f"NI-DAQ not detected ({e}). Falling back to Simulation Mode."))
                self.simulated_mode = True
                if task:
                    try: task.close()
                    except: pass
                task = None
        else:
            self.simulated_mode = True
            self.status_queue.put(("STATUS", "Running in Simulation Mode (Virtual Signal Generator)"))

        while self.running:
            try:
                # 1. Read Data
                if task and not self.simulated_mode:
                    new_data = task.read(number_of_samples_per_channel=UPDATE_STEP)
                else:
                    time.sleep(UPDATE_STEP / FS)
                    new_data = self._generate_synthetic_emg(active=False)

                new_data = np.array(new_data)
                rolling_window = np.roll(rolling_window, -UPDATE_STEP)
                rolling_window[-UPDATE_STEP:] = new_data

                # 2. Filter Signal
                v_filt = rolling_window.copy()
                for harmonic in [50, 100, 150, 200, 250, 300, 350, 400]:
                    v_filt = notch_filter(v_filt, FS, f0=harmonic)
                v_filt = bandpass_filter(v_filt, FS, lowcut=20, highcut=450)
                
                v_tkeo = tkeo(v_filt)
                env_sig = envelope(v_tkeo, window=200)

                # 3. If Calibration Active, Accumulate Data
                if self.calibration_active:
                    self.calib_buffer.extend(new_data)
                    self.data_queue.put({
                        "type": "CALIBRATION_FRAME",
                        "raw_signal": new_data,
                        "filtered_signal": v_filt[-500:],
                        "envelope": env_sig[-500:]
                    })
                    continue

                # 4. Extract 16 Features
                features = extract_features(v_filt)
                
                # Standardize with user profile if available, else default unit scale
                if self.global_mean is not None and self.global_std is not None:
                    features_scaled = (np.array(features) - self.global_mean) / self.global_std
                else:
                    features_scaled = np.array(features)
                    
                X_live = features_scaled.reshape(1, -1)

                # 5. Model Inference
                if self.model is not None and len(self.classes) > 0:
                    try:
                        probs = self.model.predict_proba(X_live)[0]
                        max_idx = np.argmax(probs)
                        confidence = probs[max_idx]
                        detected_label = self.classes[max_idx]
                    except Exception:
                        probs = np.ones(len(self.classes)) / len(self.classes)
                        confidence = 0.0
                        detected_label = "Uncertain"
                else:
                    probs = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
                    confidence = 0.5
                    detected_label = "Rest"

                # 6. Temporal Smoothing & Debounce
                prediction_history.append(detected_label)
                most_common = max(set(prediction_history), key=prediction_history.count)
                
                if confidence >= 0.45 and prediction_history.count(most_common) >= 3:
                    final_gesture = most_common
                else:
                    final_gesture = "Uncertain" if confidence < 0.35 else most_common

                # 7. Push Frame to GUI
                self.data_queue.put({
                    "type": "INFERENCE_FRAME",
                    "filtered_signal": v_filt[-500:],
                    "envelope": env_sig[-500:],
                    "probabilities": probs,
                    "classes": self.classes,
                    "confidence": confidence,
                    "gesture": final_gesture
                })

            except Exception as e:
                time.sleep(0.1)

        if task:
            try:
                task.stop()
                task.close()
            except:
                pass


# --- Main User Interface ---
class SignToSpeechInferenceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EMG Sign-to-Speech Assistive Interface | 15-Subject Generalized System")
        self.root.geometry("1280x820")
        self.root.configure(bg="#1E1E2E")
        self.root.minsize(1100, 750)

        # Queues
        self.data_queue = queue.Queue(maxsize=5)
        self.status_queue = queue.Queue(maxsize=10)

        # Workers
        self.speech_engine = SpeechEngine()
        self.engine = DAQInferenceEngine(self.data_queue, self.status_queue)
        
        # State Variables
        self.is_running = False
        self.speech_enabled = tk.BooleanVar(value=True)
        self.simulated_var = tk.BooleanVar(value=False)
        self.calibrating = False
        self.calib_step_idx = 0
        self.calib_steps = []
        self.calib_total_data = []

        self._build_theme()
        self._build_layout()
        
        # Start GUI Update Poller
        self.root.after(40, self._gui_update_loop)

    def _build_theme(self):
        self.colors = {
            "bg": "#1E1E2E",
            "card_bg": "#2A2A3E",
            "accent": "#6C5CE7",
            "accent_hover": "#5B4BC4",
            "text": "#FFFFFF",
            "text_muted": "#A0A0B0",
            "green": "#2ECC71",
            "red": "#E74C3C",
            "yellow": "#F1C40F",
            "cyan": "#00D2D3",
            "border": "#3A3A52"
        }

    def _build_layout(self):
        # Header Frame
        header = tk.Frame(self.root, bg=self.colors["card_bg"], height=70, padx=25, pady=10)
        header.pack(fill=tk.X, side=tk.TOP)
        
        title_box = tk.Frame(header, bg=self.colors["card_bg"])
        title_box.pack(side=tk.LEFT)
        
        tk.Label(title_box, text="ANTHRIQ NEUROMOTOR INTERFACE", font=("Segoe UI", 16, "bold"), fg=self.colors["accent"], bg=self.colors["card_bg"]).pack(anchor="w")
        tk.Label(title_box, text="Single-Channel Surface EMG Sign-to-Speech System (15-Subject Model)", font=("Segoe UI", 10), fg=self.colors["text_muted"], bg=self.colors["card_bg"]).pack(anchor="w")

        # Top Right Controls
        ctrl_box = tk.Frame(header, bg=self.colors["card_bg"])
        ctrl_box.pack(side=tk.RIGHT)

        self.btn_tts = tk.Checkbutton(ctrl_box, text="🔊 Speech Output", variable=self.speech_enabled, font=("Segoe UI", 10, "bold"), fg=self.colors["cyan"], bg=self.colors["card_bg"], selectcolor=self.colors["bg"], activebackground=self.colors["card_bg"], command=self._toggle_speech)
        self.btn_tts.pack(side=tk.LEFT, padx=10)

        self.btn_sim = tk.Checkbutton(ctrl_box, text="Simulated DAQ", variable=self.simulated_var, font=("Segoe UI", 10), fg=self.colors["yellow"], bg=self.colors["card_bg"], selectcolor=self.colors["bg"], activebackground=self.colors["card_bg"])
        self.btn_sim.pack(side=tk.LEFT, padx=10)

        self.btn_calibrate = tk.Button(ctrl_box, text="⚙ Start Calibration", font=("Segoe UI", 11, "bold"), bg=self.colors["accent"], fg="white", activebackground=self.colors["accent_hover"], activeforeground="white", relief=tk.FLAT, padx=15, pady=6, cursor="hand2", command=self.start_calibration_wizard)
        self.btn_calibrate.pack(side=tk.LEFT, padx=5)

        # Body Container (Split into Left: Oscilloscope + Probabilities, Right: Detected Sign + Speech + Instructions)
        body = tk.Frame(self.root, bg=self.colors["bg"], padx=15, pady=15)
        body.pack(fill=tk.BOTH, expand=True)

        # --- LEFT PANEL: Signals & Probabilities ---
        left_panel = tk.Frame(body, bg=self.colors["bg"])
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Oscilloscope Card
        osc_card = tk.Frame(left_panel, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID, padx=15, pady=10)
        osc_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        tk.Label(osc_card, text="REAL-TIME BIO-POTENTIAL OSCILLOSCOPE (Dev1/ai0 @ 1000 Hz)", font=("Segoe UI", 11, "bold"), fg=self.colors["text"], bg=self.colors["card_bg"]).pack(anchor="w", pady=(0, 5))

        self.fig = Figure(figsize=(6, 3.2), dpi=100, facecolor=self.colors["card_bg"])
        self.ax_signal = self.fig.add_subplot(211)
        self.ax_env = self.fig.add_subplot(212)
        
        for ax, title, color in [(self.ax_signal, "Bandpassed Signal (20-450 Hz)", "#00D2D3"), (self.ax_env, "TKEO Energy Activation Envelope", "#F39C12")]:
            ax.set_facecolor("#161622")
            ax.tick_params(colors=self.colors["text_muted"], labelsize=8)
            ax.set_title(title, color=self.colors["text_muted"], fontsize=9, loc="left")
            ax.grid(True, color="#2D2D42", linestyle="--", linewidth=0.5)

        self.line_sig, = self.ax_signal.plot([], [], lw=1.2, color="#00D2D3")
        self.line_env, = self.ax_env.plot([], [], lw=1.5, color="#F39C12")
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=osc_card)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Probability Distribution Card
        prob_card = tk.Frame(left_panel, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID, padx=15, pady=10)
        prob_card.pack(fill=tk.X, expand=False)
        
        tk.Label(prob_card, text="CLASSIFICATION PROBABILITY DISTRIBUTION", font=("Segoe UI", 11, "bold"), fg=self.colors["text"], bg=self.colors["card_bg"]).pack(anchor="w", pady=(0, 8))

        self.prob_bars = {}
        self.prob_labels = {}
        prob_grid = tk.Frame(prob_card, bg=self.colors["card_bg"])
        prob_grid.pack(fill=tk.X)

        classes_display = ['Fist', 'Open palm', 'Pinch', 'Wrist up', 'Wrist down', 'Rest']
        for i, cls in enumerate(classes_display):
            row = tk.Frame(prob_grid, bg=self.colors["card_bg"])
            row.pack(fill=tk.X, pady=2)
            
            lbl = tk.Label(row, text=f"{cls:<12}", font=("Segoe UI", 9, "bold"), fg=self.colors["text_muted"], bg=self.colors["card_bg"], width=12, anchor="w")
            lbl.pack(side=tk.LEFT)
            
            pbar = ttk.Progressbar(row, orient=tk.HORIZONTAL, length=220, mode='determinate')
            pbar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
            
            val_lbl = tk.Label(row, text="0%", font=("Segoe UI", 9), fg=self.colors["text_muted"], bg=self.colors["card_bg"], width=5)
            val_lbl.pack(side=tk.RIGHT)
            
            self.prob_bars[cls] = pbar
            self.prob_labels[cls] = (lbl, val_lbl)

        # --- RIGHT PANEL: Gesture Card & Speech Output ---
        right_panel = tk.Frame(body, bg=self.colors["bg"], width=460)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        # Active Sign Output Card
        sign_card = tk.Frame(right_panel, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID, padx=20, pady=20)
        sign_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(sign_card, text="DETECTED GESTURE", font=("Segoe UI", 10, "bold"), fg=self.colors["text_muted"], bg=self.colors["card_bg"]).pack(anchor="center")

        self.lbl_gesture = tk.Label(sign_card, text="REST", font=("Segoe UI", 32, "bold"), fg=self.colors["green"], bg=self.colors["card_bg"], pady=10)
        self.lbl_gesture.pack(anchor="center")

        self.lbl_conf = tk.Label(sign_card, text="Confidence: 98%", font=("Segoe UI", 12), fg=self.colors["text_muted"], bg=self.colors["card_bg"])
        self.lbl_conf.pack(anchor="center")

        # Speech Synthesizer Output Card
        speech_card = tk.Frame(right_panel, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID, padx=20, pady=20)
        speech_card.pack(fill=tk.X, pady=(0, 10))

        tk.Label(speech_card, text="SIGN-TO-SPEECH TRANSLATION", font=("Segoe UI", 10, "bold"), fg=self.colors["text_muted"], bg=self.colors["card_bg"]).pack(anchor="w")

        self.lbl_speech = tk.Label(speech_card, text='"..."', font=("Segoe UI", 16, "italic", "bold"), fg=self.colors["cyan"], bg=self.colors["card_bg"], wraplength=400, justify="center", pady=15)
        self.lbl_speech.pack(anchor="center")

        # Calibration & System Status Card
        self.status_card = tk.Frame(right_panel, bg=self.colors["card_bg"], bd=1, relief=tk.SOLID, padx=20, pady=15)
        self.status_card.pack(fill=tk.BOTH, expand=True)

        tk.Label(self.status_card, text="SYSTEM STATUS & CALIBRATION", font=("Segoe UI", 10, "bold"), fg=self.colors["text_muted"], bg=self.colors["card_bg"]).pack(anchor="w")

        self.lbl_calib_title = tk.Label(self.status_card, text="Ready for Real-Time Inference", font=("Segoe UI", 12, "bold"), fg=self.colors["text"], bg=self.colors["card_bg"], pady=6)
        self.lbl_calib_title.pack(anchor="w")

        self.lbl_calib_desc = tk.Label(self.status_card, text="Click 'Start Calibration' before live testing to calibrate your baseline muscle impedance.", font=("Segoe UI", 10), fg=self.colors["text_muted"], bg=self.colors["card_bg"], wraplength=400, justify="left")
        self.lbl_calib_desc.pack(anchor="w", pady=(0, 10))

        self.calib_progress = ttk.Progressbar(self.status_card, orient=tk.HORIZONTAL, mode='determinate')
        self.calib_progress.pack(fill=tk.X, pady=(0, 15))

        # Bottom Action Buttons
        btn_box = tk.Frame(self.status_card, bg=self.colors["card_bg"])
        btn_box.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_run = tk.Button(btn_box, text="▶ Start Live Stream", font=("Segoe UI", 11, "bold"), bg=self.colors["green"], fg="white", activebackground="#27AE60", activeforeground="white", relief=tk.FLAT, padx=15, pady=8, cursor="hand2", command=self.toggle_live_inference)
        self.btn_run.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

    def _toggle_speech(self):
        self.speech_engine.enabled = self.speech_enabled.get()

    def toggle_live_inference(self):
        if not self.is_running:
            self.is_running = True
            self.btn_run.config(text="⏹ Stop Live Stream", bg=self.colors["red"])
            self.engine.start(simulated=self.simulated_var.get())
        else:
            self.is_running = False
            self.btn_run.config(text="▶ Start Live Stream", bg=self.colors["green"])
            self.engine.stop()

    # --- GUIDED CALIBRATION WIZARD WITH PREPARATION TIME ---
    def start_calibration_wizard(self):
        if self.calibrating:
            return
        
        # Define 6 Guided Calibration Steps with Preparation Countdowns
        self.calib_steps = [
            {
                "gesture": "Rest",
                "title": "STEP 1/6: BASELINE REST CALIBRATION",
                "instruction": "Keep your arm completely relaxed on the table. Do NOT contract any muscles.",
                "prep_time": 3,
                "record_time": 5
            },
            {
                "gesture": "Fist",
                "title": "STEP 2/6: FIST GESTURE",
                "instruction": "Form a firm, tight closed fist.",
                "prep_time": 3,
                "record_time": 3
            },
            {
                "gesture": "Open palm",
                "title": "STEP 3/6: OPEN PALM GESTURE",
                "instruction": "Spread your fingers and keep palm wide open.",
                "prep_time": 3,
                "record_time": 3
            },
            {
                "gesture": "Pinch",
                "title": "STEP 4/6: PINCH GESTURE",
                "instruction": "Pinch your thumb and index finger tips firmly together.",
                "prep_time": 3,
                "record_time": 3
            },
            {
                "gesture": "Wrist up",
                "title": "STEP 5/6: WRIST UP GESTURE",
                "instruction": "Flex your wrist upward towards your forearm.",
                "prep_time": 3,
                "record_time": 3
            },
            {
                "gesture": "Wrist down",
                "title": "STEP 6/6: WRIST DOWN GESTURE",
                "instruction": "Bend your wrist downward towards your palm.",
                "prep_time": 3,
                "record_time": 3
            }
        ]
        
        self.calibrating = True
        self.calib_step_idx = 0
        self.calib_total_data = []
        self.btn_calibrate.config(state=tk.DISABLED)
        
        # Start Engine in background if not running
        if not self.is_running:
            self.is_running = True
            self.btn_run.config(text="⏹ Stop Live Stream", bg=self.colors["red"])
            self.engine.start(simulated=self.simulated_var.get())
            
        self._run_calib_step_preparation()

    def _run_calib_step_preparation(self):
        if self.calib_step_idx >= len(self.calib_steps):
            self._finish_calibration()
            return
            
        step = self.calib_steps[self.calib_step_idx]
        self.lbl_calib_title.config(text=step["title"], fg=self.colors["yellow"])
        self.lbl_gesture.config(text=step["gesture"].upper(), fg=self.colors["yellow"])
        
        # 3-Second Preparation Countdown Loop
        prep_seconds = step["prep_time"]
        self.speech_engine.speak(f"Prepare for {step['gesture']}")
        
        def countdown(remaining):
            if remaining > 0:
                self.lbl_calib_desc.config(
                    text=f"👉 PREPARE: {step['instruction']}\n\n⏳ GET READY IN: {remaining} seconds..."
                )
                self.calib_progress['value'] = (1 - (remaining / prep_seconds)) * 100
                self.root.after(1000, lambda: countdown(remaining - 1))
            else:
                self._run_calib_step_recording()

        countdown(prep_seconds)

    def _run_calib_step_recording(self):
        step = self.calib_steps[self.calib_step_idx]
        self.lbl_calib_title.config(text=f"🔴 RECORDING: {step['gesture'].upper()}", fg=self.colors["red"])
        self.speech_engine.speak(f"Hold {step['gesture']}")
        
        record_seconds = step["record_time"]
        self.engine.calib_buffer = []
        self.engine.calibration_active = True
        
        start_time = time.time()
        
        def record_tick():
            elapsed = time.time() - start_time
            remaining = max(0.0, record_seconds - elapsed)
            
            self.lbl_calib_desc.config(
                text=f"✊ NOW HOLD {step['gesture'].upper()} STEADILY!\n\n⏱ Time Remaining: {remaining:.1f}s"
            )
            self.calib_progress['value'] = (elapsed / record_seconds) * 100
            
            if elapsed < record_seconds:
                self.root.after(100, record_tick)
            else:
                self.engine.calibration_active = False
                self.calib_total_data.extend(self.engine.calib_buffer)
                self.calib_step_idx += 1
                
                # Brief 1.5s Relax Transition before next step
                self.lbl_calib_title.config(text="🛑 RELAX MUSCLES", fg=self.colors["green"])
                self.lbl_calib_desc.config(text="Great! Relax your arm completely.")
                self.root.after(1500, self._run_calib_step_preparation)

        record_tick()

    def _finish_calibration(self):
        self.lbl_calib_title.config(text="✅ CALIBRATING USER SCALING PROFILE...", fg=self.colors["cyan"])
        self.lbl_calib_desc.config(text="Extracting 16 Hjorth & spectral features to construct personal z^(p) scaler...")
        
        def compute_profile():
            raw_calib = np.array(self.calib_total_data)
            if len(raw_calib) > 1500:
                # Filter calibration data
                v_filt = raw_calib
                for harmonic in [50, 100, 150, 200, 250, 300, 350, 400]:
                    v_filt = notch_filter(v_filt, FS, f0=harmonic)
                v_filt = bandpass_filter(v_filt, FS, lowcut=20, highcut=450)
                
                # Extract chunk features
                feats_list = []
                chunk_len = int(1.5 * FS)
                for i in range(0, len(v_filt) - chunk_len, int(0.5 * FS)):
                    feats_list.append(extract_features(v_filt[i:i+chunk_len]))
                    
                feats_arr = np.array(feats_list)
                self.engine.global_mean = np.mean(feats_arr, axis=0)
                self.engine.global_std = np.std(feats_arr, axis=0) + 1e-8
                
                # Save profile
                np.save("user_calibration_profile.npy", {"mean": self.engine.global_mean, "std": self.engine.global_std})
            
            self.root.after(0, self._on_calibration_complete)

        threading.Thread(target=compute_profile, daemon=True).start()

    def _on_calibration_complete(self):
        self.calibrating = False
        self.btn_calibrate.config(state=tk.NORMAL)
        self.lbl_calib_title.config(text="🎉 CALIBRATION COMPLETE & ACTIVE!", fg=self.colors["green"])
        self.lbl_calib_desc.config(text="Personal z^(p) scaling matrix computed! You can now perform gestures freely.")
        self.calib_progress['value'] = 100
        self.speech_engine.speak("Calibration complete. System ready.")
        
        # If live inference wasn't active, start it
        if not self.is_running:
            self.toggle_live_inference()

    # --- GUI REAL-TIME POLLING LOOP ---
    def _gui_update_loop(self):
        # 1. Process Status Messages
        try:
            while not self.status_queue.empty():
                msg_type, msg = self.status_queue.get_nowait()
                if msg_type == "ERROR":
                    messagebox.showerror("Error", msg)
                elif not self.calibrating:
                    self.lbl_calib_title.config(text=msg)
        except:
            pass

        # 2. Process Real-Time Signal & Classification Frames
        try:
            while not self.data_queue.empty():
                frame = self.data_queue.get_nowait()
                
                if frame["type"] in ("INFERENCE_FRAME", "CALIBRATION_FRAME"):
                    sig = frame.get("filtered_signal", [])
                    env_sig = frame.get("envelope", [])
                    
                    if len(sig) > 0:
                        x = np.arange(len(sig))
                        self.line_sig.set_data(x, sig)
                        self.line_env.set_data(x, env_sig)
                        
                        self.ax_signal.set_xlim(0, len(sig))
                        self.ax_env.set_xlim(0, len(env_sig))
                        
                        max_val = max(0.001, np.max(np.abs(sig)) * 1.2)
                        self.ax_signal.set_ylim(-max_val, max_val)
                        self.ax_env.set_ylim(0, max(0.001, np.max(env_sig) * 1.2))
                        
                        self.canvas.draw_idle()

                if frame["type"] == "INFERENCE_FRAME" and not self.calibrating:
                    gesture = frame["gesture"]
                    conf = frame["confidence"]
                    probs = frame["probabilities"]
                    classes = frame["classes"]
                    
                    # Update Gesture Badge
                    self.lbl_gesture.config(text=gesture.upper(), fg=GESTURE_COLORS.get(gesture, self.colors["green"]))
                    self.lbl_conf.config(text=f"Confidence: {conf*100:.1f}%")
                    
                    # Update Sign-to-Speech Output
                    spoken_text = SIGN_TO_SPEECH.get(gesture, "")
                    if spoken_text:
                        self.lbl_speech.config(text=f'"{spoken_text}"', fg=self.colors["cyan"])
                        self.speech_engine.speak(spoken_text)
                    elif gesture == "Rest":
                        self.lbl_speech.config(text='"[Idle / Waiting for Sign]"', fg=self.colors["text_muted"])
                    else:
                        self.lbl_speech.config(text='"[Uncertain movement]"', fg=self.colors["yellow"])

                    # Update Probability Bars
                    for i, cls_name in enumerate(classes):
                        if cls_name in self.prob_bars:
                            p_val = probs[i] * 100
                            self.prob_bars[cls_name]['value'] = p_val
                            lbl, val_lbl = self.prob_labels[cls_name]
                            val_lbl.config(text=f"{p_val:.0f}%")
                            
                            if cls_name == gesture:
                                lbl.config(fg=self.colors["cyan"])
                                val_lbl.config(fg=self.colors["cyan"], font=("Segoe UI", 9, "bold"))
                            else:
                                lbl.config(fg=self.colors["text_muted"])
                                val_lbl.config(fg=self.colors["text_muted"], font=("Segoe UI", 9))

        except Exception:
            pass

        self.root.after(40, self._gui_update_loop)


def main():
    root = tk.Tk()
    app = SignToSpeechInferenceApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.engine.stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
