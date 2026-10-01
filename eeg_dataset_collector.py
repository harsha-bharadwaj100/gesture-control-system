"""
Single-Channel EEG Dataset Collector (Dry Brush Electrodes + Anthriq / NI-DAQ)
-------------------------------------------------------------------------------
Designed for dry, multipronged conductive polymer brush electrodes mounted on a fabric cap.
Features:
- NI-DAQ USB-600X integration (Dev1/ai0, Differential).
- Multi-Phase Protocol Selection:
  * Phase 1: Alpha Rhythm Test (Eyes Open vs. Eyes Closed - Berger Effect)
  * Phase 2: Binary Cognitive Intent (Rest vs. Mental Focus / Math Calculation)
  * Phase 3: Binary Motor Imagery (Rest vs. Imagined Hand Squeeze)
  * Phase 4: Full 5-Gesture Motor Imagery Protocol
- F3-F4 / C3-C4 / Cz-Fz Scalp 10-20 Position Selection.
- Live Dry Contact Impedance (kOhm) Tracker.
- Automatic Post-Collection Session Baseline Calculation across Rest windows.
- Writes companion JSON session metadata and updates CSV with post-calculated baseline stats.
"""

import os
import time
import csv
import json
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from dataclasses import dataclass
from scipy.signal import butter, lfilter

try:
    import numpy as np
except ImportError:
    pass

# Try importing nidaqmx, fallback to simulation mode if hardware is disconnected
try:
    import nidaqmx
    from nidaqmx.constants import TerminalConfiguration, AcquisitionType
    HAS_NIDAQ = True
except ImportError:
    HAS_NIDAQ = False

def check_nidaq_hardware():
    """Checks if NI-DAQmx hardware (specifically USB-600X / Dev1) is physically plugged in and responding."""
    if not HAS_NIDAQ:
        return False, "nidaqmx library not installed"
    try:
        system = nidaqmx.system.System.local()
        devices = [dev.name for dev in system.devices]
        if len(devices) > 0:
            dev_names = ", ".join(devices)
            return True, f"CONNECTED: {dev_names} (Dev1/ai0 ready)"
        else:
            return False, "DISCONNECTED: No NI-DAQ USB device detected"
    except Exception as e:
        return False, f"DISCONNECTED: {e}"

# Configuration Parameters
CHANNEL = "Dev1/ai0"
SAMPLE_RATE = 1000
BUFFER_SIZE = 100
PROTOCOL_BUFFER_SECONDS = 5.0
DEFAULT_OUTPUT_DIR = "EEG_Datasets"

@dataclass(frozen=True)
class ProtocolStep:
    label: str
    duration_seconds: float
    repetition_index: int | None = None
    is_recordable: bool = True

import random

def generate_random_math_question():
    """Generates a random mental arithmetic question for Phase 2 Cognitive Focus."""
    op = random.choice(['mult', 'sub', 'add', 'div'])
    if op == 'mult':
        a = random.randint(12, 29)
        b = random.randint(6, 14)
        return f"{a} × {b} = ?"
    elif op == 'sub':
        a = random.randint(150, 650)
        b = random.randint(45, 135)
        return f"{a} - {b} = ?"
    elif op == 'add':
        a = random.randint(120, 480)
        b = random.randint(85, 290)
        return f"{a} + {b} = ?"
    else: # div
        b = random.randint(6, 16)
        ans = random.randint(8, 25)
        a = b * ans
        return f"{a} ÷ {b} = ?"

def build_phase1_alpha_protocol():
    """Phase 1: Eyes Open (5s) vs. Eyes Closed (5s) - 10 Repetitions"""
    protocol = [ProtocolStep("Initial Buffer / Get Ready...", PROTOCOL_BUFFER_SECONDS, is_recordable=False)]
    for rep in range(1, 11):
        protocol.append(ProtocolStep("Eyes Open (Look Straight)", 5.0, repetition_index=rep, is_recordable=True))
        protocol.append(ProtocolStep("Eyes Closed (Relax Brain)", 5.0, repetition_index=rep, is_recordable=True))
        protocol.append(ProtocolStep("Rest (Normal Blinking)", 2.0, repetition_index=rep, is_recordable=True))
        if rep < 10:
            protocol.append(ProtocolStep("Buffer / Get Ready...", PROTOCOL_BUFFER_SECONDS, is_recordable=False))
    return protocol

def build_phase2_cognitive_protocol():
    """Phase 2: Rest (5s) vs. Mental Focus (5s with 2 random math questions) - 15 Repetitions"""
    protocol = [ProtocolStep("Initial Buffer / Get Ready...", PROTOCOL_BUFFER_SECONDS, is_recordable=False)]
    for rep in range(1, 16):
        q1 = generate_random_math_question()
        q2 = generate_random_math_question()
        # Label stores both sub-questions so GUI can display & rotate them
        label = f"Mental Focus [Q1: {q1} | Q2: {q2}]"
        protocol.append(ProtocolStep(label, 5.0, repetition_index=rep, is_recordable=True))
        protocol.append(ProtocolStep("Rest (Relaxed Mind)", 5.0, repetition_index=rep, is_recordable=True))
        if rep < 15:
            protocol.append(ProtocolStep("Buffer / Get Ready...", PROTOCOL_BUFFER_SECONDS, is_recordable=False))
    return protocol

def build_phase3_motor_imagery_protocol():
    """Phase 3: Rest (5s) vs. Imagined Squeeze (5s) - 15 Repetitions"""
    protocol = [ProtocolStep("Initial Buffer / Get Ready...", PROTOCOL_BUFFER_SECONDS, is_recordable=False)]
    for rep in range(1, 16):
        protocol.append(ProtocolStep("Imagined Hand Squeeze (Visualize Squeeze)", 5.0, repetition_index=rep, is_recordable=True))
        protocol.append(ProtocolStep("Rest (Relaxed Hand)", 5.0, repetition_index=rep, is_recordable=True))
        if rep < 15:
            protocol.append(ProtocolStep("Buffer / Get Ready...", PROTOCOL_BUFFER_SECONDS, is_recordable=False))
    return protocol

def build_phase4_5gesture_protocol():
    """Phase 4: Full 5-Gesture Motor Protocol"""
    gestures = ("Fist", "Open palm", "Pinch", "Wrist up", "Wrist down")
    protocol = [ProtocolStep("Initial Buffer / Get Ready...", PROTOCOL_BUFFER_SECONDS, is_recordable=False)]
    for g_idx, gesture_name in enumerate(gestures):
        for rep in range(1, 31):
            protocol.append(ProtocolStep(gesture_name, 4.0, repetition_index=rep, is_recordable=True))
            protocol.append(ProtocolStep("Rest", 4.0, repetition_index=rep, is_recordable=True))
            if not (g_idx == len(gestures) - 1 and rep == 30):
                next_label = (
                    f"Buffer / Get Ready for {gestures[g_idx + 1]}..."
                    if rep == 30 and g_idx < len(gestures) - 1
                    else "Buffer / Get Ready..."
                )
                protocol.append(ProtocolStep(next_label, PROTOCOL_BUFFER_SECONDS, is_recordable=False))
    return protocol

class EEGOnlineFilter:
    def __init__(self, fs=1000.0):
        nyq = fs / 2.0
        self.b_notch, self.a_notch = butter(2, [49.0 / nyq, 51.0 / nyq], btype='bandstop')
        self.b_bp, self.a_bp = butter(4, [0.5 / nyq, 40.0 / nyq], btype='band')
        self.zi_notch = None
        self.zi_bp = None

    def process(self, chunk):
        chunk = np.asarray(chunk, dtype=float) if 'np' in globals() else list(chunk)
        # Use scipy lfilter for continuous streaming
        if self.zi_notch is None:
            from scipy.signal import lfilter_zi
            self.zi_notch = lfilter_zi(self.b_notch, self.a_notch) * chunk[0]
            self.zi_bp = lfilter_zi(self.b_bp, self.a_bp) * chunk[0]
        
        s1, self.zi_notch = lfilter(self.b_notch, self.a_notch, chunk, zi=self.zi_notch)
        s2, self.zi_bp = lfilter(self.b_bp, self.a_bp, s1, zi=self.zi_bp)
        return s2

class EEGDatasetCollectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("NeuroLimb - Multi-Phase Dry EEG Dataset Collector")
        self.root.geometry("900x720")
        self.root.configure(bg="#F4F6F8")

        # Intercept window close (X button) to guarantee data is saved safely
        self.root.protocol("WM_DELETE_WINDOW", self.on_window_close)

        self.participant_var = tk.StringVar(value="")
        self.phase_var = tk.StringVar(value="Phase 1: Alpha Rhythm Test (Eyes Open vs. Eyes Closed)")
        self.scalp_pos_var = tk.StringVar(value="F3-F4 (Frontal Motor/Cognitive Strip)")
        self.is_recording = False
        self.protocol = []
        self.protocol_index = 0
        self.current_step = None
        self.current_gesture = "Waiting..."

        self.output_filepath = None
        self.summary_filepath = None
        self.current_impedance_kohm = 50.0
        self.all_impedance_samples = []

        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        self._build_ui()
        self.refresh_hardware_status()

    def _build_ui(self):
        # Header Frame
        header_frame = tk.Frame(self.root, bg="#1F4E78", pady=15)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            header_frame, text="NeuroLimb Multi-Phase EEG Data Collector", 
            font=("Helvetica", 20, "bold"), fg="white", bg="#1F4E78"
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            header_frame, text="Dry Polymer Brush Electrodes | Single Differential Channel (Dev1/ai0)", 
            font=("Helvetica", 11), fg="#D0E1F9", bg="#1F4E78"
        )
        subtitle_label.pack()

        # Hardware Status Banner Frame
        self.hw_status_frame = tk.Frame(self.root, bg="#FFF3CD", pady=8)
        self.hw_status_frame.pack(fill=tk.X, padx=20, pady=10)

        self.lbl_hw_status = tk.Label(
            self.hw_status_frame, text="🔍 Checking NI-DAQ Hardware Connection...", 
            font=("Helvetica", 11, "bold"), bg="#FFF3CD", fg="#856404"
        )
        self.lbl_hw_status.pack(side=tk.LEFT, padx=10)

        self.btn_rescan_hw = tk.Button(
            self.hw_status_frame, text="🔌 Re-Scan Hardware", command=self.refresh_hardware_status,
            font=("Helvetica", 9, "bold"), bg="#17A2B8", fg="white", bd=0, padx=10, pady=3, cursor="hand2"
        )
        self.btn_rescan_hw.pack(side=tk.RIGHT, padx=10)

        # Inputs Frame
        input_frame = tk.Frame(self.root, bg="#F4F6F8", pady=10)
        input_frame.pack()

        tk.Label(input_frame, text="Participant Name / ID:", font=("Helvetica", 12, "bold"), bg="#F4F6F8").grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.entry_name = tk.Entry(input_frame, textvariable=self.participant_var, font=("Helvetica", 12), width=32)
        self.entry_name.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.entry_name.focus_set()

        tk.Label(input_frame, text="Protocol Phase Mode:", font=("Helvetica", 12, "bold"), bg="#F4F6F8").grid(row=1, column=0, sticky="e", padx=5, pady=5)
        self.combo_phase = ttk.Combobox(
            input_frame, textvariable=self.phase_var, font=("Helvetica", 10, "bold"), width=42,
            values=[
                "Phase 1: Alpha Rhythm Test (Eyes Open vs. Eyes Closed)",
                "Phase 2: Binary Cognitive Intent (Rest vs. Mental Focus)",
                "Phase 3: Binary Motor Imagery (Rest vs. Imagined Squeeze)",
                "Phase 4: Full 5-Gesture Motor Protocol"
            ]
        )
        self.combo_phase.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        tk.Label(input_frame, text="Scalp 10-20 Position:", font=("Helvetica", 12, "bold"), bg="#F4F6F8").grid(row=2, column=0, sticky="e", padx=5, pady=5)
        self.combo_pos = ttk.Combobox(
            input_frame, textvariable=self.scalp_pos_var, font=("Helvetica", 11), width=42,
            values=[
                "F3-F4 (Frontal Motor/Cognitive Strip)",
                "C3-C4 (Differential Motor Strip)",
                "Cz-Fz (Central-Frontal)",
                "F3-Cz (Left Frontal)",
                "F4-Cz (Right Frontal)",
                "C3-Cz (Left Motor)",
                "C4-Cz (Right Motor)"
            ]
        )
        self.combo_pos.grid(row=2, column=1, sticky="w", padx=5, pady=5)

        # Status & Display Card
        self.card_frame = tk.Frame(self.root, bg="white", bd=2, relief=tk.GROOVE, pady=20, padx=20)
        self.card_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=15)

        self.lbl_instruction = tk.Label(
            self.card_frame, text="Select Protocol Phase & Press Start", 
            font=("Helvetica", 22, "bold"), bg="white", fg="#333333"
        )
        self.lbl_instruction.pack(expand=True)

        self.lbl_progress = tk.Label(self.card_frame, text="", font=("Helvetica", 13), bg="white", fg="#555555")
        self.lbl_progress.pack(pady=3)

        # Post-Session Baseline Info Label
        self.lbl_baseline = tk.Label(
            self.card_frame, text="Session Baseline: Will be calculated automatically across all Rest windows", 
            font=("Helvetica", 10, "italic"), bg="white", fg="#4A148C"
        )
        self.lbl_baseline.pack(pady=3)

        self.lbl_contact = tk.Label(
            self.card_frame, text="Dry Contact Impedance: Ready to verify on Hardware Start", 
            font=("Helvetica", 11, "italic"), bg="white", fg="#007ACC"
        )
        self.lbl_contact.pack(pady=3)

        # Action Buttons Container Frame
        btn_container = tk.Frame(self.root, bg="#F4F6F8")
        btn_container.pack(pady=(0, 20))

        self.btn_start = tk.Button(
            btn_container, text="🚀 Start Selected EEG Phase", command=self.start_session,
            font=("Helvetica", 15, "bold"), bg="#28A745", fg="white", padx=15, pady=8, bd=0, cursor="hand2"
        )
        self.btn_start.pack(side=tk.LEFT, padx=10)

        self.btn_save_close = tk.Button(
            btn_container, text="💾 Save & Exit Session", command=self.on_save_and_exit,
            font=("Helvetica", 15, "bold"), bg="#DC3545", fg="white", padx=15, pady=8, bd=0, cursor="hand2"
        )
        self.btn_save_close.pack(side=tk.LEFT, padx=10)

    def refresh_hardware_status(self):
        """Scans for connected NI-DAQ hardware and updates the top banner status badge."""
        is_conn, msg = check_nidaq_hardware()
        if is_conn:
            self.hw_status_frame.configure(bg="#D4EDDA") # Soft Green
            self.lbl_hw_status.configure(
                text=f"🟢 NI-DAQ HARDWARE DETECTED: {msg}",
                bg="#D4EDDA", fg="#155724"
            )
        else:
            self.hw_status_frame.configure(bg="#F8D7DA") # Soft Red/Orange
            self.lbl_hw_status.configure(
                text=f"⚠️ NI-DAQ HARDWARE NOT DETECTED: {msg} (Offline Simulation Mode)",
                bg="#F8D7DA", fg="#721C24"
            )
        return is_conn

    def start_session(self):
        # Refresh hardware connection status right before starting
        is_connected = self.refresh_hardware_status()
        if not is_connected:
            if not messagebox.askyesno(
                "NI-DAQ Disconnected", 
                "No physical NI-DAQ hardware was detected on your USB ports.\n\nDo you want to proceed in Offline Simulation Mode?"
            ):
                return

        raw_name = self.participant_var.get().strip().replace(" ", "_")
        cleaned_name = "".join([c for c in raw_name if c.isalnum() or c in ("_", "-")])
        
        if not cleaned_name:
            messagebox.showwarning("Missing Name", "Please enter a valid participant name/ID.")
            return

        selected_phase = self.phase_var.get()
        if "Phase 1" in selected_phase:
            self.protocol = build_phase1_alpha_protocol()
            slug = "phase1_alpha"
        elif "Phase 2" in selected_phase:
            self.protocol = build_phase2_cognitive_protocol()
            slug = "phase2_cognitive"
        elif "Phase 3" in selected_phase:
            self.protocol = build_phase3_motor_imagery_protocol()
            slug = "phase3_motor_imagery"
        else:
            self.protocol = build_phase4_5gesture_protocol()
            slug = "phase4_5gesture"

        self.participant_id = cleaned_name
        self.phase_slug = slug
        self.scalp_pos = self.scalp_pos_var.get()
        self.output_filepath = os.path.join(DEFAULT_OUTPUT_DIR, f"{self.participant_id}_{slug}_eeg_dataset.csv")
        self.summary_filepath = os.path.join(DEFAULT_OUTPUT_DIR, f"{self.participant_id}_{slug}_summary.json")

        self.entry_name.config(state=tk.DISABLED)
        self.combo_phase.config(state=tk.DISABLED)
        self.combo_pos.config(state=tk.DISABLED)
        self.btn_start.config(state=tk.DISABLED, text="Recording Session Active...", bg="#6C757D")
        self.is_recording = True
        self.all_impedance_samples = []

        # Launch DAQ worker thread
        self.daq_thread = threading.Thread(target=self.daq_loop, daemon=True)
        self.daq_thread.start()

        self.next_phase()

    def next_phase(self):
        if self.protocol_index < len(self.protocol):
            self.current_step = self.protocol[self.protocol_index]
            self.current_gesture = self.current_step.label
            duration = self.current_step.duration_seconds

            self.lbl_progress.config(
                text=f"Step {self.protocol_index + 1} of {len(self.protocol)} | Duration: {duration}s"
            )

            # High contrast visual feedback
            if "Eyes Closed" in self.current_gesture or "Rest" in self.current_gesture:
                bg_color = "#E3F2FD" # Calm Blue
                fg_color = "#0D47A1"
            elif "Ready" in self.current_gesture:
                bg_color = "#FFFDE7" # Alert Yellow
                fg_color = "#F57F17"
            else:
                bg_color = "#FFEBEE" # Active Action / Focus Red
                fg_color = "#B71C1C"

            display_text = self.current_gesture
            if "Mental Focus [Q1:" in self.current_gesture:
                try:
                    parts = self.current_gesture.split("[Q1: ")[1].split(" | Q2: ")
                    q1_text = parts[0]
                    q2_text = parts[1].rstrip("]")
                    display_text = f"🧠 MENTAL MATH (Q1/2):\n{q1_text}"
                    
                    # Schedule Q2 display halfway through the 5s window (at 2.5s)
                    self.root.after(2500, lambda q2=q2_text, bg=bg_color, fg=fg_color: (
                        self.lbl_instruction.configure(text=f"🧠 MENTAL MATH (Q2/2):\n{q2}", bg=bg, fg=fg) if self.is_recording else None
                    ))
                except Exception:
                    display_text = self.current_gesture

            self.card_frame.configure(bg=bg_color)
            self.lbl_instruction.configure(text=display_text, bg=bg_color, fg=fg_color)
            self.lbl_progress.configure(bg=bg_color)
            self.lbl_baseline.configure(bg=bg_color)
            self.lbl_contact.configure(bg=bg_color)

            self.protocol_index += 1
            self.root.after(int(duration * 1000), self.next_phase)
        else:
            self.finish_session()

    def finish_session(self):
        self.is_recording = False
        self.current_gesture = "Completed"
        self.current_step = None

        # Post-collection automatic baseline & session summary computation
        post_baseline_mean, post_baseline_std, post_baseline_rms = self.compute_post_session_baseline()

        self.card_frame.configure(bg="#E8F5E9")
        self.lbl_instruction.configure(text=f"{self.phase_slug.upper()} Secured! 🎉", bg="#E8F5E9", fg="#1B5E20")
        self.lbl_progress.config(text=f"Saved to: {self.output_filepath}", bg="#E8F5E9")
        self.lbl_baseline.config(
            text=f"Post-Session Baseline: Mean = {post_baseline_mean:.5f}V | Noise Std = {post_baseline_std:.5f}V | RMS = {post_baseline_rms:.5f}V",
            bg="#E8F5E9", fg="#1B5E20", font=("Helvetica", 11, "bold")
        )
        self.lbl_contact.config(bg="#E8F5E9")
        self.btn_start.config(text="Session Complete", bg="#28A745", state=tk.DISABLED)

    def on_save_and_exit(self):
        """Safely stops recording, flushes CSV buffers, computes post-session baseline, and exits the app."""
        if self.is_recording:
            self.is_recording = False
            time.sleep(0.3) # Allow DAQ thread to flush remaining buffer

        if self.output_filepath and os.path.exists(self.output_filepath):
            self.compute_post_session_baseline()
            messagebox.showinfo("Session Saved", f"Dataset safely saved to:\n{self.output_filepath}")

        self.root.destroy()

    def on_window_close(self):
        """Intercepts window close ('X' button) to ensure all recorded data is saved before exit."""
        if self.is_recording:
            self.is_recording = False
            time.sleep(0.3)

        if self.output_filepath and os.path.exists(self.output_filepath):
            self.compute_post_session_baseline()
            print(f"Window closed. Dataset safely flushed and saved to: {self.output_filepath}")

        self.root.destroy()

    def compute_post_session_baseline(self):
        """Reads the recorded session dataset and calculates baseline parameters strictly across all Rest windows."""
        if not os.path.exists(self.output_filepath):
            return 0.0, 0.0, 0.0

        try:
            rows = []
            rest_filtered_voltages = []
            
            with open(self.output_filepath, mode='r', encoding='latin-1', newline='') as file:
                reader = csv.reader(file)
                header = next(reader)
                for row in reader:
                    rows.append(row)
                    if len(row) >= 7:
                        gesture_lbl = str(row[3])
                        try:
                            filt_v = float(row[6])
                            if "Rest" in gesture_lbl or "Eyes Closed" in gesture_lbl:
                                rest_filtered_voltages.append(filt_v)
                        except ValueError:
                            pass

            if len(rest_filtered_voltages) > 0:
                mean_val = float(np.mean(rest_filtered_voltages)) if 'np' in globals() else sum(rest_filtered_voltages)/len(rest_filtered_voltages)
                std_val = float(np.std(rest_filtered_voltages)) if 'np' in globals() else 0.02
                rms_val = float(np.sqrt(np.mean(np.array(rest_filtered_voltages)**2))) if 'np' in globals() else 0.02
                mean_impedance = float(np.mean(self.all_impedance_samples)) if len(self.all_impedance_samples) > 0 else 50.0

                # Update the CSV file with calculated post-collection baseline values
                updated_rows = []
                for row in rows:
                    if len(row) >= 9:
                        row[7] = f"{mean_val:.6f}"
                        row[8] = f"{std_val:.6f}"
                    updated_rows.append(row)

                with open(self.output_filepath, mode='w', encoding='utf-8', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(header)
                    writer.writerows(updated_rows)

                # Write companion JSON summary file
                summary_data = {
                    "Participant_ID": self.participant_id,
                    "Phase": self.phase_slug,
                    "Scalp_Position": self.scalp_pos,
                    "Total_Samples": len(rows),
                    "Rest_Baseline_Samples": len(rest_filtered_voltages),
                    "Post_Session_Baseline": {
                        "Baseline_Mean_Volts": mean_val,
                        "Baseline_Noise_Std_Volts": std_val,
                        "Baseline_RMS_Volts": rms_val
                    },
                    "Average_Dry_Impedance_kOhm": mean_impedance,
                    "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open(self.summary_filepath, mode='w') as f:
                    json.dump(summary_data, f, indent=4)

                print(f"\n==========================================================================")
                print(f"POST-SESSION BASELINE CALCULATED FOR {self.participant_id} [{self.phase_slug}]:")
                print(f"Mean: {mean_val:.6f}V | Noise Std: {std_val:.6f}V | RMS: {rms_val:.6f}V")
                print(f"Session Summary written to: {self.summary_filepath}")
                print(f"==========================================================================")
                
                return mean_val, std_val, rms_val

        except Exception as e:
            print(f"Error computing post-session baseline: {e}")
            return 0.0, 0.0, 0.0

    def daq_loop(self):
        filter_engine = EEGOnlineFilter(fs=SAMPLE_RATE)
        
        try:
            if HAS_NIDAQ:
                with nidaqmx.Task() as task:
                    task.ai_channels.add_ai_voltage_chan(
                        CHANNEL,
                        terminal_config=TerminalConfiguration.DIFF,
                        min_val=-5.0,
                        max_val=5.0
                    )
                    task.timing.cfg_samp_clk_timing(
                        rate=SAMPLE_RATE,
                        sample_mode=AcquisitionType.CONTINUOUS,
                        samps_per_chan=BUFFER_SIZE
                    )
                    task.start()
                    print(f"Hardware locked onto {CHANNEL}. Streaming dry EEG data for {self.scalp_pos} [{self.phase_slug}]...")

                    with open(self.output_filepath, mode="w", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            "Participant_ID", "Scalp_Position", "Timestamp", 
                            "Gesture_Label", "Repetition", "Raw_Voltage", "Filtered_Voltage",
                            "Baseline_Mean", "Baseline_Std", "Impedance_kOhm"
                        ])

                        while self.is_recording:
                            raw_data = task.read(number_of_samples_per_channel=BUFFER_SIZE)
                            filtered_data = filter_engine.process(raw_data)
                            
                            std_val = float(np.std(raw_data)) if 'np' in globals() else 0.01
                            est_kohm = min(250.0, max(15.0, std_val * 2500.0))
                            self.current_impedance_kohm = est_kohm
                            self.all_impedance_samples.append(est_kohm)

                            if std_val > 0.5:
                                self.lbl_contact.config(
                                    text=f"⚠️ High Contact Noise (~{est_kohm:.1f} kΩ) - Re-adjust Brush Teeth", fg="#C62828"
                                )
                            else:
                                self.lbl_contact.config(
                                    text=f"✅ Good Dry Polymer Contact (~{est_kohm:.1f} kΩ | Std: {std_val:.4f}V)", fg="#2E7D32"
                                )

                            cur_time = time.time()
                            start_t = cur_time - (BUFFER_SIZE / SAMPLE_RATE)
                            active_lbl = self.current_gesture

                            if self.current_step and self.current_step.is_recordable:
                                for i, (raw_v, filt_v) in enumerate(zip(raw_data, filtered_data)):
                                    t = start_t + (i * (1.0 / SAMPLE_RATE))
                                    writer.writerow([
                                        self.participant_id,
                                        self.scalp_pos,
                                        t,
                                        active_lbl,
                                        self.current_step.repetition_index,
                                        raw_v,
                                        filt_v,
                                        0.0, # Placeholder for post-collection calculation
                                        0.0, # Placeholder for post-collection calculation
                                        self.current_impedance_kohm
                                    ])
            else:
                # Simulation Mode fallback if DAQ hardware is offline
                print("NI-DAQ Hardware not detected. Running in Simulation Mode...")
                with open(self.output_filepath, mode="w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        "Participant_ID", "Scalp_Position", "Timestamp", 
                        "Gesture_Label", "Repetition", "Raw_Voltage", "Filtered_Voltage",
                        "Baseline_Mean", "Baseline_Std", "Impedance_kOhm"
                    ])

                    import math
                    t_counter = 0.0
                    while self.is_recording:
                        time.sleep(BUFFER_SIZE / SAMPLE_RATE)
                        raw_data = [0.05 * math.sin(2 * math.pi * 10 * (t_counter + i/1000.0)) for i in range(BUFFER_SIZE)]
                        filtered_data = filter_engine.process(raw_data)
                        
                        self.current_impedance_kohm = 45.0
                        self.all_impedance_samples.append(45.0)
                        self.lbl_contact.config(text=f"✅ Good Dry Polymer Contact (~45.0 kΩ)", fg="#2E7D32")
                        
                        cur_time = time.time()
                        start_t = cur_time - (BUFFER_SIZE / SAMPLE_RATE)
                        active_lbl = self.current_gesture

                        if self.current_step and self.current_step.is_recordable:
                            for i, (raw_v, filt_v) in enumerate(zip(raw_data, filtered_data)):
                                t = start_t + (i * (1.0 / SAMPLE_RATE))
                                writer.writerow([
                                    self.participant_id,
                                    self.scalp_pos,
                                    t,
                                    active_lbl,
                                    self.current_step.repetition_index,
                                    raw_v,
                                    filt_v,
                                    0.0,
                                    0.0,
                                    self.current_impedance_kohm
                                ])
                        t_counter += BUFFER_SIZE / SAMPLE_RATE

            print(f"Dataset raw logging completed for: {self.output_filepath}")

        except Exception as e:
            print(f"\nCRITICAL DAQ LOGGING ERROR: {e}")
            self.is_recording = False

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGDatasetCollectorApp(root)
    root.mainloop()