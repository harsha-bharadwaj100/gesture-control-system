import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import numpy as np
import time
import collections
import joblib
import warnings
from scipy.signal import iirnotch, butter, filtfilt
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore")

# --- Configuration ---
CHANNEL = "Dev1/ai0"
SAMPLE_RATE = 1000
FILTER_BUFFER_SIZE = 1000
FEATURE_WINDOW = 300
READ_CHUNK = 100

# paths (using raw strings for Windows compatibility)
MODEL_PATH = r"final_fist_open\nadicare_xgboost_purged.pkl"
CLASSES_PATH = r"final_fist_open\nadicare_classes_xgb.npy"


# --- Signal Processing Functions (Same as original) ---
def apply_filters(data):
    b_notch, a_notch = iirnotch(50.0, 30.0, SAMPLE_RATE)
    data_notched = filtfilt(b_notch, a_notch, data)

    nyquist = SAMPLE_RATE / 2.0
    b_band, a_band = butter(4, [20.0 / nyquist, 450.0 / nyquist], btype="band")
    clean_data = filtfilt(b_band, a_band, data_notched)

    rectified = np.abs(clean_data)
    b_env, a_env = butter(4, 5.0 / nyquist, btype="low")
    envelope = filtfilt(b_env, a_env, rectified)

    return clean_data, envelope


def extract_features(window, env_window):
    # EXACTLY 13 FEATURES
    mav = np.mean(np.abs(window))
    rms = np.sqrt(np.mean(window**2))
    var = np.var(window)
    wl = np.sum(np.abs(np.diff(window)))
    zc = np.sum(np.diff(np.sign(window)) != 0)
    ssc = np.sum(np.diff(np.sign(np.diff(window))) != 0)

    env_mean = np.mean(env_window)
    env_max = np.max(env_window)

    activity = var
    diff1 = np.diff(window)
    mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
    diff2 = np.diff(diff1)
    complexity = (
        np.sqrt(np.var(diff2) / np.var(diff1)) / mobility if mobility > 0 else 0
    )

    fft_vals = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(len(window), d=1.0 / SAMPLE_RATE)
    total_power = np.sum(fft_vals)
    mean_freq = np.sum(freqs * fft_vals) / total_power if total_power > 0 else 0
    peak_freq = freqs[np.argmax(fft_vals)] if total_power > 0 else 0

    return [
        mav,
        rms,
        var,
        wl,
        zc,
        ssc,
        env_mean,
        env_max,
        activity,
        mobility,
        complexity,
        mean_freq,
        peak_freq,
    ]


# --- 3D Hand Model Data ---
def get_hand_points(pose="Rest"):
    # Simplified Skeleton: Wrist, Thumb, Index, Middle, Ring, Pinky
    # Wrist at (0,0,0)
    # Coordinates: (x, y, z)

    # Base structure
    wrist = np.array([0, 0, 0])
    knuckles = np.array(
        [
            [-0.5, 2, 0],  # Pinky
            [-0.2, 2.2, 0],  # Ring
            [0.2, 2.3, 0],  # Middle
            [0.5, 2.1, 0],  # Index
            [0.8, 1, 0.2],  # Thumb base
        ]
    )

    fingers_open = np.array(
        [
            [-0.6, 4, 0],  # Pinky tip
            [-0.25, 4.5, 0],  # Ring tip
            [0.25, 4.7, 0],  # Middle tip
            [0.6, 4.3, 0],  # Index tip
            [1.2, 2.5, 0.5],  # Thumb tip
        ]
    )

    fingers_fist = np.array(
        [
            [-0.5, 2.2, -0.5],  # Pinky curled
            [-0.2, 2.4, -0.6],  # Ring curled
            [0.2, 2.5, -0.6],  # Middle curled
            [0.5, 2.3, -0.5],  # Index curled
            [0.6, 1.5, 0.5],  # Thumb curled
        ]
    )

    # Rest is somewhere in between
    fingers_rest = fingers_open * 0.7 + fingers_fist * 0.3

    if pose == "Open Hand":
        tips = fingers_open
        color = "cyan"
    elif pose == "Fist":
        tips = fingers_fist
        color = "red"
    else:  # Rest
        tips = fingers_rest
        color = "lime"

    # Build lines (bones)
    # Wrist to Knuckles
    bones = []
    for k in knuckles:
        bones.append((wrist, k))

    # Knuckles to Tips
    for k, t in zip(knuckles, tips):
        bones.append((k, t))

    return bones, color


# --- DAQ Thread ---
class DAQWorker(threading.Thread):
    def __init__(self, data_queue, status_queue):
        super().__init__()
        self.data_queue = data_queue
        self.status_queue = status_queue
        self.running = True
        self.xgb_model = None
        self.classes = None
        self.raw_mean = 0
        self.raw_std = 1
        self.env_mean = 0
        self.env_std = 1

    def run(self):
        try:
            print("🧠 Loading Model...")
            self.xgb_model = joblib.load(MODEL_PATH)
            self.classes = np.load(CLASSES_PATH)
            print(f"✅ Loaded classes: {self.classes}")

            filter_buffer = np.zeros(FILTER_BUFFER_SIZE)
            prediction_history = collections.deque(maxlen=4)

            with nidaqmx.Task() as task:
                task.ai_channels.add_ai_voltage_chan(
                    CHANNEL,
                    terminal_config=TerminalConfiguration.DIFF,
                    min_val=-5.0,
                    max_val=5.0,
                )
                task.timing.cfg_samp_clk_timing(
                    rate=SAMPLE_RATE,
                    sample_mode=AcquisitionType.CONTINUOUS,
                    samps_per_chan=SAMPLE_RATE * 2,
                )

                self.status_queue.put("Connecting to Hardware...")
                task.start()
                self.status_queue.put("Hardware Connected.")

                # --- Calibration ---
                self.status_queue.put("CALIBRATION: Please wait...")
                time.sleep(2)

                calibration_gestures = ["Rest", "Fist", "Open Hand"]
                all_calibration_data = []

                for gesture in calibration_gestures:
                    self.status_queue.put(f"CALIBRATION: HOLD {gesture.upper()}")
                    gesture_data = []
                    for _ in range(30):
                        gesture_data.extend(
                            task.read(number_of_samples_per_channel=READ_CHUNK)
                        )
                    all_calibration_data.extend(gesture_data)
                    self.status_queue.put(f"Captured {gesture}.")

                clean_calib, env_calib = apply_filters(np.array(all_calibration_data))
                self.raw_mean = np.mean(clean_calib)
                self.raw_std = np.std(clean_calib) + 1e-7
                self.env_mean = np.mean(env_calib)
                self.env_std = np.std(env_calib) + 1e-7

                self.status_queue.put("Ready by LogicLabs")

                # --- Main Loop ---
                while self.running:
                    new_data = task.read(number_of_samples_per_channel=READ_CHUNK)
                    filter_buffer = np.roll(filter_buffer, -READ_CHUNK)
                    filter_buffer[-READ_CHUNK:] = new_data

                    clean_signal, envelope_signal = apply_filters(filter_buffer)

                    window_data = clean_signal[-FEATURE_WINDOW:]
                    env_data = envelope_signal[-FEATURE_WINDOW:]

                    norm_window = (window_data - self.raw_mean) / self.raw_std
                    norm_env = (env_data - self.env_mean) / self.env_std

                    features = extract_features(norm_window, norm_env)
                    X_live = np.array(features).reshape(1, -1)

                    probabilities = self.xgb_model.predict_proba(X_live)[0]
                    predicted_index = np.argmax(probabilities)
                    confidence = probabilities[predicted_index]
                    gesture_name = self.classes[predicted_index]

                    prediction_history.append(gesture_name)
                    most_common_gesture = max(
                        set(prediction_history), key=prediction_history.count
                    )

                    final_gesture = "Uncertain"
                    if (
                        confidence > 0.60
                        and prediction_history.count(most_common_gesture) >= 3
                    ):
                        final_gesture = most_common_gesture

                    # Send data to GUI: (processed_signal, probabilities, final_gesture_name, classes)
                    if not self.data_queue.full():
                        # Send a downsampled version of the signal for visualization to avoid lag
                        self.data_queue.put(
                            (
                                clean_signal[-500:],
                                probabilities,
                                final_gesture,
                                self.classes,
                            )
                        )

        except Exception as e:
            self.status_queue.put(f"Error: {e}")
            print(e)

    def stop(self):
        self.running = False


# --- GUI Setup ---
def run_gui():
    # Setup Data Communications
    data_queue = queue.Queue(maxsize=1)
    status_queue = queue.Queue(maxsize=10)

    # Start DAQ Thread
    daq_thread = DAQWorker(data_queue, status_queue)
    daq_thread.start()

    # Setup Plot Style
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 8))
    fig.canvas.manager.set_window_title("LogicLabs 4.0 Gesture Control System")

    # Grid Layout
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.5, 1])

    # 1. Live Signal Plot (Top Left)
    ax_signal = fig.add_subplot(gs[0, 0])
    (line_signal,) = ax_signal.plot([], [], lw=1.5, color="#00ff99")
    ax_signal.set_title(
        "Real-Time EMG Bio-Signal (Filtered)",
        color="white",
        fontsize=12,
        fontweight="bold",
    )
    ax_signal.set_ylim(
        -0.0002, 0.0002
    )  # Approximate range, auto-scaling might be better but jittery
    ax_signal.set_xlim(0, 500)
    ax_signal.grid(True, alpha=0.2)
    ax_signal.set_facecolor("#1e1e1e")

    # 2. Confidence Bar Chart (Bottom Left)
    ax_prob = fig.add_subplot(gs[1, 0])
    bars = ax_prob.bar(["Init"], [0], color="#00ccff")
    ax_prob.set_title(
        "Gesture Probability Distribution",
        color="white",
        fontsize=12,
        fontweight="bold",
    )
    ax_prob.set_ylim(0, 1.0)
    ax_prob.set_facecolor("#1e1e1e")

    # 3. 3D Hand Visualizer (Right Column - Spanning both rows)
    ax_3d = fig.add_subplot(gs[:, 1], projection="3d")
    ax_3d.set_title(
        "3D Virtual Hand Tracking", color="white", fontsize=14, fontweight="bold"
    )
    ax_3d.set_xlim(-2, 2)
    ax_3d.set_ylim(0, 5)
    ax_3d.set_zlim(-1, 1)
    ax_3d.view_init(elev=20, azim=45)
    ax_3d.set_facecolor("#121212")
    # Hide axes for cleaner look
    ax_3d.set_axis_off()

    # Text helper for Status
    status_text = fig.text(
        0.02, 0.95, "Initializing...", fontsize=14, color="#ff00ff", fontweight="bold"
    )
    gesture_text = fig.text(
        0.5,
        0.95,
        "WAITING",
        fontsize=20,
        color="yellow",
        fontweight="bold",
        ha="center",
    )

    # Update Function
    def update(frame):
        # Check Status
        try:
            while not status_queue.empty():
                msg = status_queue.get_nowait()
                status_text.set_text(msg)
        except:
            pass

        # Check Data
        try:
            if not data_queue.empty():
                signal, probs, gesture, classes = data_queue.get_nowait()

                # Update Signal
                y_data = signal
                x_data = np.arange(len(y_data))
                line_signal.set_data(x_data, y_data)
                ax_signal.set_ylim(
                    np.min(y_data) * 1.2, np.max(y_data) * 1.2
                )  # Auto scale lightly

                # Update Bar Chart
                ax_prob.clear()
                ax_prob.set_title("Gesture Probability Distribution", color="white")
                ax_prob.set_ylim(0, 1.0)
                ax_prob.grid(axis="y", alpha=0.2)
                colors = ["#444444" for _ in range(len(classes))]

                # Highlight the max bar
                max_idx = np.argmax(probs)
                colors[max_idx] = "#00ff00" if probs[max_idx] > 0.6 else "#yyyy00"

                bars = ax_prob.bar(classes, probs, color=colors)

                # Update Text
                gesture_text.set_text(f"DETECTED: {gesture.upper()}")
                if gesture == "Fist":
                    gesture_text.set_color("red")
                elif gesture == "Open Hand":
                    gesture_text.set_color("cyan")
                else:
                    gesture_text.set_color("lime")

                # Update 3D Hand
                ax_3d.clear()
                ax_3d.set_axis_off()
                ax_3d.set_xlim(-2, 2)
                ax_3d.set_ylim(0, 6)
                ax_3d.set_zlim(-1, 1)
                ax_3d.view_init(elev=30, azim=-60)

                # Get 3D Points based on gesture
                bones, hand_color = get_hand_points(gesture)

                # Draw Bones
                for start, end in bones:
                    ax_3d.plot(
                        [start[0], end[0]],
                        [start[1], end[1]],
                        [start[2], end[2]],
                        color=hand_color,
                        linewidth=4,
                        alpha=0.8,
                        marker="o",
                    )

        except Exception as e:
            pass

        return (line_signal,)

    ani = animation.FuncAnimation(fig, update, interval=50, blit=False)
    plt.tight_layout()
    plt.show()

    # Cleanup on close
    daq_thread.stop()
    daq_thread.join()


if __name__ == "__main__":
    run_gui()
