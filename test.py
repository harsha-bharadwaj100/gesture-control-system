import numpy as np
import pandas as pd
import time
import collections
import joblib
import warnings
from scipy.signal import iirnotch, butter, filtfilt
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation

warnings.filterwarnings("ignore")

# --- Configuration ---
# ⚠️ POINT THIS TO ONE OF YOUR SAVED CSV DATASETS!
CSV_FILE_PATH = r"improved_dataset\keerthana1.csv"

# paths (using raw strings for Windows compatibility)
MODEL_PATH = r"final_fist_open\nadicare_xgboost_purged.pkl"
CLASSES_PATH = r"final_fist_open\nadicare_classes_xgb.npy"

SAMPLE_RATE = 1000
FILTER_BUFFER_SIZE = 1000
FEATURE_WINDOW = 300
READ_CHUNK = 100


# --- Signal Processing Functions ---
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
    wrist = np.array([0, 0, 0])
    knuckles = np.array(
        [
            [-0.5, 2, 0],
            [-0.2, 2.2, 0],
            [0.2, 2.3, 0],
            [0.5, 2.1, 0],
            [0.8, 1, 0.2],
        ]
    )

    fingers_open = np.array(
        [
            [-0.6, 4, 0],
            [-0.25, 4.5, 0],
            [0.25, 4.7, 0],
            [0.6, 4.3, 0],
            [1.2, 2.5, 0.5],
        ]
    )

    fingers_fist = np.array(
        [
            [-0.5, 2.2, -0.5],
            [-0.2, 2.4, -0.6],
            [0.2, 2.5, -0.6],
            [0.5, 2.3, -0.5],
            [0.6, 1.5, 0.5],
        ]
    )

    fingers_rest = fingers_open * 0.7 + fingers_fist * 0.3

    if pose == "Open Hand":
        tips = fingers_open
        color = "cyan"
    elif pose == "Fist":
        tips = fingers_fist
        color = "red"
    else:
        tips = fingers_rest
        color = "lime"

    bones = []
    for k in knuckles:
        bones.append((wrist, k))
    for k, t in zip(knuckles, tips):
        bones.append((k, t))

    return bones, color


# --- Simulation Data Worker ---
class SimulationWorker(threading.Thread):
    def __init__(self, data_queue, status_queue):
        super().__init__()
        self.data_queue = data_queue
        self.status_queue = status_queue
        self.running = True

    def run(self):
        try:
            print("🧠 Loading Model & Simulation Data...")
            self.xgb_model = joblib.load(MODEL_PATH)
            self.classes = np.load(CLASSES_PATH)

            # Load the CSV exactly as it was recorded
            df = pd.read_csv(CSV_FILE_PATH)
            # Filter out the transition text to prevent crashes
            allowed_labels = ["Rest (Baseline)", "Rest", "Fist", "Open Hand"]
            df = df[df["Gesture_Label"].isin(allowed_labels)]
            voltages = df["Voltage"].values

            self.status_queue.put("Initializing Simulator...")
            time.sleep(1)

            # Auto-Calibrate using the first 3 seconds of the CSV
            self.status_queue.put("CALIBRATION: Processing Dataset Baseline...")
            clean_calib, env_calib = apply_filters(voltages[:3000])
            self.raw_mean = np.mean(clean_calib)
            self.raw_std = np.std(clean_calib) + 1e-7
            self.env_mean = np.mean(env_calib)
            self.env_std = np.std(env_calib) + 1e-7

            # self.status_queue.put("Ready by LogicLabs (Simulation)")

            filter_buffer = np.zeros(FILTER_BUFFER_SIZE)
            prediction_history = collections.deque(maxlen=4)
            idx = 0

            while self.running:
                # Loop the CSV data if it reaches the end
                if idx + READ_CHUNK > len(voltages):
                    idx = 0

                new_data = voltages[idx : idx + READ_CHUNK]
                idx += READ_CHUNK

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

                if not self.data_queue.full():
                    self.data_queue.put(
                        (
                            clean_signal[-500:],
                            probabilities,
                            final_gesture,
                            self.classes,
                        )
                    )

                # Sleep to mimic real-time 1000Hz hardware sampling speed
                time.sleep(0.1)

        except Exception as e:
            self.status_queue.put(f"Error: {e}")
            print(f"CRITICAL ERROR: {e}")

    def stop(self):
        self.running = False


# --- GUI Setup ---
def run_gui():
    data_queue = queue.Queue(maxsize=1)
    status_queue = queue.Queue(maxsize=10)

    sim_thread = SimulationWorker(data_queue, status_queue)
    sim_thread.start()

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 8))
    fig.canvas.manager.set_window_title("Gesture Control System")

    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[1.5, 1])

    ax_signal = fig.add_subplot(gs[0, 0])
    (line_signal,) = ax_signal.plot([], [], lw=1.5, color="#00ff99")
    ax_signal.set_title(
        "Real-Time EMG Bio-Signal (Filtered)",
        color="white",
        fontsize=12,
        fontweight="bold",
    )
    ax_signal.set_ylim(-0.0002, 0.0002)
    ax_signal.set_xlim(0, 500)
    ax_signal.grid(True, alpha=0.2)
    ax_signal.set_facecolor("#1e1e1e")

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

    ax_3d = fig.add_subplot(gs[:, 1], projection="3d")
    ax_3d.set_title(
        "3D Virtual Hand Tracking", color="white", fontsize=14, fontweight="bold"
    )
    ax_3d.set_xlim(-2, 2)
    ax_3d.set_ylim(0, 5)
    ax_3d.set_zlim(-1, 1)
    ax_3d.view_init(elev=20, azim=45)
    ax_3d.set_facecolor("#121212")
    ax_3d.set_axis_off()

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

    def update(frame):
        try:
            while not status_queue.empty():
                msg = status_queue.get_nowait()
                status_text.set_text(msg)
        except:
            pass

        try:
            if not data_queue.empty():
                signal, probs, gesture, classes = data_queue.get_nowait()

                y_data = signal
                x_data = np.arange(len(y_data))
                line_signal.set_data(x_data, y_data)
                ax_signal.set_ylim(np.min(y_data) * 1.2, np.max(y_data) * 1.2)

                ax_prob.clear()
                ax_prob.set_title("Gesture Probability Distribution", color="white")
                ax_prob.set_ylim(0, 1.0)
                ax_prob.grid(axis="y", alpha=0.2)
                colors = ["#444444" for _ in range(len(classes))]

                max_idx = np.argmax(probs)
                colors[max_idx] = "#00ff00" if probs[max_idx] > 0.6 else "#yyyy00"
                bars = ax_prob.bar(classes, probs, color=colors)

                gesture_text.set_text(f"DETECTED: {gesture.upper()}")
                if gesture == "Fist":
                    gesture_text.set_color("red")
                elif gesture == "Open Hand":
                    gesture_text.set_color("cyan")
                else:
                    gesture_text.set_color("lime")

                ax_3d.clear()
                ax_3d.set_axis_off()
                ax_3d.set_xlim(-2, 2)
                ax_3d.set_ylim(0, 6)
                ax_3d.set_zlim(-1, 1)
                ax_3d.view_init(elev=30, azim=-60)

                bones, hand_color = get_hand_points(gesture)

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

    sim_thread.stop()
    sim_thread.join()


if __name__ == "__main__":
    run_gui()
