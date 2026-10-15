import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import numpy as np
import time
import collections
import warnings
from scipy.signal import iirnotch, butter, filtfilt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --- Configuration ---
CHANNEL = "Dev1/ai0"
SAMPLE_RATE = 1000
FILTER_BUFFER_SIZE = 1000
FEATURE_WINDOW = 300
READ_CHUNK = 100
STEP_SIZE = 50

# Anti-Flicker EMA Smoothing Factor (0.0 to 1.0)
# Lower = Smoother but slightly slower reaction time. 0.25 is the sweet spot.
SMOOTHING_FACTOR = 0.25

print("🧠 Initiating Stable SVM Zero-Shot BCI Engine...")


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


def run_stable_bci():
    filter_buffer = np.zeros(FILTER_BUFFER_SIZE)
    # We drop the heavy deque buffer because EMA handles the smoothing now!

    try:
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

            print(f"\n🚀 Hardware locked onto {CHANNEL}. Starting task...")
            task.start()

            print("\n" + "=" * 50)
            print("🚀 PHASE 1: STABLE BIOLOGICAL CALIBRATION")
            print("=" * 50)

            # CHANGE THESE TO "Thumbs Up", "Index Finger" for the other demo!
            gestures = ["Rest", "Fist", "Open Hand"]
            raw_training_data = {}

            for gesture in gestures:
                print(f"\n⚠️ GET READY: {gesture}")
                time.sleep(2.0)

                print(f"🔴 HOLD {gesture.upper()} NOW! (Hold perfectly steady...)")
                gesture_data = []
                for _ in range(40):  # 4 seconds
                    gesture_data.extend(
                        task.read(number_of_samples_per_channel=READ_CHUNK)
                    )

                clean_calib, env_calib = apply_filters(np.array(gesture_data))
                raw_training_data[gesture] = (clean_calib, env_calib)
                print(f"✅ Captured.")

            print("\n⚙️ PHASE 2: SVM BOUNDARY MAPPING...")
            X_train_list = []
            y_train_list = []

            for gesture, (clean_sig, env_sig) in raw_training_data.items():
                for i in range(600, len(clean_sig) - FEATURE_WINDOW, STEP_SIZE):
                    window = clean_sig[i : i + FEATURE_WINDOW]
                    env_window = env_sig[i : i + FEATURE_WINDOW]

                    features = extract_features(window, env_window)
                    X_train_list.append(features)
                    y_train_list.append(gesture)

            X_train = np.array(X_train_list)
            y_train = np.array(y_train_list)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            print("   -> Training Support Vector Machine (RBF Kernel)...")
            # SVM is vastly superior for tiny datasets. C=10 forces strict margins.
            local_model = SVC(
                kernel="rbf", C=10.0, gamma="scale", probability=True, random_state=42
            )
            local_model.fit(X_train_scaled, y_train)

            # Initialize our EMA probability array
            smoothed_probs = np.zeros(len(local_model.classes_))

            print("=" * 50)
            print(f"✅ SYSTEM ARMED! Rock-solid SVM locked to your arm.")
            print("💪 Start making gestures! (Press Ctrl+C to quit)\n")

            task.read(number_of_samples_per_channel=task.in_stream.avail_samp_per_chan)

            while True:
                new_data = task.read(number_of_samples_per_channel=READ_CHUNK)
                filter_buffer = np.roll(filter_buffer, -READ_CHUNK)
                filter_buffer[-READ_CHUNK:] = new_data

                clean_signal, envelope_signal = apply_filters(filter_buffer)

                window_data = clean_signal[-FEATURE_WINDOW:]
                env_data = envelope_signal[-FEATURE_WINDOW:]

                features = extract_features(window_data, env_data)

                X_live = scaler.transform(np.array(features).reshape(1, -1))

                raw_probabilities = local_model.predict_proba(X_live)[0]

                # --- THE MAGIC: EMA PROBABILITY SMOOTHING ---
                smoothed_probs = (SMOOTHING_FACTOR * raw_probabilities) + (
                    (1 - SMOOTHING_FACTOR) * smoothed_probs
                )

                predicted_index = np.argmax(smoothed_probs)
                confidence = smoothed_probs[predicted_index]
                most_common_gesture = local_model.classes_[predicted_index]

                # UI Output (Notice how clean the logic is now)
                if confidence > 0.60:
                    if most_common_gesture == "Fist":
                        print(
                            f"✊ [ {most_common_gesture:<12} ]  (Conf: {confidence*100:.0f}%)    ",
                            end="\r",
                        )
                    elif most_common_gesture == "Open Hand":
                        print(
                            f"🖐️ [ {most_common_gesture:<12} ]  (Conf: {confidence*100:.0f}%)    ",
                            end="\r",
                        )
                    else:
                        print(
                            f"〰️ [ {most_common_gesture:<12} ]  (Conf: {confidence*100:.0f}%)    ",
                            end="\r",
                        )
                else:
                    print(
                        f"⏳ [ {'Transitioning':<12} ]                                  ",
                        end="\r",
                    )

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
    except KeyboardInterrupt:
        print("\n\n🛑 Live inference stopped.")


if __name__ == "__main__":
    run_stable_bci()
