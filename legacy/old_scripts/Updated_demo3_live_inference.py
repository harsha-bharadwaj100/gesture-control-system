import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import numpy as np
import time
import collections
import warnings
from scipy.signal import iirnotch, butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --- Configuration ---
CHANNEL = "Dev1/ai0"
SAMPLE_RATE = 1000
FILTER_BUFFER_SIZE = 1000
FEATURE_WINDOW = 300
READ_CHUNK = 100
STEP_SIZE = 50  # For training window generation

print("🧠 Initiating Nadicare Zero-Shot Local BCI Engine...")


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
    # 13 Polarity-Invariant Features
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


def run_instant_bci():
    filter_buffer = np.zeros(FILTER_BUFFER_SIZE)
    prediction_history = collections.deque(maxlen=4)

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
            print("🚀 PHASE 1: BIOLOGICAL DATA CAPTURE")
            print("=" * 50)

            gestures = ["Rest", "Fist", "Open Hand"]
            raw_training_data = {}

            for gesture in gestures:
                print(f"\n⚠️ GET READY: {gesture}")
                time.sleep(2.5)

                print(f"🔴 HOLD {gesture.upper()} NOW! (Recording 4 seconds...)")
                gesture_data = []
                # Record 4 seconds to give us plenty of clean data windows
                for _ in range(40):
                    gesture_data.extend(
                        task.read(number_of_samples_per_channel=READ_CHUNK)
                    )

                # Apply filters to the raw chunk immediately
                clean_calib, env_calib = apply_filters(np.array(gesture_data))
                raw_training_data[gesture] = (clean_calib, env_calib)
                print(f"✅ Captured.")

            print("\n⚙️ PHASE 2: INSTANT NEURAL MAPPING...")
            X_train_list = []
            y_train_list = []

            # Create the training windows directly in memory
            for gesture, (clean_sig, env_sig) in raw_training_data.items():
                # Skip the first 600ms (reaction time purge!)
                for i in range(600, len(clean_sig) - FEATURE_WINDOW, STEP_SIZE):
                    window = clean_sig[i : i + FEATURE_WINDOW]
                    env_window = env_sig[i : i + FEATURE_WINDOW]

                    features = extract_features(window, env_window)
                    X_train_list.append(features)
                    y_train_list.append(gesture)

            X_train = np.array(X_train_list)
            y_train = np.array(y_train_list)

            # Normalize the local data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)

            # Train a robust Random Forest instantly
            print("   -> Training Local Random Forest Classifier...")
            local_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            )
            local_model.fit(X_train_scaled, y_train)

            print("=" * 50)
            print(f"✅ SYSTEM ARMED! Model accuracy locked to your current arm state.")
            print("💪 Start making gestures! (Press Ctrl+C to quit)\n")

            # Flush the buffer before starting live inference
            task.read(number_of_samples_per_channel=task.in_stream.avail_samp_per_chan)

            while True:
                new_data = task.read(number_of_samples_per_channel=READ_CHUNK)
                filter_buffer = np.roll(filter_buffer, -READ_CHUNK)
                filter_buffer[-READ_CHUNK:] = new_data

                clean_signal, envelope_signal = apply_filters(filter_buffer)

                window_data = clean_signal[-FEATURE_WINDOW:]
                env_data = envelope_signal[-FEATURE_WINDOW:]

                features = extract_features(window_data, env_data)

                # Scale using the INSTANT model's scaler
                X_live = scaler.transform(np.array(features).reshape(1, -1))

                probabilities = local_model.predict_proba(X_live)[0]
                predicted_index = np.argmax(probabilities)
                confidence = probabilities[predicted_index]
                gesture_name = local_model.classes_[predicted_index]

                prediction_history.append(gesture_name)
                most_common_gesture = max(
                    set(prediction_history), key=prediction_history.count
                )

                if (
                    confidence > 0.65
                    and prediction_history.count(most_common_gesture) >= 3
                ):
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
                        f"⏳ [ {'Processing':<12} ]                                  ",
                        end="\r",
                    )

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
    except KeyboardInterrupt:
        print("\n\n🛑 Live inference stopped.")


if __name__ == "__main__":
    run_instant_bci()
