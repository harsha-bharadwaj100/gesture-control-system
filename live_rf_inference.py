import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import numpy as np
import time
import collections
import joblib
import warnings

# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuration ---
CHANNEL = "Dev1/ai0"
SAMPLE_RATE = 1000
WINDOW_SIZE = 200
BUFFER_SIZE = 100

print("🧠 Loading Robust Random Forest Model...")
rf_model = joblib.load("nadicare_rf_model.pkl")
classes = np.load("gesture_classes_rf.npy")
print(f"✅ Loaded classes: {classes}")


def extract_features(window):
    mav = np.mean(np.abs(window))
    rms = np.sqrt(np.mean(window**2))
    var = np.var(window)
    wl = np.sum(np.abs(np.diff(window)))
    zc = np.sum(np.diff(np.sign(window)) != 0)
    return [mav, rms, var, wl, zc]


def run_live_inference():
    rolling_window = np.zeros(WINDOW_SIZE)
    prediction_history = collections.deque(maxlen=5)

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

            print(f"\n🚀 Hardware locked onto {CHANNEL}.")
            task.start()

            # --- 2-STEP DYNAMIC CALIBRATION ---
            print("\n🛑 STEP 1: KEEP ARM COMPLETELY RELAXED...")
            print("Calibrating baseline noise in 3... 2... 1...")
            time.sleep(1)

            baseline_data = []
            for _ in range(30):  # 3 seconds of rest
                baseline_data.extend(
                    task.read(number_of_samples_per_channel=BUFFER_SIZE)
                )
            global_mean = np.mean(baseline_data)

            print("\n✊ STEP 2: MAKE A STRONG FIST AND HOLD IT!")
            print("Calibrating maximum amplitude in 3... 2... 1...")
            time.sleep(1)

            active_data = []
            for _ in range(30):  # 3 seconds of flexing
                active_data.extend(task.read(number_of_samples_per_channel=BUFFER_SIZE))

            # We use the active data to find the standard deviation (scaling factor)
            global_std = np.std(active_data) + 1e-7

            print(f"\n✅ Dynamic Calibration complete!")
            print(f"Offset: {global_mean:.4f}V | Scale: {global_std:.4f}")
            print("💪 Start making gestures! (Press Ctrl+C to quit)\n")
            # ----------------------------------

            while True:
                new_data = task.read(number_of_samples_per_channel=BUFFER_SIZE)
                rolling_window = np.roll(rolling_window, -BUFFER_SIZE)
                rolling_window[-BUFFER_SIZE:] = new_data

                # Normalize using the 2-step MVC baseline
                normalized_window = (rolling_window - global_mean) / global_std

                features = extract_features(normalized_window)
                X_live = np.array(features).reshape(1, -1)

                probabilities = rf_model.predict_proba(X_live)[0]
                predicted_index = np.argmax(probabilities)
                confidence = probabilities[predicted_index]
                gesture_name = classes[predicted_index]

                prediction_history.append(gesture_name)
                most_common_gesture = max(
                    set(prediction_history), key=prediction_history.count
                )

                if (
                    confidence > 0.50
                    and prediction_history.count(most_common_gesture) >= 3
                ):
                    print(
                        f"✅ [ {most_common_gesture:<12} ]  (Conf: {confidence*100:.0f}%)    ",
                        end="\r",
                    )
                else:
                    print(
                        f"⏳ [ {'Uncertain':<12} ]                                  ",
                        end="\r",
                    )

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
    except KeyboardInterrupt:
        print("\n\n🛑 Live inference stopped.")


if __name__ == "__main__":
    run_live_inference()
