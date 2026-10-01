import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import numpy as np
import time
import collections
import joblib
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import skew, kurtosis
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
CHANNEL = "Dev1/ai0"
FS = 1000
WINDOW_SIZE = 1500  # 1.5 seconds to match training exactly
UPDATE_STEP = 200   # Slide by 200ms
BUFFER_SIZE = 200

print("[*] Loading Generalized Triple Ensemble Model...")
try:
    voting_clf = joblib.load("generalized_ensemble_model.pkl")
    classes = np.load("generalized_classes.npy")
    print(f"[+] Loaded classes: {classes}")
except FileNotFoundError:
    print("[!] Model files not found! Please run `uv run python generalized_model.py` first to generate the models.")
    exit(1)

def notch_filter(data, fs, f0, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, data)

def bandpass_filter(data, fs, lowcut=20.0, highcut=450.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def tkeo(data):
    y = np.zeros_like(data)
    y[1:-1] = data[1:-1]**2 - data[:-2] * data[2:]
    return y

def envelope(data, window=50):
    return pd.Series(np.abs(data)).rolling(window, min_periods=1, center=True).mean().values

def hjorth_parameters(signal):
    if len(signal) <= 1:
        return 0, 0, 0
    activity = np.var(signal)
    if activity == 0:
        return 0, 0, 0
    diff1 = np.diff(signal)
    mobility = np.sqrt(np.var(diff1) / activity)
    if mobility == 0:
        return activity, mobility, 0
    diff2 = np.diff(diff1)
    complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility
    return activity, mobility, complexity

def spectral_entropy(signal, fs):
    fft_vals = np.abs(np.fft.rfft(signal))
    psd = fft_vals ** 2
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]
    return -np.sum(psd_norm * np.log2(psd_norm))

def extract_features(signal, fs=FS):
    if len(signal) == 0:
        return [0]*16
    
    mav = np.mean(np.abs(signal))
    rms = np.sqrt(np.mean(signal**2))
    var = np.var(signal)
    wl = np.sum(np.abs(np.diff(signal)))
    zc = np.sum(np.diff(np.signbit(signal)))
    ssc = np.sum(np.diff(np.sign(np.diff(signal))) != 0)
    
    env = envelope(signal)
    env_mean = np.mean(env)
    env_max = np.max(env)
    
    sk = skew(signal)
    ku = kurtosis(signal)
    
    activity, mobility, complexity = hjorth_parameters(signal)
    
    freqs = np.fft.rfftfreq(len(signal), d=1/fs)
    fft_vals = np.abs(np.fft.rfft(signal))
    total_power = np.sum(fft_vals)
    if total_power == 0:
        mean_freq = 0
        peak_freq = 0
    else:
        mean_freq = np.sum(freqs * fft_vals) / total_power
        peak_freq = freqs[np.argmax(fft_vals)]
        
    se = spectral_entropy(signal, fs)
        
    return [mav, rms, var, wl, zc, ssc, env_mean, env_max, activity, mobility, complexity, mean_freq, peak_freq, se, sk, ku]

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
                rate=FS,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=FS * 2,
            )

            print(f"\n[*] Hardware locked onto {CHANNEL}.")
            task.start()

            # --- LIVE STANDARDIZATION CALIBRATION ---
            print("\n[!] STEP 1: KEEP ARM COMPLETELY RELAXED...")
            print("Calibrating baseline... 3... 2... 1...")
            time.sleep(1)

            calib_data = []
            for _ in range(50):  # 5 seconds of rest
                calib_data.extend(task.read(number_of_samples_per_channel=100))

            print("\n[!] STEP 2: GESTURE CALIBRATION")
            gestures_to_calibrate = ["Fist", "Open Palm", "Pinch", "Wrist Up", "Wrist Down"]
            
            for gesture in gestures_to_calibrate:
                print(f"👉 Prepare to make: {gesture} in 2... 1...")
                time.sleep(2)
                print(f"✊ HOLD {gesture}!")
                for _ in range(20):  # 2 seconds of holding the gesture
                    calib_data.extend(task.read(number_of_samples_per_channel=100))
                print("🛑 Relax.")
                time.sleep(1)

            # Calculate calibration features
            calib_data = np.array(calib_data)
            v_filt = calib_data
            for harmonic in [50, 100, 150, 200, 250, 300, 350, 400]:
                v_filt = notch_filter(v_filt, FS, f0=harmonic)
            v_filt = bandpass_filter(v_filt, FS, lowcut=20, highcut=450)
            
            # Extract features in chunks to build a live scaler
            calib_features = []
            chunk_size = int(1.5 * FS)
            for i in range(0, len(calib_data) - chunk_size, int(0.5 * FS)):
                segment = v_filt[i:i+chunk_size]
                calib_features.append(extract_features(segment))
                
            calib_features = np.array(calib_features)
            global_mean = np.mean(calib_features, axis=0)
            global_std = np.std(calib_features, axis=0) + 1e-8

            print(f"\n[+] Dynamic Calibration complete!")
            print("[*] Start making gestures! (Press Ctrl+C to quit)\n")
            # ----------------------------------

            while True:
                new_data = task.read(number_of_samples_per_channel=UPDATE_STEP)
                rolling_window = np.roll(rolling_window, -UPDATE_STEP)
                rolling_window[-UPDATE_STEP:] = new_data

                # Live Filtering Pipeline
                v_filt = rolling_window.copy()
                for harmonic in [50, 100, 150, 200, 250, 300, 350, 400]:
                    v_filt = notch_filter(v_filt, FS, f0=harmonic)
                v_filt = bandpass_filter(v_filt, FS, lowcut=20, highcut=450)

                # Extract 16 features
                features = extract_features(v_filt)
                
                # Standardize using calibration mean/std
                features_scaled = (np.array(features) - global_mean) / global_std
                X_live = features_scaled.reshape(1, -1)

                probabilities = voting_clf.predict_proba(X_live)[0]
                predicted_index = np.argmax(probabilities)
                confidence = probabilities[predicted_index]
                gesture_name = classes[predicted_index]

                prediction_history.append(gesture_name)
                most_common_gesture = max(
                    set(prediction_history), key=prediction_history.count
                )

                if (
                    confidence > 0.45
                    and prediction_history.count(most_common_gesture) >= 3
                ):
                    print(
                        f"[+] [ {most_common_gesture:<12} ]  (Conf: {confidence*100:.0f}%)    ",
                        end="\r",
                    )
                else:
                    print(
                        f"[*] [ {'Uncertain':<12} ]                                  ",
                        end="\r",
                    )

    except Exception as e:
        print(f"\n[!] ERROR: {e}")
    except KeyboardInterrupt:
        print("\n\n[!] Live inference stopped.")

if __name__ == "__main__":
    run_live_inference()
