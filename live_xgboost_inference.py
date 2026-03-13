import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import numpy as np
import time
import collections
import joblib
import warnings
from scipy.signal import iirnotch, butter, filtfilt
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore") 

# --- Configuration ---
CHANNEL = "Dev1/ai0"
SAMPLE_RATE = 1000
FILTER_BUFFER_SIZE = 1000  
FEATURE_WINDOW = 300       
READ_CHUNK = 100           

print("🧠 Loading 5-Gesture Global XGBoost Model...")
xgb_model = joblib.load("nadicare_xgboost_model.pkl") 
classes = np.load("nadicare_classes_xgb.npy")
print(f"✅ Loaded classes: {classes}")

def apply_filters(data):
    b_notch, a_notch = iirnotch(50.0, 30.0, SAMPLE_RATE)
    data_notched = filtfilt(b_notch, a_notch, data)
    
    nyquist = SAMPLE_RATE / 2.0
    b_band, a_band = butter(4, [20.0 / nyquist, 450.0 / nyquist], btype='band')
    clean_data = filtfilt(b_band, a_band, data_notched)
    
    rectified = np.abs(clean_data)
    b_env, a_env = butter(4, 5.0 / nyquist, btype='low')
    envelope = filtfilt(b_env, a_env, rectified)
    
    return clean_data, envelope

def extract_features(window, env_window):
    raw_mean = np.mean(window)
    skew_val = skew(window)
    kurt_val = kurtosis(window)

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
    complexity = np.sqrt(np.var(diff2) / np.var(diff1)) / mobility if mobility > 0 else 0

    fft_vals = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(len(window), d=1.0/SAMPLE_RATE)
    total_power = np.sum(fft_vals)
    mean_freq = np.sum(freqs * fft_vals) / total_power if total_power > 0 else 0
    peak_freq = freqs[np.argmax(fft_vals)] if total_power > 0 else 0

    return [raw_mean, skew_val, kurt_val, mav, rms, var, wl, zc, ssc, env_mean, env_max, activity, mobility, complexity, mean_freq, peak_freq]

def run_live_inference():
    filter_buffer = np.zeros(FILTER_BUFFER_SIZE)
    prediction_history = collections.deque(maxlen=5) 
    
    try:
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
                samps_per_chan=SAMPLE_RATE * 2 
            )

            print(f"\n🚀 Hardware locked onto {CHANNEL}.")
            task.start()
            
            # --- THE "CRAZY" FIX: GUIDED ONBOARDING CALIBRATION ---
            print("\n" + "="*50)
            print("🚀 USER ONBOARDING SEQUENCE INITIATED")
            print("="*50)
            
            calibration_gestures = ["Rest", "Fist", "Open Hand", "Thumbs Up", "Index Finger"]
            all_calibration_data = []
            
            for gesture in calibration_gestures:
                print(f"\n⚠️ GET READY: {gesture}")
                time.sleep(2)  # Give them time to prepare
                
                print(f"🔴 HOLD {gesture.upper()} NOW!")
                gesture_data = []
                # Record 3 seconds of this specific gesture
                for _ in range(30):
                    gesture_data.extend(task.read(number_of_samples_per_channel=READ_CHUNK))
                
                all_calibration_data.extend(gesture_data)
                print(f"✅ {gesture} captured. Relax.")
            
            print("\n⚙️ Processing user profile...")
            # Filter the massive 15-second combined array
            clean_calib, env_calib = apply_filters(np.array(all_calibration_data))
            
            # Calculate the flawless, balanced mathematical center!
            raw_mean = np.mean(clean_calib)
            raw_std = np.std(clean_calib) + 1e-7
            env_mean = np.mean(env_calib)
            env_std = np.std(env_calib) + 1e-7
            
            print("="*50)
            print(f"✅ Calibration complete! Model is locked to your arm.")
            print("💪 Start making gestures! (Press Ctrl+C to quit)\n")
            # --------------------------------------------------------
            
            while True:
                new_data = task.read(number_of_samples_per_channel=READ_CHUNK)
                filter_buffer = np.roll(filter_buffer, -READ_CHUNK)
                filter_buffer[-READ_CHUNK:] = new_data
                
                clean_signal, envelope_signal = apply_filters(filter_buffer)
                
                window_data = clean_signal[-FEATURE_WINDOW:]
                env_data = envelope_signal[-FEATURE_WINDOW:]
                
                # Normalize using the perfectly balanced onboarded parameters
                norm_window = (window_data - raw_mean) / raw_std
                norm_env = (env_data - env_mean) / env_std
                
                features = extract_features(norm_window, norm_env)
                X_live = np.array(features).reshape(1, -1)
                
                probabilities = xgb_model.predict_proba(X_live)[0]
                predicted_index = np.argmax(probabilities)
                confidence = probabilities[predicted_index]
                gesture_name = classes[predicted_index]
                
                prediction_history.append(gesture_name)
                most_common_gesture = max(set(prediction_history), key=prediction_history.count)
                
                if confidence > 0.50 and prediction_history.count(most_common_gesture) >= 3:
                    if most_common_gesture == "Fist":
                        print(f"✊ [ {most_common_gesture:<14} ]  (Conf: {confidence*100:.0f}%)    ", end='\r')
                    elif most_common_gesture == "Open Hand":
                        print(f"🖐️ [ {most_common_gesture:<14} ]  (Conf: {confidence*100:.0f}%)    ", end='\r')
                    elif most_common_gesture == "Thumbs Up":
                        print(f"👍 [ {most_common_gesture:<14} ]  (Conf: {confidence*100:.0f}%)    ", end='\r')
                    elif most_common_gesture == "Index Finger":
                        print(f"☝️ [ {most_common_gesture:<14} ]  (Conf: {confidence*100:.0f}%)    ", end='\r')
                    else:
                        print(f"〰️ [ {most_common_gesture:<14} ]  (Conf: {confidence*100:.0f}%)    ", end='\r')
                else:
                    print(f"⏳ [ {'Processing':<14} ]                                  ", end='\r')

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
    except KeyboardInterrupt:
        print("\n\n🛑 Live inference stopped.")

if __name__ == "__main__":
    run_live_inference()