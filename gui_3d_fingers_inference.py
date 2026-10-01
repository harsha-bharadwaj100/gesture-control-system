import threading
import time
import collections
import joblib
import numpy as np
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
from scipy.signal import iirnotch, butter, filtfilt
from scipy.stats import skew, kurtosis
import warnings
import sys

# Try importing ursina
try:
    from ursina import *
except ImportError:
    print("❌ Ursina engine is missing. Please run: pip install ursina")
    sys.exit(1)

warnings.filterwarnings("ignore")

# --- Configuration ---
CHANNEL = "Dev1/ai0"
SAMPLE_RATE = 1000
FILTER_BUFFER_SIZE = 1000
FEATURE_WINDOW = 300
READ_CHUNK = 100

# --- Shared State ---
class SharedState:
    def __init__(self):
        self.running = True
        self.status_message = "Initializing..."
        self.current_gesture = "Initializing"
        self.confidence = 0.0
        self.prediction_history = collections.deque(maxlen=5) # 5 for fingers
        self.calibrating = True
        self.calibration_progress = 0.0
        self.normalization_params = {} 

state = SharedState()

# --- Load Model ---
try:
    print("🧠 Loading Fine-Motor Finger Model...")
    xgb_model = joblib.load("nadicare_finger_model.pkl")
    classes = np.load("nadicare_classes_finger.npy")
    print(f"✅ Loaded classes: {classes}")
except Exception as e:
    print(f"❌ critical error loading model: {e}")
    sys.exit(1)


# --- DSP & Feature Extraction ---
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
    # 16 Features for Fingers
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
    complexity = (
        np.sqrt(np.var(diff2) / np.var(diff1)) / mobility if mobility > 0 else 0
    )

    fft_vals = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(len(window), d=1.0 / SAMPLE_RATE)
    total_power = np.sum(fft_vals)
    mean_freq = np.sum(freqs * fft_vals) / total_power if total_power > 0 else 0
    peak_freq = freqs[np.argmax(fft_vals)] if total_power > 0 else 0

    return [raw_mean, skew_val, kurt_val, mav, rms, var, wl, zc, ssc, env_mean, env_max, 
            activity, mobility, complexity, mean_freq, peak_freq]

# --- DAQ Thread ---
def daq_process():
    filter_buffer = np.zeros(FILTER_BUFFER_SIZE)
    
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

            print(f"🚀 Hardware locked onto {CHANNEL}.")
            task.start()

            # --- Calibration Sequence ---
            calibration_gestures = ["Rest", "Thumbs Up", "Index Finger"]
            all_calibration_data = []

            for i, gesture in enumerate(calibration_gestures):
                state.status_message = f"CALIBRATING: {gesture.upper()}\nGet Ready..."
                time.sleep(3) 
                
                state.status_message = f"CALIBRATING: {gesture.upper()}\nHOLD NOW!"
                gesture_data = []
                
                for p in range(30):
                    chunk = task.read(number_of_samples_per_channel=READ_CHUNK)
                    gesture_data.extend(chunk)
                    state.calibration_progress = (p + 1) / 30.0
                
                all_calibration_data.extend(gesture_data)
                state.status_message = f"Captured {gesture}. Relax."
                time.sleep(1)

            state.status_message = "Processing finger profile..."
            clean_calib, env_calib = apply_filters(np.array(all_calibration_data))

            state.normalization_params['raw_mean'] = np.mean(clean_calib)
            state.normalization_params['raw_std'] = np.std(clean_calib) + 1e-7
            state.normalization_params['env_mean'] = np.mean(env_calib)
            state.normalization_params['env_std'] = np.std(env_calib) + 1e-7

            state.calibrating = False
            state.status_message = "Fine Motor Inference Active"

            # --- Main Loop ---
            raw_mean = state.normalization_params['raw_mean']
            raw_std = state.normalization_params['raw_std']
            env_mean = state.normalization_params['env_mean']
            env_std = state.normalization_params['env_std']

            while state.running:
                new_data = task.read(number_of_samples_per_channel=READ_CHUNK)
                filter_buffer = np.roll(filter_buffer, -READ_CHUNK)
                filter_buffer[-READ_CHUNK:] = new_data

                clean_signal, envelope_signal = apply_filters(filter_buffer)

                window_data = clean_signal[-FEATURE_WINDOW:]
                env_data = envelope_signal[-FEATURE_WINDOW:]

                norm_window = (window_data - raw_mean) / raw_std
                norm_env = (env_data - env_mean) / env_std

                features = extract_features(norm_window, norm_env)
                X_live = np.array(features).reshape(1, -1)

                probabilities = xgb_model.predict_proba(X_live)[0]
                predicted_index = np.argmax(probabilities)
                conf = probabilities[predicted_index]
                gesture_name = classes[predicted_index]

                state.prediction_history.append(gesture_name)
                most_common = max(set(state.prediction_history), key=state.prediction_history.count)

                if conf > 0.55 and state.prediction_history.count(most_common) >= 3:
                     state.current_gesture = most_common
                     state.confidence = conf
                else:
                    pass
                    
    except Exception as e:
        state.status_message = f"Error: {str(e)}"
        print(e)
    except KeyboardInterrupt:
        state.running = False

# --- Ursina 3D GUI Setup ---
app = Ursina()

# Design: Cyberpunk / Medical High Tech
window.color = color.rgb(10, 20, 30) # Dark Navy
window.title = "NadiCare 3D - Fine Motor Interface"

camera.position = (0, 2, -12)
camera.rotation_x = 10

# --- 3D Hand Entity ---
class RobotHand_Fine(Entity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.mat_palm = color.rgb(60, 60, 80)
        self.mat_joint = color.rgb(255, 165, 0) # Orange Neon
        self.mat_finger = color.rgb(200, 200, 200)
        
        self.palm = Entity(parent=self, model='cube', scale=(3, 0.5, 3), color=self.mat_palm)
        self.fingers = []
        
        # Fingers: Pinky to Index
        configs = [
            (-1.2, 1.5, 2.5, "Pinky"),
            (-0.4, 1.6, 2.8, "Ring"),
            (0.4, 1.6, 3.0, "Middle"),
            (1.2, 1.5, 2.8, "Index"),
        ]
        
        for x, z, length, name in configs:
            self.create_finger(x, z, length, name)
            
        self.create_thumb()

    def create_finger(self, x, z, length, name):
        joint1 = Entity(parent=self.palm, position=(x, 0, 1.5), color=self.mat_joint, model='sphere', scale=0.4)
        seg1 = Entity(parent=joint1, model='cube', origin=(0, 0, -0.5), scale=(0.3, 0.3, length/2), position=(0,0,0), color=self.mat_finger)
        joint2 = Entity(parent=seg1, position=(0, 0, 1), color=self.mat_joint, model='sphere', scale=1.2)
        seg2 = Entity(parent=joint2, model='cube', origin=(0, 0, -0.5), scale=(1, 1, 1), position=(0,0,0), color=self.mat_finger)
        
        self.fingers.append({'joints': [joint1, joint2], 'type': 'finger', 'name': name})

    def create_thumb(self):
        joint1 = Entity(parent=self.palm, position=(1.8, 0, -0.5), rotation=(0, 45, 0), color=self.mat_joint, model='sphere', scale=0.5)
        seg1 = Entity(parent=joint1, model='cube', origin=(0, 0, -0.5), scale=(0.4, 0.4, 1.2), position=(0,0,0), color=self.mat_finger)
        joint2 = Entity(parent=seg1, position=(0, 0, 1), color=self.mat_joint, model='sphere', scale=1.1)
        seg2 = Entity(parent=joint2, model='cube', origin=(0, 0, -0.5), scale=(1, 1, 0.8), position=(0,0,0), color=self.mat_finger)
        
        self.fingers.append({'joints': [joint1, joint2], 'type': 'thumb', 'name': 'Thumb'})

    def animate_gesture(self, gesture):
        speed = 8 * time.dt
        
        # Default State: Loose Fist / Relaxed
        # Target angles for each finger [BottomJoint, TopJoint]
        targets = {
            "Thumb": [10, 10],   # Slightly in
            "Index": [30, 30],   # Curled
            "Middle": [30, 30],
            "Ring": [30, 30],
            "Pinky": [30, 30]
        }
        
        if gesture == "Index Finger":
            # Index points out, others curled
            targets["Index"] = [0, 0]
            targets["Middle"] = [90, 90]
            targets["Ring"] = [90, 90]
            targets["Pinky"] = [90, 90]
            targets["Thumb"] = [45, 45] # Thumb tucked
            
        elif gesture == "Thumbs Up":
            # Thumb up (out), others curled into fist
            targets["Thumb"] = [-45, 0] # Rotate out/up
            targets["Index"] = [90, 90]
            targets["Middle"] = [90, 90]
            targets["Ring"] = [90, 90]
            targets["Pinky"] = [90, 90]
            
        elif gesture == "Rest":
            # Loose
            pass 

        for finger in self.fingers:
            name = finger['name']
            t_angles = targets.get(name, [0,0])
            joints = finger['joints']
            
            if name == 'Thumb':
                joints[0].rotation_z = lerp(joints[0].rotation_z, t_angles[0], speed)
                joints[1].rotation_x = lerp(joints[1].rotation_x, t_angles[1], speed)
            else:
                joints[0].rotation_x = lerp(joints[0].rotation_x, t_angles[0], speed)
                joints[1].rotation_x = lerp(joints[1].rotation_x, t_angles[1], speed)

hand_model = RobotHand_Fine(position=(0, -1, 0))

# --- UI Overlay ---
title_text = Text(text="FINGER CONTROL INTERFACE", position=(-0.8, 0.45), scale=1.5, color=color.orange)
status_text = Text(text="Initializing...", position=(-0.8, 0.38), scale=1.2, color=color.white)
gesture_icon = Entity(parent=camera.ui, model='quad', scale=0.3, position=(0.6, 0.2), color=color.white, texture='white_cube')
gesture_text = Text(text="WAITING", position=(0, 0.35), origin=(0,0), scale=3, color=color.rgb(255, 165, 0))

# Confidence Radar (Simple Bar)
conf_bar_bg = Entity(parent=camera.ui, model='quad', scale=(0.5, 0.02), position=(0, -0.4), color=color.gray)
conf_bar = Entity(parent=camera.ui, model='quad', scale=(0, 0.02), position=(-0.25, -0.4), origin=(-0.5, 0), color=color.orange)


def update():
    status_text.text = state.status_message
    
    if state.calibrating:
        gesture_text.text = "CALIBRATION"
        gesture_text.color = color.yellow
        hand_model.animate_gesture("Rest")
        conf_bar.scale_x = state.calibration_progress * 0.5
    else:
        gesture_text.text = state.current_gesture.upper()
        hand_model.animate_gesture(state.current_gesture)
        
        target_scale = state.confidence * 0.5
        conf_bar.scale_x = lerp(conf_bar.scale_x, target_scale, 0.1)

# --- Start ---
t = threading.Thread(target=daq_process, daemon=True)
t.start()

app.run()
