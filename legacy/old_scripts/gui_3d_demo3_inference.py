import threading
import time
import collections
import joblib
import numpy as np
import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
from scipy.signal import iirnotch, butter, filtfilt
import warnings
import sys

# Try importing ursina, if not present, notify user
try:
    from ursina import *
except ImportError:
    print("❌ Ursina engine is missing. Please run: pip install ursina")
    sys.exit(1)

warnings.filterwarnings("ignore")

# --- Configuration (Kept from original) ---
CHANNEL = "Dev1/ai0"
SAMPLE_RATE = 1000
FILTER_BUFFER_SIZE = 1000
FEATURE_WINDOW = 300
READ_CHUNK = 100

# --- Shared State for Threading ---
class SharedState:
    def __init__(self):
        self.running = True
        self.status_message = "Initializing..."
        self.current_gesture = "Initializing"
        self.confidence = 0.0
        self.prediction_history = collections.deque(maxlen=4)
        
        # Data for monitoring (optional visualization of signal)
        self.raw_data = []
        
        # Calibration state
        self.calibrating = True
        self.calibration_step = 0  # 0:Start, 1:Rest, 2:Fist, 3:Open
        self.calibration_progress = 0.0
        self.normalization_params = {} # mean, std

state = SharedState()

# --- Load Model ---
try:
    print("🧠 Loading Ultimate 4-Person Gross Motor Model...")
    xgb_model = joblib.load(r"final_fist_open\nadicare_xgboost_purged.pkl")
    classes = np.load(r"final_fist_open\nadicare_classes_xgb.npy")
    print(f"✅ Loaded classes: {classes}")
except Exception as e:
    print(f"❌ critical error loading model: {e}")
    # Create dummy for testing UI if model fails (remove in production)
    # xgb_model = None 
    # classes = ["Rest", "Fist", "Open Hand"]
    sys.exit(1)


# --- DSP & Feature Extraction (Exact copy works best) ---
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

    return [mav, rms, var, wl, zc, ssc, env_mean, env_max, activity, mobility, complexity, mean_freq, peak_freq]

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
            calibration_gestures = ["Rest", "Fist", "Open Hand"]
            all_calibration_data = []

            for i, gesture in enumerate(calibration_gestures):
                state.calibration_step = i + 1
                state.status_message = f"CALIBRATING: {gesture.upper()}\nGet Ready..."
                time.sleep(3) 
                
                state.status_message = f"CALIBRATING: {gesture.upper()}\nHOLD NOW!"
                gesture_data = []
                
                # Collect 30 chunks
                for p in range(30):
                    chunk = task.read(number_of_samples_per_channel=READ_CHUNK)
                    gesture_data.extend(chunk)
                    state.calibration_progress = (p + 1) / 30.0
                
                all_calibration_data.extend(gesture_data)
                state.status_message = f"Captured {gesture}. Relax."
                time.sleep(1)

            state.status_message = "Processing profile..."
            clean_calib, env_calib = apply_filters(np.array(all_calibration_data))

            state.normalization_params['raw_mean'] = np.mean(clean_calib)
            state.normalization_params['raw_std'] = np.std(clean_calib) + 1e-7
            state.normalization_params['env_mean'] = np.mean(env_calib)
            state.normalization_params['env_std'] = np.std(env_calib) + 1e-7

            state.calibrating = False
            state.status_message = "Live Inference Active"

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

                if conf > 0.60 and state.prediction_history.count(most_common) >= 3:
                     state.current_gesture = most_common
                     state.confidence = conf
                else:
                    # Keep previous or set to uncertain
                    pass
                    
    except Exception as e:
        state.status_message = f"Error: {str(e)}"
        print(e)

# --- Ursina 3D GUI Setup ---
app = Ursina()

# Professional Dark Theme
window.color = color.rgb(15, 15, 20)
window.title = "NadiCare 3D - Gross Motor Interface"
window.borderless = False
window.fullscreen = False

# Camera Setup
camera.position = (0, 3, -10)
camera.rotation_x = 15

# --- 3D Hand Entity ---
class RobotHand(Entity):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Materials
        self.mat_palm = color.rgb(40, 40, 50)
        self.mat_joint = color.rgb(0, 200, 255) # Cyan neon
        
        # Palm
        self.palm = Entity(parent=self, model='cube', scale=(3, 0.5, 3), color=self.mat_palm)
        
        # Fingers (List of lists of segments)
        self.fingers = []
        
        # Finger Setup: Pos X, Pos Z, Length, Name
        configs = [
            (-1.2, 1.5, 2.5, "Pinky"),
            (-0.4, 1.6, 2.8, "Ring"),
            (0.4, 1.6, 3.0, "Middle"),
            (1.2, 1.5, 2.8, "Index"),
        ]
        
        for x, z, length, name in configs:
            self.create_finger(x, z, length)
            
        # Thumb (Special placement)
        self.create_thumb()

    def create_finger(self, x, z, length):
        # Base Joint
        joint1 = Entity(parent=self.palm, position=(x, 0, 1.5), color=self.mat_joint, model='sphere', scale=0.4)
        # Segment 1
        seg1 = Entity(parent=joint1, model='cube', origin=(0, 0, -0.5), scale=(0.3, 0.3, length/2), position=(0,0,0), color=color.gray)
        # Mid Joint
        joint2 = Entity(parent=seg1, position=(0, 0, 1), color=self.mat_joint, model='sphere', scale=1.2) # Relative scale
        # Segment 2
        seg2 = Entity(parent=joint2, model='cube', origin=(0, 0, -0.5), scale=(1, 1, 1), position=(0,0,0), color=color.gray)
        
        self.fingers.append({'joints': [joint1, joint2], 'type': 'finger'})

    def create_thumb(self):
        # Thumb attached to side
        joint1 = Entity(parent=self.palm, position=(1.8, 0, -0.5), rotation=(0, 45, 0), color=self.mat_joint, model='sphere', scale=0.5)
        seg1 = Entity(parent=joint1, model='cube', origin=(0, 0, -0.5), scale=(0.4, 0.4, 1.2), position=(0,0,0), color=color.gray)
        joint2 = Entity(parent=seg1, position=(0, 0, 1), color=self.mat_joint, model='sphere', scale=1.1)
        seg2 = Entity(parent=joint2, model='cube', origin=(0, 0, -0.5), scale=(1, 1, 0.8), position=(0,0,0), color=color.gray)
        
        self.fingers.append({'joints': [joint1, joint2], 'type': 'thumb'})

    def animate_gesture(self, gesture):
        target_curl = 0
        spread = 0
        
        if gesture == "Fist":
            target_curl = 90
            spread = -5 # squeeze
        elif gesture == "Open Hand":
            target_curl = 0
            spread = 15 # splay
        elif gesture == "Rest":
            target_curl = 30
            spread = 0
        
        # Smooth interpolation
        speed = 5 * time.dt
        
        for i, finger in enumerate(self.fingers):
            joints = finger['joints']
            
            # Curl Logic
            # Thumb moves differently
            if finger['type'] == 'thumb':
                rot = 45 if gesture == "Fist" else 10
                joints[0].rotation_z = lerp(joints[0].rotation_z, rot, speed) # Fold thumb in
                joints[1].rotation_x = lerp(joints[1].rotation_x, target_curl, speed)
            else:
                # Main fingers curl around X axis
                joints[0].rotation_x = lerp(joints[0].rotation_x, target_curl, speed)
                joints[1].rotation_x = lerp(joints[1].rotation_x, target_curl * 1.2, speed) # Tip curls more
                
                # Splay (Rotation around Y)
                # Calculate spread factor based on finger index (0=Pinky, 3=Index) - Center is 1.5
                center_offset = (i - 1.5) 
                target_splay = center_offset * spread
                joints[0].rotation_y = lerp(joints[0].rotation_y, target_splay, speed)


hand_model = RobotHand(position=(0, -1, 0))

# --- UI Overlay ---
title_text = Text(text="GESTURE CONTROL SYSTEM", position=(-0.85, 0.45), scale=1.5, color=color.cyan)
status_text = Text(text="Initializing...", position=(-0.85, 0.38), scale=1.2, color=color.white)
gesture_text = Text(text="WAITING", position=(0, 0.35), origin=(0,0), scale=3, color=color.rgb(255, 100, 0))
conf_bar_bg = Entity(parent=camera.ui, model='quad', scale=(0.6, 0.05), position=(0, 0.25), color=color.rgb(50,50,50))
conf_bar = Entity(parent=camera.ui, model='quad', scale=(0, 0.05), position=(-0.3, 0.25), origin=(-0.5, 0), color=color.cyan)
conf_text = Text(text="0%", position=(0.32, 0.25), origin=(-0.5, 0), color=color.cyan)

# Particles for effect
def spawn_particles():
    if state.current_gesture != "Rest" and not state.calibrating:
        e = Entity(model='sphere', color=color.cyan, scale=0.1, position=(random.uniform(-1,1), -2, random.uniform(-1,1)))
        e.animate_position(e.position + (0, 5, 0), duration=2, curve=curve.linear)
        e.fadeOut(duration=2, delay=0)
        destroy(e, delay=2)


def update():
    # Update Status Text
    status_text.text = state.status_message
    
    # Update Gesture Display
    if state.calibrating:
        gesture_text.text = "CALIBRATION"
        gesture_text.color = color.yellow
        hand_model.animate_gesture("Rest")
        conf_bar.scale_x = state.calibration_progress * 0.6
        conf_text.text = f"{int(state.calibration_progress*100)}%"
    else:
        gesture_text.text = state.current_gesture.upper()
        
        # Color Coding
        if state.current_gesture == "Fist":
            gesture_text.color = color.red
        elif state.current_gesture == "Open Hand":
            gesture_text.color = color.green
        else:
            gesture_text.color = color.gray
            
        hand_model.animate_gesture(state.current_gesture)
        
        # Confidence Bar
        target_scale = state.confidence * 0.6
        conf_bar.scale_x = lerp(conf_bar.scale_x, target_scale, 0.1)
        conf_text.text = f"{int(state.confidence*100)}%"
        
        if state.confidence > 0.8:
            spawn_particles()
    
    # Hand idle animation (breathing)
    hand_model.y = -1 + math.sin(time.time() * 2) * 0.05
    hand_model.rotation_y = math.sin(time.time()) * 5

# --- Start ---
# Start DAQ in background
t = threading.Thread(target=daq_process, daemon=True)
t.start()

# Start GUI
app.run()
