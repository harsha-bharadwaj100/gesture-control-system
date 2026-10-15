import nidaqmx
from nidaqmx.constants import TerminalConfiguration, AcquisitionType
import numpy as np
import time
import os
import collections

# Suppress annoying TensorFlow terminal warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

# --- Configuration ---
CHANNEL = "Dev1/ai0"
SAMPLE_RATE = 1000
WINDOW_SIZE = 200
BUFFER_SIZE = 100
print("🧠 Loading Nadicare Generalized Model...")
model = load_model("nadicare_generalized_model.h5")
classes = np.load("gesture_classes.npy")
print(f"✅ Loaded classes: {classes}")

import collections

# Update this at the top of your script
BUFFER_SIZE = 100  # Pulls 0.1 seconds of data at a time to prevent buffer overflow

def run_live_inference():
    rolling_window = np.zeros(WINDOW_SIZE)
    prediction_history = collections.deque(maxlen=5)
    
    try:
        with nidaqmx.Task() as task:
            task.ai_channels.add_ai_voltage_chan(
                CHANNEL,
                terminal_config=TerminalConfiguration.DIFF,
                min_val=-5.0,
                max_val=5.0
            )
            
            # The hardware buffer needs to be large enough to hold continuous data
            task.timing.cfg_samp_clk_timing(
                rate=SAMPLE_RATE,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=SAMPLE_RATE * 2  # Set hardware buffer to 2 seconds
            )

            print(f"\n🚀 Hardware locked onto {CHANNEL}.")
            task.start()
            
            # --- THE FIX: BASELINE CALIBRATION ---
            print("\n🛑 KEEP ARM COMPLETELY RELAXED! Calibrating baseline noise...")
            time.sleep(1) # Let the user settle
            
            baseline_data = []
            # Read 30 chunks of 100 samples (3 seconds of data)
            for _ in range(30):
                baseline_data.extend(task.read(number_of_samples_per_channel=BUFFER_SIZE))
                
            global_mean = np.mean(baseline_data)
            global_std = np.std(baseline_data) + 1e-7
            
            print(f"✅ Calibration complete! (Mean: {global_mean:.4f}V, Std: {global_std:.4f})")
            print("💪 Start making gestures! (Press Ctrl+C to quit)\n")
            # -------------------------------------
            
            while True:
                # 1. Read 100 samples
                new_data = task.read(number_of_samples_per_channel=BUFFER_SIZE)
                
                # 2. Slide the window
                rolling_window = np.roll(rolling_window, -BUFFER_SIZE)
                rolling_window[-BUFFER_SIZE:] = new_data
                
                # 3. FIXED NORMALIZATION (Using global baseline, not local window)
                normalized_window = (rolling_window - global_mean) / global_std
                
                # 4. Prepare shape
                X_live = normalized_window.reshape(1, WINDOW_SIZE, 1)
                
                # 5. Fast Predict
                predictions = model(X_live, training=False).numpy()[0]
                
                predicted_index = np.argmax(predictions)
                confidence = predictions[predicted_index]
                gesture_name = classes[predicted_index]
                
                # 6. Add to history for temporal smoothing
                prediction_history.append(gesture_name)
                most_common_gesture = max(set(prediction_history), key=prediction_history.count)
                
                # 7. Stable Output
                if confidence > 0.50 and prediction_history.count(most_common_gesture) >= 3:
                    print(f"✅ Prediction: [ {most_common_gesture:<12} ]  (Raw Conf: {confidence*100:.0f}%)    ", end='\r')
                else:
                    print(f"⏳ Processing: [ {'Uncertain':<12} ]                                  ", end='\r')

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
    except KeyboardInterrupt:
        print("\n\n🛑 Live inference stopped.")

if __name__ == "__main__":
    run_live_inference()