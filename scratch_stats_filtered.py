import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

SAMPLE_RATE = 1000

def apply_filters(data):
    # 50Hz notch filter
    b_notch, a_notch = iirnotch(50.0, 30.0, SAMPLE_RATE)
    data_notched = filtfilt(b_notch, a_notch, data)
    
    # 20-450Hz bandpass filter
    nyquist = SAMPLE_RATE / 2.0
    b_band, a_band = butter(4, [20.0 / nyquist, 450.0 / nyquist], btype='band')
    clean_data = filtfilt(b_band, a_band, data_notched)
    
    return clean_data

# Load dataset
df = pd.read_csv('All_Datasets/harsha5gesture_dataset.csv')
print("Loaded dataset.")

# Apply filter per repetition per gesture
grouped = df.groupby(['Gesture_Label', 'Repetition'])
filtered_voltages = []

for name, group in grouped:
    v = group['Voltage'].values
    v_filt = apply_filters(v)
    filtered_voltages.extend(v_filt)

df['Filtered_Voltage'] = filtered_voltages

df['Time_Idx'] = df.groupby(['Gesture_Label', 'Repetition']).cumcount()
profile = df.groupby('Time_Idx')['Filtered_Voltage'].apply(lambda x: np.mean(np.abs(x))).reset_index()

for idx in [0, 100, 200, 300, 400, 500, 600, 1000, 2000, 3000, 3900]:
    row = profile[profile['Time_Idx'] == idx]
    if not row.empty:
        val = row["Filtered_Voltage"].values[0]
        print(f"Sample {idx:4d} ({idx}ms): Filtered MAV = {val:.6f}")

# Let's save a plot of the filtered profile
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(profile['Time_Idx'], profile['Filtered_Voltage'])
plt.title('Filtered EMG Activation Profile Over 4-Second Window (All Gestures, Harsha)')
plt.xlabel('Sample Index (ms since start of prompt)')
plt.ylabel('Mean Absolute Voltage (Filtered)')
plt.grid(True)
plt.savefig('filtered_activation_profile.png')
print("Saved filtered_activation_profile.png")
