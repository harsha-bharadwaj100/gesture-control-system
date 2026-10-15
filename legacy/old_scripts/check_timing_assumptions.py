import pandas as pd
import numpy as np
import glob
import os
from scipy.signal import butter, filtfilt, iirnotch

SAMPLE_RATE = 1000

def apply_filters_with_detrend(data):
    # Detrend (subtract mean) to remove DC offset before filtering
    data_detrended = data - np.mean(data)
    
    # 50Hz notch filter
    b_notch, a_notch = iirnotch(50.0, 30.0, SAMPLE_RATE)
    data_notched = filtfilt(b_notch, a_notch, data_detrended)
    
    # 20-450Hz bandpass filter
    nyquist = SAMPLE_RATE / 2.0
    b_band, a_band = butter(4, [20.0 / nyquist, 450.0 / nyquist], btype='band')
    clean_data = filtfilt(b_band, a_band, data_notched)
    
    return clean_data

def analyze_onsets_offsets(file_path):
    participant = os.path.basename(file_path).replace('5gesture_dataset.csv', '').replace('.csv', '')
    df = pd.read_csv(file_path)
    
    df['Time_Idx'] = df.groupby(['Gesture_Label', 'Repetition']).cumcount()
    results = []
    grouped = df.groupby(['Gesture_Label', 'Repetition'])
    
    for (gesture, rep), group in grouped:
        v = group['Voltage'].values
        if len(v) < 1000:
            continue
        v_filt = apply_filters_with_detrend(v)
        
        # Calculate MAV with 100ms window
        win_size = 100
        mav = np.array([np.mean(np.abs(v_filt[i:i+win_size])) for i in range(len(v_filt) - win_size + 1)])
        
        # Noise floor: average of the quietest 200ms in the segment
        # In a 4-second block, let's look at all 200ms non-overlapping segments
        segments_200 = [np.mean(np.abs(v_filt[i:i+200])) for i in range(0, len(v_filt) - 200, 100)]
        noise_floor = np.min(segments_200)
        
        # Peak level
        peak_val = np.max(mav)
        
        # Let's see if there is any contraction
        # Threshold: noise floor + 20% of (peak - noise_floor)
        # But we also set a minimum threshold of 0.002V (2mV) to prevent triggering on pure noise
        threshold = max(noise_floor + 0.20 * (peak_val - noise_floor), 0.003)
        
        is_flat = (peak_val - noise_floor) < 0.002 or peak_val < 0.003
        
        onset = None
        offset = None
        
        if not is_flat:
            # Find when MAV crosses the threshold
            active_indices = np.where(mav > threshold)[0]
            if len(active_indices) > 0:
                # Add window size / 2 to align center of window
                onset = active_indices[0] + win_size // 2
                offset = active_indices[-1] + win_size // 2
        
        results.append({
            'Participant': participant,
            'Gesture': gesture,
            'Repetition': rep,
            'Noise_Floor': noise_floor,
            'Peak_MAV': peak_val,
            'Is_Flat': is_flat,
            'Onset_ms': onset,
            'Offset_ms': offset
        })
        
    return pd.DataFrame(results)

def main():
    csv_files = sorted(glob.glob('All_Datasets/*.csv'))
    print(f"Found {len(csv_files)} datasets. Analyzing with detrending...")
    
    summary_list = []
    for f in csv_files:
        p_name = os.path.basename(f).replace('5gesture_dataset.csv', '')
        res_df = analyze_onsets_offsets(f)
        summary_list.append(res_df)
        
        flat_pct = res_df['Is_Flat'].mean() * 100
        active_df = res_df[~res_df['Is_Flat']]
        if len(active_df) > 0:
            avg_onset = active_df['Onset_ms'].median()
            avg_offset = active_df['Offset_ms'].median()
        else:
            avg_onset = np.nan
            avg_offset = np.nan
            
        print(f"Participant: {p_name}")
        print(f"  -> Flat trials: {flat_pct:.1f}%")
        print(f"  -> Median Onset: {avg_onset} ms")
        print(f"  -> Median Offset: {avg_offset} ms")
        print(f"  -> Active trials analyzed: {len(active_df)} / {len(res_df)}")
        print("-" * 50)

if __name__ == "__main__":
    main()
