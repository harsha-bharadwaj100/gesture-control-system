import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = "All_Datasets"
OUTPUT_FILE = "extracted_features.csv"

def extract_features(signal):
    """Calculate RMS, MAV, Waveform Length, and Zero Crossings."""
    if len(signal) == 0:
        return [0, 0, 0, 0]
    
    rms = np.sqrt(np.mean(signal**2))
    mav = np.mean(np.abs(signal))
    wl = np.sum(np.abs(np.diff(signal)))
    zc = np.sum(np.diff(np.signbit(signal)))
    
    return [rms, mav, wl, zc]

def load_and_extract(data_dir, window_size=500, overlap=250):
    """
    Reads all CSVs, applies sliding window feature extraction.
    """
    data = []
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"Found {len(csv_files)} datasets. Extracting features...")
    
    for file in csv_files:
        participant = os.path.basename(file).replace('5gesture_dataset.csv', '').replace('.csv', '')
        df = pd.read_csv(file)
        
        if 'Voltage' not in df.columns or 'Gesture_Label' not in df.columns:
            continue
            
        grouped = df.groupby(['Gesture_Label', 'Repetition'])
        for (gesture, rep), group in grouped:
            voltage = group['Voltage'].values
            
            step = window_size - overlap
            for start in range(0, len(voltage) - window_size + 1, step):
                window = voltage[start:start + window_size]
                rms, mav, wl, zc = extract_features(window)
                data.append({
                    'Participant': participant,
                    'Gesture_Label': gesture,
                    'Repetition': rep,
                    'RMS': rms,
                    'MAV': mav,
                    'WL': wl,
                    'ZC': zc
                })
                
    return pd.DataFrame(data)

if __name__ == "__main__":
    df_features = load_and_extract(DATA_DIR, window_size=500, overlap=250)
    df_features.to_csv(OUTPUT_FILE, index=False)
    print(f"Extraction complete! Saved {len(df_features)} samples to {OUTPUT_FILE}")
