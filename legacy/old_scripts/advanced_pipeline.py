import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import skew, kurtosis
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = "All_Datasets"
FS = 1000  # Assumed sampling frequency 1000Hz based on timestamps

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
    """Teager-Kaiser Energy Operator"""
    y = np.zeros_like(data)
    y[1:-1] = data[1:-1]**2 - data[:-2] * data[2:]
    return y

def envelope(data, window=50):
    """Simple moving average envelope"""
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

def extract_features(signal, fs=FS):
    if len(signal) == 0:
        return [0]*13
    
    mav = np.mean(np.abs(signal))
    rms = np.sqrt(np.mean(signal**2))
    var = np.var(signal)
    wl = np.sum(np.abs(np.diff(signal)))
    zc = np.sum(np.diff(np.signbit(signal)))
    ssc = np.sum(np.diff(np.sign(np.diff(signal))) != 0)
    
    env = envelope(signal)
    env_mean = np.mean(env)
    env_max = np.max(env)
    
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
        
    return [mav, rms, var, wl, zc, ssc, env_mean, env_max, activity, mobility, complexity, mean_freq, peak_freq]

def process_and_extract():
    all_X = []
    all_y = []
    
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Processing {len(files)} files with extreme recovery...")
    
    for f in files:
        df = pd.read_csv(f)
        if 'Voltage' not in df.columns or 'Gesture_Label' not in df.columns:
            continue
            
        raw_voltage = df['Voltage'].values
        
        v_filt = raw_voltage
        # Very aggressive 50Hz harmonic obliteration
        for harmonic in [50, 100, 150, 200, 250, 300, 350, 400]:
            v_filt = notch_filter(v_filt, FS, f0=harmonic)
        v_filt = bandpass_filter(v_filt, FS, lowcut=20, highcut=450)
        
        v_tkeo = tkeo(v_filt)
        env = envelope(v_tkeo, window=200) 
        
        df['Recovered_Voltage'] = v_filt
        df['Energy_Env'] = env
        
        # We will keep ALL trials (since user said use them all) but we will extract multiple segments
        grouped = df.groupby(['Gesture_Label', 'Repetition'])
        
        for (gesture, rep), group in grouped:
            if gesture.lower() == 'rest' or 'baseline' in gesture.lower():
                continue
                
            group_env = group['Energy_Env'].values
            window_size = int(1.5 * FS)
            
            if len(group_env) > window_size:
                max_energy_idx = np.argmax(pd.Series(group_env).rolling(window_size).sum().dropna().values)
                active_segment = group['Recovered_Voltage'].values[max_energy_idx : max_energy_idx + window_size]
                
                feats_active = extract_features(active_segment)
                all_X.append(feats_active)
                all_y.append(gesture)
                
                # To balance the dataset automatically, we only harvest ONE rest sample for every FIVE gestures
                # Since there are 5 gestures in the dataset, if we harvest 1 rest per gesture repetition, Rest is 50%
                # If we only harvest rest from rep % 5 == 0, Rest becomes ~16% (perfectly balanced with 5 gestures)
                if rep % 5 == 0:
                    min_energy_idx = np.argmin(pd.Series(group_env).rolling(window_size).sum().dropna().values)
                    rest_segment = group['Recovered_Voltage'].values[min_energy_idx : min_energy_idx + window_size]
                    
                    feats_rest = extract_features(rest_segment)
                    all_X.append(feats_rest)
                    all_y.append('Rest')
            else:
                feats = extract_features(group['Recovered_Voltage'].values)
                all_X.append(feats)
                all_y.append(gesture)
                
    return np.array(all_X), np.array(all_y)

def main():
    X, y = process_and_extract()
    print(f"Total samples extracted: {len(X)}")
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("Classes:", le.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Compute sample weights to force the model to care equally about all gestures
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    print("\nTraining Ensemble with Class Balancing...")
    
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_estimators=400, max_depth=12, learning_rate=0.05)
    rf_clf = RandomForestClassifier(n_estimators=400, max_depth=20, class_weight='balanced', random_state=42)
    
    voting_clf = VotingClassifier(estimators=[('xgb', xgb_clf), ('rf', rf_clf)], voting='soft')
    
    # Fit individual models first so we can apply sample weights to XGBoost
    xgb_clf.fit(X_train_scaled, y_train, sample_weight=sample_weights)
    rf_clf.fit(X_train_scaled, y_train)
    
    # Manually ensemble predictions to get past the VotingClassifier sample_weight limitation
    xgb_preds = xgb_clf.predict_proba(X_test_scaled)
    rf_preds = rf_clf.predict_proba(X_test_scaled)
    
    ensemble_preds = (xgb_preds + rf_preds) / 2.0
    y_pred = np.argmax(ensemble_preds, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    
    print("\n==============================")
    print(f"FINAL ACCURACY: {acc*100:.2f}%")
    print("==============================\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

if __name__ == "__main__":
    main()
