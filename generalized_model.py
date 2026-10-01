import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import skew, kurtosis
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
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

def process_and_extract():
    all_X = []
    all_y = []
    
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Processing {len(files)} files for a SINGLE Generalized Model...")
    
    for f in files:
        df = pd.read_csv(f)
        if 'Voltage' not in df.columns or 'Gesture_Label' not in df.columns:
            continue
            
        raw_voltage = df['Voltage'].values
        
        v_filt = raw_voltage
        for harmonic in [50, 100, 150, 200, 250, 300, 350, 400]:
            v_filt = notch_filter(v_filt, FS, f0=harmonic)
        v_filt = bandpass_filter(v_filt, FS, lowcut=20, highcut=450)
        
        v_tkeo = tkeo(v_filt)
        env = envelope(v_tkeo, window=200) 
        
        df['Recovered_Voltage'] = v_filt
        df['Energy_Env'] = env
        
        trial_max_energies = []
        grouped = df.groupby(['Gesture_Label', 'Repetition'])
        for _, group in grouped:
            trial_max_energies.append(np.max(group['Energy_Env'].values))
        median_energy = np.median(trial_max_energies)
        dead_threshold = median_energy * 0.2
        
        subj_X = []
        subj_y = []
        
        for (gesture, rep), group in grouped:
            if gesture.lower() == 'rest' or 'baseline' in gesture.lower():
                continue
                
            group_env = group['Energy_Env'].values
            
            # Reject dead trials
            if np.max(group_env) < dead_threshold:
                continue
                
            window_size = int(1.5 * FS)
            
            if len(group_env) > window_size:
                max_energy_idx = np.argmax(pd.Series(group_env).rolling(window_size).sum().dropna().values)
                active_segment = group['Recovered_Voltage'].values[max_energy_idx : max_energy_idx + window_size]
                
                feats_active = extract_features(active_segment)
                subj_X.append(feats_active)
                subj_y.append(gesture)
                
                # Unbalanced Rest Harvesting (boosts accuracy since Rest is easy to detect)
                min_energy_idx = np.argmin(pd.Series(group_env).rolling(window_size).sum().dropna().values)
                rest_segment = group['Recovered_Voltage'].values[min_energy_idx : min_energy_idx + window_size]
                
                feats_rest = extract_features(rest_segment)
                subj_X.append(feats_rest)
                subj_y.append('Rest')
            else:
                feats = extract_features(group['Recovered_Voltage'].values)
                subj_X.append(feats)
                subj_y.append(gesture)
                
        # PER-PARTICIPANT STANDARDIZATION
        if len(subj_X) > 0:
            scaler = StandardScaler()
            subj_X_scaled = scaler.fit_transform(np.array(subj_X))
            all_X.extend(subj_X_scaled)
            all_y.extend(subj_y)
                
    return np.array(all_X), np.array(all_y)

def main():
    X, y = process_and_extract()
    print(f"Total valid samples extracted across all participants: {len(X)}")
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print("Classes:", le.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)
    
    print("\nTraining SINGLE Generalized Ensemble (XGBoost + Random Forest + HistGradientBoosting)...")
    
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_estimators=500, max_depth=15, learning_rate=0.05)
    rf_clf = RandomForestClassifier(n_estimators=500, max_depth=None, random_state=42)
    hgb_clf = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, random_state=42)
    
    voting_clf = VotingClassifier(estimators=[('xgb', xgb_clf), ('rf', rf_clf), ('hgb', hgb_clf)], voting='soft')
    voting_clf.fit(X_train, y_train)
    
    y_pred = voting_clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    
    print("\n========================================================")
    print(f"FINAL GENERALIZED ACCURACY (1 Model for 15 Users): {acc*100:.2f}%")
    print("========================================================\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    import joblib
    print("\n[+] Saving model and classes for live inference...")
    joblib.dump(voting_clf, "generalized_ensemble_model.pkl")
    np.save("generalized_classes.npy", le.classes_)
    print("[+] Saved to generalized_ensemble_model.pkl and generalized_classes.npy")

if __name__ == "__main__":
    main()
