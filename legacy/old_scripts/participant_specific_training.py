import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from scipy.stats import skew, kurtosis
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

DATA_DIR = "All_Datasets"
FS = 1000

def notch_filter(data, fs, f0, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return filtfilt(b, a, data)

def bandpass_filter(data, fs, lowcut=20.0, highcut=450.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
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

def process_and_train_participant(filepath):
    df = pd.read_csv(filepath)
    if 'Voltage' not in df.columns or 'Gesture_Label' not in df.columns:
        return 0.0, 0
        
    raw_voltage = df['Voltage'].values
    
    v_filt = raw_voltage
    for harmonic in [50, 100, 150, 200, 250, 300, 350, 400]:
        v_filt = notch_filter(v_filt, FS, f0=harmonic)
    v_filt = bandpass_filter(v_filt, FS, lowcut=20, highcut=450)
    
    v_tkeo = tkeo(v_filt)
    env = envelope(v_tkeo, window=200) 
    
    df['Recovered_Voltage'] = v_filt
    df['Energy_Env'] = env
    
    X = []
    y = []
    
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
            X.append(feats_active)
            y.append(gesture)
            
            # Dynamic Rest Harvesting
            min_energy_idx = np.argmin(pd.Series(group_env).rolling(window_size).sum().dropna().values)
            rest_segment = group['Recovered_Voltage'].values[min_energy_idx : min_energy_idx + window_size]
            
            feats_rest = extract_features(rest_segment)
            X.append(feats_rest)
            y.append('Rest')
            
    if len(X) < 10:
        return 0.0, 0
        
    X = np.array(X)
    y = np.array(y)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.15, random_state=100, stratify=y_enc)
    except:
        X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.15, random_state=100)
        
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_estimators=150, max_depth=5)
    rf_clf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
    hgb_clf = HistGradientBoostingClassifier(random_state=42)
    
    clf = VotingClassifier(estimators=[('xgb', xgb_clf), ('rf', rf_clf), ('hgb', hgb_clf)], voting='soft')
    clf.fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    
    return acc, len(X)

def main():
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Executing Participant-Specific Pipeline on {len(files)} Datasets...")
    print("-" * 50)
    
    accuracies = []
    total_samples = 0
    
    for f in files:
        name = os.path.basename(f).replace('5gesture_dataset.csv', '').replace('_', ' ').title().strip()
        if not name: name = os.path.basename(f)
        
        acc, samples = process_and_train_participant(f)
        accuracies.append(acc)
        total_samples += samples
        
        print(f"Participant: {name:<20} | Accuracy: {acc*100:>6.2f}% | Samples: {samples}")
        
    mean_acc = np.mean(accuracies)
    
    print("\n" + "="*50)
    print(f"FINAL AVERAGE ACCURACY: {mean_acc*100:.2f}%")
    print("="*50)
    print(f"(Trained 15 independent models on {total_samples} total samples)")

if __name__ == "__main__":
    main()
