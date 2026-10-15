import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "All_Datasets"
FS = 1000

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
    y = np.zeros_like(data)
    y[1:-1] = data[1:-1]**2 - data[:-2] * data[2:]
    return y

def envelope(data, window=50):
    return pd.Series(np.abs(data)).rolling(window, min_periods=1, center=True).mean().values

def process_and_extract_sequences():
    all_X = []
    all_y = []
    
    files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Processing {len(files)} files for Generalized 1D CNN...")
    
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
        dead_threshold = median_energy * 0.05
        
        subj_X = []
        subj_y = []
        
        for (gesture, rep), group in grouped:
            if gesture.lower() == 'rest' or 'baseline' in gesture.lower():
                continue
                
            group_env = group['Energy_Env'].values
            
            if np.max(group_env) < dead_threshold:
                continue
                
            window_size = int(1.5 * FS) # 1500 timesteps
            
            if len(group_env) > window_size:
                max_energy_idx = np.argmax(pd.Series(group_env).rolling(window_size).sum().dropna().values)
                active_segment = group['Recovered_Voltage'].values[max_energy_idx : max_energy_idx + window_size]
                
                # Zero padding if slightly short
                if len(active_segment) < window_size:
                    active_segment = np.pad(active_segment, (0, window_size - len(active_segment)), 'constant')
                else:
                    active_segment = active_segment[:window_size]
                    
                subj_X.append(active_segment)
                subj_y.append(gesture)
                
                # Balanced Rest Harvesting
                if rep % 5 == 0:
                    min_energy_idx = np.argmin(pd.Series(group_env).rolling(window_size).sum().dropna().values)
                    rest_segment = group['Recovered_Voltage'].values[min_energy_idx : min_energy_idx + window_size]
                    
                    if len(rest_segment) < window_size:
                        rest_segment = np.pad(rest_segment, (0, window_size - len(rest_segment)), 'constant')
                    else:
                        rest_segment = rest_segment[:window_size]
                        
                    subj_X.append(rest_segment)
                    subj_y.append('Rest')
                    
        # Per-participant Sequence Standardization
        if len(subj_X) > 0:
            subj_X = np.array(subj_X)
            # Standardize across all timesteps for this participant
            mean_val = np.mean(subj_X)
            std_val = np.std(subj_X)
            subj_X_scaled = (subj_X - mean_val) / (std_val + 1e-8)
            
            all_X.extend(subj_X_scaled)
            all_y.extend(subj_y)
                
    return np.array(all_X), np.array(all_y)

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=64, kernel_size=10, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),
        Dropout(0.3),
        
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=4),
        Dropout(0.3),
        
        LSTM(64, return_sequences=False),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    X, y = process_and_extract_sequences()
    print(f"Total valid sequences extracted: {len(X)}")
    
    # Reshape for CNN input: (samples, timesteps, features=1)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    num_classes = len(le.classes_)
    print("Classes:", le.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.15, random_state=42, stratify=y_enc)
    
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    
    print("\nTraining Generalized 1D CNN + LSTM Deep Learning Model...")
    
    model = build_model((X.shape[1], 1), num_classes)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=1
    )
    
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    
    print("\n========================================================")
    print(f"FINAL DEEP LEARNING GENERALIZED ACCURACY: {acc*100:.2f}%")
    print("========================================================\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

if __name__ == "__main__":
    main()
