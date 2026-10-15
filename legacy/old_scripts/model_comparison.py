import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D

DATA_DIR = "All_Datasets"

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
    Assumes CSV has 'Voltage' and 'Gesture_Label'.
    """
    X = []
    y = []
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"Found {len(csv_files)} datasets.")
    
    for file in csv_files:
        df = pd.read_csv(file)
        
        if 'Voltage' not in df.columns or 'Gesture_Label' not in df.columns:
            continue
            
        # Group by Gesture and Repetition to ensure we don't mix them
        grouped = df.groupby(['Gesture_Label', 'Repetition'])
        
        for (gesture, rep), group in grouped:
            voltage = group['Voltage'].values
            
            # Sliding window over the repetition
            step = window_size - overlap
            for start in range(0, len(voltage) - window_size + 1, step):
                window = voltage[start:start + window_size]
                features = extract_features(window)
                X.append(features)
                y.append(gesture)
                
    return np.array(X), np.array(y)

def main():
    print("Extracting features (RMS, MAV, Waveform Length, Zero Crossings)...")
    X, y = load_and_extract(DATA_DIR, window_size=500, overlap=250)
    
    if len(X) == 0:
        print("No data found or extracted. Please check the dataset path.")
        return

    print(f"Extracted {len(X)} samples with 4 features each.")

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    print(f"Classes: {le.classes_}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # 1. Support Vector Machine (SVM)
    print("\n--- Training SVM ---")
    svm_model = SVC(kernel='rbf', random_state=42)
    svm_model.fit(X_train_scaled, y_train)
    y_pred_svm = svm_model.predict(X_test_scaled)
    acc_svm = accuracy_score(y_test, y_pred_svm)
    results['SVM'] = acc_svm
    print(f"SVM Accuracy: {acc_svm:.4f}")

    # 2. Random Forest (RF)
    print("\n--- Training Random Forest ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    y_pred_rf = rf_model.predict(X_test_scaled)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    results['Random Forest'] = acc_rf
    print(f"Random Forest Accuracy: {acc_rf:.4f}")

    # 3. K-Nearest Neighbors (KNN)
    print("\n--- Training KNN ---")
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    y_pred_knn = knn_model.predict(X_test_scaled)
    acc_knn = accuracy_score(y_test, y_pred_knn)
    results['KNN'] = acc_knn
    print(f"KNN Accuracy: {acc_knn:.4f}")

    # 4. Convolutional Neural Network (CNN)
    # Reshape for 1D CNN: (samples, timesteps, features) -> (samples, 4, 1)
    print("\n--- Training CNN ---")
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    # Convert to categorical for categorical_crossentropy
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

    cnn_model = Sequential([
        Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Train CNN
    cnn_model.fit(X_train_cnn, y_train_cat, epochs=15, batch_size=64, validation_split=0.2, verbose=1)
    
    # Evaluate CNN
    loss, acc_cnn = cnn_model.evaluate(X_test_cnn, y_test_cat, verbose=0)
    results['CNN'] = acc_cnn
    print(f"CNN Accuracy: {acc_cnn:.4f}")

    # --- Final Comparison ---
    print("\n==============================")
    print("MODEL COMPARISON SUMMARY")
    print("==============================")
    for model_name, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{model_name}: {acc*100:.2f}%")

if __name__ == "__main__":
    main()
