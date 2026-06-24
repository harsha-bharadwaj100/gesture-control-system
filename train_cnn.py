import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D

FEATURES_FILE = "extracted_features.csv"
RESULTS_DIR = "Results"

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    if not os.path.exists(FEATURES_FILE):
        print(f"Error: {FEATURES_FILE} not found. Run extract_features.py first.")
        return
        
    df = pd.read_csv(FEATURES_FILE)
    X = df[['RMS', 'MAV', 'WL', 'ZC']].values
    y = df['Gesture_Label'].values
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

    print("Training CNN...")
    model = Sequential([
        Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_train_cnn.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train_cnn, y_train_cat, epochs=15, batch_size=64, validation_split=0.2, verbose=1)
    
    loss, acc = model.evaluate(X_test_cnn, y_test_cat, verbose=0)
    print(f"CNN Accuracy: {acc:.4f}")
    
    with open(os.path.join(RESULTS_DIR, "cnn_result.json"), "w") as f:
        json.dump({"Model": "CNN", "Accuracy": acc}, f)

if __name__ == "__main__":
    main()
