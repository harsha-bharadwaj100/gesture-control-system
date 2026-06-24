import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training KNN...")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"KNN Accuracy: {acc:.4f}")
    
    with open(os.path.join(RESULTS_DIR, "knn_result.json"), "w") as f:
        json.dump({"Model": "KNN", "Accuracy": acc}, f)

if __name__ == "__main__":
    main()
