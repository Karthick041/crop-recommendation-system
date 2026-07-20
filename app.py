import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# 1. Load Data Safely
def load_data(filename):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Missing {filename}! Place it in the same folder.")
    print("✨ Reading dataset...")
    return pd.read_csv(filename)

# 2. Train and Save Model
def train_pipeline(df):
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("🧠 Training the Random Forest AI model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"✅ Model Trained! Evaluation Accuracy: {acc * 100:.2f}%")
    
    # Save the brains for later deployment
    joblib.dump(model, 'crop_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("💾 Saved 'crop_model.pkl' and 'scaler.pkl' to your project folder.\n")

# 3. Make Recommendations
def get_recommendation(n, p, k, temp, humid, ph, rain):
    model = joblib.load('crop_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    features = np.array([[n, p, k, temp, humid, ph, rain]])
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    classes = model.classes_
    
    print("=" * 45)
    print(f"🌱 RECOMMENDED CROP: {prediction.upper()}")
    print("=" * 45)
    
    # Show alternative choices
    top_indices = np.argsort(probabilities)[-3:][::-1]
    print("Top alternative possibilities for your soil:")
    for idx in top_indices:
        print(f" - {classes[idx]}: {probabilities[idx]*100:.1f}% match")

if __name__ == "__main__":
    # Run the automated pipeline
    try:
        data = load_data('Crop_recommendation.csv')
        train_pipeline(data)
        
        # Test Case Example
        print("Testing system with real-time custom soil data inputs...")
        get_recommendation(n=60, p=20, k=50, temp=5, humid=82.1, ph=7.5, rain=50.4)
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")