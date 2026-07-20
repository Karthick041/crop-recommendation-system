import streamlit as st
import numpy as np
import joblib

# Set page title and layout
st.set_page_config(page_title="Crop Recommendation System", page_icon="🌱", layout="centered")

# App Header
st.title("🌱 Smart Crop Recommendation System")
st.write("Enter your soil chemical composition and weather parameters to find the best crop for your field.")

# Load Saved AI Model and Scaler safely
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('crop_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error("Error: Could not load model files. Make sure 'crop_model.pkl' and 'scaler.pkl' are in the same directory.")
        return None, None

model, scaler = load_assets()

if model and scaler:
    # Organize input fields into columns for a clean UI
    st.subheader("📊 Soil & Climate Inputs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90, help="Nitrogen ratio in soil")
        p = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=42, help="Phosphorus ratio in soil")
        k = st.number_input("Potassium (K)", min_value=0, max_value=200, value=43, help="Potassium ratio in soil")
        ph = st.number_input("Soil pH Level", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

    with col2:
        temp = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=23.5, step=0.1)
        humid = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.1)
        rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=202.4, step=0.1)

    # Button to predict
    if st.button("🌾 Get Recommendation", use_container_width=True):
        # Format input data
        inputs = np.array([[n, p, k, temp, humid, ph, rain]])
        
        # Scale inputs using saved scaler
        inputs_scaled = scaler.transform(inputs)
        
        # Make predictions
        prediction = model.predict(inputs_scaled)[0]
        probabilities = model.predict_proba(inputs_scaled)[0]
        classes = model.classes_
        
        # Display Result
        st.success(f"### 🎉 Recommended Crop: **{prediction.upper()}**")
        
        # Show top 3 possibilities in a clean expander block
        st.subheader("Top Alternative Matches")
        top_indices = np.argsort(probabilities)[-3:][::-1]
        
        for idx in top_indices:
            crop_name = classes[idx].capitalize()
            confidence = probabilities[idx] * 100
            st.write(f"• **{crop_name}**: {confidence:.1f}% match")
            st.progress(float(probabilities[idx]))
            