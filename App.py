%%writefile app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Delivery Delay Predictor",
    page_icon="🚚",
    layout="centered"
)

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('logi.sav')

model = load_model()

# App header
st.title("🚚 Delivery Delay Predictor")
st.write("Enter the shipment details below to predict the likelihood of a delivery delay.")

# User Inputs
st.header("Shipment Details")

col1, col2 = st.columns(2)

with col1:
    delivery_distance = st.number_input("Delivery Distance (miles)", min_value=0.0, max_value=500.0, value=25.0, step=1.0)
    traffic_congestion = st.slider("Traffic Congestion Level (1-5)", min_value=1, max_value=5, value=3)
    weather_condition = st.slider("Weather Condition Score (1-5)", min_value=1, max_value=5, value=2)
    delivery_slot = st.slider("Delivery Slot (1-4)", min_value=1, max_value=4, value=2)
    driver_experience = st.number_input("Driver Experience (Years)", min_value=0, max_value=50, value=5)
    num_stops = st.number_input("Number of Stops", min_value=0, max_value=20, value=3)

with col2:
    vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0, max_value=30, value=4)
    road_condition_score = st.slider("Road Condition Score (1-5)", min_value=1, max_value=5, value=3)
    package_weight = st.number_input("Package Weight (lbs)", min_value=0.0, max_value=500.0, value=15.0, step=0.5)
    fuel_efficiency = st.number_input("Fuel Efficiency (mpg)", min_value=1.0, max_value=100.0, value=15.0, step=0.5)
    warehouse_processing_time = st.number_input("Warehouse Processing Time (mins)", min_value=0, max_value=1440, value=45)

# Predict Button
if st.button("Predict Delay", type="primary"):
    # Create feature array with correct names as used during model training
    input_data = pd.DataFrame([{
        'Delivery_Distance': delivery_distance,
        'Traffic_Congestion': traffic_congestion,
        'Weather_Condition': weather_condition,
        'Delivery_Slot': delivery_slot,
        'Driver_Experience': driver_experience,
        'Num_Stops': num_stops,
        'Vehicle_Age': vehicle_age,
        'Road_Condition_Score': road_condition_score,
        'Package_Weight': package_weight,
        'Fuel_Efficiency': fuel_efficiency,
        'Warehouse_Processing_Time': warehouse_processing_time
    }])

    # Generate prediction and probabilities
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    st.subheader("Prediction Results")
    
    if prediction == 1:
        st.error(f"🚨 **Predicted: Delivery Delay** (Probability: {probabilities[1]:.2%})")
    else:
        st.success(f"✅ **Predicted: On Time** (Probability of no delay: {probabilities[0]:.2%})")
