import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def load_model():
    return joblib.load("rf_model.pkl")

def load_scaler():
    return joblib.load("scaler.pkl")

def main():
    st.title("Medical Condition Predictor")
    st.write("Enter the required parameters to predict the medical condition.")
    
    # Load model and scaler
    model = load_model()
    scaler = load_scaler()
    
    # Define input fields
    columns = ["Heart_Rate", "Step_Count", "Sleep_Hours", "BMI", "Blood_Pressure",
               "Cholesterol_Level", "Daily_Calories", "Hydration_Level", "Activity_Level",
               "Stress_Level", "Diet_Score"]
    
    input_data = {}
    for col in columns:
        input_data[col] = st.number_input(f"Enter {col}", value=0.0, format="%.2f")
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Apply scaling
    input_scaled = scaler.transform(input_df)
    
    # Predict button
    if st.button("Predict"):
        prediction = model.predict(input_scaled)
        st.success(f"Predicted Medical Condition: {prediction[0]}")

if __name__ == "__main__":
    main()
