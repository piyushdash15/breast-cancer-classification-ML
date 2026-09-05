import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("breast_cancer_model.pkl")

st.title("Breast Cancer Classification")
st.write("Enter the tumor measurements below.")

radius_mean = st.number_input("Radius Mean", min_value=0.0)
texture_mean = st.number_input("Texture Mean", min_value=0.0)
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0)
area_mean = st.number_input("Area Mean", min_value=0.0)

# Add the remaining 26 features here

if st.button("Predict"):

    input_data = pd.DataFrame([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        # remaining 26 features
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error("Prediction: Malignant")
    else:
        st.success("Prediction: Benign")

    st.write(f"Malignant probability: {probability:.2%}")
