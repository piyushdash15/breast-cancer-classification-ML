import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline
model = joblib.load("breast_cancer_model.pkl")

# Page title
st.title("🔬 Breast Cancer Classification")
st.write("Enter the tumor measurements below to get a prediction.")

st.warning(
    "This application is for educational purposes only and is not a medical diagnostic tool."
)

# -------------------------------------------------
# Mean Features
# -------------------------------------------------

st.header("Mean Features")

radius_mean = st.number_input("Radius Mean", min_value=0.0)
texture_mean = st.number_input("Texture Mean", min_value=0.0)
perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0)
area_mean = st.number_input("Area Mean", min_value=0.0)
smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0)
compactness_mean = st.number_input("Compactness Mean", min_value=0.0)
concavity_mean = st.number_input("Concavity Mean", min_value=0.0)
concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0)
symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0)
fractal_dimension_mean = st.number_input("Fractal Dimension Mean", min_value=0.0)

# -------------------------------------------------
# Standard Error Features
# -------------------------------------------------

st.header("Standard Error Features")

radius_se = st.number_input("Radius SE", min_value=0.0)
texture_se = st.number_input("Texture SE", min_value=0.0)
perimeter_se = st.number_input("Perimeter SE", min_value=0.0)
area_se = st.number_input("Area SE", min_value=0.0)
smoothness_se = st.number_input("Smoothness SE", min_value=0.0)
compactness_se = st.number_input("Compactness SE", min_value=0.0)
concavity_se = st.number_input("Concavity SE", min_value=0.0)
concave_points_se = st.number_input("Concave Points SE", min_value=0.0)
symmetry_se = st.number_input("Symmetry SE", min_value=0.0)
fractal_dimension_se = st.number_input("Fractal Dimension SE", min_value=0.0)

# -------------------------------------------------
# Worst Features
# -------------------------------------------------

st.header("Worst Features")

radius_worst = st.number_input("Radius Worst", min_value=0.0)
texture_worst = st.number_input("Texture Worst", min_value=0.0)
perimeter_worst = st.number_input("Perimeter Worst", min_value=0.0)
area_worst = st.number_input("Area Worst", min_value=0.0)
smoothness_worst = st.number_input("Smoothness Worst", min_value=0.0)
compactness_worst = st.number_input("Compactness Worst", min_value=0.0)
concavity_worst = st.number_input("Concavity Worst", min_value=0.0)
concave_points_worst = st.number_input("Concave Points Worst", min_value=0.0)
symmetry_worst = st.number_input("Symmetry Worst", min_value=0.0)
fractal_dimension_worst = st.number_input(
    "Fractal Dimension Worst", min_value=0.0
)

# -------------------------------------------------
# Prediction
# -------------------------------------------------

if st.button("🔍 Predict"):

    # Create input DataFrame with EXACTLY 30 features
    input_data = pd.DataFrame([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        fractal_dimension_mean,

        radius_se,
        texture_se,
        perimeter_se,
        area_se,
        smoothness_se,
        compactness_se,
        concavity_se,
        concave_points_se,
        symmetry_se,
        fractal_dimension_se,

        radius_worst,
        texture_worst,
        perimeter_worst,
        area_worst,
        smoothness_worst,
        compactness_worst,
        concavity_worst,
        concave_points_worst,
        symmetry_worst,
        fractal_dimension_worst
    ]], columns=[
        "radius_mean",
        "texture_mean",
        "perimeter_mean",
        "area_mean",
        "smoothness_mean",
        "compactness_mean",
        "concavity_mean",
        "concave points_mean",
        "symmetry_mean",
        "fractal_dimension_mean",

        "radius_se",
        "texture_se",
        "perimeter_se",
        "area_se",
        "smoothness_se",
        "compactness_se",
        "concavity_se",
        "concave points_se",
        "symmetry_se",
        "fractal_dimension_se",

        "radius_worst",
        "texture_worst",
        "perimeter_worst",
        "area_worst",
        "smoothness_worst",
        "compactness_worst",
        "concavity_worst",
        "concave points_worst",
        "symmetry_worst",
        "fractal_dimension_worst"
    ])

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display result
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Prediction: Malignant")
    else:
        st.success("✅ Prediction: Benign")

    st.write(
        f"Malignant Probability: **{probability:.2%}**"
    )
