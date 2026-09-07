# -*- coding: utf-8 -*-
"""
Breast Cancer Diagnostic AI System - Single Page Streamlit Web Application
Architecture: Scikit-learn Pipeline (StandardScaler + LogisticRegression)
Dataset: Wisconsin Diagnostic Breast Cancer (WDBC)
Mode: Streamlined 10-Parameter Input with Automated Background Feature Synthesis
"""


import time
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import joblib
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Breast Cancer AI Diagnostic System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# High-Contrast Clinical Design & CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
        line-height: 1.2;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .disclaimer-box {
        background-color: #FEF3C7;
        border-left: 5px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 6px;
        color: #92400E;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .card-benign {
        background-color: #ECFDF5 !important;
        border: 2px solid #10B981 !important;
        border-radius: 10px;
        padding: 24px;
        text-align: center;
        color: #065F46 !important;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .card-benign * {
        color: #065F46 !important;
    }
    .card-malignant {
        background-color: #FEF2F2 !important;
        border: 2px solid #EF4444 !important;
        border-radius: 10px;
        padding: 24px;
        text-align: center;
        color: #991B1B !important;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
    .card-malignant * {
        color: #991B1B !important;
    }
    .pred-title {
        font-size: 1.9rem;
        font-weight: 800;
        margin-bottom: 6px;
    }
    .confidence-text {
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 8px;
    }
    .preset-banner {
        background-color: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 1.2rem;
        color: #1E40AF;
        font-size: 0.95rem;
    }
    .auto-badge {
        background-color: #E0E7FF;
        color: #3730A3;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        display: inline-block;
        margin-left: 8px;
    }
    .feature-card-header {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 14px;
        font-weight: 700;
        color: #1E3A8A;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Exact Feature Names & Group Definitions
# ---------------------------------------------------------
FEATURE_NAMES = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean',
    'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# The 10 features exposed to the user
MEAN_FEATURES = FEATURE_NAMES[0:10]

# The 20 features calculated automatically behind the scenes
SE_FEATURES = FEATURE_NAMES[10:20]
WORST_FEATURES = FEATURE_NAMES[20:30]
AUTO_CALCULATED_FEATURES = SE_FEATURES + WORST_FEATURES

# Human-readable labels for the 10 Mean features
MEAN_LABELS = {
    'radius_mean': 'Radius Mean (mm)',
    'texture_mean': 'Texture Mean (Gray-scale SD)',
    'perimeter_mean': 'Perimeter Mean (mm)',
    'area_mean': 'Area Mean (sq mm)',
    'smoothness_mean': 'Smoothness Mean (Local Variation)',
    'compactness_mean': 'Compactness Mean (Perimeter² / Area - 1.0)',
    'concavity_mean': 'Concavity Mean (Severity of Concave Portions)',
    'concave points_mean': 'Concave Points Mean (Number of Points)',
    'symmetry_mean': 'Symmetry Mean',
    'fractal_dimension_mean': 'Fractal Dimension Mean (Coastline Approx)'
}

# ---------------------------------------------------------
# Preloaded Benchmark Cases (10 Mean Parameters)
# ---------------------------------------------------------
BENIGN_PRESET = {
    'radius_mean': 13.54,
    'texture_mean': 14.36,
    'perimeter_mean': 87.46,
    'area_mean': 566.3,
    'smoothness_mean': 0.09779,
    'compactness_mean': 0.08129,
    'concavity_mean': 0.06664,
    'concave points_mean': 0.04781,
    'symmetry_mean': 0.1885,
    'fractal_dimension_mean': 0.05766
}

MALIGNANT_PRESET = {
    'radius_mean': 17.99,
    'texture_mean': 10.38,
    'perimeter_mean': 122.8,
    'area_mean': 1001.0,
    'smoothness_mean': 0.1184,
    'compactness_mean': 0.2776,
    'concavity_mean': 0.3001,
    'concave points_mean': 0.1471,
    'symmetry_mean': 0.2419,
    'fractal_dimension_mean': 0.07871
}

DATASET_AVERAGE_PRESET = {
    'radius_mean': 14.127,
    'texture_mean': 19.290,
    'perimeter_mean': 91.969,
    'area_mean': 654.889,
    'smoothness_mean': 0.09636,
    'compactness_mean': 0.10434,
    'concavity_mean': 0.08880,
    'concave points_mean': 0.04892,
    'symmetry_mean': 0.18116,
    'fractal_dimension_mean': 0.06280
}

PRESETS = {
    "🟢 Benign Sample Preset": BENIGN_PRESET,
    "🔴 Malignant Sample Preset": MALIGNANT_PRESET,
    "📊 Dataset Average": DATASET_AVERAGE_PRESET
}

# ---------------------------------------------------------
# Cached Model & Statistical Synthesizer Loading
# ---------------------------------------------------------
MODEL_FILENAME = "breast_cancer_model.pkl"
DATA_FILENAME = "data.csv"

@st.cache_resource(show_spinner="Loading machine learning pipeline...")
def load_trained_model():
    """Load and cache the trained Scikit-learn Logistic Regression Pipeline."""
    current_dir = Path(__file__).resolve().parent
    potential_paths = [
        current_dir / MODEL_FILENAME,
        Path.cwd() / MODEL_FILENAME,
        Path(MODEL_FILENAME)
    ]
    for p in potential_paths:
        if p.exists():
            return joblib.load(p), p
    return None, None

@st.cache_resource(show_spinner="Fitting background feature synthesizer on WDBC dataset...")
def load_feature_synthesizer():
    """
    Fits a multivariate statistical estimator using 10 Mean Features
    """
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir / DATA_FILENAME if (current_dir / DATA_FILENAME).exists() else Path(DATA_FILENAME)

    if not data_path.exists():
        return None

    df = pd.read_csv(data_path)
    # Fit multivariate linear regression from 10 Mean -> 20 Other features
    synthesizer = LinearRegression()
    synthesizer.fit(df[MEAN_FEATURES], df[AUTO_CALCULATED_FEATURES])
    return synthesizer

pipeline, model_resolved_path = load_trained_model()
feature_synthesizer = load_feature_synthesizer()
img_path = (Path(__file__).with_name('img.png')).absolute()

# ---------------------------------------------------------
# Sidebar: System Metadata & Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image(img_path, width=200)
    st.title("Model Based on")
    st.markdown("**Wisconsin Breast Cancer Dataset**")
    st.divider()
    st.markdown("### Features")
    st.markdown("""
    - **Required Inputs:** **10 Mean Features**
    - **Auto-Calculated:** **20 SE & Worst Features from Dataset**
    - **Classifier:** `StandardScaler` + `LogisticRegression`
    """)

    st.divider()
    st.markdown("### Engine Status")
    if pipeline is not None:
        st.success(f"Loaded: `{MODEL_FILENAME}`")
    else:
        st.error(f"Missing: `{MODEL_FILENAME}`")

    if feature_synthesizer is not None:
        st.success("Feature Synthesizer: `Active`")
    else:
        st.warning("Feature Synthesizer: `data.csv missing`")

    st.divider()

# ---------------------------------------------------------
# Main Header & Prominent Medical Disclaimer
# ---------------------------------------------------------
st.markdown('<div class="main-header">Breast Cancer AI Diagnostic System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Streamlined Clinical Classification via 10 Mean Cytological Features ',
    unsafe_allow_html=True
)

st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>Medical Disclaimer:</strong> This web application is an AI/ML educational and research prototype trained on the 
    Wisconsin Diagnostic Breast Cancer (WDBC) dataset.Predictions generated by this tool must <strong>NEVER</strong> be used as a standalone diagnostic determination or substitute for formal 
    clinical histopathology, radiological evaluation, or consultation with licensed healthcare professionals.
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Guardrail: Check for Model Existence
# ---------------------------------------------------------
if pipeline is None:
    st.error(f"""
    ❌ **Trained Model Artifact Not Found: `{MODEL_FILENAME}`**

    The inference engine requires the trained model file `{MODEL_FILENAME}` in the application directory.

    **Instructions to generate the model:**
    1. Verify that `data.csv` is present in your project directory.
    2. Run the training script in your shell:
       ```bash
       python breast_cancer_detection_using_ml.py
       ```
    3. Refresh this page once `breast_cancer_model.pkl` is saved.
    """)
    st.stop()

# ---------------------------------------------------------
# Benchmark Presets & Session State Initialization
# ---------------------------------------------------------
st.markdown("### 1. Benchmark Case Presets")

def on_preset_change():
    """Callback to sync session state with the chosen preset."""
    selected = st.session_state.selected_preset_choice
    preset_dict = PRESETS[selected]
    for key, val in preset_dict.items():
        st.session_state[f"input_{key}"] = float(val)

# Initialize session state with default preset if not present
if "selected_preset_choice" not in st.session_state:
    st.session_state.selected_preset_choice = "🟢 Benign Sample Preset"
    on_preset_change()

col_preset, col_info = st.columns([2, 3])
with col_preset:
    st.selectbox(
        "Select Benchmark Profile:",
        options=list(PRESETS.keys()),
        key="selected_preset_choice",
        on_change=on_preset_change,
        help="Quickly populate the 10 Mean cytological measurements with default test cases."
    )

with col_info:
    st.markdown("""
    <div class="preset-banner">
        💡 Enter or adjust 10 Mean cytological features below. 
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Input Features Section (10 Mean Features Only)
# ---------------------------------------------------------
st.markdown("""
<div class="feature-card-header">
    Mean Cytological Features
</div>
""", unsafe_allow_html=True)
st.caption("Average values computed across all cell nuclei observed in the biopsy fine needle aspirate (FNA).")

user_inputs_mean = {}
cols_mean = st.columns(3)

for idx, feat in enumerate(MEAN_FEATURES):
    target_col = cols_mean[idx % 3]
    default_val = st.session_state.get(f"input_{feat}", float(BENIGN_PRESET[feat]))
    label = MEAN_LABELS.get(feat, feat)

    # Context-aware step and decimal formatting
    step_val = 0.01 if default_val >= 1.0 else 0.001
    fmt = "%.2f" if default_val >= 10.0 else ("%.4f" if default_val >= 1.0 else "%.5f")

    user_inputs_mean[feat] = target_col.number_input(
        label,
        value=float(default_val),
        step=step_val,
        format=fmt,
        key=f"input_{feat}"
    )

st.divider()

# ---------------------------------------------------------
# Diagnostic Inference Execution
# ---------------------------------------------------------
submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
with submit_col2:
    predict_clicked = st.button(
        "🖥️ Run Diagnostic Prediction",
        type="secondary",
        use_container_width=True
    )

# Run prediction when user clicks or on initial page load
if predict_clicked or "has_run" not in st.session_state:
    st.session_state.has_run = True

    # 1. Construct 10-feature DataFrame from user inputs
    input_mean_df = pd.DataFrame(
        [{feat: float(user_inputs_mean[feat]) for feat in MEAN_FEATURES}],
        columns=MEAN_FEATURES
    )

    # 2. Automatically calculate remaining 20 features behind the scenes
    if feature_synthesizer is not None:
        predicted_other = feature_synthesizer.predict(input_mean_df)
        calculated_other_df = pd.DataFrame(predicted_other, columns=AUTO_CALCULATED_FEATURES)
    else:
        # Fallback: fill with dataset mean defaults if synthesizer is unavailable
        calculated_other_df = pd.DataFrame(
            [[DATASET_AVERAGE_PRESET.get(col, 0.0) for col in AUTO_CALCULATED_FEATURES]],
            columns=AUTO_CALCULATED_FEATURES
        )

    # 3. Assemble full 30-feature vector in exact column sequence required by model
    full_30_df = pd.concat([input_mean_df, calculated_other_df], axis=1)[FEATURE_NAMES]

    # 4. Execute inference and time latency
    start_t = time.perf_counter()
    prediction = int(pipeline.predict(full_30_df)[0])
    probabilities = pipeline.predict_proba(full_30_df)[0]
    latency_ms = (time.perf_counter() - start_t) * 1000.0

    # 5. Extract probabilities (Index 0 = Benign [B], Index 1 = Malignant [M])
    benign_prob = float(probabilities[0])
    malignant_prob = float(probabilities[1])

    st.markdown("### 2. Diagnostic Inference Results")

    # 6. Display high-contrast custom HTML result card
    if prediction == 0:
        confidence = benign_prob * 100.0
        st.markdown(f"""
        <div class="card-benign">
            <div style="font-size: 3rem; line-height: 1;">🟢</div>
            <div class="pred-title">DIAGNOSTIC PREDICTION: BENIGN</div>
            <div class="confidence-text">Model Confidence: <strong>{confidence:.2f}%</strong></div>
            <div style="margin-top: 10px; font-size: 1.05rem; font-weight: 500;">
                Cellular morphometric features indicate low risk / non-malignant tissue characteristics.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        confidence = malignant_prob * 100.0
        st.markdown(f"""
        <div class="card-malignant">
            <div style="font-size: 3rem; line-height: 1;">🔴</div>
            <div class="pred-title">DIAGNOSTIC PREDICTION: MALIGNANT</div>
            <div class="confidence-text">Model Confidence: <strong>{confidence:.2f}%</strong></div>
            <div style="margin-top: 10px; font-size: 1.05rem; font-weight: 500;">
                Severe cellular atypia and morphometric characteristics strongly indicate malignant neoplasm.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 7. Probability distribution bar & metrics
    st.markdown("#### Probability Distribution & Risk Index")
    st.progress(
        malignant_prob,
        text=f"Malignancy Risk Index: {malignant_prob * 100.0:.2f}% (Benign Probability: {benign_prob * 100.0:.2f}%)"
    )

    m1, m2, m3 = st.columns(3)
    m1.metric(
        label="Malignant Probability",
        value=f"{malignant_prob * 100.0:.2f}%",
        delta_color="inverse"
    )
    m2.metric(
        label="Benign Probability",
        value=f"{benign_prob * 100.0:.2f}%"
    )
    m3.metric(
        label="Inference Latency",
        value=f"{latency_ms:.2f} ms"
    )

    # 8. Full Transparency: Behind the Scenes Auto-Calculated Features
    with st.expander("View 20 Auto-Calculated SE & Worst Features"):
        st.markdown("""
        **Streamlit calculates the remaining 20 Standard Error (heterogeneity) and Worst parameters from dataset automatically:
        """)
        
        tab_calc1, tab_calc2 = st.tabs(["Standard Error Features (10)", "Worst (Extreme) Features (10)"])
        with tab_calc1:
            st.dataframe(calculated_other_df[SE_FEATURES].style.format("{:.6f}"), use_container_width=True)
        with tab_calc2:
            st.dataframe(calculated_other_df[WORST_FEATURES].style.format("{:.4f}"), use_container_width=True)

        st.caption("Complete 30-feature vector passed into `pipeline.predict()`:")
        st.dataframe(full_30_df.style.format("{:.4f}"), use_container_width=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.caption(
    "Breast Cancer Diagnostic AI "
)
