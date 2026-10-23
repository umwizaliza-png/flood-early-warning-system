
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# 1. Page setup and model loading
# --------------------------------------------------
st.set_page_config(
    page_title="FloodGuard AI | Flood Risk Intelligence",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
model = joblib.load(BASE_DIR / "flood_logistic_regression.joblib")

with open(BASE_DIR / "model_metadata.json", "r", encoding="utf-8") as file:
    metadata = json.load(file)

# --------------------------------------------------
# 2. Professional dashboard styling
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: #0B1220;
        color: #F1F5F9;
    }

    [data-testid="stSidebar"] {
        background: #101C2D;
        border-right: 1px solid #26364C;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label {
        color: #F1F5F9 !important;
    }

    .hero {
        background: linear-gradient(120deg, #17365C, #20577A);
        padding: 34px 38px;
        border: 1px solid #32769A;
        border-radius: 22px;
        margin-bottom: 24px;
    }

    .eyebrow {
        color: #F5C76A;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 15px;
    }

    .hero h1 {
        color: #FFFFFF;
        font-size: clamp(28px, 4vw, 42px);
        font-weight: 800;
        margin: 0 0 12px 0;
        line-height: 1.2;
    }

    .hero p {
        color: #D5E8F5;
        font-size: 15px;
        line-height: 1.8;
        margin-bottom: 0;
    }

    .section-heading {
        color: #F5C76A;
        font-size: 20px;
        font-weight: 800;
        margin: 28px 0 16px 0;
    }

    .risk-card {
        background: #14243A;
        border: 1px solid #304965;
        border-left: 6px solid #65BCE8;
        border-radius: 18px;
        padding: 26px;
        margin: 8px 0 18px 0;
    }

    .risk-high {
        border-left-color: #FF7373;
        background: #301F2B;
    }

    .risk-low {
        border-left-color: #58D6A0;
        background: #142D2B;
    }

    .risk-label {
        color: #A9C6E0;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 1.4px;
        margin-bottom: 12px;
    }

    .risk-card h2 {
        color: #FFFFFF;
        font-size: 27px;
        margin: 0 0 12px 0;
    }

    .risk-card p {
        color: #D4E0ED;
        line-height: 1.7;
        margin-bottom: 0;
    }

    .metric-card {
        background: #121F32;
        border: 1px solid #293D55;
        border-radius: 15px;
        padding: 20px;
        min-height: 125px;
    }

    .metric-label {
        color: #AFC4DA;
        font-size: 13px;
        margin-bottom: 12px;
    }

    .metric-value {
        color: #FFFFFF;
        font-size: 27px;
        font-weight: 800;
    }

    .metric-note {
        color: #8FA9C4;
        font-size: 12px;
        margin-top: 7px;
    }

    .info-card {
        background: #121F32;
        border: 1px solid #293D55;
        border-radius: 15px;
        padding: 22px;
        height: 100%;
    }

    .info-card h3 {
        color: #F5C76A;
        font-size: 17px;
        margin-top: 0;
    }

    .info-card p {
        color: #CBD8E6;
        font-size: 14px;
        line-height: 1.8;
    }

    .prototype-note {
        background: #302B1D;
        border: 1px solid #796438;
        border-radius: 12px;
        padding: 16px 18px;
        color: #F6DE9F;
        line-height: 1.7;
        font-size: 13px;
        margin-top: 24px;
    }

    div.stButton > button,
    div.stFormSubmitButton > button {
        background: #F5C76A;
        color: #102039;
        border: none;
        border-radius: 10px;
        font-weight: 800;
        min-height: 44px;
        width: 100%;
    }

    div.stButton > button:hover,
    div.stFormSubmitButton > button:hover {
        background: #FFDA8B;
        color: #102039;
        border: none;
    }

    [data-testid="stMetric"] {
        background: #121F32;
        border: 1px solid #293D55;
        border-radius: 12px;
        padding: 16px;
    }

    [data-testid="stMetricLabel"] {
        color: #AFC4DA !important;
    }

    [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
    }

    hr {
        border-color: #293D55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 3. Sidebar control centre
# --------------------------------------------------
st.sidebar.markdown(
    """
    <div style="padding: 10px 0 22px 0;">
        <div style="font-size: 27px;">🌧️</div>
        <h2 style="margin: 6px 0; color: #FFFFFF;">
            Control Centre
        </h2>
        <p style="color: #AFC4DA; font-size: 13px; line-height: 1.7;">
            Adjust the environmental measurements and assess
            the model's flood-risk classification.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar.form("flood_prediction_form"):
    st.markdown("### Environmental inputs")

    water_level = st.slider(
        "Water level (m)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        help="Water level measurement in metres.",
    )

    rainfall = st.slider(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=10.0,
        step=1.0,
        help="Rainfall measurement in millimetres.",
    )

    soil_moisture = st.slider(
        "Soil moisture (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0,
        help="Soil moisture percentage.",
    )

    submitted = st.form_submit_button("Run Flood Risk Prediction")

if "flood_inputs" not in st.session_state:
    st.session_state.flood_inputs = {
        "WaterLevel_m": 1.0,
        "Rainfall_mm": 10.0,
        "SoilMoisture_pct": 20.0,
    }

if submitted:
    st.session_state.flood_inputs = {
        "WaterLevel_m": water_level,
        "Rainfall_mm": rainfall,
        "SoilMoisture_pct": soil_moisture,
    }

# --------------------------------------------------
# 4. Prepare input and run the trained model
# --------------------------------------------------
input_data = pd.DataFrame([st.session_state.flood_inputs])

prediction = int(model.predict(input_data)[0])
probability = float(model.predict_proba(input_data)[0][1])

current_inputs = st.session_state.flood_inputs

# --------------------------------------------------
# 5. Dashboard banner
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">
            Smart monitoring · Flood risk intelligence
        </div>
        <h1>🌧️ FloodGuard AI</h1>
        <p>
            An intelligent flood early-warning prototype that
            estimates flood-risk categories using rainfall,
            water level, and soil moisture measurements.
            Explore environmental conditions and review the
            machine-learning model's recommendation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-heading">01 / Flood risk assessment</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------
# 6. Main prediction recommendation
# --------------------------------------------------
if prediction == 1:
    risk_title = "HIGH FLOOD RISK"
    risk_class = "risk-high"
    risk_icon = "⚠️"
    risk_description = (
        "The trained model classifies these measurements "
        "as high flood risk. Review the inputs and seek "
        "independent local information before making decisions."
    )
else:
    risk_title = "LOW FLOOD RISK"
    risk_class = "risk-low"
    risk_icon = "✓"
    risk_description = (
        "The trained model classifies these measurements "
        "as low flood risk. This result does not guarantee "
        "that flooding will not occur."
    )

st.markdown(
    f"""
    <div class="risk-card {risk_class}">
        <div class="risk-label">MODEL RECOMMENDATION</div>
        <h2>{risk_icon} {risk_title}</h2>
        <p>{risk_description}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Showing the model's prediction for the selected inputs. "
    "Use the sidebar controls and select Run Flood Risk Prediction "
    "to evaluate another scenario."
)

# --------------------------------------------------
# 7. Environmental metric cards
# --------------------------------------------------
st.markdown(
    '<div class="section-heading">02 / Environmental measurements</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">🌊 Water level</div>
            <div class="metric-value">
                {current_inputs["WaterLevel_m"]:.2f} m
            </div>
            <div class="metric-note">Measured water height</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">🌧️ Rainfall</div>
            <div class="metric-value">
                {current_inputs["Rainfall_mm"]:.0f} mm
            </div>
            <div class="metric-note">Rainfall measurement</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">🌱 Soil moisture</div>
            <div class="metric-value">
                {current_inputs["SoilMoisture_pct"]:.0f}%
            </div>
            <div class="metric-note">Moisture percentage</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">📊 High-risk estimate</div>
            <div class="metric-value">{probability:.1%}</div>
            <div class="metric-note">Model-estimated probability</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# 8. Model information
# --------------------------------------------------
st.markdown(
    '<div class="section-heading">03 / About the model</div>',
    unsafe_allow_html=True,
)

info1, info2 = st.columns(2)

with info1:
    st.markdown(
        """
        <div class="info-card">
            <h3>🧠 Machine-learning approach</h3>
            <p>
                This prototype uses Logistic Regression to
                classify the supplied environmental measurements
                into low-risk or high-risk categories.
            </p>
            <p>
                Inputs: water level, rainfall, and soil moisture.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with info2:
    st.markdown(
        """
        <div class="info-card">
            <h3>🗂️ Dataset and validation</h3>
            <p>
                The model was trained using historical secondary
                data from Metro Manila, Philippines.
            </p>
            <p>
                The model has not been validated for Rwanda or
                for operational flood-warning decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# --------------------------------------------------
# 9. Responsible-use notice and footer
# --------------------------------------------------
st.markdown(
    """
    <div class="prototype-note">
        <strong>⚠️ Academic prototype only</strong><br>
        The model-estimated probability is not a calibrated
        real-world probability of a flood occurring. Predictions
        must not be used as the sole basis for emergency decisions.
        Consult official local warnings and emergency services.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#8FA9C4;
                font-size:12px; padding:8px 0 20px 0;">
        FLOODGUARD AI · MACHINE LEARNING · ACADEMIC PROTOTYPE
    </div>
    """,
    unsafe_allow_html=True,
)
