
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / "flood_logistic_regression.joblib")

with open(BASE_DIR / "model_metadata.json", "r") as f:
    metadata = json.load(f)

st.set_page_config(
    page_title="Flood Early-Warning System",
    page_icon="🌧️",
    layout="centered"
)

st.title("🌧️ Flood Early-Warning System")
st.write(
    "Enter the environmental measurements below to estimate "
    "the flood-risk category."
)

st.warning(
    "Academic prototype only. This model uses historical data "
    "from Metro Manila, Philippines. It has not been validated "
    "for Rwanda or for real-world emergency warnings."
)

with st.form("flood_prediction_form"):
    water_level = st.number_input(
        "Water level (m)",
        min_value=0.0,
        max_value=100.0,
        value=1.0,
        step=0.1
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=1000.0,
        value=10.0,
        step=0.5
    )

    soil_moisture = st.number_input(
        "Soil moisture (%)",
        min_value=0.0,
        max_value=100.0,
        value=20.0,
        step=1.0
    )

    submitted = st.form_submit_button("Predict Flood Risk")

if submitted:
    input_data = pd.DataFrame([{
        "WaterLevel_m": water_level,
        "Rainfall_mm": rainfall,
        "SoilMoisture_pct": soil_moisture
    }])

    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    st.subheader("Prediction Result")

    st.metric(
        "Estimated High Flood probability",
        f"{probability:.1%}"
    )

    if prediction == 1:
        st.error("Predicted category: HIGH FLOOD RISK")
    else:
        st.success("Predicted category: LOW FLOOD RISK")

    st.caption(
        "This probability is a model estimate, not a calibrated "
        "real-world probability of a flood occurring."
    )
