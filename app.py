
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="CPE Risk Predictor", layout="wide", page_icon="🧪")

st.title("CPE (Carbapenemase-producing Enterobacterales) Risk Predictor")

# Colab / Notebook 환경에서는 __file__이 없으므로 직접 지정
MODEL_PATH = Path("cpe_model.pkl")
if not MODEL_PATH.exists():
    st.error("❌ Model file 'cpe_model.pkl' not found in current directory.")
    st.stop()

try:
    with open(MODEL_PATH, "rb") as f:
        model_blob = pickle.load(f)
except Exception as e:
    st.error(f"❌ Error loading model file: {e}")
    st.stop()

model = model_blob.get("model")
model_features = model_blob.get("features", [])
threshold = float(model_blob.get("threshold", 0.45))

if model is None or not model_features:
    st.error("❌ Model object or feature list is missing in the pickle file.")
    st.stop()

# User input UI dynamically generated from model_features
st.header("Input Features")
user_input = []
for feature in model_features:
    if "days" in feature.lower():
        val = st.number_input(feature, min_value=0, value=0)
    else:
        val = st.selectbox(feature, [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    user_input.append(val)

# Convert to numpy array
input_vector = np.array([user_input])

# Prediction
if st.button("Predict"):
    try:
        prob = model.predict_proba(input_vector)[0][1]
        risk_label = "High risk" if prob >= threshold else "Low risk"
        st.markdown(f"**{risk_label}** ({prob:.2%})")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
