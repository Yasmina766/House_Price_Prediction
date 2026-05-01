import streamlit as st
import pickle as pkl
import numpy as np
import pandas as pd

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")

@st.cache_resource
def load_model():
    model = pkl.load(open("model/house_price_model.pkl", "rb"))
    scaler = pkl.load(open("model/scaler.pkl", "rb"))
    return model, scaler

model, scaler = load_model()

st.title("🏠 House Sale Price Prediction")
st.markdown("Edit the key property details below:")

st.subheader("📍 Location")
col1, col2 = st.columns(2)
with col1:
    longitude = st.slider("Longitude", -124.0, -114.0, -119.0, step=0.01)
with col2:
    latitude = st.slider("Latitude", 32.0, 42.0, 36.0, step=0.01)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ["INLAND", "NEAR BAY", "NEAR OCEAN", "<1H OCEAN", "ISLAND"]
)

st.subheader("🏡 Property Details")
col3, col4 = st.columns(2)
with col3:
    housing_median_age = st.slider("Housing Median Age (years)", 1, 52, 25)
    total_rooms = st.number_input("Total Rooms", min_value=100, max_value=10000, value=2000, step=50)
with col4:
    total_bedrooms = st.number_input("Total Bedrooms", min_value=50, max_value=3000, value=400, step=10)
    households = st.number_input("Households", min_value=50, max_value=5000, value=500, step=10)

st.subheader("👥 Population & Income")
col5, col6 = st.columns(2)
with col5:
    population = st.number_input("Population", min_value=100, max_value=20000, value=1200, step=100)
with col6:
    median_income = st.slider("Median Income (tens of $1000)", 0.5, 15.0, 4.0, step=0.1)

if st.button("🔮 Predict Sale Price", use_container_width=True):
    ocean_cols = {
        "INLAND": [0, 0, 0, 0, 1],
        "NEAR BAY": [0, 0, 1, 0, 0],
        "NEAR OCEAN": [0, 1, 0, 0, 0],
        "<1H OCEAN": [1, 0, 0, 0, 0],
        "ISLAND": [0, 0, 0, 1, 0],
    }
    ocean_encoded = ocean_cols[ocean_proximity]

    features = np.array([[
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        median_income,
        *ocean_encoded
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    st.success(f"💰 Estimated Sale Price: **${prediction:,.0f}**")
    st.caption("Prediction powered by XGBoost · R² = 84%")
