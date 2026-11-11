from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent


@st.cache_resource
def load_predictor():
    with (ROOT / "model.pkl").open("rb") as model_file:
        model = pickle.load(model_file)
    with (ROOT / "columns.pkl").open("rb") as columns_file:
        columns = pickle.load(columns_file)
    return model, columns


def estimate_yield(model, columns, country, crop, year, rainfall, temperature, pesticides):
    features = pd.DataFrame(0.0, index=[0], columns=columns)
    features["Year"] = year
    features["average_rain_fall_mm_per_year"] = rainfall
    features["avg_temp"] = temperature
    features["log_pesticides"] = np.log1p(pesticides)
    features[f"Area_{country}"] = 1
    features[f"Item_{crop}"] = 1
    return np.expm1(model.predict(features)[0])


st.set_page_config(page_title="Crop yield estimator | AgTech", layout="centered")
model, columns = load_predictor()
countries = sorted(name[5:] for name in columns if name.startswith("Area_"))
crops = sorted(name[5:] for name in columns if name.startswith("Item_"))

st.title("Crop yield estimator")
st.write("Explore how crop yield varies by country and growing conditions. "
         "Change any value to update the estimate.")

with st.container(border=True):
    st.subheader("Crop and location")
    location_col, crop_col, year_col = st.columns([2, 2, 1])
    country = location_col.selectbox("Country", countries, index=countries.index("Ghana"))
    crop = crop_col.selectbox("Crop", crops, index=crops.index("Maize"))
    year = year_col.number_input("Year", min_value=1990, max_value=2030, value=2020)

    st.subheader("Annual conditions")
    st.caption("The starting values are examples. Enter figures for the selected country and year.")
    rain_col, temp_col, pesticide_col = st.columns(3)
    rainfall = rain_col.number_input(
        "Annual rainfall (mm)", min_value=0.0, max_value=5000.0, value=1000.0
    )
    temperature = temp_col.number_input(
        "Mean temperature (°C)", min_value=0.0, max_value=50.0, value=25.0, step=0.1
    )
    pesticides = pesticide_col.number_input(
        "Pesticide use (tonnes)", min_value=0.0, max_value=500000.0, value=1000.0,
        help="Use the country's total for the year, in metric tonnes.",
    )

prediction = estimate_yield(model, columns, country, crop, year, rainfall, temperature, pesticides)
with st.container(border=True):
    st.subheader("Estimated yield")
    st.write(f"{crop} in {country}, {year}")
    if np.isfinite(prediction) and prediction >= 0:
        st.metric("Tonnes per hectare", f"{prediction / 10000:,.2f}")
        st.caption(f"Equivalent to {prediction:,.0f} hectograms per hectare (hg/ha).")
    else:
        st.write("The model could not produce a usable yield for these values. "
                 "Check the annual conditions and try again.")

with st.expander("How to read this estimate"):
    st.write("Yield is the amount harvested per hectare. The regression model uses historical "
             "country-level data on crops, rainfall, temperature and pesticide use. "
             "An individual farm's harvest can vary with soil, variety and management.")
    st.write("To compare scenarios, change one input at a time. Differences between estimates "
             "reflect patterns in the training data.")

st.divider()
st.caption("Built by [Elvis Frimpong](https://elvisfrimpong-da.github.io/connect/) · "
           "[View project on GitHub](https://github.com/elvisfrimpong-da/agtech-yield-prediction)")
