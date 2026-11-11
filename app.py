from html import escape
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


def icon(name):
    paths = {
        "sprout": '<path d="M12 22v-8M12 16C6 16 3 12 3 7c6 0 9 3 9 9Zm0-4c0-6 3-9 9-9 0 6-3 9-9 9Z"/>',
        "rain": '<path d="M6 14a4 4 0 0 1-.6-7.9A6 6 0 0 1 16.6 6a4 4 0 1 1 1.4 8M7 17l-1 3m6-3-1 3m6-3-1 3"/>',
        "temp": '<path d="M9 14.3V5a3 3 0 0 1 6 0v9.3a5 5 0 1 1-6 0ZM12 9v8m6-11h3m-3 4h2"/><circle cx="12" cy="18" r="1"/>',
        "drop": '<path d="M12 2S5 10.7 5 15a7 7 0 0 0 14 0c0-4.3-7-13-7-13Z"/><path d="M9 15a3 3 0 0 0 3 3"/>',
        "harvest": '<path d="M5 21 18 8M15 11c-1-4 0-7 5-9 1 5 0 8-5 9ZM11 15c-4-1-5-4-4-8 4 1 6 4 4 8Zm1-1c4-2 7-1 9 1-3 4-6 5-10 2"/>',
    }
    return (f'<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{paths[name]}</svg>')


def kpi(name, label, value, unit):
    return (f'<div class="kpi {name}"><span class="symbol">{icon(name)}</span>'
            f'<span class="label">{label}</span><strong>{value}</strong><small>{unit}</small></div>')


st.set_page_config(page_title="Crop yield estimator | AgTech", layout="wide")
st.html("<style>" + (ROOT / "dashboard.css").read_text() + "</style>")
model, columns = load_predictor()
countries = sorted(name[5:] for name in columns if name.startswith("Area_"))
crops = sorted(name[5:] for name in columns if name.startswith("Item_"))

st.html(f'<header class="mast"><span class="brand">{icon("sprout")} AgTech</span>'
        '<h1>Crop yield estimator</h1></header>')
overview = st.container()

with st.container(border=True):
    st.subheader("Adjust your scenario")
    st.caption("Start with the example below or enter your own figures. Estimates update as you edit.")
    location, plant, date, size = st.columns(4)
    country = location.selectbox("Country", countries, index=countries.index("Ghana"))
    crop = plant.selectbox("Crop", crops, index=crops.index("Maize"))
    year = date.number_input("Year", 1990, 2030, 2020)
    hectares = size.number_input("Harvested area (ha)", min_value=0.01, value=1.0, step=0.25,
                                 help="Used only to convert yield into a total harvest.")
    rain, temp, spray = st.columns(3)
    rainfall = rain.number_input("Annual rainfall (mm)", 0.0, 5000.0, 1000.0)
    temperature = temp.number_input("Mean temperature (°C)", 0.0, 50.0, 25.0, step=0.1)
    pesticides = spray.number_input("National pesticide use (tonnes)", 0.0, 500000.0, 1000.0)

prediction = estimate_yield(model, columns, country, crop, year, rainfall, temperature, pesticides)
valid = np.isfinite(prediction) and prediction >= 0
yield_tonnes = prediction / 10000
yield_label = f"{yield_tonnes:,.2f}" if valid else "Unavailable"
total_label = f"{yield_tonnes * hectares:,.2f}" if valid else "Unavailable"

with overview:
    st.html(f'<section class="yield-panel"><div class="field-art" aria-hidden="true"></div>'
            f'<div class="yield-copy"><p>{escape(crop)} &nbsp;·&nbsp; {escape(country)} &nbsp;·&nbsp; {year}</p>'
            f'<h2>Estimated yield</h2><div class="yield-line"><b>{yield_label}</b>'
            '<span>tonnes per hectare</span></div></div></section>')
    st.html('<div class="kpi-grid">'
            + kpi("harvest", "Estimated harvest", total_label, f"tonnes across {hectares:,g} ha")
            + kpi("rain", "Annual rainfall", f"{rainfall:,.2f}".rstrip("0").rstrip("."), "mm / year")
            + kpi("temp", "Mean temperature", f"{temperature:g}", "°C")
            + kpi("drop", "National pesticide use", f"{pesticides:,.2f}".rstrip("0").rstrip("."), "tonnes / year")
            + '</div>')
    if not valid:
        st.warning("The model could not produce a usable yield. Check the annual conditions.")

with st.expander("How are yield and harvest calculated?"):
    st.write("The model estimates yield from historical country-level crop and climate data. "
             "Estimated harvest is yield multiplied by the harvested area you enter. "
             "Actual field yields depend on soil, variety and management.")
    if valid:
        st.caption(f"Model output for this scenario: {prediction:,.0f} hectograms per hectare.")

st.caption("Built by [Elvis Frimpong](https://elvisfrimpong-da.github.io/connect/) · "
           "[View project on GitHub](https://github.com/elvisfrimpong-da/agtech-yield-prediction)")
