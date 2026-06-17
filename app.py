import streamlit as st
import pandas as pd
import numpy as np
import pickle


# PAGE CONFIG

st.set_page_config(
    page_title="AgTech Yield Predictor",
    layout="wide"
)

# LOAD MODEL

model = pickle.load(open("model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))


# STYLING (UPDATED)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(120deg, #0b1d13, #102820, #0E1117);
    color: white;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f2f1f, #0b1d13);
}
.stButton>button {
    background: linear-gradient(90deg, #4CAF50, #2e7d32);
    color: white;
    border-radius: 12px;
}
.card {
    background: rgba(16, 40, 32, 0.85);
    padding: 25px;
    border-radius: 15px;
    text-align: center;
}
.metric-card {
    background: rgba(20, 60, 45, 0.85);
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    transition: 0.3s ease;
}
.metric-card:hover {
    transform: scale(1.05);
}
h1, h2, h3 {
    color: #A5D6A7;
}
</style>
""", unsafe_allow_html=True)


# TITLE

st.markdown("""
<h1 style='text-align: center;'> AgTech Crop Yield Intelligence</h1>
<p style='text-align: center;'>AI-powered insights for smarter agriculture</p>
""", unsafe_allow_html=True)


# SIDEBAR

st.sidebar.header(" Input Features")

year = st.sidebar.slider("Year", 1990, 2030, 2020)
rainfall = st.sidebar.number_input("Rainfall (mm/year)", 0.0, 5000.0, 1000.0)
temp = st.sidebar.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
pesticides = st.sidebar.number_input("Pesticides (tonnes)", 0.0, 500000.0, 1000.0)

areas = [col.replace("Area_", "") for col in columns if col.startswith("Area_")]
items = [col.replace("Item_", "") for col in columns if col.startswith("Item_")]

area = st.sidebar.selectbox("Country", areas)
item = st.sidebar.selectbox("Crop", items)


# INPUT DATA

input_data = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

input_data['Year'] = year
input_data['average_rain_fall_mm_per_year'] = rainfall
input_data['avg_temp'] = temp
input_data['log_pesticides'] = np.log1p(pesticides)

if f"Area_{area}" in input_data.columns:
    input_data[f"Area_{area}"] = 1

if f"Item_{item}" in input_data.columns:
    input_data[f"Item_{item}"] = 1


# LAYOUT

col1, col2 = st.columns([1.2, 1])


# PREDICTION + INTERPRETATION

with col1:
    st.subheader(" Crop Yield Prediction")

    if st.button("🚀 Predict Yield"):
        pred = np.expm1(model.predict(input_data)[0])

        st.markdown(f"""
        <div class="card">
            <h2>Estimated Crop Yield</h2>
            <h1 style="color:#4CAF50;">{pred:,.2f}</h1>
            <p>hg/ha</p>
        </div>
        """, unsafe_allow_html=True)

        # 🔥 INTERPRETATION
        if pred < 20000:
            st.warning("⚠️ Low yield expected. Consider improving conditions.")
        elif pred < 50000:
            st.info("ℹ️ Moderate yield expected under current conditions.")
        else:
            st.success("✅ High yield expected. Conditions are favorable.")


# FIXED CHART SCALING

with col2:
    st.subheader("📊 Environmental Insights")

    chart_data = pd.DataFrame({
        "Feature": ["Rainfall", "Temperature", "Pesticides"],
        "Value": [
            rainfall / 1000,      # scaled
            temp,
            pesticides / 10000    # scaled
        ]
    })

    st.bar_chart(chart_data.set_index("Feature"))


# METRICS

st.markdown("### 📌 Key Inputs")

m1, m2, m3 = st.columns(3)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🌧️ Rainfall</h3>
        <h2>{rainfall} mm</h2>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🌡️ Temperature</h3>
        <h2>{temp} °C</h2>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <h3>🧪 Pesticides</h3>
        <h2>{pesticides} t</h2>
    </div>
    """, unsafe_allow_html=True)


# CONTEXT

st.markdown(f"""
### 🌍 Context  
Analyzing **{item}** production in **{area}**
""")


# CONNECT

st.markdown("---")
c1, c2 = st.columns([1,2])

with c1:
    st.image("elvis_frimpong_connect.png", width=180)

with c2:
    st.markdown("""
    <div class="card">
        <h3>🤝 Connect With Me</h3>
        <p>Let’s collaborate in AgTech & Data Science.</p>
        <a href="https://elvisfrimpong-da.github.io/connect/" target="_blank">
        🔗 Visit My Page
        </a>
    </div>
    """, unsafe_allow_html=True)


# FOOTER

st.markdown("---")
st.markdown("Built with Streamlit by Elvis | AgTech ML Project")