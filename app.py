# Interactive Streamlit App for Crop Recommendation
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="🌾 Crop Recommendation System",
    page_icon="🌱",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f5fff5;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #2e7d32;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #4f4f4f;
        margin-bottom: 30px;
    }

    .result-box {
        background-color: #e8f5e9;
        padding: 25px;
        border-radius: 15px;
        border: 2px solid #81c784;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        color: #1b5e20;
    }

    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.08);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load Dataset
# -----------------------------
@st.cache_data

def load_data():
    return pd.read_csv("Crop_recommendation.csv")


df = load_data()

# -----------------------------
# Train Model
# -----------------------------
X = df.drop("label", axis=1)
y = df["label"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y_encoded)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="title">🌾 Smart Crop Recommendation System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Predict the most suitable crop based on soil and weather conditions</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("🧪 Enter Soil & Weather Details")

N = st.sidebar.slider("Nitrogen (N)", 0, 150, 50)
P = st.sidebar.slider("Phosphorus (P)", 0, 150, 50)
K = st.sidebar.slider("Potassium (K)", 0, 210, 40)

temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0)
ph = st.sidebar.slider("pH Value", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

# -----------------------------
# Input DataFrame
# -----------------------------
input_data = pd.DataFrame(
    {
        "N": [N],
        "P": [P],
        "K": [K],
        "temperature": [temperature],
        "humidity": [humidity],
        "ph": [ph],
        "rainfall": [rainfall],
    }
)

# -----------------------------
# Main Layout
# -----------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Current Input Values")
    st.dataframe(input_data, use_container_width=True)

    st.subheader("📊 Soil Nutrient Overview")
    chart_df = pd.DataFrame(
        {
            "Nutrients": ["Nitrogen", "Phosphorus", "Potassium"],
            "Values": [N, P, K],
        }
    )

    st.bar_chart(chart_df.set_index("Nutrients"))

with col2:
    st.subheader("🌤 Weather Conditions")

    weather_df = pd.DataFrame(
        {
            "Factors": ["Temperature", "Humidity", "pH", "Rainfall"],
            "Values": [temperature, humidity, ph, rainfall],
        }
    )

    st.line_chart(weather_df.set_index("Factors"))

    st.subheader("🤖 Predict Best Crop")

    if st.button("🔍 Recommend Crop"):
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)
        crop_name = label_encoder.inverse_transform(prediction)[0]

        st.markdown(
            f'''
            <div class="result-box">
                Recommended Crop 🌱<br><br>
                {crop_name.upper()}
            </div>
            ''',
            unsafe_allow_html=True,
        )

# -----------------------------
# Dataset Preview
# -----------------------------
st.markdown("---")
st.subheader("📂 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    """
    <center>
    Built with ❤️ using Streamlit and Machine Learning
    </center>
    """,
    unsafe_allow_html=True,
)
