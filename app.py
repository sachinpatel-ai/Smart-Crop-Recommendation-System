import streamlit as st
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ==================================================================
# PAGE CONFIG
# ==================================================================
st.set_page_config(
    page_title="Smart Agriculture | AI Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================================================================
# PATHS
# ==================================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "crop_model.joblib"
CSV_PATH = BASE_DIR / "Crop_recommendation.csv"

FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


# ==================================================================
# MODEL LOADING (cached resource — loaded once)
# ==================================================================
@st.cache_resource(show_spinner=False)
def load_model(path: Path):
    """Load the pre-trained RandomForestClassifier. No training happens here."""
    if not path.exists():
        return None, f"crop_model.joblib was not found at: {path}"
    try:
        m = joblib.load(path)
        return m, None
    except Exception as exc:
        return None, f"Failed to load crop_model.joblib: {exc}"


# ==================================================================
# PREPROCESSING PIPELINE (cached data — mirrors the training notebook)
#
#   1. Load CSV
#   2. IQR-based clipping on numeric features (computed on full dataset,
#      exactly as in the notebook, BEFORE the train/test split)
#   3. LabelEncoder fit on the 'label' column
#   4. train_test_split(test_size=0.2, random_state=42)
#   5. StandardScaler fit on X_train (post-clip)
#
# The fitted LabelEncoder and StandardScaler are reused at inference
# time so that live predictions go through identical preprocessing to
# what crop_model.joblib was trained on.
# ==================================================================
@st.cache_data(show_spinner=False)
def build_preprocessing_pipeline(csv_path: Path):
    if not csv_path.exists():
        return None, f"Crop_recommendation(1).csv was not found at: {csv_path}"

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return None, f"Failed to read CSV: {exc}"

    missing_cols = [c for c in FEATURE_ORDER + ["label"] if c not in df.columns]
    if missing_cols:
        return None, f"Dataset is missing expected column(s): {missing_cols}"

    try:
        # ---- Step 1: IQR-based clipping (bounds derived from full df) ----
        clip_bounds = {}
        for col in FEATURE_ORDER:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            clip_bounds[col] = (lower, upper)
            df[col] = df[col].clip(lower=lower, upper=upper)

        # ---- Step 2: Label encoding ----
        le = LabelEncoder()
        df["label"] = le.fit_transform(df["label"])

        # ---- Step 3: Train/test split (mirrors notebook, random_state=42) ----
        X = df.drop("label", axis=1)
        y = df["label"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # ---- Step 4: StandardScaler fit on X_train ----
        scaler = StandardScaler()
        scaler.fit(X_train[FEATURE_ORDER])

    except Exception as exc:
        return None, f"Preprocessing pipeline failed: {exc}"

    pipeline = {
        "clip_bounds": clip_bounds,
        "label_encoder": le,
        "scaler": scaler,
        "known_labels": sorted(le.classes_.tolist()),
    }
    return pipeline, None


model, model_error = load_model(MODEL_PATH)
pipeline, pipeline_error = build_preprocessing_pipeline(CSV_PATH)


# ==================================================================
# INFERENCE HELPERS
# ==================================================================
def apply_clipping(input_df: pd.DataFrame, clip_bounds: dict) -> pd.DataFrame:
    clipped = input_df.copy()
    for col in FEATURE_ORDER:
        lower, upper = clip_bounds[col]
        clipped[col] = clipped[col].clip(lower=lower, upper=upper)
    return clipped


def run_prediction(model, pipeline, input_values: list):
    """
    Runs the full inference pipeline and returns a dict with every
    intermediate artifact needed for both the result card and the
    diagnostics panel. Raises a RuntimeError tagged with the failing
    stage on any error, so the caller can show a precise message.
    """
    diagnostics = {}

    # ---- Stage: build DataFrame ----
    try:
        input_df = pd.DataFrame([input_values], columns=FEATURE_ORDER)
        diagnostics["raw_input"] = input_df.iloc[0].to_dict()
    except Exception as exc:
        raise RuntimeError(f"[DataFrame construction] {exc}") from exc

    # ---- Stage: IQR clipping ----
    try:
        clipped_df = apply_clipping(input_df, pipeline["clip_bounds"])
        diagnostics["clipped_input"] = clipped_df.iloc[0].to_dict()
    except Exception as exc:
        raise RuntimeError(f"[IQR clipping] {exc}") from exc

    # ---- Stage: StandardScaler ----
    try:
        scaled_input = pipeline["scaler"].transform(clipped_df[FEATURE_ORDER])
        diagnostics["scaled_input"] = dict(zip(FEATURE_ORDER, scaled_input[0].tolist()))
    except Exception as exc:
        raise RuntimeError(f"[StandardScaler transform] {exc}") from exc

    # ---- Stage: model.predict ----
    try:
        raw_prediction = model.predict(scaled_input)
        diagnostics["raw_prediction_class"] = int(raw_prediction[0])
    except Exception as exc:
        raise RuntimeError(f"[model.predict] {exc}") from exc

    # ---- Stage: label decoding ----
    try:
        crop_name = pipeline["label_encoder"].inverse_transform(raw_prediction)[0]
        diagnostics["decoded_crop"] = crop_name
    except Exception as exc:
        raise RuntimeError(f"[LabelEncoder.inverse_transform] {exc}") from exc

    return crop_name, diagnostics


# ==================================================================
# GLOBAL STYLING
# ==================================================================
BACKGROUND_IMAGE_URL = (
    "https://images.unsplash.com/photo-1500382017468-9049fed747ef?"
    "auto=format&fit=crop&w=1950&q=80"
)

st.markdown(
    f"""
    <style>
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    div[data-testid="stToolbar"] {{visibility: hidden; height: 0; position: fixed;}}
    div[data-testid="stDecoration"] {{visibility: hidden;}}
    div[data-testid="stStatusWidget"] {{visibility: hidden;}}

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Playfair+Display:wght@600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
    }}

    .stApp {{
        background:
            linear-gradient(180deg, rgba(6, 38, 22, 0.86) 0%, rgba(8, 48, 28, 0.82) 45%, rgba(4, 28, 16, 0.93) 100%),
            url('{BACKGROUND_IMAGE_URL}');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }}

    .hero-wrapper {{
        text-align: center;
        padding: 2.8rem 1.5rem 2.2rem 1.5rem;
        margin-bottom: 2rem;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(255,255,255,0.10), rgba(255,255,255,0.03));
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,0.18);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }}

    .hero-badge {{
        display: inline-block;
        padding: 0.4rem 1.1rem;
        border-radius: 50px;
        background: rgba(120, 200, 130, 0.18);
        border: 1px solid rgba(150, 220, 150, 0.4);
        color: #d9f7d0;
        font-size: 0.9rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }}

    .hero-title {{
        font-family: 'Playfair Display', serif;
        font-size: 2.85rem;
        font-weight: 800;
        background: linear-gradient(90deg, #eafff0, #a8e6a3 40%, #6fbf73);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.2rem 0 0.6rem 0;
        line-height: 1.15;
    }}

    .hero-subtitle {{
        color: #e7f5e6;
        font-size: 1.08rem;
        max-width: 640px;
        margin: 0 auto;
        line-height: 1.6;
        opacity: 0.92;
        font-weight: 300;
    }}

    .glass-card {{
        background: linear-gradient(135deg, rgba(255,255,255,0.12), rgba(255,255,255,0.04));
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 24px;
        padding: 2rem 2.2rem 1.2rem 2.2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: box-shadow 0.3s ease;
    }}

    .glass-card:hover {{
        box-shadow: 0 12px 40px rgba(0,0,0,0.42);
    }}

    .card-title {{
        font-size: 1.5rem;
        font-weight: 700;
        color: #f2fff0;
        margin-bottom: 0.3rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .card-subtitle {{
        color: #cfe9cb;
        font-size: 0.92rem;
        margin-bottom: 1.4rem;
        font-weight: 300;
        opacity: 0.85;
    }}

    label, .stNumberInput label {{
        color: #eafbe6 !important;
        font-weight: 500 !important;
        font-size: 0.92rem !important;
    }}

    div[data-baseweb="input"] {{
        background: rgba(255,255,255,0.9) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.4) !important;
    }}

    input[type="number"] {{
        color: #1e3a1e !important;
        font-weight: 600 !important;
    }}

    div.stButton > button {{
        width: 100%;
        background: linear-gradient(135deg, #3fa34d, #2c7a3d);
        color: white;
        font-size: 1.15rem;
        font-weight: 600;
        letter-spacing: 0.4px;
        padding: 0.9rem 1rem;
        border-radius: 16px;
        border: none;
        box-shadow: 0 6px 20px rgba(46, 125, 50, 0.45);
        transition: all 0.25s ease-in-out;
        margin-top: 0.6rem;
    }}

    div.stButton > button:hover {{
        transform: translateY(-3px) scale(1.01);
        box-shadow: 0 10px 28px rgba(46, 125, 50, 0.6);
        background: linear-gradient(135deg, #47b358, #338a45);
        color: white;
        border: none;
    }}

    div.stButton > button:active {{
        transform: translateY(0px) scale(0.99);
    }}

    .result-card {{
        text-align: center;
        padding: 2.4rem 1.5rem;
        margin-top: 1.5rem;
        border-radius: 26px;
        background: linear-gradient(135deg, rgba(63, 163, 77, 0.35), rgba(23, 87, 34, 0.45));
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(180, 255, 170, 0.4);
        box-shadow: 0 10px 36px rgba(0,0,0,0.4);
        animation: fadeInUp 0.6s ease;
    }}

    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(18px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    .result-label {{
        color: #d7f5d1;
        font-size: 1rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-weight: 500;
        margin-bottom: 0.6rem;
        opacity: 0.9;
    }}

    .result-crop {{
        font-family: 'Playfair Display', serif;
        font-size: 3.2rem;
        font-weight: 800;
        color: #ffffff;
        text-shadow: 0 4px 18px rgba(0,0,0,0.35);
        margin: 0;
        text-transform: capitalize;
    }}

    .field-note {{
        margin-top: 1.6rem;
        padding: 1.1rem 1.4rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.08);
        border-left: 4px solid #6fbf73;
        color: #e6f5e2;
        font-size: 0.9rem;
        line-height: 1.55;
        backdrop-filter: blur(10px);
    }}

    .field-note b {{
        color: #b7f0ad;
    }}

    .app-footer {{
        text-align: center;
        color: #cfe9cb;
        opacity: 0.6;
        font-size: 0.8rem;
        margin-top: 2.5rem;
        font-weight: 300;
    }}

    @media (max-width: 768px) {{
        .hero-title {{ font-size: 2rem; }}
        .hero-subtitle {{ font-size: 0.95rem; }}
        .result-crop {{ font-size: 2.2rem; }}
        .glass-card {{ padding: 1.4rem 1.2rem 0.8rem 1.2rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==================================================================
# HERO SECTION
# ==================================================================
st.markdown(
    """
    <div class="hero-wrapper">
        <div class="hero-badge">🌱 SMART AGRICULTURE</div>
        <div class="hero-title">AI-Powered Crop Recommendation</div>
        <div class="hero-subtitle">
            Discover the most suitable crop for your soil and weather conditions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================================================================
# STARTUP ERROR MESSAGES
# ==================================================================
if model_error:
    st.error(f"❌ {model_error}\n\nPlease place `crop_model.joblib` in the same folder as `app.py`.")

if pipeline_error:
    st.error(
        f"❌ {pipeline_error}\n\n"
        "Please place `Crop_recommendation(1).csv` in the same folder as `app.py`. "
        "This file is required to reproduce the IQR clipping, LabelEncoder, and "
        "StandardScaler used during training."
    )

pipeline_ready = model is not None and pipeline is not None

# ==================================================================
# INPUT CARD
# ==================================================================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🌾 Farm Conditions</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="card-subtitle">Provide accurate soil and weather readings for the best recommendation.</div>',
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)

with col1:
    nitrogen = st.number_input(
        "Nitrogen (N)", min_value=0.0, max_value=300.0, value=90.0, step=1.0,
        help="Nitrogen content ratio in soil"
    )
    temperature = st.number_input(
        "Temperature (°C)", min_value=-10.0, max_value=60.0, value=20.879744, step=0.1,
        help="Average ambient temperature", format="%.4f"
    )
    ph = st.number_input(
        "Soil pH", min_value=0.0, max_value=14.0, value=6.502985, step=0.1,
        help="Soil acidity/alkalinity level", format="%.4f"
    )

with col2:
    phosphorus = st.number_input(
        "Phosphorus (P)", min_value=0.0, max_value=300.0, value=42.0, step=1.0,
        help="Phosphorus content ratio in soil"
    )
    humidity = st.number_input(
        "Humidity (%)", min_value=0.0, max_value=100.0, value=82.002744, step=1.0,
        help="Relative humidity percentage", format="%.4f"
    )
    rainfall = st.number_input(
        "Rainfall (mm)", min_value=0.0, max_value=1000.0, value=202.935536, step=1.0,
        help="Average rainfall in millimeters", format="%.4f"
    )

with col3:
    potassium = st.number_input(
        "Potassium (K)", min_value=0.0, max_value=300.0, value=43.0, step=1.0,
        help="Potassium content ratio in soil"
    )
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background: rgba(255,255,255,0.08); border-radius: 14px;
        padding: 0.9rem; margin-top: 0.4rem; font-size: 0.82rem; color:#e2f5df;
        border: 1px solid rgba(255,255,255,0.12);">
        💡 <b>Tip:</b> Defaults reflect typical dataset averages — adjust
        them to match your own farm readings.
        </div>
        """,
        unsafe_allow_html=True,
    )

predict_clicked = st.button(
    "🌿 Recommend Best Crop",
    use_container_width=True,
    disabled=not pipeline_ready,
)

st.markdown("</div>", unsafe_allow_html=True)

# ==================================================================
# PREDICTION
# ==================================================================
if predict_clicked:
    if not pipeline_ready:
        st.error("Cannot make a prediction — model or preprocessing pipeline failed to load. See errors above.")
    else:
        input_values = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
        try:
            with st.spinner("🌱 Analyzing your farm conditions..."):
                crop_name, diagnostics = run_prediction(model, pipeline, input_values)

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">🌾 Recommended Crop</div>
                    <div class="result-crop">{crop_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                """
                <div class="field-note">
                <b>💡 Farming Tip:</b> This recommendation is generated from soil nutrient,
                temperature, humidity, pH, and rainfall conditions. Consider local farming
                conditions before planting.
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("🔍 Model Diagnostics"):
                st.write("**Model type:**", type(model).__name__)
                st.write("**Number of input features:**", len(FEATURE_ORDER))
                st.write("**Input feature names (in order):**", FEATURE_ORDER)
                st.write("**Raw prediction class (encoded):**", diagnostics["raw_prediction_class"])
                st.write("**Decoded crop name:**", diagnostics["decoded_crop"])
                st.write("**Raw input values:**")
                st.json(diagnostics["raw_input"])
                st.write("**Input after IQR clipping:**")
                st.json(diagnostics["clipped_input"])
                st.write("**Input after StandardScaler:**")
                st.json(diagnostics["scaled_input"])
                st.write("**Known crop labels (from LabelEncoder):**")
                st.write(pipeline["known_labels"])

        except RuntimeError as e:
            st.error(f"❌ Prediction failed at stage: {e}")
            with st.expander("🔍 Debug details", expanded=True):
                st.write("**Feature values submitted:**")
                st.json(dict(zip(FEATURE_ORDER, input_values)))
                st.write("**Full exception:**")
                st.exception(e)
        except Exception as e:
            st.error("❌ An unexpected error occurred during prediction.")
            with st.expander("🔍 Debug details", expanded=True):
                st.write("**Feature values submitted:**")
                st.json(dict(zip(FEATURE_ORDER, input_values)))
                st.write("**Full exception:**")
                st.exception(e)

# ==================================================================
# FOOTER
# ==================================================================
st.markdown(
    """
    <div class="app-footer">
        🌍 Smart Agriculture · Powered by Machine Learning · Built By Scahin Patel
    </div>
    """,
    unsafe_allow_html=True,
)
