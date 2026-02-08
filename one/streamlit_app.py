import os

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from src.data_collection import fetch_gold_data
from src.data_preprocessing import GoldDataPreprocessor
from src.models import GoldPricePredictor
from src.train import main as train_models


def compute_prediction(model_name: str = "Linear Regression", days_ahead: int = 1):
    """Compute gold price prediction and return structured values for the UI."""
    preprocessor = GoldDataPreprocessor()
    df = preprocessor.load_data("data/gold_prices.csv")
    df = preprocessor.create_features(df)

    predictor = GoldPricePredictor()

    if model_name == "LSTM":
        predictor.load_model("LSTM", "models/lstm.keras")
        X, y = preprocessor.prepare_sequences(df)
        X_pred = X[-1:]
        prediction_scaled = predictor.models["LSTM"].predict(X_pred)
        n_features = preprocessor.scaler.n_features_in_
        dummy = np.zeros((1, n_features))
        dummy[0, 0] = prediction_scaled[0, 0]
        prediction = preprocessor.scaler.inverse_transform(dummy)[0, 0]
    else:
        predictor.load_model(model_name, f"models/{model_name.lower().replace(' ', '_')}.pkl")
        X, y, feature_cols = preprocessor.prepare_traditional_ml(df, forecast_days=days_ahead)
        X_pred = X[-1:]
        prediction = predictor.models[model_name].predict(X_pred)[0]

    current_price = df["Close"].iloc[-1]

    # Per-gram conversion
    OUNCE_TO_GRAM = 31.1034768
    current_price_usd_per_gram = current_price / OUNCE_TO_GRAM
    prediction_usd_per_gram = prediction / OUNCE_TO_GRAM

    # FX conversion
    try:
        fx = yf.Ticker("USDINR=X")
        fx_data = fx.history(period="5d")
        usd_inr = float(fx_data["Close"].iloc[-1])
    except Exception:
        usd_inr = None

    if usd_inr is not None:
        current_price_inr_per_gram = current_price_usd_per_gram * usd_inr
        prediction_inr_per_gram = prediction_usd_per_gram * usd_inr
    else:
        current_price_inr_per_gram = None
        prediction_inr_per_gram = None

    recent_history = df[["Close"]].tail(180)

    return {
        "current_usd_oz": float(current_price),
        "prediction_usd_oz": float(prediction),
        "current_usd_g": float(current_price_usd_per_gram),
        "prediction_usd_g": float(prediction_usd_per_gram),
        "usd_inr": usd_inr,
        "current_inr_g": current_price_inr_per_gram,
        "prediction_inr_g": prediction_inr_per_gram,
        "history": recent_history,
    }


def compute_all_models_predictions(days_ahead: int):
    """Get predictions from all models for the comparison table (like R Shiny)."""
    preprocessor = GoldDataPreprocessor()
    df = preprocessor.load_data("data/gold_prices.csv")
    df = preprocessor.create_features(df)
    current_price = float(df["Close"].iloc[-1])
    OUNCE_TO_GRAM = 31.1034768
    try:
        fx = yf.Ticker("USDINR=X")
        fx_data = fx.history(period="5d")
        usd_inr = float(fx_data["Close"].iloc[-1])
    except Exception:
        usd_inr = None

    # Prepare traditional ML input once (same X for all non-LSTM models)
    df_trad = df.copy()
    X_trad, _, _ = preprocessor.prepare_traditional_ml(df_trad, forecast_days=days_ahead)
    X_pred_trad = X_trad[-1:]

    model_names = [
        "Random Forest", "Linear Regression", "Ridge Regression",
        "Gradient Boosting", "SVR", "LSTM",
    ]
    rows = []
    predictor = GoldPricePredictor()

    for name in model_names:
        try:
            if name == "LSTM":
                prep_lstm = GoldDataPreprocessor()
                predictor.load_model("LSTM", "models/lstm.keras")
                X, _ = prep_lstm.prepare_sequences(df)
                X_pred = X[-1:]
                pred_scaled = predictor.models["LSTM"].predict(X_pred)
                n_f = prep_lstm.scaler.n_features_in_
                dummy = np.zeros((1, n_f))
                dummy[0, 0] = pred_scaled[0, 0]
                pred_oz = float(prep_lstm.scaler.inverse_transform(dummy)[0, 0])
            else:
                predictor.load_model(name, f"models/{name.lower().replace(' ', '_')}.pkl")
                pred_oz = float(predictor.models[name].predict(X_pred_trad)[0])

            pred_g = pred_oz / OUNCE_TO_GRAM
            pred_inr_g = (pred_g * usd_inr) if usd_inr else None
            rows.append({
                "Model": name,
                "Predicted_USD_oz": round(pred_oz, 4),
                "Predicted_USD_g": round(pred_g, 4),
                "Predicted_INR_g": round(pred_inr_g, 4) if pred_inr_g is not None else None,
            })
        except Exception:
            rows.append({
                "Model": name,
                "Predicted_USD_oz": None,
                "Predicted_USD_g": None,
                "Predicted_INR_g": None,
            })

    return pd.DataFrame(rows), current_price, usd_inr


def ensure_backend_ready():
    """Fetch data and train models once, if needed."""
    data_exists = os.path.exists("data/gold_prices.csv")
    models_exist = os.path.exists("models/random_forest.pkl") and os.path.exists("models/lstm.keras")

    if not data_exists:
        fetch_gold_data(period="10y")
        data_exists = True

    if data_exists and not models_exist:
        train_models()


st.set_page_config(page_title="Gold Price Prediction", page_icon="🪙", layout="centered")

# Background gold image and glassmorphism card
st.markdown(
    """
    <style>
    .stApp {
        background-image:
            linear-gradient(rgba(0, 0, 0, 0.55), rgba(0, 0, 0, 0.85)),
            url("https://images.pexels.com/photos/843700/pexels-photo-843700.jpeg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }
    .block-container {
        backdrop-filter: blur(10px);
        background: rgba(15, 12, 41, 0.75);
        padding: 2.5rem 2.5rem 4rem 2.5rem;
        border-radius: 1.25rem;
        border: 1px solid rgba(255, 215, 0, 0.7);
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.8);
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFD700 !important;  /* gold titles */
    }
    p, label, span {
        color: #F5F5F5 !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #FFD700, #FFA500);
        color: #000000 !important;
        border: none;
        padding: 0.6rem 1.2rem;
        border-radius: 999px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #FFE55C, #FFB347);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Gold Price Prediction")
st.image("https://images.pexels.com/photos/843700/pexels-photo-843700.jpeg", width=140)
st.write("One click to **fetch data, train models, predict**, and visualize gold prices (including **per gram in INR**). Scroll down a little after clicking **Predict** to see the results.")

# One-time backend initialization
if "backend_ready" not in st.session_state:
    with st.spinner("Preparing data and training models (first run only)..."):
        ensure_backend_ready()
    st.session_state["backend_ready"] = True

model_name = st.selectbox(
    "Choose model for prediction (Linear Regression / Ridge give stable results)",
    ["Linear Regression", "Ridge Regression", "Random Forest", "Gradient Boosting", "SVR", "LSTM"],
    index=0,
)

days_ahead = st.slider("Days ahead to predict", min_value=1, max_value=7, value=1)

if st.button("Predict"):
    with st.spinner("Computing prediction..."):
        result = compute_prediction(model_name=model_name, days_ahead=days_ahead)
        all_df, _current_oz, _usd_inr = compute_all_models_predictions(days_ahead)

    st.subheader("Prediction Result (for selected model)")
    st.write(f"**Model:** {model_name} | **Days ahead:** {days_ahead}")

    st.subheader("Latest price (per gram in INR)")
    if result["current_inr_g"] is not None:
        delta_inr = result["prediction_inr_g"] - result["current_inr_g"]
        st.metric(
            label="Predicted price (INR per gram)",
            value=f"Rs {result['prediction_inr_g']:,.2f}",
            delta=f"{delta_inr:,.2f} Rs",
        )
        st.metric(
            label="Current price (INR per gram)",
            value=f"Rs {result['current_inr_g']:,.2f}",
        )
    else:
        st.warning("Could not fetch USD/INR rate. INR values are not available right now.")

    st.subheader("Details (USD)")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Current price (USD/oz):** ${result['current_usd_oz']:.2f}")
        st.write(f"**Current price (USD/g):** ${result['current_usd_g']:.2f}")
    with col2:
        st.write(f"**Predicted price (USD/oz):** ${result['prediction_usd_oz']:.2f}")
        st.write(f"**Predicted price (USD/g):** ${result['prediction_usd_g']:.2f}")

    if result["usd_inr"] is not None:
        st.write(f"**USD/INR rate used:** Rs {result['usd_inr']:.2f} per $1")

    st.subheader("All Models - Prediction Table")
    st.dataframe(all_df, use_container_width=True)
    st.caption("Linear Regression and Ridge Regression usually give the most realistic prices. Random Forest / SVR can show unrealistic values.")

    st.subheader("Recent gold price history (USD/oz)")
    history = result["history"]
    history = history.rename(columns={"Close": "USD/oz"})
    st.line_chart(history)

