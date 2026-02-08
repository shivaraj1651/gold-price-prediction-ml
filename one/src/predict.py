import pandas as pd
import numpy as np
import os
import sys
import yfinance as yf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import GoldDataPreprocessor
from src.models import GoldPricePredictor
import joblib

def predict_future_price(model_name='Linear Regression', days_ahead=1):
    """
    Predict future gold price
    
    Args:
        model_name: Name of the model to use
        days_ahead: Number of days ahead to predict
    """
    # Load data
    preprocessor = GoldDataPreprocessor()
    df = preprocessor.load_data("data/gold_prices.csv")
    df = preprocessor.create_features(df)
    
    # Load model
    predictor = GoldPricePredictor()
    
    if model_name == 'LSTM':
        # LSTM model is saved in native Keras format
        predictor.load_model(model_name, "models/lstm.keras")
        # Prepare LSTM data
        X, y = preprocessor.prepare_sequences(df)
        X_pred = X[-1:]  # Last sequence
        prediction_scaled = predictor.models[model_name].predict(X_pred)
        # For LSTM, inverse transform the first (Close price) feature
        n_features = preprocessor.scaler.n_features_in_
        dummy = np.zeros((1, n_features))
        dummy[0, 0] = prediction_scaled[0, 0]
        prediction = preprocessor.scaler.inverse_transform(dummy)[0, 0]
    else:
        predictor.load_model(model_name, f"models/{model_name.lower().replace(' ', '_')}.pkl")
        # Prepare traditional ML data
        X, y, feature_cols = preprocessor.prepare_traditional_ml(df, forecast_days=days_ahead)
        X_pred = X[-1:]  # Last sample
        # For traditional ML models, the target is in original price units
        prediction = predictor.models[model_name].predict(X_pred)[0]
    
    current_price = df['Close'].iloc[-1]

    # Unit conversion: price per ounce -> per gram
    OUNCE_TO_GRAM = 31.1034768
    current_price_usd_per_gram = current_price / OUNCE_TO_GRAM
    prediction_usd_per_gram = prediction / OUNCE_TO_GRAM

    # Fetch latest USD/INR exchange rate
    try:
        fx = yf.Ticker("USDINR=X")
        fx_data = fx.history(period="5d")
        usd_inr = fx_data["Close"].iloc[-1]
    except Exception:
        usd_inr = None

    # Convert to INR if FX rate is available
    if usd_inr is not None:
        current_price_inr = current_price * usd_inr
        prediction_inr = prediction * usd_inr

        current_price_inr_per_gram = current_price_usd_per_gram * usd_inr
        prediction_inr_per_gram = prediction_usd_per_gram * usd_inr
    else:
        current_price_inr = None
        prediction_inr = None
        current_price_inr_per_gram = None
        prediction_inr_per_gram = None

    print(f"\n{'='*50}")
    print(f"GOLD PRICE PREDICTION")
    print(f"{'='*50}")
    print(f"Model Used: {model_name}")
    print(f"Current Price (USD/oz): ${current_price:.2f}")
    print(f"Predicted Price ({days_ahead} day(s) ahead, USD/oz): ${prediction:.2f}")
    print(f"Current Price (USD/g): ${current_price_usd_per_gram:.2f}")
    print(f"Predicted Price ({days_ahead} day(s) ahead, USD/g): ${prediction_usd_per_gram:.2f}")
    print(f"Expected Change (USD/oz): ${prediction - current_price:.2f} ({((prediction/current_price - 1) * 100):.2f}%)")

    if usd_inr is not None:
        print(f"\nUSD/INR rate used: Rs {usd_inr:.2f} per $1")
        print(f"Current Price (INR/oz): Rs {current_price_inr:.2f}")
        print(f"Predicted Price ({days_ahead} day(s) ahead, INR/oz): Rs {prediction_inr:.2f}")
        print(f"Current Price (INR/g): Rs {current_price_inr_per_gram:.2f}")
        print(f"Predicted Price ({days_ahead} day(s) ahead, INR/g): Rs {prediction_inr_per_gram:.2f}")
        print(f"Expected Change (INR/oz): Rs {prediction_inr - current_price_inr:.2f}")
    else:
        print("\nCould not fetch USD/INR rate; INR values not shown.")

    print(f"{'='*50}\n")
    
    return prediction

if __name__ == "__main__":
    # Use Linear Regression for stable, realistic predictions (like R project)
    predict_future_price(model_name='Linear Regression', days_ahead=1)

