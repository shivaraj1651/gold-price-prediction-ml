import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def fetch_gold_data(symbol="GC=F", period="10y"):
    """
    Fetch gold price data from Yahoo Finance
    
    Args:
        symbol: Gold futures symbol (default: GC=F)
        period: Time period (1y, 5y, 10y, max)
    
    Returns:
        DataFrame with gold price data
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        gold = yf.Ticker(symbol)
        data = gold.history(period=period)
        
        # Save to CSV
        data.to_csv("data/gold_prices.csv")
        print(f"Data saved successfully! Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

if __name__ == "__main__":
    print("Fetching gold price data...")
    data = fetch_gold_data()
    if data is not None:
        print(data.head())
        print("\nData columns:", data.columns.tolist())

