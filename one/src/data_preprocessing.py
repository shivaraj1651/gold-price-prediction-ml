import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class GoldDataPreprocessor:
    def __init__(self, lookback=60):
        """
        Initialize preprocessor
        
        Args:
            lookback: Number of days to look back for prediction
        """
        self.lookback = lookback
        self.scaler = MinMaxScaler()
        
    def load_data(self, filepath="data/gold_prices.csv"):
        """Load gold price data from CSV"""
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df
    
    def create_features(self, df):
        """Create technical indicators and features"""
        # Use Close price as target
        df['Close'] = df['Close']
        
        # Technical indicators
        df['SMA_7'] = df['Close'].rolling(window=7).mean()
        df['SMA_30'] = df['Close'].rolling(window=30).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Volume features
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price change features
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def prepare_sequences(self, df, target_col='Close'):
        """
        Prepare sequences for time series prediction
        
        Returns:
            X: Features (samples, lookback, features)
            y: Targets (samples,)
        """
        # Select feature columns
        feature_cols = ['Close', 'Open', 'High', 'Low', 'Volume', 
                       'SMA_7', 'SMA_30', 'EMA_12', 'EMA_26', 
                       'RSI', 'MACD', 'MACD_signal', 'BB_middle', 
                       'BB_upper', 'BB_lower', 'Volume_Ratio', 
                       'Price_Change', 'High_Low_Ratio']
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in df.columns]
        data = df[available_cols].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i])
            y.append(scaled_data[i, 0])  # Close price is first column
        
        return np.array(X), np.array(y)
    
    def prepare_traditional_ml(self, df, target_col='Close', forecast_days=1):
        """
        Prepare data for traditional ML algorithms
        
        Returns:
            X: Features (samples, features)
            y: Targets (samples,)
        """
        # Create lag features
        for lag in [1, 2, 3, 5, 7, 14, 30]:
            df[f'Close_lag_{lag}'] = df[target_col].shift(lag)
        
        # Target: future price
        df['Target'] = df[target_col].shift(-forecast_days)
        
        # Select feature columns
        feature_cols = [col for col in df.columns 
                       if col not in ['Target', 'Date'] and not col.startswith('Unnamed')]
        
        df = df.dropna()
        
        X = df[feature_cols].values
        y = df['Target'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, feature_cols

