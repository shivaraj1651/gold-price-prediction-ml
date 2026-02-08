import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import os

class GoldPricePredictor:
    def __init__(self):
        self.models = {}
        self.model_names = []
        
    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        """Train Linear Regression model"""
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models['Linear Regression'] = model
        return {'mse': mse, 'mae': mae, 'r2': r2, 'predictions': y_pred}
    
    def train_ridge_regression(self, X_train, y_train, X_test, y_test, alpha=1.0):
        """Train Ridge Regression model"""
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models['Ridge Regression'] = model
        return {'mse': mse, 'mae': mae, 'r2': r2, 'predictions': y_pred}
    
    def train_random_forest(self, X_train, y_train, X_test, y_test, n_estimators=100):
        """Train Random Forest model"""
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models['Random Forest'] = model
        return {'mse': mse, 'mae': mae, 'r2': r2, 'predictions': y_pred}
    
    def train_gradient_boosting(self, X_train, y_train, X_test, y_test, n_estimators=100):
        """Train Gradient Boosting model"""
        model = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models['Gradient Boosting'] = model
        return {'mse': mse, 'mae': mae, 'r2': r2, 'predictions': y_pred}
    
    def train_svr(self, X_train, y_train, X_test, y_test, kernel='rbf'):
        """Train Support Vector Regression model"""
        model = SVR(kernel=kernel, C=1.0, epsilon=0.1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models['SVR'] = model
        return {'mse': mse, 'mae': mae, 'r2': r2, 'predictions': y_pred}
    
    def train_lstm(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Train LSTM model for time series prediction"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        history = model.fit(X_train, y_train, 
                           epochs=epochs, 
                           batch_size=batch_size,
                           validation_data=(X_test, y_test),
                           verbose=1)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.models['LSTM'] = model
        return {'mse': mse, 'mae': mae, 'r2': r2, 'predictions': y_pred, 'history': history}
    
    def save_model(self, model_name, filepath):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if model_name in self.models:
            if model_name == 'LSTM':
                # Keras requires a `.keras` or `.h5` extension
                self.models[model_name].save(filepath)
            else:
                joblib.dump(self.models[model_name], filepath)
            print(f"Model {model_name} saved to {filepath}")
        else:
            print(f"Model {model_name} not found!")
    
    def load_model(self, model_name, filepath):
        """Load trained model"""
        if model_name == 'LSTM':
            from tensorflow.keras.models import load_model
            self.models[model_name] = load_model(filepath)
        else:
            self.models[model_name] = joblib.load(filepath)
        print(f"Model {model_name} loaded from {filepath}")

