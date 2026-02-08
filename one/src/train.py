import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import GoldDataPreprocessor
from src.models import GoldPricePredictor

def main():
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = GoldDataPreprocessor(lookback=60)
    
    # Load and preprocess data
    print("Loading data...")
    df = preprocessor.load_data("data/gold_prices.csv")
    
    print("Creating features...")
    df = preprocessor.create_features(df)
    
    # Split data
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # Prepare data for traditional ML
    print("Preparing data for traditional ML models...")
    X_train, y_train, feature_cols = preprocessor.prepare_traditional_ml(train_df)
    X_test, y_test, _ = preprocessor.prepare_traditional_ml(test_df)
    
    # Prepare data for LSTM
    print("Preparing data for LSTM...")
    X_train_lstm, y_train_lstm = preprocessor.prepare_sequences(train_df)
    X_test_lstm, y_test_lstm = preprocessor.prepare_sequences(test_df)
    
    # Split LSTM data
    X_train_lstm, X_val_lstm, y_train_lstm, y_val_lstm = train_test_split(
        X_train_lstm, y_train_lstm, test_size=0.2, random_state=42
    )
    
    # Initialize predictor
    predictor = GoldPricePredictor()
    
    # Train all models
    results = {}
    
    print("\n" + "="*50)
    print("Training Linear Regression...")
    results['Linear Regression'] = predictor.train_linear_regression(X_train, y_train, X_test, y_test)
    
    print("\nTraining Ridge Regression...")
    results['Ridge Regression'] = predictor.train_ridge_regression(X_train, y_train, X_test, y_test)
    
    print("\nTraining Random Forest...")
    results['Random Forest'] = predictor.train_random_forest(X_train, y_train, X_test, y_test)
    
    print("\nTraining Gradient Boosting...")
    results['Gradient Boosting'] = predictor.train_gradient_boosting(X_train, y_train, X_test, y_test)
    
    print("\nTraining SVR...")
    results['SVR'] = predictor.train_svr(X_train, y_train, X_test, y_test)
    
    print("\nTraining LSTM...")
    results['LSTM'] = predictor.train_lstm(X_train_lstm, y_train_lstm, X_val_lstm, y_val_lstm, epochs=50)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*50)
    print(f"{'Model':<25} {'MSE':<15} {'MAE':<15} {'R² Score':<15}")
    print("-"*50)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['mse']:<15.4f} {metrics['mae']:<15.4f} {metrics['r2']:<15.4f}")
    
    # Save models
    print("\nSaving models...")
    for model_name in predictor.models.keys():
        # Use correct extension per model type
        ext = ".keras" if model_name == "LSTM" else ".pkl"
        predictor.save_model(model_name, f"models/{model_name.lower().replace(' ', '_')}{ext}")
    
    # Plot predictions
    plot_predictions(results, y_test, y_val_lstm)
    
    return predictor, results

def plot_predictions(results, y_test_traditional, y_test_lstm):
    """Plot predictions vs actual values"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    model_names = ['Linear Regression', 'Ridge Regression', 'Random Forest', 
                   'Gradient Boosting', 'SVR', 'LSTM']
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        if model_name == 'LSTM':
            y_test = y_test_lstm
            y_pred = results[model_name]['predictions'].flatten()
        else:
            y_test = y_test_traditional
            y_pred = results[model_name]['predictions']
        
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Price')
        ax.set_ylabel('Predicted Price')
        ax.set_title(f'{model_name}\nR² = {results[model_name]["r2"]:.4f}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/predictions_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPredictions plot saved to results/predictions_comparison.png")
    plt.close()

if __name__ == "__main__":
    main()

