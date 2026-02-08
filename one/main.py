from src.data_collection import fetch_gold_data
from src.train import main as train_models
from src.predict import predict_future_price

if __name__ == "__main__":
    print("Gold Price Prediction System")
    print("="*50)
    
    # Step 1: Fetch data
    print("\nStep 1: Fetching gold price data...")
    fetch_gold_data(period="10y")
    
    # Step 2: Train models
    print("\nStep 2: Training models...")
    predictor, results = train_models()
    
    # Step 3: Make prediction
    print("\nStep 3: Making prediction...")
    predict_future_price(model_name='Linear Regression', days_ahead=1)

