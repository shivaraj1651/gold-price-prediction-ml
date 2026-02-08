# gold-price-prediction-ml
Gold price prediction using 6 ML algorithms (Linear Regression, Ridge, Random Forest, Gradient Boosting, SVR, LSTM). Fetches data from Yahoo Finance, builds technical indicators, trains models, and shows predictions in INR per gram via Streamlit. Python, scikit-learn, TensorFlow, Streamlit.


# Gold Price Prediction using Machine Learning

Predict gold prices using 6 ML algorithms and view results in INR per gram via a Streamlit web app.

## Features
- Fetches gold price data from Yahoo Finance (10 years)
- 6 algorithms: Linear Regression, Ridge, Random Forest, Gradient Boosting, SVR, LSTM
- Technical indicators: RSI, MACD, Bollinger Bands, moving averages
- Predictions in USD/oz and INR per gram (live USD/INR)
- Streamlit UI with model selection and price chart

## Setup
pip install -r requirements.txt

Directories data/, models/, results/ are created automatically when you run the project.

How to Run
Backend (full pipeline: fetch → train → predict)

Runs data fetch, trains all 6 models, saves them, and prints a sample prediction (Linear Regression).
python main.py

Frontend (Streamlit UI)
Starts the web app. On first run it will fetch data and train models if needed. Then you can choose model, days ahead, and click Predict to see current vs predicted price (INR per gram), all-models table, and chart.

python -m streamlit run streamlit_app.py

Open the URL shown in the terminal (e.g. http://localhost:8501) in your browser.

Project Structure
Path	Role

main.py	Run
full backend: fetch + train + predict
streamlit_app.py	Run frontend (Streamlit UI)
src/data_collection.py	Fetch gold data from Yahoo Finance → data/gold_prices.csv
src/data_preprocessing.py	Feature engineering, train/test prep, LSTM sequences
src/models.py	Define and train all 6 models
src/train.py	Load data, split, train all models, save to models/
src/predict.py	Load model + data, predict, convert to INR per gram
data/	Gold price CSV
models/	Saved models (.pkl and lstm.keras)
results/	Plots (e.g. predictions comparison)

Tech Stack
Python, pandas, NumPy, scikit-learn, TensorFlow/Keras, Streamlit, yfinance
---The same content is saved in your project as **`README.md`**, so you can open that file and copy it, or use the block above. It’s one README that covers both **backend** (`python main.py`) and **frontend** (`python -m streamlit run streamlit_app.py`).
