# Gold Price Prediction Project – Detailed Explanation

This document explains **how the project works end-to-end**: data flow, preprocessing, models, training, prediction, and the frontend.

---

## 1. High-Level Flow

```
Fetch data (Yahoo) → Save CSV → Load & create features → Split train/test
       → Train 6 models (Linear, Ridge, RF, GB, SVR, LSTM) → Save models
       → Predict (selected model) → Convert USD→INR, oz→gram → Show in UI/CLI
```

- **Backend**: `main.py` runs fetch → train → predict (default: Linear Regression).
- **Frontend**: Streamlit app can auto fetch/train on first run, then lets you choose model, days ahead, and click **Predict** to see result + “All Models” table + chart.

---

## 2. Data Collection (`src/data_collection.py`)

**What it does**
- Uses **Yahoo Finance** (`yfinance`) to download historical gold price data.
- **Symbol**: `GC=F` (gold futures in **USD per troy ounce**).
- **Period**: e.g. 10 years (`period="10y"`).

**Output**
- Saves a CSV: `data/gold_prices.csv`.
- Columns: **Date (index), Open, High, Low, Close, Volume** (and sometimes Dividends, Stock Splits).

**Why this data**
- Close price is the main series we want to predict.
- Volume and OHLC are used later to build features.

---

## 3. Feature Engineering (`src/data_preprocessing.py`)

The class `GoldDataPreprocessor` does two kinds of things: (1) build a rich feature set from raw OHLCV, and (2) prepare two different formats for the two types of models.

### 3.1 Creating Features (`create_features`)

From the raw dataframe it adds:

| Feature | Meaning |
|--------|---------|
| **SMA_7, SMA_30** | Simple moving average of Close over 7 and 30 days |
| **EMA_12, EMA_26** | Exponential moving average (12 and 26 days) |
| **RSI** | Relative Strength Index (14-day): momentum, 0–100 |
| **MACD** | EMA_12 − EMA_26 |
| **MACD_signal** | 9-day EMA of MACD |
| **BB_middle, BB_upper, BB_lower** | Bollinger Bands (20-day mean ± 2× std) |
| **Volume_SMA, Volume_Ratio** | 20-day average volume and ratio vs that average |
| **Price_Change** | Day-to-day % change in Close |
| **High_Low_Ratio** | High / Low for the day |

Then it drops rows with NaN (from rolling windows), so we get a clean table where each row is one day with all indicators.

**Why**
- Gives the model more signal than “just past closes”: trend (moving averages), momentum (RSI, MACD), volatility (Bollinger), and volume.

### 3.2 Two Ways to Prepare Data for Models

We have **two model families**, so we prepare data in **two formats**.

---

#### A) Traditional ML format (`prepare_traditional_ml`)

**Used for**: Linear Regression, Ridge, Random Forest, Gradient Boosting, SVR.

**Steps**
1. **Lags**: For each day, add past Close prices:  
   `Close_lag_1`, `Close_lag_2`, `Close_lag_3`, `Close_lag_5`, `Close_lag_7`, `Close_lag_14`, `Close_lag_30`.
2. **Target**: For “predict N days ahead”,  
   `Target = Close.shift(-forecast_days)`  
   So for each row, target = Close price N days later.
3. **Features**: All columns except `Target` (and Date/Unnamed): lags + OHLC + Volume + all the technical indicators above.
4. **Drop NaN**: Rows with missing lags or target are removed.
5. **Scale**: `MinMaxScaler` is fit on the feature matrix and transforms it. **Target (y) is not scaled** – we predict raw USD/oz.

**Shape**
- **X**: `(number_of_samples, number_of_features)` — one row per day, many columns.
- **y**: `(number_of_samples,)` — one future Close price per row.

So the model learns: “given today’s lags + indicators, predict Close N days from now.”

---

#### B) LSTM format (`prepare_sequences`)

**Used for**: LSTM (time-series neural network).

**Steps**
1. **Feature set**: A fixed list of columns (Close, Open, High, Low, Volume + technical indicators). Only existing columns are used.
2. **Scale**: The same set of columns is scaled with `MinMaxScaler` (each column 0–1).
3. **Sequences**: For each time index `i` (from `lookback` to end):
   - **X[i]** = last `lookback` days (e.g. 60) of scaled values → shape `(lookback, number_of_features)`.
   - **y[i]** = scaled Close at time `i` (i.e. “next day” relative to the window).

**Shape**
- **X**: `(samples, lookback, features)` — e.g. (N, 60, 18).
- **y**: `(samples,)` — scaled Close for the next day.

So the LSTM learns: “given the last 60 days of (scaled) prices and indicators, predict the next day’s (scaled) Close.” At prediction time we inverse-transform that scaled value back to USD/oz.

---

## 4. Training Pipeline (`src/train.py`)

**Steps**
1. Load `data/gold_prices.csv` and run `create_features`.
2. **Time-based split**: first 80% = train, last 20% = test (no shuffling, to respect time order).
3. **Traditional ML**:
   - Call `prepare_traditional_ml` on train and on test (each gets its own scaler fit on that subset in the current code; test uses recent dates).
   - Get `X_train, y_train`, `X_test, y_test`.
4. **LSTM**:
   - Call `prepare_sequences` on train and test.
   - Split train into train/validation (e.g. 80/20) for early stopping / monitoring.
5. **Train 6 models** (in `src/models.py`):
   - **Linear Regression**: ordinary least squares.
   - **Ridge Regression**: linear with L2 penalty (`alpha=1`).
   - **Random Forest**: 100 trees.
   - **Gradient Boosting**: 100 trees, sequential.
   - **SVR**: Support Vector Regression (RBF kernel).
   - **LSTM**: 3 LSTM layers (50 units) + Dropout, then Dense(1), MSE loss, Adam optimizer, 50 epochs.
6. **Evaluate**: MSE, MAE, R² on test/validation.
7. **Save**:  
   - Traditional models → `models/<name>.pkl` (joblib).  
   - LSTM → `models/lstm.keras`.

**Why Linear Regression / Ridge often “look correct”**
- They predict a linear combination of features; with lags and indicators they tend to stay near the recent price range.
- Tree models (RF, GB) and SVR can overfit or extrapolate poorly on this setup, sometimes giving negative or unrealistic prices; the app defaults to **Linear Regression** and shows an “All Models” table so you can compare.

---

## 5. Prediction (`src/predict.py` and Streamlit)

**Inputs**
- **Model name** (e.g. Linear Regression, LSTM).
- **Days ahead** (e.g. 1 or 7).

**Steps**
1. Load the same CSV, run `create_features`.
2. **If LSTM**:
   - Load `models/lstm.keras`.
   - Build sequences with `prepare_sequences`, take the **last** sequence `X[-1:]`.
   - Model outputs scaled next-day Close; put it in a dummy vector and **inverse-transform** with the same scaler → get **prediction in USD/oz**.
3. **If traditional ML**:
   - Load the right `.pkl` model.
   - Call `prepare_traditional_ml` with the chosen `forecast_days`; take the **last** row `X[-1:]`.
   - Model output is already in **USD/oz** (target was never scaled).
4. **Current price**: last row’s `Close` in the preprocessed dataframe (USD/oz).
5. **Convert to per gram**:  
   `price_per_gram = price_per_oz / 31.1034768` (troy oz to grams).
6. **Convert to INR**: Fetch latest **USD/INR** from Yahoo (`USDINR=X`), then  
   `price_inr = price_usd * usd_inr`.

**Output**
- Current and predicted price in:
  - USD/oz and USD/g
  - INR/oz and INR/g (if USD/INR is available)
- Same logic is used in the Streamlit app to show the main “Prediction Result” and to fill the “All Models - Prediction Table”.

---

## 6. Frontend – Streamlit (`streamlit_app.py`)

**On first run**
- If `data/gold_prices.csv` is missing → call `fetch_gold_data`.
- If any of the saved models are missing → call `train_models()` so all 6 models exist.

**UI**
- Dropdown: **model** (default: Linear Regression).
- Slider: **days ahead** (1–7).
- Button: **Predict**.

**When you click Predict**
1. **Selected model**: `compute_prediction(model_name, days_ahead)` runs as above and returns current/predicted USD/oz, USD/g, INR/g, history, etc.
2. **All models**: `compute_all_models_predictions(days_ahead)` runs prediction for all 6 models (reusing one traditional-ML feature vector for non-LSTM, and a separate LSTM preprocessor for LSTM) and builds a table: Model, Predicted_USD_oz, Predicted_USD_g, Predicted_INR_g.
3. **Display**:
   - “Prediction Result (for selected model)” with model name and days ahead.
   - Main metrics: **current and predicted price (INR per gram)** and delta.
   - Details: USD/oz, USD/g, USD/INR rate.
   - **“All Models - Prediction Table”** (like your R Shiny app).
   - **Recent gold price history**: line chart of last ~180 days (USD/oz).

**Styling**
- Gold-themed background image, glass-style card, gold/orange button so the app looks consistent and the Predict button is clear.

---

## 7. How to Run the Whole Project

**One command (backend: fetch + train + predict with Linear Regression)**
```bash
cd "F:\3rd sem MCA Notes\one"
python main.py
```

**One command (frontend: fetch/train if needed, then open UI)**
```bash
cd "F:\3rd sem MCA Notes\one"
python -m streamlit run streamlit_app.py
```
Then in the browser: choose model (e.g. Linear Regression), set days ahead, click **Predict**, and scroll to see result, table, and chart.

---

## 8. File and Data Flow Summary

| File / Folder | Role |
|---------------|------|
| `src/data_collection.py` | Fetch gold (GC=F) from Yahoo → `data/gold_prices.csv` |
| `src/data_preprocessing.py` | Load CSV, create features, prepare traditional-ML (X,y) and LSTM (sequences) |
| `src/models.py` | Define and train Linear, Ridge, RF, GB, SVR, LSTM; save/load |
| `src/train.py` | Load data → split → prepare both formats → train all 6 → save → optional plot |
| `src/predict.py` | Load data + model → predict USD/oz → convert to USD/g and INR/g → print or return |
| `streamlit_app.py` | First-run fetch/train, then UI: model choice, Predict, result + table + chart |
| `main.py` | Orchestrate: fetch → train → predict (default Linear Regression) |
| `data/gold_prices.csv` | Input time series (OHLCV) |
| `models/*.pkl`, `models/lstm.keras` | Trained models used at prediction time |

---

## 9. Summary

- **Data**: Gold futures in USD/oz from Yahoo; we add technical indicators and lags.
- **Models**: Six algorithms (Linear, Ridge, RF, GB, SVR, LSTM) trained on two data layouts (flat table vs sequences).
- **Prediction**: One selected model predicts future Close in USD/oz; we convert to USD/g and INR/g using live USD/INR.
- **Default**: Linear Regression is used by default so you get stable, realistic-looking predictions; the “All Models” table lets you compare and see that some models (e.g. RF, SVR) can give unrealistic values on this setup.
- **Frontend**: Streamlit runs the full pipeline (fetch/train if needed) and shows result, all-model table, and history chart in a single flow.

This is how the project works in detail from data to prediction and UI.
