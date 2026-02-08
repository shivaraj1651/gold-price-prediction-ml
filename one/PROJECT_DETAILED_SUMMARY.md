# Gold Price Prediction – Detailed Summary, Algorithms & Accuracy

This document explains **how the project works**, **how many algorithms** are used, **what each algorithm does**, and **how accuracy is measured** for each.

---

## 1. How the Project Works (Step by Step)

### Step 1: Data collection
- The app downloads **gold futures price** (USD per troy ounce) from **Yahoo Finance** (symbol `GC=F`).
- It fetches about **10 years** of daily data (Open, High, Low, Close, Volume) and saves it as `data/gold_prices.csv`.

### Step 2: Feature engineering
- From raw OHLCV we build **technical indicators**: moving averages (SMA 7/30, EMA 12/26), **RSI**, **MACD**, **Bollinger Bands**, volume ratio, price change, etc.
- For **traditional ML** (Linear, Ridge, RF, GB, SVR): we add **lag features** (Close 1, 2, 3, 5, 7, 14, 30 days ago) and set **target** = Close price N days ahead. Features are scaled; target is kept in USD/oz.
- For **LSTM**: we build **sequences** of the last **60 days** of (scaled) prices and indicators; target = next day’s scaled Close.

### Step 3: Train–test split
- Data is split **by time**: first **80%** for **training**, last **20%** for **testing** (no shuffling).
- So we train on roughly 8 years and evaluate on roughly 2 years.

### Step 4: Training
- **Five traditional models** are trained on the flat table (X = features, y = future Close).
- **One LSTM** is trained on 60-day sequences (input = 60 timesteps, output = next day Close).
- For each model we compute **MSE**, **MAE**, and **R²** on the test/validation set and save the model.

### Step 5: Prediction
- User picks a **model** and **days ahead** (e.g. 1 or 7).
- We load the latest data, build the same features, and pass the **last** sample (or last 60-day window for LSTM) to the chosen model.
- Model outputs **predicted Close in USD/oz**. We then convert to **USD per gram** (÷ 31.103) and **INR per gram** (× live USD/INR rate).

### Step 6: Frontend (Streamlit)
- User can **fetch data**, **train models** (if not already done), then **select model** and **days ahead** and click **Predict**.
- The app shows **current vs predicted price (INR per gram)**, an **“All Models”** comparison table, and a **recent price chart**.

---

## 2. How Many Algorithms Are Used?

**Total: 6 algorithms.**

| # | Algorithm           | Type              | Used for prediction of |
|---|--------------------|-------------------|-------------------------|
| 1 | Linear Regression  | Traditional ML    | Next day(s) Close price |
| 2 | Ridge Regression   | Traditional ML    | Next day(s) Close price |
| 3 | Random Forest      | Traditional ML    | Next day(s) Close price |
| 4 | Gradient Boosting  | Traditional ML    | Next day(s) Close price |
| 5 | SVR                | Traditional ML    | Next day(s) Close price |
| 6 | LSTM               | Deep learning     | Next day Close price   |

All predict the **same thing** (future gold Close price in USD/oz); we then convert to INR per gram in the app.

---

## 3. Brief Explanation of Each Algorithm

### 1. Linear Regression
- **Idea**: Fits a **linear equation** (weighted sum of features + intercept) to predict future Close.
- **Pros**: Simple, fast, stable; often gives realistic predictions (no negative prices).
- **Cons**: Can only capture linear relationships; may miss complex patterns.

### 2. Ridge Regression
- **Idea**: Same as Linear Regression but adds an **L2 penalty** on the weights to reduce overfitting.
- **Pros**: More stable than plain linear regression when we have many features; still gives sensible predictions.
- **Cons**: Still linear; may underfit if the relationship is very nonlinear.

### 3. Random Forest
- **Idea**: Builds many **decision trees** on random subsets of data and features; prediction = average of all trees.
- **Pros**: Can capture nonlinear patterns; no need to scale features for trees.
- **Cons**: On this gold dataset it often **extrapolates poorly** (test period has higher prices), so it can give **unrealistic or negative** predicted prices.

### 4. Gradient Boosting
- **Idea**: Builds **trees one by one**; each new tree corrects the errors of the previous ones.
- **Pros**: Often strong on tabular data.
- **Cons**: Same as Random Forest here: can give **unrealistic predictions** on the test period (high/negative values).

### 5. SVR (Support Vector Regression)
- **Idea**: Finds a **tube** around the data (epsilon) and fits a function that keeps most points inside the tube.
- **Pros**: Works in high dimensions; uses kernel (e.g. RBF) for nonlinearity.
- **Cons**: On this project it often gives **very bad or negative** predictions on the test set; sensitive to scaling and parameters.

### 6. LSTM (Long Short-Term Memory)
- **Idea**: **Recurrent neural network** that reads a **sequence** of 60 days and predicts the next day’s Close.
- **Pros**: Designed for time series; can learn temporal patterns; in our runs it often gets **high R²** on validation.
- **Cons**: Needs more data and tuning; slower to train; results depend on how the sequence data is prepared.

---

## 4. How Accuracy Is Measured (MSE, MAE, R²)

For **regression** we do **not** use “classification accuracy”. We use three metrics:

| Metric | Full name              | Meaning |
|--------|------------------------|--------|
| **MSE** | Mean Squared Error     | Average of (actual − predicted)². Lower is better. In USD². |
| **MAE** | Mean Absolute Error    | Average of \|actual − predicted\|. Lower is better. In USD. |
| **R²**  | R-squared              | How much of the variation in actual prices is explained by the model. 1 = perfect, 0 = no better than mean, &lt; 0 = worse than mean. |

- **Good model**: **Low MSE, low MAE, R² close to 1** (or at least positive and high).
- **Bad model**: **High MSE, high MAE, R² negative or very low**.

When you run **`python main.py`** (or **`python src/train.py`**), the script prints a **MODEL PERFORMANCE SUMMARY** table with **MSE**, **MAE**, and **R²** for each of the 6 algorithms on the **test/validation** set.

---

## 5. Typical Accuracy (What You Might See)

Actual numbers depend on your data and run. In general in this project:

- **Linear Regression & Ridge**: Often **reasonable R²** (e.g. ~0.99 on some splits) and **realistic predictions** (close to current price). **MSE/MAE** can be in the thousands because gold is in thousands of USD/oz.
- **Random Forest, Gradient Boosting, SVR**: Often **negative R²** and **very high MSE/MAE** on the test set, and sometimes **negative or unrealistic** predicted prices. So by these metrics they perform **worse** on this task.
- **LSTM**: Often **high R²** (e.g. ~0.99) and **low MSE/MAE** on the **validation** set (because it sees sequences from a similar time range). So by the printed metrics it can look **best**.

So:
- **For “accuracy” in terms of R²/MSE/MAE**: LSTM and Linear/Ridge often look good; RF, GB, SVR often look bad on the test set.
- **For “realistic” predictions (e.g. for reports or UI)**: We **default to Linear Regression** and show an “All Models” table so you can compare.

---

## 6. Where to See the Exact Accuracy of Each Algorithm

1. Run the full pipeline:
   ```bash
   cd "F:\3rd sem MCA Notes\one"
   python main.py
   ```
2. In the console, after training, look for the block:
   ```text
   MODEL PERFORMANCE SUMMARY
   Model                     MSE             MAE             R² Score
   --------------------------------------------------
   Linear Regression         ...             ...             ...
   Ridge Regression          ...             ...             ...
   ...
   LSTM                      ...             ...             ...
   ```
   Those three columns are the **accuracy (performance)** of each algorithm on the test/validation set.

You can copy that table into your report or presentation and say: “We measured performance using MSE, MAE, and R²; here are the results for each of the 6 algorithms.”

---

## 7. Short Summary Table (for Report/Presentation)

| Algorithm          | Type        | Brief idea                    | Typical role in this project        |
|--------------------|------------|-------------------------------|-------------------------------------|
| Linear Regression  | Traditional| Linear combination of features| Stable, realistic predictions; default in app. |
| Ridge Regression  | Traditional| Linear + L2 penalty           | Same; often similar or slightly better than Linear. |
| Random Forest     | Traditional| Many trees, average           | Often poor R² and unrealistic predictions here. |
| Gradient Boosting | Traditional| Sequential correction by trees| Same as RF in this project.         |
| SVR               | Traditional| Support vectors + kernel      | Often worst R² and very unrealistic here. |
| LSTM              | Deep learning | 60-day sequence → next day | Often best R² on validation; can be used for comparison. |

**Accuracy**: For each algorithm, **MSE, MAE, and R²** are printed in the **MODEL PERFORMANCE SUMMARY** when you run **`python main.py`**. Use that table as the “accuracy of each algorithm” in your report.
