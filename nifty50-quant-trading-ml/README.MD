# 📈 NIFTY 50 Weekly Directional Prediction & Backtesting Model

An end-to-end quantitative machine learning pipeline that uses an engineered Random Forest Classifier to predict week-over-week directional movements of India's benchmark index (NIFTY 50). The project translates predictive signals into a vectorized compounding trading strategy.

---

## 🚀 Key Performance Results

* **Out-of-Sample Predictive Accuracy:** **59.18%**
* **Strategy Cumulative Return:** **39.42%** over a 196-week unseen testing window.
* **Benchmark Outperformance:** Successfully beat both the unoptimized baseline model (**21.14%**) and the underlying passive **NIFTY 50 Buy & Hold** index strategy (**38.63%**).

---

## 🛠️ Technical Workflow & Machine Learning Pipeline

### 1. Feature Engineering & Lag Analysis
* Transformed raw price data into **Percentage Returns (`.pct_change()`)** to normalize historical price scales and evaluate week-over-week performance.
* Engineered a sequence of **historical return lags (`Lag_1`, `Lag_2`, etc.)** to create a rolling window of recent market history. This allowed the Random Forest to analyze immediate weekly momentum and short-term trends without introducing lookahead bias.

### 2. Rigorous Evaluation Protocol (No Lookahead Bias)
* **Chronological Splitting:** Avoided standard random train/test splits, which corrupt time-series data via data leakage. Instead, used absolute chronological slicing to ensure the model only trained on past data to predict future weeks.
* **Hyperparameter Optimization:** Utilized `GridSearchCV` to exhaustively tune structural parameters.

### 3. Variance Reduction & Regularization
* The structural turning point in out-of-sample performance was constraining tree depth (`max_depth=3`). Capping tree growth intentionally introduced bias to drastically lower model variance. This design choice prevented the classifier from overfitting to short-term market noise and forced it to map long-term structural momentum.

---

## 📊 Quantitative Backtest Framework

The predictive signals were fed into a custom, vectorized backtesting engine with the following properties:
* **Binary Execution Engine:** Strong buy signals coordinate a 100% allocation to the index, while short/neutral signals rotate the capital entirely into risk-free cash.
* **Compounding Math:** Assumes 100% geometric compounding of weekly returns ($1 + R_t$).

> ⚠️ **Live Execution Note:** Achieving the backtest's performance profile in production requires long-term execution adherence across hundreds of iterations to allow the 59.18% statistical edge to normalize against short-term drawdowns.

---

## 🗂️ Project Structure & Reproduction
* **Environment:** Developed and executed inside Google Colab.
* **Core Stack:** `pandas`, `numpy`, `scikit-learn` (`RandomForestClassifier`, `GridSearchCV`), `matplotlib`, `seaborn`, `yfinance`.
* **Primary Notebook:** `NIFTY50_Directional_Prediction.ipynb`
