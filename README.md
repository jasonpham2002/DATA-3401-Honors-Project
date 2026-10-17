# DATA 3401 Honors Project — Quantitative Trading Backtester & ML Market Regime Classifier

This repository contains the complete implementation for the **DATA 3401 (Python for Data Science 1)** Honors Contract Project. It connects an Object-Oriented Quantitative Trading Backtesting System (**Project 1**) with a from-scratch NumPy Logistic Regression Market Regime Classifier (**Project 2**) to evaluate whether machine learning regime filtering improves risk-adjusted trading performance on the **SPDR S&P 500 ETF Trust (`SPY`)**.

---

## Repository Structure

```text
DATA-3401-Honors-Project/
├── Market_Regime_Prediction/                # Project 2: ML Market Regime Classifier
│   ├── preprocessing_pipeline.py            # Feature engineering, target labeling, zero-leakage scaling
│   └── LogisticRegression.py                # Pure-NumPy Logistic Regression (sigmoid, BCE loss, GD, metrics)
├── Trading Strategies Backtesting System/   # Project 1: Quantitative Trading Backtester
│   ├── pipeline.py                          # yfinance / CSV ingestion, MultiIndex normalization (df.xs)
│   ├── strategy.py                          # OOP Strategy hierarchy (MACrossover, RSIMeanReversion, MLHybridStrategy)
│   ├── backtester.py                        # Daily portfolio simulation, performance metrics & visualizations
│   ├── SPY_5_years                          # Cached 5-year daily SPY dataset (1,254 trading days)
│   ├── TradingNotebook.ipynb                # End-to-end execution, visualizations, and written research report
│   └── README.md                            # Project 1 module documentation
└── requirements.txt                         # Core Python dependencies
```

---

## Project Overview

### Project 1: Quantitative Trading Backtester (`Trading Strategies Backtesting System/`)
- **`DataPipeline` (`pipeline.py`)**: Fetches split-adjusted historical market data from `yfinance` or loads cached CSV files (`SPY_5_years`), normalizes single-string or list ticker inputs with `isinstance()`, extracts single-ticker cross-sections from MultiIndex DataFrames via `df.xs()`, and forward-fills missing bars.
- **`Strategy` Hierarchy (`strategy.py`)**:
  - `Strategy`: Abstract base class defining `generate_signal()`, `execute_trade()`, and `calculate_returns()` / `calculate_return()`.
  - `MACrossover`: Generates buy signals (`1`) when the 20-day Simple Moving Average crosses above the 50-day Simple Moving Average (`SMA_20 > SMA_50`), and exits (`0`) otherwise.
  - `RSIMeanReversion`: Computes the 14-day Relative Strength Index (RSI) from scratch in NumPy, entering long positions when `RSI < 30` (oversold) and exiting to cash when `RSI > 70` (overbought).
  - `MLHybridStrategy`: Combines technical trend signals (`MA Crossover == 1` or `Close > SMA_20`) with the Project 2 ML classifier, allowing market exposure only when the ML model independently predicts a Bullish (`1`) regime.
- **`Backtest` Engine (`backtester.py`)**: Simulates day-by-day execution with `$100,000` initial capital, logs every buy/sell transaction, tracks daily mark-to-market net worth, computes **Total Return**, **Annualized Sharpe Ratio**, **Max Drawdown**, **Win Rate**, and **Profit Factor**, and plots equity, signal-shaded price, and drawdown curves.

### Project 2: ML Market Regime Classifier (`Market_Regime_Prediction/`)
- **`Pipeline` (`preprocessing_pipeline.py`)**: Engineers four predictive technical features (`ret_5d`, `vol_20d`, `rsi_14`, `sma50_ratio`), constructs a 5-day forward regime target (`Close[t+5] > SMA_50[t]`), and performs chronological 80/20 train/test splitting with zero-leakage Z-score standardization fit strictly on training data.
- **`LogisticRegression` (`LogisticRegression.py`)**: Built **100% from scratch in NumPy** (no `scikit-learn` or `pandas` dependencies), implementing numerically stable `sigmoid_()`, class-weighted Binary Cross-Entropy `compute_cost()`, vectorized `compute_gradient()`, batch `gradient_descent()`, 3-day pure-NumPy probability smoothing, and evaluation metrics (`Accuracy`, `Precision`, `Recall`, `F1-Score`, `Confusion Matrix`, and `Baseline (Always Bullish)`).

---

## Key Empirical Results (Out-of-Sample Test Period: 240 Trading Days)

### 1. Project 2: ML Classifier Out-of-Sample Performance
| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **83.75%** |
| **Baseline (Always Bullish)** | 80.83% |
| **Precision** | 86.05% |
| **Recall** | 95.36% |
| **F1-Score** | 90.46% |

### 2. Project 1 + 2: Out-of-Sample Backtest Comparison (`$100,000` Initial Capital)
| Strategy | Total Return | Sharpe Ratio | Max Drawdown | Win Rate | Profit Factor | Total Trades |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Buy & Hold Benchmark (SPY)** | 17.76% | 1.39 | 8.88% | — | — | 1 |
| **Dual MA Crossover (20/50)** | 13.72% | 1.27 | 5.05% | 100.00% | $\infty$ | 2 |
| **RSI Mean Reversion (14-d, 30/70)** | 8.92% | 1.50 | 4.55% | 100.00% | $\infty$ | 2 |
| **ML Hybrid Strategy (MA + Regime Filter)** | **16.96%** | **1.54** | **4.47%** | **100.00%** | **$\infty$** | **3** |

---

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch Jupyter and open [`Trading Strategies Backtesting System/TradingNotebook.ipynb`](Trading%20Strategies%20Backtesting%20System/TradingNotebook.ipynb) to view and execute all code, charts, and the five-chapter comparative analysis.