# Quantitative Trading Backtesting System (Project 1)

This directory contains the Object-Oriented Quantitative Trading Backtesting Engine and the integrated research notebook (`TradingNotebook.ipynb`) for the DATA 3401 Honors Project.

---

## Module Architecture

### 1. `pipeline.py` (`DataPipeline`)
- Downloads split-adjusted historical OHLCV data via `yfinance` (`auto_adjust=True`) or loads cached CSV data (`SPY_5_years`).
- Normalizes ticker inputs (`str` or `list`) using `isinstance()` and exposes `self.interval` / `self.intervals`.
- Automatically detects and extracts single-ticker cross-sections from MultiIndex columns using `df.xs()`, followed by forward-filling (`.ffill()`).

### 2. `strategy.py` (`Strategy` Hierarchy)
All trading strategies inherit from the `Strategy` base class, which defines `generate_signal()`, `execute_trade()`, and `calculate_returns()` / `calculate_return()`:
- **`MACrossover`**: Dual Simple Moving Average crossover strategy (`short_window=20`, `long_window=50`). Triggers a Buy signal (`1`) when `SMA_20 > SMA_50` and exits to cash (`0`) when `SMA_20 <= SMA_50`.
- **`RSIMeanReversion`**: Manually calculates the 14-day Relative Strength Index (RSI) using NumPy arrays (`np.where`). Triggers a Buy signal (`1`) when `RSI < 30` (oversold) and exits to cash (`0`) when `RSI > 70` (overbought), holding the prior position state in between.
- **`MLHybridStrategy`**: Integrates the Project 2 NumPy `LogisticRegression` classifier as a regime filter. Allows long exposure (`1`) only when technical trend conditions are active (`MA Crossover == 1` or `Close > SMA_20`) **and** the ML classifier independently predicts a Bullish (`1`) market regime.

### 3. `backtester.py` (`Backtest`)
- Simulates daily trading execution over historical price series starting with `$100,000` cash.
- Logs each Buy and Sell transaction (`Action`, `Shares`, `Price`, `Profit`) in `self.trade_history` and tracks daily mark-to-market portfolio value in `self.tot_net_worth`.
- **`cal_metrics()`**: Computes **Total Return**, **Annualized Sharpe Ratio** ($\sqrt{252}$ scaling), **Max Drawdown**, **Win Rate**, **Profit Factor**, and the **Buy & Hold Benchmark Return**.
- **`plot_result()` & `plot_drawdown()`**: Generates portfolio equity curves, price charts with green trade-position shading, and drawdown severity curves.

### 4. `TradingNotebook.ipynb`
End-to-end Jupyter Notebook containing:
- **Chapter 1**: Data Pipeline & Exploratory Data Analysis (5-year `SPY` price, 20/50 SMAs, and return distribution).
- **Chapter 2**: Project 1 Standalone Strategy Backtesting (`MACrossover` and `RSIMeanReversion`).
- **Chapter 3**: Project 2 ML Market Regime Classifier (feature engineering, training loss convergence, feature importance, confusion matrix, and regime prediction overlay).
- **Chapter 4**: System Integration (`MLHybridStrategy` backtest).
- **Chapter 5**: Comprehensive Comparative Analysis & Written Research Report.
