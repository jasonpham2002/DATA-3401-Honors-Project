# Trading strategies backtesting system

---
## Trading strategies
+ Dual SMA (20/50): 
  + When 20-day MA > 50-day MA, buy. Sell when the opposite happens.
+ Mean reversion RSI:
  + Entry: 200-SMA below price, 2-RSI < 30
  + Exit: 2-RSI > 70
### OOP Notice:
Each trading strategy will have its own subclass, from the main `Strategy` class. The base class will have the following methods as rudimentary:
+ Buy method (use technical indicators and/or models to help catch buying signals).
+ Sell method (use technical indicators and/or models to help catch selling signals).
+ Execute trade
+ Calculate return

The subclasses will each have signals trigger (buy, sell command)
#### Other classes
+ Backtesting engine
+ Data Pipeline
+ Performance metrics
+ Cross-validation
## EDA
Data source: yfinance
