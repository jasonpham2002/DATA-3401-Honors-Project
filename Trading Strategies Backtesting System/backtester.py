import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from pipeline import DataPipeline
import numpy as np
from strategy import Strategy, MACrossover, RSIMeanReversion
class Backtest:
    def __init__(self, asset_data,strategy,starting_cash=100000):
        self.current_cash = starting_cash
        self.trade_history = []
        self.cur_own_stocks = 0
        self.returns = 0
        self.data = asset_data
        self.strat = strategy

    def run(self):
        self.strat.generate_signal()
        self.strat.execute_trade()

        for index, row in self.data.iterrows():
            if row['position'] == 1:
                self.cur_own_stocks = self.current_cash // self.data['Close']





