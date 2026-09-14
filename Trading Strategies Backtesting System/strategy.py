import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from pipeline import DataPipeline
import numpy as np

class Strategy:
    def __init__(self, asset_data):
        self.starting_cash = 100000
        self.current_cash = 100000
        self.trade_history = []
        self.position = 0
        self.data = asset_data
    def buy(self):
        pass
    def sell(self):
        pass
    def execute_trade(self):
        pass
    def calculate_return(self):
        self.data['asset_returns'] = self.data['Close'].pct_change()
        self.data['strategy_returns'] = self.data['asset_returns'] * self.data['signal'].shift(1) #calculate the returns for the money you spend invested
        # 1 is invested, 0 is sitting in cash


    def generate_signal(self):
        pass

class MACrossover(Strategy):
    def __init__(self, asset_data, short_window=20, long_window=50):
        super().__init__(asset_data)
        self.short_window=short_window
        self.long_window=long_window

    def generate_signal(self):
        self.data['short_average_close'] = self.data['Close'].rolling(window=self.short_window).mean()
        self.data['long_average_close'] = self.data['Close'].rolling(window=self.long_window).mean()

        #when ma_short > ma_long --> 1, otherwise 0
        self.data['signal'] = np.where(self.data['short_average_close']>self.data['long_average_close'],1,0)


    def execute_trade(self):
        # idea: subtraction is 0 (1-1, 0-0 --> higher - higher, lower - lower), no pos change --> hold
        # subtraction is 1 (1-0 --> higher - lower), pos is buy because price is up
        # subtraction is -1 (0-1--> higher - lower), pos is buy because price is down
        self.data['position'] = self.data['signal'].diff()


class RSIMeanReversion(Strategy):
    def __init__(self, asset_data):
        super().__init__(asset_data)

    def generate_signal(self):
        self.data['SMA200'] = self.data['Close'].rolling(window=200).meand()
        RS = self.data['Close'].pct_change().rolling(window=)