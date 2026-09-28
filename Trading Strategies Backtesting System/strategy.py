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
        # idea: subtraction is 0 (1-1, 0-0 --> higher - higher, lower - lower), no pos change --> hold
        # subtraction is 1 (1-0 --> higher - lower), pos is buy because price is up
        # subtraction is -1 (0-1--> higher - lower), pos is sell because price is down
        self.data['position'] = self.data['signal'].diff()

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

class RSIMeanReversion(Strategy):
    def __init__(self, asset_data, time_window=14):
        super().__init__(asset_data)
        self.interval = time_window

    def generate_signal(self):
        #SMA200 and SMA5 computation
        self.data['SMA200'] = self.data['Close'].rolling(window=200).mean()
        self.data['SMA5'] = self.data['Close'].rolling(window=5).mean()

        #daily changes for later isolation of daily gains and losses
        self.data['daily_changes'] = self.data['Close'].diff()

        #Writing gains and losses columns to calculate Relative Strength = gains/losses
        self.data['gains'] = np.where(self.data['daily_changes']>0, self.data['daily_changes'], 0)
        self.data['losses'] = np.where(self.data['daily_changes'] < 0, self.data['daily_changes']*-1, 0)

        #avg gains and losses computation
        self.data['avg_gains'] = self.data['gains'].rolling(window=self.interval).mean()
        self.data['avg_losses'] = self.data['losses'].rolling(window=self.interval).mean()

        #RSI computation
        self.data[f'RSI_{self.interval}'] = 100-(100/(1+(self.data['avg_gains']/self.data['avg_losses'])))

        conditions = [
            (self.data[f'RSI_{self.interval}'] < 10) & (self.data['SMA200'] < self.data['Close']),
        (self.data[f'RSI_{self.interval}']>70) & (self.data['SMA5'] > self.data['Close'])
        ]

        choices = [1,0]
        self.data['signal'] = np.select(conditions, choices, default=np.nan)
        self.data['signal'] = self.data['signal'].ffill()

