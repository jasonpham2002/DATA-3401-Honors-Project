import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

class Strategy:
    def __init__(self):
        pass
    def Buy(self):
        pass
    def Sell(self):
        pass
    def Trade(self):
        pass
    def CalReturn(self):
        pass

class DataPipeline:
    def __init__(self, start_date, end_date, intervals, ticker):
        self.start_date = start_date
        self.end_date = end_date
        self.intervals = intervals

        self.data = None

        if isinstance(ticker, str):
            self.ticker = ticker.split()
        elif isinstance(ticker, list):
            self.ticker = ticker
        else:
            raise TypeError("Ticker must be a list or a string of tickers")


    def download_data(self):

        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval=self.intervals,group_by='ticker', auto_adjust=True)
        if self.data is None or self.data.empty:
            raise ValueError(f"Data didn't download and is empty for the ticker '{self.ticker}'")

        # MultiIndex checker
        if isinstance(self.data.columns, pd.MultiIndex) and len(self.ticker) == 1:
            self.data.columns = self.data.columns.droplevel(0)

        print(f"Successfully downloaded {len(self.data)} rows for {self.ticker}.")
        return self.data



