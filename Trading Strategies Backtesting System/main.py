import yfinance as yf
from datetime import datetime, timedelta
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
    def __init__(self, startdate, enddate, intervals, ticker):
        self.startdate = startdate
        self.enddate = enddate
        self.intervals = intervals
        self.ticker = ticker
        self.data = None


    def downloaddata(self):
        self.data = yf.download(self.ticker, start=self.startdate, end=self.enddate, interval=self.intervals, auto_adjust=True)
        if self.data is None or self.data.empty:
            raise ValueError(f"Data didn't download and is empty for the ticker '{self.ticker}'")

        print(f"Successfully downloaded {len(self.data)} rows for {self.ticker}.")
        return self.data

