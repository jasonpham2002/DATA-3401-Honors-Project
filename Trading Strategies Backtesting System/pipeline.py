import pandas as pd
import yfinance as yf

class DataPipeline:
    def __init__(self, start_date, end_date, intervals, ticker):
        self.start_date = start_date
        self.end_date = end_date
        self.intervals = intervals

        self.data = None

        if isinstance(ticker, str): #Checking if the input is a string (not list)
            self.ticker = ticker.split() #convert string into a single-list
        elif isinstance(ticker, list): #if the input is a list, then proceed normally
            self.ticker = ticker
        else:
            raise TypeError("Ticker must be a list or a string of tickers")

    def download_data(self):
        # Download data from yfinance/ error handling
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date, interval=self.intervals,
                                group_by='ticker', auto_adjust=True)
        if self.data is None or self.data.empty:
            raise ValueError(f"Data didn't download and is empty for the ticker '{self.ticker}'")

        # MultiIndex checker
        if isinstance(self.data.columns, pd.MultiIndex) and len(self.ticker) == 1:
            self.data.columns = self.data.columns.droplevel(0)

        self.data = self.data.ffill()  # foward fill NaN values for when the asset's market stops for a period.

        print(f"Successfully downloaded {len(self.data)} rows for {self.ticker}.")
        self.data.to_csv(f"{"_".join(self.ticker)}_Market")

        return self.data

    def fetch_data(self, file_path):
        self.data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return self.data

    def features_init(self):
        self.data['5D_Return'] = self.data['Close'].pct_change(period=5)
