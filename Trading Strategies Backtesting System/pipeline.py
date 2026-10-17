import pandas as pd
import yfinance as yf

class DataPipeline:
    """
    Data ingestion and cleaning pipeline for financial market backtesting.
    Fetches real-time market data from yfinance or loads cached CSV files.
    Applies input normalization, MultiIndex flattening, and forward-filling for continuous data.
    """
    def __init__(self, start_date, end_date, intervals="1d", ticker="SPY"):
        self.start_date = start_date
        self.end_date = end_date
        self.interval = intervals
        self.data = None

        if isinstance(ticker, str):  # Wrap single ticker string into a list
            self.ticker = [t.strip() for t in ticker.replace(',', ' ').split() if t.strip()]
        elif isinstance(ticker, list):
            self.ticker = ticker
        else:
            raise TypeError("Ticker must be a list or a string of tickers")

    @property
    def intervals(self):
        return self.interval

    @intervals.setter
    def intervals(self, value):
        self.interval = value

    def download_data(self):
        """Downloads data from yfinance with auto-adjusted prices for stock splits."""
        self.data = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            interval=self.interval,
            group_by='ticker',
            auto_adjust=True
        )
        if self.data is None or self.data.empty:
            raise ValueError(f"Data didn't download and is empty for the ticker '{self.ticker}'")

        self._normalize_columns()

        # Forward fill missing bars (e.g. holidays / halt periods)
        self.data = self.data.ffill().dropna(how='all')

        print(f"Successfully downloaded {len(self.data)} rows for {self.ticker}.")
        self.data.to_csv(f"{'_'.join(self.ticker)}_Market")
        return self.data

    def fetch_data(self, file_path):
        """Loads and parses cached historical price CSV files."""
        # Detect whether CSV was saved with a two-level MultiIndex header
        sample = pd.read_csv(file_path, nrows=2)
        if hasattr(file_path, 'seek'):
            file_path.seek(0)

        price_cols = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}
        first_row_vals = set(str(v) for v in sample.iloc[0].values) if len(sample) > 0 else set()
        is_multi_header = (
            sample.columns[0] in ('Price', 'Ticker')
            or (len(sample) > 0 and str(sample.iloc[0, 0]) in ('Price', 'Ticker'))
            or bool(price_cols.intersection(first_row_vals))
        )
        if is_multi_header:
            self.data = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
        else:
            self.data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        self._normalize_columns()
        self.data = self.data.ffill().dropna(how='all')
        return self.data

    def _normalize_columns(self):
        """Normalizes single-ticker MultiIndex DataFrames to standard OHLCV columns using df.xs()."""
        if isinstance(self.data.columns, pd.MultiIndex) and len(self.ticker) == 1:
            price_cols = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}
            level_0 = set(self.data.columns.get_level_values(0))
            ticker_level = 1 if price_cols.intersection(level_0) else 0
            level_tickers = self.data.columns.get_level_values(ticker_level)
            target_ticker = self.ticker[0] if self.ticker[0] in level_tickers else level_tickers[0]
            self.data = self.data.xs(target_ticker, axis=1, level=ticker_level).copy()
