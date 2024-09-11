import yfinance as yf
import pandas as pd
from typing import Optional

class StockDataDownloader:
    def __init__(self, ticker: str):
        """
        Initialize the StockDataDownloader with a stock ticker symbol.
        
        Args:
        ticker (str): The ticker symbol of the stock (e.g., '7203.T' for Toyota).
        """
        self.ticker = ticker
        self.data = pd.DataFrame()

    def download_data(self, start_date: str, end_date: str, interval: str = '1d') -> None:
        """
        Download historical stock data from Yahoo Finance.
        
        Args:
        start_date (str): The start date for the historical data (format 'YYYY-MM-DD').
        end_date (str): The end date for the historical data (format 'YYYY-MM-DD').
        interval (str): Data interval. Options include '1d', '1wk', '1mo'. Default is '1d'.
        """
        try:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(start=start_date, end=end_date, interval=interval)
            print(f"Data downloaded for {self.ticker} from {start_date} to {end_date}.")
        except Exception as e:
            print(f"Error downloading data: {e}")

    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Get the downloaded stock data.
        
        Returns:
        pd.DataFrame: The historical stock data as a DataFrame, or None if no data is downloaded.
        """
        if not self.data.empty:
            return self.data
        else:
            print("No data available. Please download data first.")
            return None
