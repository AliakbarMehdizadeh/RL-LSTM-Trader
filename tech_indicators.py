import pandas as pd
import talib

class TechnicalIndicators:
    def __init__(self, data: pd.DataFrame, price_column: str):
        """
        Initialize the TechnicalIndicators class.
        
        Args:
        data (pd.DataFrame): DataFrame containing the stock price data.
        price_column (str): The column name for the stock price in the DataFrame.
        """
        self.data = data.copy()
        self.price_column = price_column

    def calculate_moving_averages(self, ma7_window: int = 7, ma21_window: int = 21, short_span: int = 12, long_span: int = 26) -> None:
        """Calculate 7-day and 21-day moving averages."""
        self.data['ma7'] = self.data[self.price_column].rolling(window=ma7_window).mean()
        self.data['ma21'] = self.data[self.price_column].rolling(window=ma21_window).mean()
        self.data['12ema'] = self.data[self.price_column].ewm(span=short_span, adjust=False).mean()
        self.data['26ema'] = self.data[self.price_column].ewm(span=long_span, adjust=False).mean()

    def calculate_bollinger_bands(self, window: int = 20) -> None:
        """Calculate Bollinger Bands based on a moving average and standard deviation."""
        self.data['20sd'] = self.data[self.price_column].rolling(window=window).std()
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['Bollinger_Upper'] = self.data['SMA_20'] + (self.data['Close'].rolling(window=20).std() * 2)
        self.data['Bollinger_Lower'] = self.data['SMA_20'] - (self.data['Close'].rolling(window=20).std() * 2)
        
        # Calculate ATR
        self.data['ATR'] = talib.ATR(self.data['High'], self.data['Low'], self.data['Close'], timeperiod=14)

    def calculate_momentum(self) -> None:
        """Calculate Momentum as the difference between current price and previous price."""        
        # Calculate RSI
        self.data['RSI'] = talib.RSI(self.data['Close'], timeperiod=14)
        # Calculate MACD
        self.data['MACD'], self.data['MACD_signal'], self.data['MACD_hist'] = talib.MACD(self.data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
        
        # Calculate volume moving average
        self.data['Volume_SMA_20'] = self.data['Volume'].rolling(window=20).mean()

    def calculate_all_indicators(self) -> pd.DataFrame:
        """Calculate all technical indicators and return the updated DataFrame."""
        self.calculate_moving_averages()
        self.calculate_bollinger_bands()
        self.calculate_momentum()
        return self.data
