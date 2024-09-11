import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import talib

class TechnicalIndicatorsPlotter:
    def __init__(self, data: pd.DataFrame, price_column: str):
        """
        Initialize the TechnicalIndicatorsPlotter class with data and price column name.
        
        Args:
        data (pd.DataFrame): DataFrame containing the stock price data.
        price_column (str): The column name for the stock price in the DataFrame.
        """
        self.data = data.copy()
        self.price_column = price_column

    def plot_price_and_moving_averages(self):
        """Plot stock price along with the 7-day, 21-day moving averages, and EMAs."""
        plt.figure(figsize=(14, 7))
        plt.plot(self.data[self.price_column], label=f'{self.price_column}', color='blue')
        plt.plot(self.data['ma7'], label='7-day MA', color='green', linestyle='--')
        plt.plot(self.data['ma21'], label='21-day MA', color='red', linestyle='--')
        plt.plot(self.data['12ema'], label='12-day EMA', color='purple', linestyle='-.')
        plt.plot(self.data['26ema'], label='26-day EMA', color='orange', linestyle='-.')
        plt.title(f"{self.price_column} Price with Moving Averages")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_bollinger_bands(self):
        """Plot stock price along with Bollinger Bands."""
        plt.figure(figsize=(14, 7))
        plt.plot(self.data[self.price_column], label=f'{self.price_column}', color='blue')
        plt.plot(self.data['SMA_20'], label='20-day SMA', color='green', linestyle='--')
        plt.plot(self.data['Bollinger_Upper'], label='Bollinger Upper Band', color='orange', linestyle='-.')
        plt.plot(self.data['Bollinger_Lower'], label='Bollinger Lower Band', color='orange', linestyle='-.')
        plt.fill_between(self.data.index, self.data['Bollinger_Upper'], self.data['Bollinger_Lower'], color='orange', alpha=0.1)
        plt.title(f"{self.price_column} Price with Bollinger Bands")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_macd(self):
        """Plot the MACD line, Signal line, and Histogram."""
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['MACD'], label='MACD', color='blue')
        plt.plot(self.data['MACD_signal'], label='Signal Line', color='red')
        plt.bar(self.data.index, self.data['MACD_hist'], label='MACD Histogram', color='green')
        plt.title("MACD and Signal Line")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_rsi(self):
        """Plot the RSI (Relative Strength Index)."""
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['RSI'], label='RSI', color='purple')
        plt.axhline(70, linestyle='--', color='red', alpha=0.5)
        plt.axhline(30, linestyle='--', color='green', alpha=0.5)
        plt.title("RSI (Relative Strength Index)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_atr(self):
        """Plot the ATR (Average True Range)."""
        plt.figure(figsize=(14, 7))
        plt.plot(self.data['ATR'], label='ATR', color='orange')
        plt.title("Average True Range (ATR)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_volume(self):
        """Plot the Volume and its 20-day moving average."""
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.bar(self.data.index, self.data['Volume'], color='blue', alpha=0.6, label='Volume')
        ax.plot(self.data['Volume_SMA_20'], color='red', linestyle='--', label='20-day Volume MA')
        ax.set_title("Stock Volume with 20-day Moving Average")
        ax.legend()
        ax.grid(True)
        plt.show()

    def plot_all(self):
        """Plot all technical indicators."""
        self.plot_price_and_moving_averages()
        self.plot_bollinger_bands()
        self.plot_macd()
        self.plot_rsi()
        self.plot_atr()
        self.plot_volume()
