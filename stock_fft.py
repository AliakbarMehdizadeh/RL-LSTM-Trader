import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class StockFFT:
    def __init__(self, stock_data):
        """
        Initialize the StockFFT class with stock data.
        
        Parameters:
        stock_data (pd.DataFrame): DataFrame containing the stock prices with a 'Close' column.
        """
        self.input = stock_data
        self.stock_data = stock_data[['Close']].reset_index().drop('Date', axis=1)
        self.close_prices = np.asarray(self.stock_data['Close'].tolist())
        self.fft_results = np.fft.fft(self.close_prices)
        self.fft_df = self._create_fft_df()

    def _create_fft_df(self):
        """
        Create a DataFrame to store FFT results, including the magnitude and phase.
        
        Returns:
        pd.DataFrame: DataFrame containing the FFT results.
        """
        fft_df = pd.DataFrame({'fft': self.fft_results})
        fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
        fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
        return fft_df

    def plot_fft_components(self, components_list=[3, 6, 9, 100]):
        """
        Plot the inverse FFT for different numbers of components.
        
        Parameters:
        components_list (list): List of integers representing the number of FFT components to retain.
        """
        plt.figure(figsize=(14, 7), dpi=100)
        fft_list = np.asarray(self.fft_df['fft'].tolist())
        
        for num_ in components_list:
            fft_list_m = np.copy(fft_list)
            fft_list_m[num_:-num_] = 0  # Zero out all but the first and last 'num_' components
            reconstructed = np.fft.ifft(fft_list_m).real
            plt.plot(reconstructed, label=f'Fourier transform with {num_} components')
        
        # Plot the original 'Close' prices
        plt.plot(self.stock_data['Close'], label='Real')
        
        # Add labels, title, and legend
        plt.xlabel('Days')
        plt.ylabel('USD')
        plt.title('Stock Prices & Fourier Transforms')
        plt.legend()

        # Show the plot
        plt.show()

    def add_fft_predictions(self, components_list=[3, 6, 9, 100]):
        """
        Add FFT-based predictions to the original dataset for comparison.
        
        Parameters:
        components_list (list): List of integers representing the number of FFT components to retain.
        
        Returns:
        pd.DataFrame: DataFrame containing the original and FFT-based predictions.
        """
        fft_predictions_df = pd.DataFrame(index=self.stock_data.index)
        
        for num_ in components_list:
            fft_list = np.copy(self.fft_results)
            fft_list[num_:-num_] = 0  # Zero out all but the first and last 'num_' components
            reconstructed = np.fft.ifft(fft_list).real
            fft_predictions_df[f'FFT_{num_}_components'] = reconstructed

        fft_predictions_df['Date'] = self.input.index
        fft_predictions_df.set_index('Date', inplace=True)
        #fft_predictions_df = fft_predictions_df.drop('Close',axis=1)
        
        merge_stock_df = pd.merge(self.input,fft_predictions_df, left_on='Date', right_on='Date')
        
        return merge_stock_df
