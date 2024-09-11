import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

class TradingModelTester:
    def __init__(self, model, test_dataset, environment_class, results_folder='results'):
        self.model = model
        self.test_dataset = test_dataset
        self.environment_class = environment_class
        self.test_env = self.environment_class(self.test_dataset)
        self.test_portfolio_values = []
        self.test_stock_holdings = []
        self.test_stock_prices = []
        
        # Ensure to set initial_balance for metrics calculation
        self.initial_balance = self.test_env.initial_balance
        
        # Ensure results folder exists
        self.results_folder = results_folder
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)
    
    def run_testing(self):
        # Reset the environment and start testing
        obs = self.test_env.reset()
        done = False
        
        # Run the model in the testing environment
        for step in range(len(self.test_dataset)):
            action, _states = self.model.predict(obs)  # Get the action from the trained model
            obs, reward, done, info = self.test_env.step(action)  # Step through the environment
            
            # Collect testing values
            self.test_portfolio_values.append(info["portfolio_value"])
            self.test_stock_holdings.append(info["stock_holdings"])
            self.test_stock_prices.append(info["stock_prices"])
            
            if done:
                break  # Stop when the environment reaches the end of the dataset
    
    def plot_results(self):
        # Plot and save the portfolio values, stock holdings, and stock prices
        plt.figure(figsize=(14, 8))
        
        # Portfolio Value Plot
        plt.subplot(3, 1, 1)
        plt.plot(self.test_portfolio_values, label='Portfolio Value', color='blue')
        plt.title('Portfolio Value During Testing')
        plt.xlabel('Timestep')
        plt.ylabel('Portfolio Value')
        plt.legend()
        
        # Stock Holdings Plot
        plt.subplot(3, 1, 2)
        plt.plot(self.test_stock_holdings, label='Stock Holdings', color='green')
        plt.title('Stock Holdings During Testing')
        plt.xlabel('Timestep')
        plt.ylabel('Number of Stocks')
        plt.legend()
        
        # Stock Prices Plot
        plt.subplot(3, 1, 3)
        plt.plot(self.test_stock_prices, label='Stock Prices', color='red')
        plt.title('Stock Prices During Testing')
        plt.xlabel('Timestep')
        plt.ylabel('Stock Price')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_folder, 'results_plot.png'))
        plt.close()
    
    def plot_returns_distribution(self):
        # Access portfolio returns at the end of testing
        portfolio_returns = self.test_env.portfolio_returns
        
        sns.set_style('darkgrid')
        plt.figure(figsize=(14, 3))
        sns.histplot(portfolio_returns, kde=True)
        plt.xlabel('Returns')
        plt.title('Returns Distribution')
        plt.savefig(os.path.join(self.results_folder, 'returns_distribution.png'))
        plt.close()
    
    def plot_action_distribution(self):
        # Retrieve action counts
        action_counts = self.test_env.action_counts
        
        conversion_dict = {'0': 'buy', '1': 'hold', '2': 'sell'}
        
        # Normalize the action counts
        total_actions = sum(action_counts.values())
        normalized_counts = {conversion_dict[k]: v / total_actions for k, v in action_counts.items()}
        
        # Convert to DataFrame for seaborn
        df = pd.DataFrame(list(normalized_counts.items()), columns=['Action', 'Frequency'])
        
        # Plot using seaborn
        plt.figure(figsize=(8, 3))
        sns.barplot(x='Action', y='Frequency', data=df)
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.title('Action Distribution')
        plt.savefig(os.path.join(self.results_folder, 'action_distribution.png'))
        plt.close()
    
    def calculate_and_print_metrics(self):
        """
        Calculate and print common trading metrics.
        """
        metrics = self.test_env.calculate_metrics()
        print("Trading Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")
