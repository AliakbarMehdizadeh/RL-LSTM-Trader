import gym
from gym import spaces
import numpy as np
import pandas as pd
from scipy.stats import norm

class TradingEnv(gym.Env):
    def __init__(self, data, stop_loss_pct=0.10, take_profit_pct=0.25):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.balance = 10000  # Starting balance
        self.initial_balance = self.balance
        self.portfolio_value = self.balance
        self.previous_portfolio_value = self.balance

        self.positions = 0  # Number of shares owned
        self.transaction_cost_pct = 0.001  # 0.1% per trade

        # Store the price at which shares were bought to calculate stop-loss/take-profit
        self.entry_price = None
        self.action_counts = {0: 0, 1: 0, 2: 0}
        self.portfolio_returns = []
        self.portfolio_values = []  # Track portfolio values over time

        # Stop-loss and take-profit percentages
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.action_space = spaces.Discrete(3)  # Buy, Hold, Sell
        self.observation_space = spaces.Box(low=0, high=1, shape=(len(self.data.columns),), dtype=np.float32)
        
    def reset(self):
        self.current_step = 0
        self.initial_balance = 10000
        self.positions = 0
        self.entry_price = None
        self.action_counts = {'0': 0, '1': 0, '2': 0}
        self.portfolio_returns = []
        self.portfolio_values = []
        self.balance = self.balance
        self.portfolio_value = self.balance
        self.previous_portfolio_value = self.balance
        
        return self._next_observation()
    
    def _next_observation(self):
        return self.data.iloc[self.current_step].values

    def _take_action(self, action, current_price):
        if action == 0:  # Buy
            max_buy_amount = 0.30 * self.balance  # Only buy 30% of available balance
            shares_to_buy = max_buy_amount // current_price
            if shares_to_buy > 0:
                self.positions += shares_to_buy
                total_cost = shares_to_buy * current_price
                transaction_cost = total_cost * self.transaction_cost_pct
                self.balance -= total_cost + transaction_cost

                # Set the entry price when the agent buys
                self.entry_price = current_price

        elif action == 2:  # Sell
            shares_to_sell = int(0.30 * self.positions)  # Only sell 30% of holdings
            if shares_to_sell > 0:
                self.positions -= shares_to_sell
                total_revenue = shares_to_sell * current_price
                transaction_cost = total_revenue * self.transaction_cost_pct
                self.balance += total_revenue - transaction_cost
                
                # Reset the entry price when position is sold
                self.entry_price = None

        # If action == 1 (Hold), nothing changes
        
        # Track action counts
        self.action_counts[str(action)] += 1

        # Check for stop-loss or take-profit
        self._check_risk_management(current_price)

    def _check_risk_management(self, current_price):
        """
        Automatically sell if stop-loss or take-profit thresholds are reached.
        """
        if self.entry_price is not None:  # Only check if we have an open position
            price_change = (current_price - self.entry_price) / self.entry_price
            
            # Stop-loss: Sell if the price drops below the threshold
            if price_change <= -self.stop_loss_pct:
                self._sell_all(current_price)
                #print(f"Stop-loss triggered at {current_price}")
            
            # Take-profit: Sell if the price increases beyond the threshold
            elif price_change >= self.take_profit_pct:
                self._sell_all(current_price)
                #print(f"Take-profit triggered at {current_price}")

    def _sell_all(self, current_price):
        """
        Helper function to sell all shares at the current price.
        """
        if self.positions > 0:
            total_revenue = self.positions * current_price
            transaction_cost = total_revenue * self.transaction_cost_pct
            self.balance += total_revenue - transaction_cost
            self.positions = 0
            self.entry_price = None  # Reset the entry price since position is closed

    def step(self, action):
        current_price = self.data['Close'].iloc[self.current_step]

        # Take the action (buy, sell, or hold)
        self._take_action(action, current_price)

        # Advance to the next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        # Calculate portfolio value (balance + stock holdings value)
        self.portfolio_value = self.balance + self.positions * current_price
        self.portfolio_values.append(self.portfolio_value)

        # Portfolio Return
        portfolio_return = (self.portfolio_value - self.previous_portfolio_value) / self.previous_portfolio_value
        self.portfolio_returns.append(portfolio_return)

        # Reward: percentage change in portfolio value
        reward = self.portfolio_value - self.previous_portfolio_value

        self.previous_portfolio_value = self.portfolio_value

        # Update initial balance for next step
        #self.initial_balance = self.portfolio_value

        # Return the next observation, reward, done, and extra info
        return self._next_observation(), reward, done, {
            "portfolio_value": self.portfolio_value,
            "stock_holdings": self.positions,
            "stock_prices": current_price,
            "balance": self.balance,
        }

    def calculate_metrics(self):
        """
        Calculate and return common trading metrics.
        """
        if len(self.portfolio_returns) == 0:
            return {
                "Cumulative Returns": 0,
                "Sharpe Ratio": 0,
                "Maximum Drawdown": 0,
                "Win/Loss Ratio": 0
            }
        
        # Convert returns to a numpy array
        returns = np.array(self.portfolio_returns)

        # Cumulative Returns
        cumulative_returns = (self.portfolio_values[-1] - self.initial_balance) / self.initial_balance
        # Sharpe Ratio
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = mean_return / std_return if std_return != 0 else 0

        # Maximum Drawdown
        portfolio_values_array = np.array(self.portfolio_values)
        rolling_max = np.maximum.accumulate(portfolio_values_array)
        drawdowns = (rolling_max - portfolio_values_array) / rolling_max
        max_drawdown = np.max(drawdowns)

        # Win/Loss Ratio
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        win_loss_ratio = (len(positive_returns) / len(negative_returns)) if len(negative_returns) > 0 else len(positive_returns)

        return {
            "Cumulative Returns": cumulative_returns,
            "Sharpe Ratio": sharpe_ratio,
            "Maximum Drawdown": max_drawdown,
            "Win/Loss Ratio": win_loss_ratio
        }

    def plot_results(self, result_df):
        """
        Plot the actual vs predicted values (to be used for additional metrics visualization).
        """
        plt.figure(figsize=(14, 5))
        plt.plot(result_df.index, result_df['Actual'], color='blue', label='Actual Prices')
        plt.plot(result_df.index, result_df['Predicted'], color='red', label='Predicted Prices')
        plt.title('Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def evaluate(self):
        """
        Evaluate and print trading metrics.
        """
        metrics = self.calculate_metrics()
        print("Trading Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path)

    def predict_new_data(self, new_df):
        """
        Predict future stock prices using new data.
        
        Parameters:
        new_df (pd.DataFrame): New stock data to predict on.

        Returns:
        result_df (pd.DataFrame): DataFrame with actual and predicted prices.
        """
        # Scale the new data using the same scaler fitted on training data
        new_scaled_data = self.scaler.transform(new_df)
        
        # Create sequences for the new data
        X_new, _ = self._create_sequences(new_scaled_data)
        
        # Make predictions on new data
        y_pred_new = self.model.predict(X_new)
        
        # Inverse transform the predictions
        y_pred_new_rescaled = self.scaler.inverse_transform(
            np.concatenate((np.zeros((y_pred_new.shape[0], new_df.shape[1] - 1)), y_pred_new), axis=1)
        )[:, -1]
        
        result_df = pd.DataFrame({
            'Date': new_df.index,
            'Actual': new_df['Close'],
            'Predicted': y_pred_new_rescaled
        }).set_index('Date')
        
        return result_df