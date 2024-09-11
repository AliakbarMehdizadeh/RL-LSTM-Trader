import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import keras_tuner as kt
import shutil
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow import keras
import keras_tuner as kt

class StockPredictionLSTM:
    def __init__(self, df, target_column='Close', look_back=30):
        self.df = df
        self.X  = self.df.values
        self.y  = self.df[target_column].values.reshape(-1, 1)
        self.target_column = target_column
        self.look_back = look_back

    # Function to create LSTM sequences
    def create_sequences(self, X, y):
        X_sequences, y_sequences = [], []
        
        for i in range(len(X) - self.look_back):
            X_seq = X[i:i + self.look_back]
            y_seq = y[i + self.look_back]
            
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
        
        return np.array(X_sequences), np.array(y_sequences)

    def build_model(self, hp):
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units=hp.Int('units_lstm1', min_value=32, max_value=128, step=32),
                       return_sequences=True, 
                       input_shape=(self.look_back, self.X.shape[1])))
        model.add(Dropout(hp.Float('dropout_rate_1', min_value=0.1, max_value=0.3, step=0.1)))

        # Second LSTM layer
        model.add(LSTM(units=hp.Int('units_lstm2', min_value=32, max_value=128, step=32),
                       return_sequences=True))
        model.add(Dropout(hp.Float('dropout_rate_2', min_value=0.1, max_value=0.3, step=0.1)))

        model.add(LSTM(units=hp.Int('units_lstm3', min_value=32, max_value=128, step=32),
                       return_sequences=False))
        model.add(Dropout(hp.Float('dropout_rate_3', min_value=0.1, max_value=0.3, step=0.1)))

        #model.add(Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.4, step=0.1)))
        model.add(Dense(units=hp.Int('dense_units', min_value=50, max_value=150, step=50)))
        model.add(Dropout(hp.Float('dropout_rate_4', min_value=0.1, max_value=0.3, step=0.1)))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Learning rate for optimizer
        learning_rate = hp.Float('learning_rate', min_value=1e-3, max_value=1e-1, sampling='LOG')
        
        # Compile the model
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        
        return model

    def build_model_direct(self):
        model = Sequential()
        
        # Define fixed hyperparameters
        units_lstm1 = 50
        dropout_rate_1 = 0.1
        units_lstm2 = 50
        dropout_rate_2 = 0.1
        units_lstm3 = 50
        dropout_rate_3 = 0.1
        dense_units = 50
        dropout_rate_4 = 0.1
        learning_rate = 1e-3
        
        # First LSTM layer
        model.add(LSTM(units=units_lstm1, 
                       return_sequences=True, 
                       input_shape=(self.look_back, self.X.shape[1])))
        model.add(Dropout(dropout_rate_1))
    
        # Second LSTM layer
        model.add(LSTM(units=units_lstm2, 
                       return_sequences=True))
        model.add(Dropout(dropout_rate_2))
    
        # Third LSTM layer
        model.add(LSTM(units=units_lstm3, 
                       return_sequences=False))
        model.add(Dropout(dropout_rate_3))
    
        # Dense layer
        model.add(Dense(units=dense_units))
        model.add(Dropout(dropout_rate_4))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile the model with a fixed learning rate
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        
        return model

    def train(self, epochs=50, batch_size=32, tuner=False, max_trials=5, executions_per_trial=1):
        
        # Initialize TimeSeriesSplit (we'll use 5 splits for example)
        tscv = TimeSeriesSplit(n_splits=5)

        # Define callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath='best_model.h5',
                save_best_only=True,
                monitor='val_loss',
                save_weights_only=False,
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
        ]
        
        if tuner:
            shutil.rmtree('tuner_logs', ignore_errors=True)
            tuner = kt.RandomSearch(
                self.build_model,
                objective='val_mean_squared_error',
                max_trials=max_trials,
                executions_per_trial=executions_per_trial,
                directory='tuner_logs',
                project_name='lstm_tuning'
            )

                    
            for train_index, test_index in tscv.split(self.X):
        
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]

                self.scaler_x = RobustScaler()
                self.scaler_y = RobustScaler()

                X_train_scaled = self.scaler_x.fit_transform(X_train)
                X_test_scaled = self.scaler_x.transform(X_test)

                y_train_scaled = self.scaler_y.fit_transform(y_train)
                y_test_scaled = self.scaler_y.transform(y_test)

                # Create LSTM sequences for training and testing
                X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
                X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test_scaled)
        
                tuner.search(X_train_seq, y_train_seq, epochs=epochs, validation_data=(X_test_seq, y_test_seq), batch_size=batch_size, callbacks=callbacks)
            
            # Get the best hyperparameters and set the model
            best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
            
            # Print available hyperparameters to debug
            print("Available hyperparameters:")
            for key in best_hps.values:
                print(f"{key}: {best_hps.get(key)}")
    
            self.model = tuner.hypermodel.build(best_hps)
            
            for train_index, test_index in tscv.split(self.X):
                
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]

                self.scaler_x = RobustScaler()
                self.scaler_y = RobustScaler()

                X_train_scaled = self.scaler_x.fit_transform(X_train)
                X_test_scaled = self.scaler_x.transform(X_test)

                y_train_scaled = self.scaler_y.fit_transform(y_train)
                y_test_scaled = self.scaler_y.transform(y_test)

                # Create LSTM sequences for training and testing
                X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
                X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test_scaled)
                
                self.model.fit(X_train_seq, y_train_seq, epochs=epochs, validation_data=(X_test_seq, y_test_seq), batch_size=batch_size, callbacks=callbacks)
                
            # Save the last test split data for predictions
            self.X_test = X_test_seq
            self.y_test = y_test_seq
        else:
            # Normal training without tuning
            self.model = self.build_model_direct()
            print("model is training...")

            for train_index, test_index in tscv.split(self.X):
                X_train, X_test = self.X[train_index], self.X[test_index]
                y_train, y_test = self.y[train_index], self.y[test_index]

                self.scaler_x = RobustScaler()
                self.scaler_y = RobustScaler()

                X_train_scaled = self.scaler_x.fit_transform(X_train)
                X_test_scaled = self.scaler_x.transform(X_test)

                y_train_scaled = self.scaler_y.fit_transform(y_train)
                y_test_scaled = self.scaler_y.transform(y_test)

                # Create LSTM sequences for training and testing
                X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
                X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test_scaled)
                self.model.fit(X_train_seq, y_train_seq, epochs=epochs, batch_size=batch_size, validation_data=(X_test_seq, y_test_seq), callbacks=callbacks, verbose =0)
            
            print("model training is done")

            # Save the last test split data for predictions
            self.X_test = X_test_seq
            self.y_test = y_test_seq

    def evaluate(self):
        if self.model is None:
            raise ValueError("Model has not been trained or built yet.")
        test_loss, test_mse = self.model.evaluate(self.X_test, self.y_test)
        print(f"Test Loss: {test_loss}")
        print(f"Test MSE: {test_mse}")

    def predict(self):
        if self.X_test is None or self.y_test is None:
            raise ValueError("Model must be trained before prediction.")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Inverse scaling to get actual price values
        y_pred_rescaled = self.scaler_y.inverse_transform(np.concatenate((np.zeros((y_pred.shape[0], self.df.shape[1] - 1)), y_pred), axis=1))[:, -1]
        y_test_rescaled = self.scaler_y.inverse_transform(np.concatenate((np.zeros((self.y_test.shape[0], self.df.shape[1] - 1)), self.y_test.reshape(-1, 1)), axis=1))[:, -1]
        
        # Get the corresponding date index for the test set
        test_dates = self.df.index[-len(self.y_test):]
        
        # Create a DataFrame with Date as index and two columns: Actual and Predicted
        result_df = pd.DataFrame({
            'Actual': y_test_rescaled,
            'Predicted': y_pred_rescaled
        }, index=test_dates)

        
        return result_df

    def plot_results(self, result_df):
        # Plot the actual vs predicted values
        plt.figure(figsize=(14, 5))
        plt.plot(result_df.index, result_df['Actual'], color='blue', label='Actual Prices')
        plt.plot(result_df.index, result_df['Predicted'], color='red', label='Predicted Prices')
        plt.title('Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    def evaluate(self, result_df):
        # Calculate and return the mean squared error
        mse = mean_squared_error(result_df['Actual'], result_df['Predicted'])
        print(f"Mean Squared Error: {mse}")
        return mse

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path)

    def predict_new_data(self, new_df, target_column='Close'):
        """
        Predict future stock prices using new data.
        
        Parameters:
        new_df (pd.DataFrame): New stock data to predict on.

        Returns:
        result_df (pd.DataFrame): DataFrame with actual and predicted prices.
        """
        self.X_test  = new_df.values
        self.y_test  = new_df[target_column].values.reshape(-1, 1)
        
        self.scaler_x = RobustScaler()
        self.scaler_y = RobustScaler()

        X_train_scaled = self.scaler_x.fit_transform(self.X)
        X_test_scaled = self.scaler_x.transform(self.X_test)

        y_train_scaled = self.scaler_y.fit_transform(self.y)
        y_test_scaled = self.scaler_y.transform(self.y_test)

        # Create LSTM sequences
        X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test_scaled)  
              
        # Make predictions on new data
        y_pred_new = self.model.predict(X_test_seq)
        
        # Inverse transform the predictions
        y_pred_new_rescaled = self.scaler_y.inverse_transform(
            np.concatenate((np.zeros((y_pred_new.shape[0], new_df.shape[1] - 1)), y_pred_new), axis=1)
        )[:, -1]
        
        # Create a DataFrame to hold the results
        result_df_new = pd.DataFrame({
            'Predicted': y_pred_new_rescaled
        }, index=new_df.index[-len(y_pred_new):])  # Assuming new_df has a date index
        
        # If actual values are available in new_df, include them in the result DataFrame
        if self.target_column in new_df.columns:
            result_df_new['Actual'] = new_df[self.target_column].iloc[-len(y_pred_new):].values
        
        return result_df_new

    def predict_and_plot_new_data(self, new_df):
        """
        Predict and plot future stock prices using new data.
        
        Parameters:
        new_df (pd.DataFrame): New stock data to predict and plot.
        """
        # Predict future stock prices
        result_df_new = self.predict_new_data(new_df)
        
        # Plot the actual and predicted prices
        plt.figure(figsize=(14, 5))
        if 'Actual' in result_df_new.columns:
            plt.plot(result_df_new.index, result_df_new['Actual'], color='blue', label='Actual Prices')
        plt.plot(result_df_new.index, result_df_new['Predicted'], color='red', label='Predicted Prices')
        
        plt.title('Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

        return result_df_new