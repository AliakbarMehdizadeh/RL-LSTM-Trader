import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

from datagatherer import StockDataDownloader
from tech_indicators import TechnicalIndicators
from indicators_plotter import TechnicalIndicatorsPlotter
from stock_fft import StockFFT
from lstm_creator import StockPredictionLSTM
from TradeEnv import TradingEnv
from TradeTestEnv import TradingModelTester
from config import END_DATE,START_DATE,TICKER_SYMBOL
import shimmy
from stable_baselines3 import PPO

if __name__ == "__main__":
    
    #stock ticker
    ticker_symbol = TICKER_SYMBOL
    # data retrieval start and end
    start_date = START_DATE
    end_date = END_DATE

    # downloading data
    downloader = StockDataDownloader(ticker=ticker_symbol)
    downloader.download_data(start_date=start_date, end_date=end_date)
    stock_data = downloader.get_data()
    print('stock data downloaded')

    # split data for test and train
    split_index = int(len(stock_data) * 0.8)
    
    train_stock_data = stock_data.iloc[:split_index]
    test_stock_data = stock_data.iloc[split_index:]

    # add basic indicators
    train_indicators = TechnicalIndicators(train_stock_data, price_column='Close')
    test_indicators = TechnicalIndicators(test_stock_data, price_column='Close')

    train_stock_indicators = train_indicators.calculate_all_indicators()
    test_stock_indicators = test_indicators.calculate_all_indicators()

    # add fourie components

    train_stock_fft = StockFFT(train_stock_indicators)
    test_stock_fft = StockFFT(test_stock_indicators)
    
    train_stock_indicators_fft = train_stock_fft.add_fft_predictions(components_list=[3, 9, 27, 81])
    test_stock_indicators_fft = test_stock_fft.add_fft_predictions(components_list=[3, 9, 27, 81])

    print('Indicators calculated')

    # Preprocessing 

    # drop rows with None values 
    train_stock_indicators_fft_processed = train_stock_indicators_fft.dropna()
    test_stock_indicators_fft_processed = test_stock_indicators_fft.dropna()

    # removing highly correlated indicators
    train_corr_matrix = train_stock_indicators_fft_processed.corr()
    target_corr = train_corr_matrix['Close']
    threshold = 0.95
    correlated_columns = list(target_corr[target_corr.abs() > threshold].drop('Close').index)

    cols_to_delete= ['Dividends', 'Stock Splits'] + correlated_columns
    
    train_stock_indicators_fft_processed = train_stock_indicators_fft_processed.drop(cols_to_delete,axis=1)
    test_stock_indicators_fft_processed = test_stock_indicators_fft_processed.drop(cols_to_delete,axis=1)

    # LSTM training
    print('LSTM training started')

    df = train_stock_indicators_fft_processed.copy()

    # Initialize the model
    lstm_model = StockPredictionLSTM(df, target_column='Close')
    
    # Train the model with hyperparameter tuning using keras_tuner
    lstm_model.train(epochs=200, batch_size=8, tuner=False, max_trials=10, executions_per_trial=1)
    
    # Get predictions and plot
    #train_result_df = lstm_model.predict()

    # adding LSTM preds to our df as next day prediction

    output = lstm_model.predict_new_data(train_stock_indicators_fft_processed)
    RL_train_dataset = output.merge( train_stock_indicators_fft_processed, how='left', left_index=True, right_index=True)
    RL_train_dataset['NextDPred'] = RL_train_dataset['Predicted'].shift(-1)
    
    output = lstm_model.predict_new_data(test_stock_indicators_fft_processed)
    RL_test_dataset = output.merge( test_stock_indicators_fft_processed, how='left', left_index=True, right_index=True)
    RL_test_dataset['NextDPred'] = RL_test_dataset['Predicted'].shift(-1)

    print('LSTM training finished')

    # RL Training
    print('RL training started')

    # Initialize environment and model
    env = TradingEnv(RL_train_dataset)
    model = PPO("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=100000)
    print('RL training finished')
    # RL Test

    # Create an instance of the tester
    tester = TradingModelTester(model, RL_test_dataset[:325], TradingEnv)
    # Run the testing
    tester.run_testing()
    tester.plot_results()
    tester.plot_returns_distribution()
    tester.plot_action_distribution()
    tester.calculate_and_print_metrics()
    print('Check the result folder for AI trader performance graphs')