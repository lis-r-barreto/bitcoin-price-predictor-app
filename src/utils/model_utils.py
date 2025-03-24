from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from utils.data_process_utils import create_sequences


def evaluate_lstm(df, column='log_return', window_size=50, epochs=20, batch_size=32, train_size_ratio=0.8, save_path="models/lstm_model.h5"):
    """
    Trains and evaluates an LSTM model for time series forecasting.
    Saves the trained LSTM model to a file.

    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        column (str): The column to use for LSTM modeling.
        window_size (int): The size of the window for each sequence.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        train_size_ratio (float): The proportion of data to use for training.
        save_path (str): Path to save the LSTM model.

    Returns:
        tuple: RMSE, MAE, and the trained LSTM model.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[f"{column}_scaled"] = scaler.fit_transform(df[[column]])

    train_size = int(len(df) * train_size_ratio)
    train_data = df[f"{column}_scaled"].values[:train_size]
    test_data = df[f"{column}_scaled"].values[train_size:]

    X_train, y_train = create_sequences(train_data, window_size)
    X_test, y_test = create_sequences(test_data, window_size)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
        LSTM(50, return_sequences=False),
        Dense(25, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_test, y_test))

    model.save(save_path)

    predicted_returns = model.predict(X_test)

    predicted_returns = scaler.inverse_transform(predicted_returns)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(y_test_actual, predicted_returns))
    mae = mean_absolute_error(y_test_actual, predicted_returns)

    return rmse, mae, model


def prepare_lstm_data(df, column='log_return', window_size=50):
    """
    Prepare data for LSTM predictions.

    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        column (str): The column to use for LSTM modeling.
        window_size (int): The size of the window for LSTM sequences.

    Returns:
        tuple: Scaled data, test sequences (X_test), and scaler object.
    """

    scaler = MinMaxScaler(feature_range=(-1, 1))  
    scaled_data = scaler.fit_transform(df[[column]])


    X_test = []
    for i in range(window_size, len(scaled_data)):
        X_test.append(scaled_data[i - window_size:i, 0])  

    X_test = np.array(X_test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))  

    return scaled_data, X_test, scaler


def predict_next_x_days_lstm(df, window_size=50, x=7):
    """
    Predict next x days using LSTM model.

    Args:
        model_path (str): Path to the saved LSTM .h5 model
        df (pd.DataFrame): Original Bitcoin data DataFrame
        window_size (int): Number of past days used for predictions
        x (int): Number of days to forecast

    Returns:
        list: List of tuples [(date, prediction), ...]
    """
    model = load_model("/models/lstm_model.h5", custom_objects={"mse": MeanSquaredError()})
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    df['log_return_scaled'] = scaler.fit_transform(df[['log_return']])
    
    last_data = df['log_return_scaled'].values[-window_size:]
    input_data = last_data.reshape(1, window_size, 1)
    
    predictions = []
    last_date = pd.to_datetime(df.index[-1])
    
    for _ in range(x):
        pred = model.predict(input_data)
        pred_price = scaler.inverse_transform(pred)[0][0]
        pred_price = np.exp(pred_price) * df['price'][-1]
        
        next_date = last_date + timedelta(days=1)
        predictions.append((next_date.strftime('%Y-%m-%d'), pred_price))
        
        input_data = np.roll(input_data, -1, axis=1)
        input_data[0, -1, 0] = pred
        
        last_date = next_date
    
    return predictions
