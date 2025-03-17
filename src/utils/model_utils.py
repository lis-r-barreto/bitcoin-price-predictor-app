import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
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
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df[f"{column}_scaled"] = scaler.fit_transform(df[[column]])

    # Split into train and test sets
    train_size = int(len(df) * train_size_ratio)
    train_data = df[f"{column}_scaled"].values[:train_size]
    test_data = df[f"{column}_scaled"].values[train_size:]

    # Create sequences
    X_train, y_train = create_sequences(train_data, window_size)
    X_test, y_test = create_sequences(test_data, window_size)

    # Reshape for LSTM input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Build LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(window_size, 1)),
        LSTM(50, return_sequences=False),
        Dense(25, activation="relu"),
        Dense(1)
    ])

    # Compile and train the model
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(X_test, y_test))

    # Save the trained LSTM model
    model.save(save_path)

    # Make predictions
    predicted_returns = model.predict(X_test)

    # Inverse transform the predictions
    predicted_returns = scaler.inverse_transform(predicted_returns)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate error metrics
    rmse = np.sqrt(mean_squared_error(y_test_actual, predicted_returns))
    mae = mean_absolute_error(y_test_actual, predicted_returns)

    return rmse, mae, model

# Function to prepare data for LSTM predictions
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
    # Normalize the data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))  # Match scaling range with evaluate_lstm
    scaled_data = scaler.fit_transform(df[[column]])

    # Create sequences for testing
    X_test = []
    for i in range(window_size, len(scaled_data)):
        X_test.append(scaled_data[i - window_size:i, 0])  # Extract sequences of length `window_size`

    # Convert to numpy array and reshape for LSTM input
    X_test = np.array(X_test)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))  # Reshape to 3D for LSTM

    return scaled_data, X_test, scaler