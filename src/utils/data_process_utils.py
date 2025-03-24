import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

def load_data(data_path):
    """Load and preprocess the dataset."""
    df = pd.read_parquet(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    df['log_return'] = np.log(df['price']) - np.log(df['price'].shift(1))
    df = df.dropna()
    return df

def normalize_data(data, feature_column):
    """Normalize data using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data[f"{feature_column}_scaled"] = scaler.fit_transform(data[[feature_column]])
    return data, scaler

def extract_data():
    """Extract historical Bitcoin data and save as Parquet."""
    df_crypto = yf.download("BTC-USD", start="2010-01-01", end="2025-03-12")
    df_crypto = df_crypto[["Close"]].reset_index()
    df_crypto.columns = ["date", "price"]
    df_crypto.to_parquet('data/yfinance_20250312.parquet')
    return 'data/yfinance_20250312.parquet'

def count_duplicates(df):
    """Check for duplicate values in the DataFrame."""
    duplicates = df.duplicated(keep=False)
    duplicates_count = duplicates.sum()

    if duplicates_count > 0:
        print("Duplicate values found:")
        print(df[duplicates])
    else:
        print("No duplicate values found.")

    return duplicates_count

def validate_frequency(df):
    """Validate the frequency of the data in the DataFrame."""
    date_diffs = df['date'].diff().dt.days
    return date_diffs.value_counts()


def create_dataset(df):
    df_2 = df.copy()
    df_2["log_return"] = np.log(df_2["price"]) - np.log(df_2["price"].shift(1))
    print(df_2.head())
    print(df_2.tail())
    df_2 = df_2.dropna()
    return df_2


def create_sequences(data, window_size=50):
    """
    Prepares data for LSTM by creating sequences of a given window size.

    Args:
        data (array-like): The input data.
        window_size (int): The size of the window for each sequence.

    Returns:
        tuple: Arrays of input sequences (X) and corresponding targets (y).
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

def main():
    data_path = extract_data()
    df = load_data(data_path)

    print("Number of duplicates:", count_duplicates(df))

    frequency = validate_frequency(df)
    print("\nFrequency of date differences:")
    print(frequency)

    grouped_df = df.groupby('date').agg({'price': 'mean'}).reset_index()
    grouped_df = grouped_df.set_index('date')

    print("\nDataFrame grouped by date with average prices:")
    print(grouped_df)

    grouped_df.to_parquet('data/crypto_data_processed.parquet', engine='pyarrow', compression='snappy')
    print("\nDataFrame saved as 'crypto_data_processed.parquet' successfully!")

if __name__ == "__main__":
    main()
