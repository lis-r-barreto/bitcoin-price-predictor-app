import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model  # For loading LSTM models
from tensorflow.keras.losses import MeanSquaredError
from utils.data_viz_utils import (
    line_plot,
    seasonality_analysis,
    volatility_analysis,
    technical_indicators,
    combined_indicators,
    plot_lstm_predictions,
)

from utils.data_process_utils import create_dataset

from utils.model_utils import prepare_lstm_data

# Function to load data from a Parquet file
@st.cache_data
def load_data(parquet_file):
    return pd.read_parquet(parquet_file)

# Streamlit app
def main():
    st.set_page_config(page_title="Bitcoin Price Predictor App", layout="wide")
    st.title("Bitcoin Price Predictor App")
    st.markdown(
        """
        Welcome to the **Bitcoin Price Predictor App**!
        
        This app provides an interactive way to explore Bitcoin price data, 
        analyze trends, and visualize technical indicators. The data used spans from **2010 to March 2025** and is sourced from **[Yahoo Finance](https://finance.yahoo.com/quote/BTC-USD/history/?p=BTC-USD)**. 
        Use the tabs below to navigate through the features. 📊
        """
    )

    # Load data
    parquet_file = "data/crypto_data_processed.parquet"  # Replace with your file path
    df = load_data(parquet_file)
    df_2 = create_dataset(df)

    # Load pre-trained models
    lstm_model = load_model(
        "models/lstm_model.h5",
        custom_objects={"mse": MeanSquaredError()}
    )

    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "📊 Technical Indicators", "🔮 Model Prediction"])

    # Tab 1: Data Analysis
    with tab1:
        st.session_state.current_tab = 0
        st.header("Data Analysis")
        st.markdown(
            """
            Explore the Bitcoin price data through various visualizations, including price trends, 
            distribution analysis, seasonality, and volatility.
            """
        )

        # Line Plot
        st.subheader("Price Over Time")
        st.markdown("Visualize the price trend of Bitcoin over time.")
        line_plot(df)


        # Seasonal Decomposition
        st.subheader("Seasonality Analysis")
        st.markdown(
            """
            Decompose the time series into **trend**, **seasonal**, and **residual** components to better understand 
            the underlying patterns in the data.
            """
        )
        period = st.number_input(
            "Select the period for seasonal decomposition (e.g., 7 for weekly):",
            min_value=1,
            value=7,
            step=1,
        )
        seasonality_analysis(df, column="price", period=period)

        # Volatility Analysis
        st.subheader("Volatility Analysis")
        st.markdown(
            """
            Analyze the rolling volatility of the Bitcoin price to understand price fluctuations over time.
            """
        )
        window = st.number_input(
            "Select the rolling window size for volatility calculation:",
            min_value=1,
            value=30,
            step=1,
        )
        volatility_analysis(df, column="price", window=window)

    # Tab 2: Technical Indicators
    with tab2:
        st.session_state.current_tab = 1
        st.header("Technical Indicators")
        st.markdown(
            """
            Analyze key technical indicators such as **Simple Moving Averages (SMA)**, **Relative Strength Index (RSI)**, 
            and **Moving Average Convergence Divergence (MACD)** to gain insights into market trends.
            """
        )
        technical_indicators(df)

        st.markdown(
            """
            View all technical indicators combined in a single visualization for a comprehensive analysis of the Bitcoin market.
            """
        )

        combined_indicators(df)

    # Tab 4: Model Prediction
    with tab3:
        st.session_state.current_tab = 2
        st.header("Model Prediction")
        st.markdown(
            """
            Use pre-trained machine learning models (ARIMA, LSTM, Prophet) to predict Bitcoin prices and visualize the results.
            """
        )

        # Select column for prediction
        price_column = 'price'  # Replace with the column name you want to predict

        # Button to run predictions
        if st.button("Run Predictions 🤖"):
            st.write("### Running Predictions...")

            # Prepare data for LSTM
            st.write("#### Preparing Data for LSTM...")
            _, X_test, scaler = prepare_lstm_data(df_2, column=price_column)

            # Load the pre-trained LSTM model
            st.write("#### Loading Pre-Trained LSTM Model...")
            lstm_model = load_model(
                "models/lstm_model.h5",
                custom_objects={"mse": MeanSquaredError()}
            )

            # Make predictions
            st.write("#### Making Predictions with LSTM...")
            lstm_predictions = lstm_model.predict(X_test)
            lstm_predictions = scaler.inverse_transform(lstm_predictions)  # Inverse transform to original scale

            # Plot the results
            plot_lstm_predictions(df_2, lstm_predictions, title="LSTM Predictions vs Actual Prices")
    
    if st.session_state.current_tab == 2:
        st.session_state.current_tab = 2

def display_credits():
    st.markdown(
        """
        <footer style='text-align: center; margin-top: 50px;'>
            <hr>
            <p>This application was developed by:</p>
            <p>
                <a href="https://www.linkedin.com/in/andr%C3%A9-prestes-67bb71181/">André Prestes</a> | 
                <a href="https://www.linkedin.com/in/felipe-f-porto/">Felipe Porto</a> | 
                <a href="https://www.linkedin.com/in/lis-r-barreto/">Lis Barreto</a>
            </p>
        </footer>
        """,
        unsafe_allow_html=True
    )

# Run the app
if __name__ == "__main__":
    main()
    display_credits()