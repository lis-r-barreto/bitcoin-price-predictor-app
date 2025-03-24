from datetime import timedelta
import streamlit as st
import numpy as np
import pandas as pd
from utils.data_viz_utils import (
    line_plot,
    seasonality_analysis,
    volatility_analysis,
    technical_indicators,
    combined_indicators,
    create_pred_visualization,
)

from utils.data_process_utils import create_dataset

from utils.model_utils import predict_next_x_days_lstm


@st.cache_data
def load_data(parquet_file):
    return pd.read_parquet(parquet_file)

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

    parquet_file = "/data/crypto_data_processed.parquet"
    df = load_data(parquet_file)
    df_2 = create_dataset(df)

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

    # Tab 3: Model Prediction
    with tab3:
        st.session_state.current_tab = 2
        st.header("Prediction Controls")
        forecast_days = st.slider("Prediction Horizon (Days)", min_value=1, max_value=30, value=7, key="forecast_days")

        if st.button("Generate Forecast", use_container_width=True):
                with st.spinner("Computing predictions..."):
                    predictions = predict_next_x_days_lstm(
                        df=df_2,
                        window_size=50,
                        x=forecast_days
                    )
                    
                    pred_dates = [df_2.index[-1] + timedelta(days=i+1) for i in range(forecast_days)]
                    pred_prices = [p[1] for p in predictions]
                    
                    # Format results with historical context
                    full_df = pd.DataFrame({
                        'Date': pred_dates,
                        'Predicted Price': pred_prices
                    }).set_index('Date')
                    
                
                st.dataframe(
                    full_df.style.format({"Predicted Price": "${:,.2f}"}),
                    height=200
                )

                fig = create_pred_visualization(full_df)
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric(
                    "Next Day Prediction",
                    f"${pred_prices[0]:,.2f}",
                    delta=f"{(pred_prices[0]/df['price'].iloc[-1]-1):+,.2%}"
                )
                st.metric(
                    "Forecast Avg. Change",
                    f"{np.mean(np.diff(pred_prices))/np.mean(pred_prices):+,.2%}"
                )
                
    
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
