# Bitcoin Price Predictor App

Welcome to the **Bitcoin Price Predictor App**! This interactive data application is designed to provide insights into Bitcoin price trends, analyze technical indicators, and predict future prices using machine learning models. The app is built with **Streamlit**, making it easy to use and visually appealing.

---

## Features

### 📈 **Overview**
- **Price Trends**: Visualize Bitcoin price trends over time.
- **Seasonality Analysis**: Decompose the time series into trend, seasonal, and residual components.
- **Volatility Analysis**: Analyze rolling volatility to understand price fluctuations.

### 📊 **Technical Indicators**
- Explore key technical indicators such as:
  - **Simple Moving Averages (SMA)**
  - **Relative Strength Index (RSI)**
  - **Moving Average Convergence Divergence (MACD)**
- View all indicators combined for a comprehensive market analysis.

### 🔮 **Model Prediction**
- Predict Bitcoin prices for the next 1 to 30 days using an **LSTM (Long Short-Term Memory)** model.
- Visualize predictions alongside historical data.
- Key metrics include:
  - **Next Day Prediction**
  - **Forecast Average Change**

---

## How to Run the App

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/lis-r-barreto/bitcoin-price-predictor-app/
   cd bitcoin-price-predictor-app/src/bitcoin_price_predictor_app/
   ```

2. **Install Dependencies**:
   Ensure you have Python installed. Then, use Poetry to install the required libraries:
   ```bash
   poetry install
   ```

3. **Run the App**:
   Start the Streamlit app by running:
   ```bash
   streamlit run app.py
   ```

4. **Access the App**:
   Open your browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

---

## Data Source

The app uses Bitcoin price data from **[Yahoo Finance](https://finance.yahoo.com/quote/BTC-USD/history/?p=BTC-USD)**, spanning from **2010 to March 2025**. The data is preprocessed and stored in a Parquet file for efficient loading.

---

## App Structure

### Tabs
1. **Overview**:
   - Visualize price trends, seasonality, and volatility.
   - Customize analysis parameters (e.g., rolling window size, seasonal decomposition period).

2. **Technical Indicators**:
   - Analyze individual and combined technical indicators for market insights.

3. **Model Prediction**:
   - Generate forecasts for the next 1 to 30 days.
   - View predictions in a table and interactive chart.

### Utilities
- **Data Visualization**: Custom plots for trends, seasonality, and technical indicators.
- **Machine Learning**: LSTM model for time series forecasting.
---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
