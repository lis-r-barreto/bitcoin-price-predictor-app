import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from statsmodels.tsa.seasonal import seasonal_decompose

def line_plot(df):
    """
    Creates a line plot of price over time.

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        None: Displays the plot in Streamlit.
    """
    line_fig = px.line(
        df,
        x=df.index,
        y="price",
        title="Cryptocurrency Price Over Time",
        labels={"x": "Date", "price": "Price"}
    )
    line_fig.update_layout(xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(line_fig)

def box_plot(df):
    """
    Creates a box plot of price distribution.

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        None: Displays the plot in Streamlit.
    """
    box_fig = px.box(
        df,
        y="price",
        title="Boxplot of Price Distribution",
        labels={"price": "Price"}
    )
    st.plotly_chart(box_fig)

def histogram(df):
    """
    Creates a histogram of price distribution.

    Args:
        df (pd.DataFrame): DataFrame containing the data.

    Returns:
        None: Displays the plot in Streamlit.
    """
    hist_fig = px.histogram(
        df,
        x="price",
        nbins=20,
        title="Histogram of Price Distribution",
        labels={"price": "Price"}
    )
    hist_fig.update_layout(xaxis_title="Price", yaxis_title="Frequency")
    st.plotly_chart(hist_fig)

def seasonality_analysis(df, column, period):
    """
    Performs seasonal decomposition and visualizes the components using Plotly.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name to analyze.
        period (int): Period for seasonal decomposition (e.g., 7 for weekly).

    Returns:
        None: Displays the decomposition plots in Streamlit.
    """
    decomposition = seasonal_decompose(df[column], model='additive', period=period)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=decomposition.trend,
        mode='lines',
        name='Trend',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=decomposition.seasonal,
        mode='lines',
        name='Seasonal',
        line=dict(color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=decomposition.resid,
        mode='lines',
        name='Residual',
        line=dict(color='green')
    ))

    fig.update_layout(
        title="Seasonal Decomposition of Time Series",
        xaxis_title="Date",
        yaxis_title="Value",
        legend_title="Components",
        height=600,
        width=1000
    )
    st.plotly_chart(fig)

def volatility_analysis(df, column, window=30):
    """
    Calculates and visualizes the volatility of a time series using a rolling window.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        column (str): Column name to analyze.
        window (int): Rolling window size for volatility calculation.

    Returns:
        None: Displays the volatility plot in Streamlit.
    """

    df_analysis = pd.DataFrame(df[column])

    df_analysis['returns'] = df[column].pct_change()

    df_analysis['volatility'] = df_analysis['returns'].rolling(window=window).std()

    fig = px.line(
        df_analysis,
        x=df_analysis.index,
        y='volatility',
        title=f'Volatility Analysis (Rolling Window = {window})',
        labels={'volatility': 'Volatility', 'index': 'Date'},
        template='plotly_white'
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Volatility",
        height=600,
        width=1000
    )

    st.plotly_chart(fig)

def technical_indicators(df):
    """
    Calculates and visualizes technical indicators (SMA, RSI, MACD) using Plotly.

    Args:
        df (pd.DataFrame): DataFrame containing the price data.

    Returns:
        None: Displays the indicators in Streamlit.
    """
    df_analysis = pd.DataFrame(df['price'])

    df_analysis['MA_7'] = SMAIndicator(df_analysis['price'], window=7).sma_indicator()
    df_analysis['MA_30'] = SMAIndicator(df_analysis['price'], window=30).sma_indicator()

    df_analysis['RSI'] = RSIIndicator(df_analysis['price'], window=14).rsi()

    macd = MACD(df_analysis['price'])
    df_analysis['MACD'] = macd.macd()
    df_analysis['MACD_signal'] = macd.macd_signal()
    df_analysis['MACD_hist'] = macd.macd_diff()

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df_analysis.index, y=df_analysis['price'], mode='lines', name='Price'))
    fig_price.add_trace(go.Scatter(x=df_analysis.index, y=df_analysis['MA_7'], mode='lines', name='MA 7'))
    fig_price.add_trace(go.Scatter(x=df_analysis.index, y=df_analysis['MA_30'], mode='lines', name='MA 30'))
    fig_price.update_layout(
        title="Price with Moving Averages",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Legend",
        template="plotly_white"
    )
    st.plotly_chart(fig_price)

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df_analysis.index, y=df_analysis['RSI'], mode='lines', name='RSI'))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig_rsi.update_layout(
        title="Relative Strength Index (RSI)",
        xaxis_title="Date",
        yaxis_title="RSI",
        template="plotly_white"
    )
    st.plotly_chart(fig_rsi)

    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df_analysis.index, y=df_analysis['MACD'], mode='lines', name='MACD'))
    fig_macd.add_trace(go.Scatter(x=df_analysis.index, y=df_analysis['MACD_signal'], mode='lines', name='MACD Signal'))
    fig_macd.add_trace(go.Bar(x=df_analysis.index, y=df_analysis['MACD_hist'], name='MACD Histogram'))
    fig_macd.update_layout(
        title="MACD (Moving Average Convergence Divergence)",
        xaxis_title="Date",
        yaxis_title="Value",
        template="plotly_white"
    )
    st.plotly_chart(fig_macd)


def combined_indicators(df):
    """
    Creates a combined plot with Price, Moving Averages, RSI, MACD, and MACD Histogram.

    Args:
        df (pd.DataFrame): DataFrame containing the price data.

    Returns:
        None: Displays the combined plot in Streamlit.
    """
    df['MA_7'] = SMAIndicator(df['price'], window=7).sma_indicator()
    df['MA_30'] = SMAIndicator(df['price'], window=30).sma_indicator()
    df['RSI'] = RSIIndicator(df['price'], window=14).rsi()

    macd = MACD(df['price'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_hist'] = macd.macd_diff()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['price'], mode='lines', name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_7'], mode='lines', name='MA 7'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_30'], mode='lines', name='MA 30'))

    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")


    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], mode='lines', name='MACD Signal'))


    fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], name='MACD Histogram'))

    fig.update_layout(
        title="Combined Indicators",
        xaxis_title="Date",
        yaxis_title="Value",
        height=800,
        width=1000,
        template="plotly_white",
        legend_title="Indicators",
        grid=dict(rows=4, columns=1, pattern="independent")
    )

    st.plotly_chart(fig)

def create_pred_visualization(full_df):  
    fig = go.Figure()
    fig.add_trace(go.Scatter(
                    x=full_df.index,
                    y=full_df['Predicted Price'],
                    mode='lines+markers',
                    line=dict(color='red', dash='dot'),
                    name="Prediction",
                    hovertemplate=(
                        "Date: %{x|%Y-%m-%d}<br>"
                        "Predicted Price: $%{y:,.2f}"
                    )
                ))
                
    fig.update_layout(
                    title="Bitcoin Price Forecast",
                    xaxis_title="Date",
                    yaxis_title="Predicted Price ($)",
                    yaxis_tickformat="$,.2f",
                    showlegend=True
                )

    st.plotly_chart(fig, use_container_width=True)
