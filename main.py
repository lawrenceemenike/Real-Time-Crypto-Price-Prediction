import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import logging

# Function to fetch historical data
def fetch_crypto_data(symbol, start_date, end_date):
    data = yf.download(f"{symbol}-USD",start=start_date, end=end_date)
    return data

# Function to calculate technical indicators
def add_technical_indicators(df):
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['Close'])
    df['Bollinger_High'] = ta.volatility.bollinger_hband(df['Close'])
    df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['Close'])
    return df

# Function to prepare featues for model training
def prepare_features(df):
    df['Returns'] = df['Close'].pct_change()
    df['Target'] = df['Returns'].shift(-1)  # This creates a NaN in the last row
    
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'SMA_20', 'EMA_20', 'RSI', 'MACD', 'Bollinger_High', 'Bollinger_Low']
    
    # Create feature dataframe
    X = df[features]
    y = df['Target']
    
    # Drop rows with NaN values in either X or y
    valid_indices = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_indices]
    y = y[valid_indices]
    
    return X, y

# Function to train the model
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, X_test, y_test

# Function to evaluate the model 
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return mse, mae, r2

# Function to make predictions 
def make_prediction(model, latest_data):
    return model.predict(latest_data.reshape(1, -1))[0]  

# Function to plot price and indicators
def plot_price_and_indicators(df):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], name='EMA 20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_High'], name='Bollinger High', line=dict(color='green', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['Bollinger_Low'], name='Bollinger Low', line=dict(color='red', dash='dash')), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=800, title='Price and Technical Indicators', xaxis_rangeslider_visible=False)
    return fig


# Streamlit app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    st.set_page_config(page_title="Crypto Price Prediction App", layout="wide")
    st.title("Crypto Price Prediction App")
    
    # Sidebar
    st.sidebar.header("Settings")
    crypto_symbol = st.sidebar.selectbox("Select Cryptocurrency", ["BTC", "ETH", "ADA", "XRP", "DOT"])
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2023-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))
    
    try:
        # Fetch and process data
        data = fetch_crypto_data(crypto_symbol, start_date, end_date)
        logger.info(f"Data shape after fetching: {data.shape}")

        if data.empty:
            st.error("No data available for the selected date range and cryptocurrency.")
            return

        if data.isnull().values.any():
            st.warning("The dataset contains some missing values. They will be handled during processing.")

        data_with_indicators = add_technical_indicators(data)
        logger.info(f"Data shape after adding indicators: {data_with_indicators.shape}")

        X, y = prepare_features(data_with_indicators)
        logger.info(f"X shape after preparation: {X.shape}")
        logger.info(f"y shape after preparation: {y.shape}")

        if len(X) != len(y):
            raise ValueError(f"Mismatch in number of samples: X has {len(X)}, y has {len(y)}")
        
        # Train model
        model, X_test, y_test = train_model(X, y)
        
        # Evaluate model
        mse, mae, r2 = evaluate_model(model, X_test, y_test)

        # Make prediction
        latest_data = X.iloc[-1].values
        prediction = make_prediction(model, latest_data)  # Fixed typo

        # Display KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}", f"{data['Returns'].iloc[-1]*100:.2f}%")
        col2.metric("Predicted Return", f"{prediction*100:.2f}%")
        col3.metric("30-Day SMA", f"${data['SMA_20'].iloc[-1]:.2f}")
        col4.metric("RSI", f"{data['RSI'].iloc[-1]:.2f}")

        # Plot price and indicators
        st.plotly_chart(plot_price_and_indicators(data_with_indicators), use_container_width=True)

        # Model performance
        st.header("Model Performance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Squared Error", f"{mse:.4f}")
        col2.metric("Mean Absolute Error", f"{mae:.4f}")
        col3.metric("R-squared", f"{r2:.4f}")

        # Recommendations
        st.header("Recommendations")
        rsi = data['RSI'].iloc[-1]
        if rsi > 70:
            st.warning("RSI indicates overbought conditions. Consider taking profits or waiting for a pullback before entering new positions.")
        elif rsi < 30:
            st.success("RSI indicates oversold conditions. This might be a good opportunity to accumulate or enter new positions.")
        else:
            st.info("RSI is in a neutral zone. Monitor other indicators and market conditions for trading decisions.")

        if prediction > 0:
            st.success(f"The model predicts a positive return of {prediction*100:.2f}% for the next period. Consider a bullish strategy.")
        else:
            st.warning(f"The model predicts a negative return of {prediction*100:.2f}% for the next period. Consider a bearish or hedging strategy.")

        st.warning("Disclaimer: These predictions and recommendations are based on historical data and should not be the sole basis for investment decisions. Always do your own research and consider seeking advice from financial professionals.")

    except ValueError as ve:
        st.error(f"Data preparation error: {str(ve)}")
        st.info("Please try adjusting the date range or check the data source.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please try again with different parameters or contact support if the issue persists.")

if __name__ == "__main__":
    main()