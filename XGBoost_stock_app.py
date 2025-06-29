import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import plotly.graph_objects as go
from datetime import timedelta

st.title("Stock Price Prediction with XGBoost")

ticker = st.text_input("Enter Stock Ticker:", "5411.T")
future_days = st.slider("Future days to predict:", 1, 100, value=30)

if st.button("Run Prediction"):
    data = yf.download(ticker, period="2y")
    close = data['Close'].dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(close)

    # --- create sequences ---
    seq_len = 30
    X, y = [], []
    for i in range(len(scaled_close) - seq_len):
        X.append(scaled_close[i:i+seq_len].flatten())
        y.append(scaled_close[i+seq_len][0])
    X, y = np.array(X), np.array(y)

    # --- train/test split ---
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- train model ---
    model = XGBRegressor(n_estimators=100, learning_rate=0.05)
    model.fit(X_train, y_train)

    # --- future prediction ---
    last_seq = X_test[-1].reshape(1, -1)
    future_preds = []
    for _ in range(future_days):
        pred = model.predict(last_seq)[0]
        pred = np.clip(pred, last_seq[0][-1] * 0.85, last_seq[0][-1] * 1.15)
        future_preds.append(pred)
        last_seq = np.append(last_seq[:, 1:], [[pred]], axis=1)

    # --- inverse scaling ---
    future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    test_preds_inv = scaler.inverse_transform(model.predict(X_test).reshape(-1, 1)).flatten()
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # --- plot ---
    past_dates = data.index[-len(y_test):]
    future_dates = pd.date_range(past_dates[-1] + timedelta(days=1), periods=future_days)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=close.flatten(), name='Actual Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=past_dates, y=test_preds_inv, name='Test Predicted', line=dict(color='orange', dash='dash')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds_inv, name='Future Prediction', line=dict(color='green')))
    fig.update_layout(title=f"{ticker} - XGBoost Prediction", xaxis_title="Date", yaxis_title="Close Price (JPY)")
    st.plotly_chart(fig)
