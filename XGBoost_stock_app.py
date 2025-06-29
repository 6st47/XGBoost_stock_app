import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
import plotly.graph_objects as go
from datetime import timedelta

# --- データ取得 ---
st.title("Stock Price Prediction with LSTM")
ticker = st.text_input("Enter Stock Ticker:", "5411.T")
future_days = st.slider("Future days to predict:", 1, 100, value=30)

if st.button("Run Prediction"):
    data = yf.download(ticker, period="2y")
    close = data['Close'].dropna().values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(close)

    # --- シーケンス作成 ---
    seq_len = 30
    X, y = [], []
    for i in range(len(scaled_close) - seq_len):
        X.append(scaled_close[i:i+seq_len])
        y.append(scaled_close[i+seq_len])
    X, y = np.array(X), np.array(y)

    # --- 訓練・テスト分割 ---
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # --- LSTMモデル構築 ---
    model = Sequential([
        Input(shape=(seq_len, 1)),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), verbose=0)

    # --- 未来予測 ---
    last_seq = X_test[-1]
    future_preds = []
    for _ in range(future_days):
        pred = model.predict(last_seq.reshape(1, seq_len, 1), verbose=0)
        # クリッピング（直近終値の±15%に限定）
        pred = np.clip(pred, last_seq[-1] * 0.85, last_seq[-1] * 1.15)
        future_preds.append(pred[0, 0])
        last_seq = np.append(last_seq[1:], pred, axis=0)

    # --- 逆正規化 ---
    future_preds_inv = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    test_preds_inv = scaler.inverse_transform(model.predict(X_test)).flatten()
    y_test_inv = scaler.inverse_transform(y_test).flatten()

    # --- グラフ描画 ---
    past_dates = data.index[-len(y_test):]
    future_dates = pd.date_range(past_dates[-1] + timedelta(days=1), periods=future_days)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=close.flatten(), name='Actual Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=past_dates, y=test_preds_inv, name='Test Predicted', line=dict(color='orange', dash='dash')))
    fig.add_trace(go.Scatter(x=future_dates, y=future_preds_inv, name='Future Prediction', line=dict(color='green')))
    fig.update_layout(title=f"{ticker} - LSTM Prediction", xaxis_title="Date", yaxis_title="Close Price (JPY)")
    st.plotly_chart(fig)
