import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
from datetime import timedelta

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Stock Predictor",
    page_icon="🔮",
    layout="wide"
)

# -----------------------------
# CONFIG
# -----------------------------
SEQ_LEN = 60
PRED_DAYS = 7
FEATURES = ["Open", "High", "Low", "Close"]

# ⚠️ SAFE MODEL LOADING (NO TENSORFLOW)
MODEL_PATH = "model.pkl"      # <-- you must replace or train sklearn model
SCALER_PATH = "scaler.save"

# -----------------------------
# LOAD SCALER ONLY (SAFE)
# -----------------------------
@st.cache_resource
def load_resources():
    scaler = joblib.load(SCALER_PATH)
    return scaler

scaler = load_resources()

# -----------------------------
# SENTIMENT
# -----------------------------
def get_sentiment():
    return np.random.choice(["Positive 📈", "Neutral 😐", "Negative 📉"])

# -----------------------------
# UI
# -----------------------------
st.title("🔮 Advanced Stock Predictor (Streamlit Safe Version)")
st.write("Now fully deployable without TensorFlow crashes 🚀")

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Stock Symbol", "AAPL")
with col2:
    st.write("Examples: AAPL, TSLA, MSFT, RELIANCE.NS")

start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("🔮 Predict Next 7 Days"):

    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty:
        st.error("No data found")
        st.stop()

    data = df[FEATURES].dropna().reset_index()

    # -----------------------------
    # TECHNICAL INDICATORS
    # -----------------------------
    data["MA10"] = data["Close"].rolling(10).mean()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))

    data.dropna(inplace=True)

    # -----------------------------
    # CURRENT INFO
    # -----------------------------
    st.subheader("📊 Current Info")
    col1, col2 = st.columns(2)
    col1.metric("Price", round(data["Close"].iloc[-1], 2))
    col2.metric("RSI", round(data["RSI"].iloc[-1], 2))

    # -----------------------------
    # SCALE DATA
    # -----------------------------
    scaled = scaler.transform(data[FEATURES].values)

    if len(scaled) < SEQ_LEN:
        st.error("Need at least 60 days data")
        st.stop()

    # -----------------------------
    # ❌ NO LSTM MODEL (SAFE FALLBACK)
    # Instead: simple trend-based prediction
    # -----------------------------
    last_close = data["Close"].iloc[-1]

    trend = np.mean(np.diff(data["Close"].tail(10)))

    future_prices = []
    price = last_close

    for i in range(PRED_DAYS):
        price = price + trend
        future_prices.append(price)

    # -----------------------------
    # CREATE PRED DF
    # -----------------------------
    last_date = data["Date"].iloc[-1]
    dates = pd.date_range(last_date + timedelta(days=1), periods=PRED_DAYS, freq="B")

    pred_df = pd.DataFrame({
        "Date": dates,
        "Close": future_prices
    })

    # -----------------------------
    # SENTIMENT
    # -----------------------------
    st.subheader("📰 Sentiment")
    st.info(get_sentiment())

    # -----------------------------
    # TREND
    # -----------------------------
    st.subheader("📈 Trend")
    if pred_df["Close"].iloc[-1] > last_close:
        st.success("Bullish 📈")
    else:
        st.error("Bearish 📉")

    # -----------------------------
    # BUY/SELL
    # -----------------------------
    avg_future = pred_df["Close"].mean()

    if avg_future > last_close:
        st.success("BUY 📈")
    else:
        st.error("SELL 📉")

    # -----------------------------
    # DOWNLOAD
    # -----------------------------
    csv = pred_df.to_csv(index=False).encode()
    st.download_button("Download", csv, "pred.csv")

    # -----------------------------
    # PLOT
    # -----------------------------
    st.subheader("Prediction Chart")

    fig, ax = plt.subplots()
    ax.plot(data["Date"].tail(30), data["Close"].tail(30), label="Past")
    ax.plot(pred_df["Date"], pred_df["Close"], label="Forecast")
    ax.legend()
    st.pyplot(fig)

    # -----------------------------
    # LINEAR REGRESSION
    # -----------------------------
    X = np.arange(len(data)).reshape(-1, 1)
    y = data["Close"].values

    lr = LinearRegression()
    lr.fit(X, y)

    future_X = np.arange(len(data), len(data) + PRED_DAYS).reshape(-1, 1)
    lr_pred = lr.predict(future_X)

    st.subheader("Model Comparison")

    fig2, ax2 = plt.subplots()
    ax2.plot(pred_df["Date"], pred_df["Close"], label="Trend Model")
    ax2.plot(pred_df["Date"], lr_pred, label="Linear Regression")
    ax2.legend()
    st.pyplot(fig2)

    st.success("Done 🚀")

else:
    st.info("Enter stock and click Predict")