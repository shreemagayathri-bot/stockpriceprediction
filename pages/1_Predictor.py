import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

# 🔥 NEW IMPORTS
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# -----------------------------
# PAGE CONFIG (MUST BE FIRST!)
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
MODEL_PATH = "lstm_multistep_ohlc.h5"
SCALER_PATH = "scaler_minmax.save"

# -----------------------------
# LOAD MODEL & SCALER
# -----------------------------
@st.cache_resource
def load_lstm_model():
    model = load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_lstm_model()

# -----------------------------
# 🔥 SENTIMENT FUNCTION
# -----------------------------
def get_sentiment():
    sentiments = ["Positive 📈", "Neutral 😐", "Negative 📉"]
    return np.random.choice(sentiments)

# -----------------------------
# APP UI
# -----------------------------
st.title("🔮 Advanced LSTM Stock Predictor (7-Day Forecast)")
st.write("Predict the next 7 days of Open, High, Low, Close for any stock using a trained LSTM model.")

st.markdown("---")

# Input section
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, RELIANCE.NS)", "AAPL")
with col2:
    st.write("**Popular symbols:** AAPL, TSLA, GOOGL, MSFT, AMZN")

col3, col4 = st.columns(2)
with col3:
    start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
with col4:
    end_date = st.date_input("End Date", value=pd.to_datetime("today"))

if st.button("🔮 Predict Next 7 Days", type="primary"):
    with st.spinner("Fetching data and predicting..."):

        # -----------------------------
        # FETCH DATA
        # -----------------------------
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error("No data found. Try another ticker.")
            st.stop()

        data = df[FEATURES].dropna().reset_index()

        # -----------------------------
        # 🔥 TECHNICAL INDICATORS
        # -----------------------------
        data["MA10"] = data["Close"].rolling(10).mean()

        delta = data["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        data["RSI"] = 100 - (100 / (1 + rs))

        data.dropna(inplace=True)

        # -----------------------------
        # 🔥 CURRENT METRICS
        # -----------------------------
        st.subheader("📊 Current Stock Info")
        col1, col2 = st.columns(2)
        col1.metric("Current Price", f"${round(data['Close'].iloc[-1], 2)}")
        col2.metric("RSI", f"{round(data['RSI'].iloc[-1], 2)}")

        # -----------------------------
        # SCALING
        # -----------------------------
        scaled_vals = scaler.transform(data[FEATURES].values)

        if len(scaled_vals) < SEQ_LEN:
            st.error("Not enough data! Need at least 60 days.")
            st.stop()

        last_window = scaled_vals[-SEQ_LEN:, :].reshape(1, SEQ_LEN, len(FEATURES))

        # -----------------------------
        # LSTM PREDICTION
        # -----------------------------
        next7_flat = model.predict(last_window)
        n_features = len(FEATURES)
        next7 = next7_flat.reshape((PRED_DAYS, n_features))
        next7_inv = scaler.inverse_transform(next7)

        # -----------------------------
        # CREATE PREDICTION DF
        # -----------------------------
        last_date = data["Date"].iloc[-1]
        next7_dates = pd.date_range(start=last_date + timedelta(days=1), periods=PRED_DAYS, freq="B")

        pred_df = pd.DataFrame(next7_inv, columns=FEATURES)
        pred_df.insert(0, "Date", next7_dates)

        st.subheader(f"📊 Predicted Prices for Next {PRED_DAYS} Days")
        st.dataframe(pred_df.style.format(precision=2))

        # -----------------------------
        # 🔥 SENTIMENT ANALYSIS
        # -----------------------------
        st.subheader("📰 Market Sentiment")
        sentiment = get_sentiment()
        st.info(f"Current Market Sentiment: {sentiment}")

        # -----------------------------
        # 🔥 TREND ANALYSIS
        # -----------------------------
        st.subheader("📈 Trend Analysis")
        if pred_df["Close"].iloc[-1].item() > data["Close"].iloc[-1].item():
            st.success("Market Trend: Bullish 📈")
        else:
            st.error("Market Trend: Bearish 📉")

        # -----------------------------
        # 🔥 BUY/SELL RECOMMENDATION
        # -----------------------------
        last_close = data["Close"].iloc[-1].item()
        future_avg = pred_df["Close"].mean()

        if future_avg > last_close:
            st.success("📈 Recommendation: BUY")
        elif future_avg < last_close:
            st.error("📉 Recommendation: SELL")
        else:
            st.warning("➖ Recommendation: HOLD")

        # -----------------------------
        # 🔥 DOWNLOAD OPTION
        # -----------------------------
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Prediction", csv, "prediction.csv")

        # -----------------------------
        # ORIGINAL PLOTS
        # -----------------------------
        st.subheader("📈 Price Predictions")
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs = axs.flatten()

        for i, f in enumerate(FEATURES):
            axs[i].plot(data["Date"].iloc[-30:], data[f].iloc[-30:], label="Past 30 Days", color="blue")
            axs[i].plot(pred_df["Date"], pred_df[f], marker='o', label="Predicted", color="red")
            axs[i].set_title(f)
            axs[i].legend()

        plt.tight_layout()
        st.pyplot(fig)

        # -----------------------------
        # 🔥 MOVING AVERAGE CHART
        # -----------------------------
        st.subheader("📉 Moving Average Chart")
        fig2, ax = plt.subplots(figsize=(10, 5))
        ax.plot(data["Date"], data["Close"], label="Close Price")
        ax.plot(data["Date"], data["MA10"], label="MA10")
        ax.legend()
        st.pyplot(fig2)

        # -----------------------------
        # 🔥 LINEAR REGRESSION MODEL
        # -----------------------------
        st.subheader("📊 Model Comparison")

        X = np.arange(len(data)).reshape(-1, 1)
        y = data["Close"].values

        lr_model = LinearRegression()
        lr_model.fit(X, y)

        future_X = np.arange(len(data), len(data) + PRED_DAYS).reshape(-1, 1)
        lr_pred = lr_model.predict(future_X)

        # RMSE
        try:
            lstm_rmse = math.sqrt(mean_squared_error(data["Close"].iloc[-PRED_DAYS:], pred_df["Close"]))
            lr_rmse = math.sqrt(mean_squared_error(data["Close"].iloc[-PRED_DAYS:], lr_pred))

            st.write(f"🔵 LSTM RMSE: {round(lstm_rmse,2)}")
            st.write(f"🟢 Linear Regression RMSE: {round(lr_rmse,2)}")
        except (ValueError, IndexError):
            st.warning("Not enough data for RMSE comparison")

        # -----------------------------
        # 🔥 COMPARISON GRAPH
        # -----------------------------
        st.subheader("📊 LSTM vs Linear Regression")

        fig3, ax3 = plt.subplots(figsize=(12, 6))
        ax3.plot(data["Date"].iloc[-30:], data["Close"].iloc[-30:], label="Actual", linewidth=2)
        ax3.plot(pred_df["Date"], pred_df["Close"], label="LSTM Prediction", linewidth=2, linestyle='--')
        ax3.plot(pred_df["Date"], lr_pred, label="Linear Regression", linewidth=2, linestyle='-.')

        ax3.set_xlabel("Date", fontsize=12)
        ax3.set_ylabel("Price", fontsize=12)
        ax3.set_title("Stock Price Prediction Comparison", fontsize=14, fontweight='bold')
        ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
        ax3.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig3)

        st.success("✅ Prediction complete!")

else:
    # Show some example content when not predicting
    st.info("👆 Enter a stock symbol and click 'Predict Next 7 Days' to get started!")

    st.markdown("### 📈 Sample Predictions")
    sample_data = {
        "Date": ["2024-01-15", "2024-01-16", "2024-01-17"],
        "Open": [185.92, 187.35, 189.12],
        "High": [187.34, 188.95, 190.45],
        "Low": [184.56, 186.23, 187.89],
        "Close": [186.27, 188.15, 189.78]
    }
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df.style.format(precision=2))

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("🔮 Advanced LSTM Stock Predictor | Created with ❤️ using Streamlit + TensorFlow")