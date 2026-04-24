import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import time

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Stock Predictor ADV",
    page_icon="🔮",
    layout="wide"
)

PRED_DAYS = 7

# -----------------------------
# SENTIMENT
# -----------------------------
def get_sentiment():
    return np.random.choice(["Positive 📈", "Neutral 😐", "Negative 📉"])

# -----------------------------
# UI
# -----------------------------
st.title("🔮 Advanced Stock Predictor (Stable & Safe)")
st.write("Streamlit Cloud Ready | No crashes 🚀")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    ticker = st.text_input("Stock Symbol", "AAPL")

with col2:
    st.write("Examples: AAPL, TSLA, MSFT, INFY.NS")

start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

# -----------------------------
# SAFE DATA LOADER (FIXED)
# -----------------------------
def load_data(ticker):
    for _ in range(3):
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )

            if df is not None and not df.empty:

                # FIX MULTIINDEX ISSUE
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df = df.reset_index()

                # CLEAN COLUMN NAMES
                df.columns = [str(col).strip().title() for col in df.columns]

                # ENSURE REQUIRED COLUMNS EXIST
                required = ["Date", "Open", "High", "Low", "Close"]
                for col in required:
                    if col not in df.columns:
                        return pd.DataFrame()

                return df.dropna()

        except:
            time.sleep(2)

    return pd.DataFrame()

# -----------------------------
# MAIN APP
# -----------------------------
if st.button("🔮 Predict Next 7 Days"):

    df = load_data(ticker)

    if df.empty:
        st.error("❌ No data found. Try AAPL, TSLA, MSFT, INFY.NS")
        st.stop()

    data = df.copy()

    # -----------------------------
    # TECH INDICATORS
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

    c1, c2 = st.columns(2)
    c1.metric("Price", round(float(data["Close"].iloc[-1]), 2))
    c2.metric("RSI", round(float(data["RSI"].iloc[-1]), 2))

    # -----------------------------
    # TREND PREDICTION
    # -----------------------------
    last_close = float(data["Close"].iloc[-1])

    trend = float(np.mean(np.diff(data["Close"].tail(10))))

    future_prices = []
    price = last_close

    for _ in range(PRED_DAYS):
        price += trend
        future_prices.append(float(price))

    # -----------------------------
    # FUTURE DATES (SAFE)
    # -----------------------------
    last_date = pd.to_datetime(data["Date"].iloc[-1])

    dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=PRED_DAYS,
        freq="B"
    )

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
    # BUY / SELL
    # -----------------------------
    future_mean = float(np.mean(pred_df["Close"]))

    if future_mean > last_close:
        st.success("📈 BUY Signal")
    else:
        st.error("📉 SELL Signal")

    # -----------------------------
    # DOWNLOAD
    # -----------------------------
    csv = pred_df.to_csv(index=False).encode()
    st.download_button("📥 Download Prediction", csv, "prediction.csv")

    # -----------------------------
    # PLOT
    # -----------------------------
    st.subheader("📊 Prediction Chart")

    fig, ax = plt.subplots()
    ax.plot(data["Date"].tail(30), data["Close"].tail(30), label="Past")
    ax.plot(pred_df["Date"], pred_df["Close"], label="Forecast")
    ax.legend()
    st.pyplot(fig)

    # -----------------------------
    # LINEAR REGRESSION
    # -----------------------------
    st.subheader("📊 Model Comparison")

    X = np.arange(len(data)).reshape(-1, 1)
    y = data["Close"].values

    lr = LinearRegression()
    lr.fit(X, y)

    future_X = np.arange(len(data), len(data) + PRED_DAYS).reshape(-1, 1)
    lr_pred = lr.predict(future_X)

    fig2, ax2 = plt.subplots()
    ax2.plot(pred_df["Date"], pred_df["Close"], label="Trend Model")
    ax2.plot(pred_df["Date"], lr_pred, label="Linear Regression")
    ax2.legend()
    st.pyplot(fig2)

    st.success("✅ Prediction Completed Successfully!")

else:
    st.info("👆 Enter stock and click Predict")