import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math
import time

# -------------------------
# CONFIG
# -------------------------
st.set_page_config(page_title="Stock Predictor ADV", layout="wide")

st.title("📈 Stock Price Predictor (ADV - Safe Version)")
st.write("RandomForest based forecasting (Streamlit Cloud safe)")

TICKER = st.text_input("Enter Stock Ticker", "AAPL")

START_DATE = "2018-01-01"
SEQ_LEN = 60
PRED_DAYS = 7
FEATURES = ["Open", "High", "Low", "Close"]

# -------------------------
# SAFE DATA LOADER
# -------------------------
@st.cache_data
def load_data(ticker):
    for _ in range(3):
        try:
            df = yf.download(ticker, start=START_DATE, progress=False)

            if df is not None and not df.empty:
                df = df.reset_index()
                df = df[["Date"] + FEATURES]

                df = df.dropna().reset_index(drop=True)

                if len(df) > 80:
                    return df
        except:
            time.sleep(2)

    return None

data = load_data(TICKER)

if data is None:
    st.error("❌ No data found. Try AAPL, TSLA, MSFT, INFY.NS")
    st.stop()

st.success(f"Data loaded: {len(data)} rows")

# -------------------------
# SCALE
# -------------------------
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[FEATURES])

# -------------------------
# CREATE SEQUENCES
# -------------------------
def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(seq_len, len(data) - pred_len):
        X.append(data[i-seq_len:i].flatten())
        y.append(data[i:i+pred_len])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, SEQ_LEN, PRED_DAYS)

split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------
# MODEL (FIXED)
# -------------------------
model = RandomForestRegressor(
    n_estimators=120,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train.reshape(y_train.shape[0], -1))

y_pred = model.predict(X_test)

# -------------------------
# INVERSE TRANSFORM FIX
# -------------------------
def inverse_transform(data):
    return scaler.inverse_transform(data)

y_test_inv = inverse_transform(y_test.reshape(-1, len(FEATURES)))
y_pred_inv = inverse_transform(y_pred)

rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
st.subheader(f"📊 RMSE: {rmse:.2f}")

# -------------------------
# PLOT SAFE
# -------------------------
st.subheader("📉 Actual vs Predicted")

sample = 0

actual = y_test_inv.reshape(-1, PRED_DAYS, len(FEATURES))[sample]
pred = y_pred_inv.reshape(-1, PRED_DAYS, len(FEATURES))[sample]

fig, ax = plt.subplots(2, 2, figsize=(12, 8))

for i, col in enumerate(FEATURES):
    r, c = divmod(i, 2)
    ax[r, c].plot(actual[:, i], label="Actual")
    ax[r, c].plot(pred[:, i], label="Predicted")
    ax[r, c].set_title(col)
    ax[r, c].legend()

st.pyplot(fig)

# -------------------------
# FUTURE PREDICTION (FIXED)
# -------------------------
st.subheader("🔮 Next 7 Days Prediction")

last_window = scaled[-SEQ_LEN:].flatten().reshape(1, -1)

next_pred = model.predict(last_window)

next_pred = next_pred.reshape(PRED_DAYS, len(FEATURES))

next_pred = inverse_transform(next_pred)

future_dates = pd.date_range(
    start=data["Date"].iloc[-1] + pd.Timedelta(days=1),
    periods=PRED_DAYS,
    freq="B"
)

pred_df = pd.DataFrame(next_pred, columns=FEATURES)
pred_df.insert(0, "Date", future_dates)

st.dataframe(pred_df)