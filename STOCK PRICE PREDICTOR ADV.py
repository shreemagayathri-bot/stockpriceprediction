"""
Deployable Multi-step Stock Predictor (NO TensorFlow)
Predict next 7 days using RandomForest model
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math

# -------------------------
# CONFIG
# -------------------------
TICKER = "AAPL"
START_DATE = "2018-01-01"
SEQ_LEN = 60
PRED_DAYS = 7
FEATURES = ["Open", "High", "Low", "Close"]

# -------------------------
# LOAD DATA
# -------------------------
df = yf.download(TICKER, start=START_DATE)
df = df.reset_index()

data = df[["Date"] + FEATURES].dropna().reset_index(drop=True)

print("Rows:", len(data))

# -------------------------
# SCALE DATA
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
        y.append(data[i:i+pred_len].flatten())
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, SEQ_LEN, PRED_DAYS)

print("X shape:", X.shape)
print("y shape:", y.shape)

# -------------------------
# TRAIN TEST SPLIT
# -------------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -------------------------
# MODEL (NO TF)
# -------------------------
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -------------------------
# PREDICT
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# INVERSE TRANSFORM
# -------------------------
def invert(y, scaler):
    y = y.reshape(-1, len(FEATURES))
    return scaler.inverse_transform(y)

y_test_inv = invert(y_test, scaler)
y_pred_inv = invert(y_pred, scaler)

# -------------------------
# RMSE
# -------------------------
rmse = math.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print("RMSE:", rmse)

# -------------------------
# PLOT LAST SAMPLE
# -------------------------
sample = -1
actual = y_test_inv[sample].reshape(PRED_DAYS, len(FEATURES))
pred = y_pred_inv[sample].reshape(PRED_DAYS, len(FEATURES))

dates = pd.date_range(end=data["Date"].iloc[-1], periods=PRED_DAYS)

plt.figure(figsize=(12, 8))
for i, col in enumerate(FEATURES):
    plt.subplot(2, 2, i+1)
    plt.plot(actual[:, i], label="Actual")
    plt.plot(pred[:, i], label="Predicted")
    plt.title(col)
    plt.legend()

plt.tight_layout()
plt.show()

# -------------------------
# NEXT 7 DAYS PREDICTION
# -------------------------
last_window = scaled[-SEQ_LEN:].flatten().reshape(1, -1)
next_pred = model.predict(last_window)

next_pred = invert(next_pred, scaler).reshape(PRED_DAYS, len(FEATURES))

future_dates = pd.date_range(start=data["Date"].iloc[-1], periods=PRED_DAYS)

pred_df = pd.DataFrame(next_pred, columns=FEATURES)
pred_df.insert(0, "Date", future_dates)

print("\nNext 7 Days Prediction:")
print(pred_df)