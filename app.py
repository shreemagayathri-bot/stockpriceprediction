import streamlit as st

# -----------------------------
# HOME PAGE
# -----------------------------
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Advanced Stock Price Predictor")
st.markdown("---")

st.markdown("""
## Welcome to the Advanced LSTM Stock Price Predictor! 🚀

This application uses cutting-edge machine learning to predict stock prices for the next 7 days.

### 📊 Features:
- **LSTM Neural Network**: Trained on historical OHLC data
- **7-Day Forecast**: Predict Open, High, Low, Close prices
- **Technical Indicators**: RSI and Moving Averages
- **Market Sentiment**: AI-powered sentiment analysis
- **Trend Analysis**: Bullish/Bearish market predictions
- **Buy/Sell Recommendations**: Data-driven investment advice
- **Model Comparison**: LSTM vs Linear Regression comparison

### 🧭 Navigation:
Use the sidebar to navigate between different sections:
- **🏠 Home**: This overview page
- **🔮 Predictor**: Main stock prediction tool
- **ℹ️ About**: Information about the app

### 🚀 Getting Started:
1. Go to the **Predictor** page
2. Enter a stock symbol (e.g., AAPL, TSLA, GOOGL)
3. Select date range
4. Click "🔮 Predict Next 7 Days"
5. View predictions, charts, and analysis!

---
**Created with ❤️ using Streamlit + TensorFlow**
""")

# Add some visual elements
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Models", "LSTM + Linear", "Regression")
with col2:
    st.metric("Prediction", "7 Days", "Ahead")
with col3:
    st.metric("Data Source", "Yahoo Finance", "Real-time")