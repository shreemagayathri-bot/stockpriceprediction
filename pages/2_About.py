import streamlit as st

# -----------------------------
# PAGE CONFIG (MUST BE FIRST!)
# -----------------------------
st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ About the Stock Price Predictor")
st.markdown("---")

st.markdown("""
## 🤖 Advanced LSTM Stock Price Predictor

This application leverages cutting-edge machine learning technology to provide accurate stock price predictions for the next 7 days.

### 🧠 Technology Stack

**Machine Learning:**
- **LSTM Neural Network**: Long Short-Term Memory network for time series prediction
- **TensorFlow/Keras**: Deep learning framework
- **Scikit-learn**: Linear regression comparison model

**Data Processing:**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **MinMaxScaler**: Data normalization

**Visualization:**
- **Matplotlib**: Chart generation
- **Streamlit**: Interactive web interface

**Data Source:**
- **Yahoo Finance (yfinance)**: Real-time stock data

### 📊 Features

#### 🔮 Prediction Engine
- **7-Day Forecast**: Predicts Open, High, Low, Close prices
- **60-Day Lookback**: Uses 60 days of historical data for predictions
- **Multi-Feature Input**: OHLC data for comprehensive analysis

#### 📈 Technical Analysis
- **RSI (Relative Strength Index)**: Momentum oscillator
- **Moving Averages**: 10-day MA for trend analysis
- **Trend Detection**: Bullish/Bearish market signals

#### 🎯 Investment Insights
- **Market Sentiment**: AI-powered sentiment analysis
- **Buy/Sell Recommendations**: Data-driven investment advice
- **Model Comparison**: LSTM vs Linear Regression accuracy metrics

### 🔧 Model Architecture

```
Input (60 days × 4 features)
       ↓
   LSTM Layers
       ↓
   Dense Layers
       ↓
Output (7 days × 4 features)
```

**LSTM Configuration:**
- Input shape: (60, 4) - 60 timesteps, 4 features
- Hidden layers: 50 units each
- Output: Multi-step prediction for 7 days
- Loss function: Mean Squared Error
- Optimizer: Adam

### 📋 Data Requirements

- **Minimum History**: 60 trading days
- **Features Used**: Open, High, Low, Close prices
- **Data Frequency**: Daily OHLC data
- **Preprocessing**: MinMax scaling (0-1 range)

### ⚠️ Important Disclaimers

**This application is for educational and research purposes only.**

- **Not Financial Advice**: Predictions are not guaranteed to be accurate
- **Market Risks**: Stock markets are volatile and unpredictable
- **Past Performance**: No indication of future results
- **Due Diligence**: Always conduct your own research before investing

### 📞 Support

For questions or feedback:
- Check the model files are present: `lstm_multistep_ohlc.h5` and `scaler_minmax.save`
- Ensure all dependencies are installed
- Verify internet connection for data fetching

### 🙏 Acknowledgments

- **TensorFlow/Keras** for the deep learning framework
- **Yahoo Finance** for stock data API
- **Streamlit** for the amazing web app framework
- **Scikit-learn** for machine learning utilities

---
**Created with ❤️ by AI Assistant | Powered by Streamlit + TensorFlow**
""")

# Add some stats
st.markdown("### 📊 App Statistics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Models", "2", "LSTM + Linear")
with col2:
    st.metric("Prediction Days", "7", "Business Days")
with col3:
    st.metric("Data Points", "60+", "Historical Days")

st.markdown("---")
st.caption("ℹ️ About | Advanced LSTM Stock Price Predictor")