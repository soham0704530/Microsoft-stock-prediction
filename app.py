import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime

# =====================================
# 🚀 Load Dataset & Model
# =====================================

st.title("📈 Microsoft Stock Price Prediction with LSTM")

st.markdown("""
This app predicts Microsoft stock prices based on historical data using a trained **LSTM** model.
""")

# Load dataset
df = pd.read_csv("MicrosoftStock.csv")
df.columns = df.columns.str.strip().str.lower()
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df.dropna(subset=['date'], inplace=True)
df.set_index('date', inplace=True)

# Feature scaling
scaler = MinMaxScaler()
df['close_scaled'] = scaler.fit_transform(df[['close']])

# Load trained LSTM model
model = load_model("lstm_microsoft_model.h5")

# =====================================
# 🚀 Sidebar Controls
# =====================================
st.sidebar.header("Settings")

start_date = st.sidebar.date_input("Start Date", df.index.min().date())
end_date = st.sidebar.date_input("End Date", df.index.max().date())

if start_date >= end_date:
    st.sidebar.error("Start date must be before end date.")
    st.stop()

df_range = df.loc[start_date:end_date]

# =====================================
# 📊 Show Historical Data
# =====================================
st.subheader(f"📊 Microsoft Stock Prices: {start_date} → {end_date}")

st.line_chart(df_range['close'])

# =====================================
# 🚀 Predict on Test Set & Plot
# =====================================
st.subheader("✅ LSTM Predictions on Test Data")

split_idx = int(len(df) * 0.8)
close_scaled = df['close_scaled'].values
test_close = close_scaled[split_idx-60:]

def create_sequences(series, seq_len=60):
    X, y = [], []
    for i in range(len(series)-seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X), np.array(y)

X_test_lstm, y_test_lstm = create_sequences(test_close)
X_test_lstm = X_test_lstm.reshape((-1, 60, 1))

pred_scaled = model.predict(X_test_lstm, verbose=0)
pred = scaler.inverse_transform(pred_scaled)
y_test_true = scaler.inverse_transform(y_test_lstm.reshape(-1,1))

test_dates = df.index[split_idx+60:]

pred_df = pd.DataFrame({
    'Actual': y_test_true.flatten(),
    'Predicted': pred.flatten()
}, index=test_dates[-len(y_test_true):])

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(pred_df.index, pred_df['Actual'], label='Actual')
ax.plot(pred_df.index, pred_df['Predicted'], label='Predicted')
ax.set_title("Actual vs Predicted Stock Price (Test Set)")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# =====================================
# 🚀 Forecast Next 30 Days
# =====================================
st.subheader("🔷 Next 30-Day Forecast")

last_60 = close_scaled[-60:]
future_preds_scaled = []

for _ in range(30):
    seq_input = last_60[-60:].reshape((1, 60, 1))
    pred_next_scaled = model.predict(seq_input, verbose=0)[0,0]
    future_preds_scaled.append(pred_next_scaled)
    last_60 = np.append(last_60, pred_next_scaled)

future_preds = scaler.inverse_transform(np.array(future_preds_scaled).reshape(-1,1))
future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=30)

forecast_df = pd.DataFrame({
    'Predicted Price': future_preds.flatten()
}, index=future_dates)

fig2, ax2 = plt.subplots(figsize=(12,6))
ax2.plot(forecast_df.index, forecast_df['Predicted Price'], label='30-Day Forecast', color='green')
ax2.set_title("Next 30-Day Microsoft Stock Price Forecast")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)

st.write(forecast_df)

st.success("✅ App ready. Explore predictions above!")
