#streamlit run "D:\PYTHON VSCODE\six-months_python_for_data_science-mentorship-program-main\13_streamlit\10_stock_market_app\app.py"


import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Streamlit configuration
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

# Title and header
st.title('Stock Market Forecasting App')
st.subheader('Forecast stock prices for selected companies.')
st.image("https://img.freepik.com/free-vector/gradient-stock-market-concept_23-2149166910.jpg")

# Sidebar inputs
st.sidebar.header('Select Parameters')
start_date = st.sidebar.date_input('Start date', date(2020, 1, 1))
end_date = st.sidebar.date_input('End date', date(2020, 12, 31))
ticker_list = ["AAPL", "MSFT", "GOOG", "GOOGL", "META", "TSLA", "NVDA", "ADBE",
               "PYPL", "INTC", "CMCSA", "NFLX", "PEP"]
ticker = st.sidebar.selectbox('Select company', ticker_list)

# --- Download stock data ---
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

# --- Flatten MultiIndex columns if present ---
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] if col[1] == '' else f"{col[0]}_{col[1]}" for col in data.columns]

# Reset index to get Date as a column
data = data.reset_index()

st.write(f"Data from {start_date} to {end_date}")
st.write(data.head())

# --- Plot stock prices ---
st.header('Data Visualization')
numeric_cols = [col for col in data.columns if col != 'Date']
fig = px.line(data, x='Date', y=numeric_cols, title=f'{ticker} Stock Prices', width=1000, height=600)
st.plotly_chart(fig)

# --- Column selection for forecasting ---
column = st.selectbox('Select column for forecasting', numeric_cols)
data = data[['Date', column]]
st.write("Selected Data")
st.write(data.head())

# --- Stationarity check ---
st.header('Stationarity Test (ADF)')
adf_pvalue = adfuller(data[column])[1]
st.write(f"ADF p-value: {adf_pvalue:.5f}")
st.write("Stationary" if adf_pvalue < 0.05 else "Non-stationary")

# --- Decomposition ---
st.header('Seasonal Decomposition')
decomp = seasonal_decompose(data[column], model='additive', period=12)
st.pyplot(decomp.plot())

st.plotly_chart(px.line(x=data["Date"], y=decomp.trend, title='Trend', width=1000, height=400))
st.plotly_chart(px.line(x=data["Date"], y=decomp.seasonal, title='Seasonality', width=1000, height=400))
st.plotly_chart(px.line(x=data["Date"], y=decomp.resid, title='Residuals', width=1000, height=400))

# --- Model selection ---
models = ['SARIMA', 'Random Forest', 'LSTM', 'Prophet']
selected_model = st.sidebar.selectbox('Select forecasting model', models)

# --- SARIMA Model ---
if selected_model == 'SARIMA':
    p = st.slider('p', 0, 5, 2)
    d = st.slider('d', 0, 5, 1)
    q = st.slider('q', 0, 5, 2)
    seasonal_period = st.number_input('Seasonal period', 0, 24, 12)

    sarima_model = sm.tsa.statespace.SARIMAX(data[column], order=(p,d,q),
                                             seasonal_order=(p,d,q,seasonal_period)).fit()
    st.write(sarima_model.summary())

    forecast_days = st.number_input('Forecast days', 1, 365, 10)
    pred = sarima_model.get_prediction(start=len(data), end=len(data)+forecast_days).predicted_mean
    pred.index = pd.date_range(start=end_date, periods=len(pred), freq='D')
    pred = pd.DataFrame({'Date': pred.index, 'predicted': pred.values})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=pred['Date'], y=pred['predicted'], mode='lines', name='Forecast'))
    fig.update_layout(title='SARIMA Forecast', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
    st.plotly_chart(fig)

# --- Random Forest Model ---
elif selected_model == 'Random Forest':
    train_size = int(len(data)*0.8)
    train, test = data[:train_size], data[train_size:]

    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(np.arange(len(train)).reshape(-1,1), train[column])
    pred = rf.predict(np.arange(len(train), len(data)).reshape(-1,1))

    rmse = np.sqrt(mean_squared_error(test[column], pred))
    st.write(f"Test RMSE: {rmse:.5f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train['Date'], y=train[column], mode='lines', name='Train'))
    fig.add_trace(go.Scatter(x=test['Date'], y=test[column], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=test['Date'], y=pred, mode='lines', name='Predicted'))
    fig.update_layout(title='Random Forest Forecast', xaxis_title='Date', yaxis_title='Price', width=1000, height=400)
    st.plotly_chart(fig)

# --- LSTM Model ---
elif selected_model == 'LSTM':
    st.header('LSTM Forecasting')

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(data[column].values.reshape(-1,1))

    train_size = int(len(scaled_data)*0.8)
    train, test = scaled_data[:train_size], scaled_data[train_size:]

    def create_seq(dataset, seq_len):
        X, y = [], []
        for i in range(len(dataset)-seq_len):
            X.append(dataset[i:i+seq_len])
            y.append(dataset[i+seq_len])
        return np.array(X), np.array(y)

    seq_len = st.slider('Sequence length', 1, 30, 10)
    X_train, y_train = create_seq(train, seq_len)
    X_test, y_test = create_seq(test, seq_len)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)),
        LSTM(50),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

    pred = lstm_model.predict(X_test)
    pred = scaler.inverse_transform(pred)
    rmse = np.sqrt(mean_squared_error(scaler.inverse_transform(y_test.reshape(-1,1)), pred))
    st.write(f"LSTM Test RMSE: {rmse:.5f}")

# --- Prophet Model ---
elif selected_model == 'Prophet':
    st.header('Prophet Forecasting')
    prophet_df = data.rename(columns={'Date':'ds', column:'y'})
    prophet_model = Prophet()
    prophet_model.fit(prophet_df)
    future = prophet_model.make_future_dataframe(periods=365)
    forecast = prophet_model.predict(future)
    fig = prophet_model.plot(forecast)
    st.pyplot(fig)
