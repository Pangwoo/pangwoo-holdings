import yfinance as yf
import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import MinMaxScaler

def get_stock_data(ticker, start_date, end_date):
    # Yahoo Finance API
    data = yf.download(ticker, start = start_date, end = end_date)

    # remove the NANs
    data.dropna(inplace=True)

    return data

def preprocess_data(data):

    # Get the quantile 1 and 3
    Q1 = data['Close'].quantile(0.25)
    Q3 = data['Close'].quantile(0.75)

    IQR = Q3 - Q1

    # remove data out range of IQR
    data = data[~((data['Close'] < (Q1 - 1.5 * IQR)) | (data['Close'] > (Q3 + 1.5 * IQR)))]

    #
    scaler = MinMaxScaler()
    data[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])

    # 50일, 200일 단기/장기 이동 평균 계산
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()

    # 이동 평균 (5일, 10일)
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_10'] = data['Close'].rolling(window=10).mean()

    data['Target'] = data['Close'].shift(-7)
    data['Close_Lag1'] = data['Close'].shift(1)
    data['Close_Lag5'] = data['Close'].shift(5)
    data['Close_Lag15'] = data['Close'].shift(15)
    data['Close_Lag30'] = data['Close'].shift(30)

    # remove the NANs
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    # print(data['Target'])
    # 지수 이동 평균
    data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()

    # calculate RSI 
    rsi = talib.RSI(data['Close'].to_numpy().flatten(), timeperiod=14)
    data['RSI'] = rsi

    # calculate MACD
    macd, macdsignal, macdhist = talib.MACD(data['Close'].to_numpy().flatten(), fastperiod=12, slowperiod=26, signalperiod=9)
    data['MACD'] = macd
    data['MACD_Signal'] = macdsignal

    data['Price_Change'] = data['Close'].shift(-7) - data['Close']

    return data

def data_inverse_transform(ticker, start_date, end_date, predict):


    data = yf.download(ticker, start = start_date, end = end_date)
    actual_prices = data['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    scaler.fit(actual_prices)
    
    predicted_prices = scaler.inverse_transform(predict.reshape(-1, 1))

    return predicted_prices






