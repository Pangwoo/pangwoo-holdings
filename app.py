from flask import Flask, render_template, jsonify, request
import yfinance as yf
from models.model import model, get_predict
from models.dataGathering import data_inverse_transform
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz



app = Flask(__name__)


# 서울 시간대 지정
ny_tz = pytz.timezone('America/New_York')
kr_tz = pytz.timezone('Asia/Seoul')

# 현재 시간(서울 시간 기준)
now = datetime.now(kr_tz)

# 주식 리스트와 날짜 구간
stock_list = ["NVDA", "TSLA", "PLTR", "BABA", "DASH", "ADBE", "QQQ"]


start_time_line = ["2017-01-20","2017-01-20", "2021-01-20", "2022-01-20"]
end_time_line = ["2021-01-20","2021-01-20", "2025-01-20", "2025-01-20"]

start_test_date = str((now - timedelta(weeks=12)).date())
end_test_date = str((now).date())

stock_list = ["NVDA", "TSLA", "BABA", "ADBE", "QQQ", "NKE"]
crypto_list =["BTC-USD", "XRP-USD", 'DOGE-USD','ETH-USD','SOL-USD','ADA-USD']

new_stock_list = ["PLTR", "DASH"]

features = ['MA50', 'MA200', 'RSI', 'MACD', 'MACD_Signal','SMA_10', 'EMA_5', 'Close_Lag1', 'Close_Lag5', 'Volume']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_stock_predictions')
def get_stock_predictions():
    predictions = {}
    
    # 모델을 주식 리스트에 맞게 예측 결과 생성
    for ticker in stock_list:
        
        period = 7
        models = []
        for start, end in zip(start_time_line, end_time_line):
            
            # 모델 훈련
            trained_model = model(ticker, start, end)

            # 예측 실행
            models.append(trained_model)
        
        # ticker, models, start_test_date, end_test_date, pre
        predicted_data = get_predict(ticker, models, start_test_date, end_test_date,period)

        for key, value in predicted_data.items():
            predicted_data[key] = np.array(value).tolist()
        
        predictions[ticker] = predicted_data
        
    return predictions

@app.route('/get_crypto_predictions')
def get_crypto_predictions():
    predictions = {}
    
    # 모델을 주식 리스트에 맞게 예측 결과 생성
    for ticker in crypto_list:
        
        period = 7
        models = []
        for start, end in zip(start_time_line, end_time_line):
            
            # 모델 훈련
            trained_model = model(ticker, start, end)

            # 예측 실행
            models.append(trained_model)
        
        # ticker, models, start_test_date, end_test_date, pre
        predicted_data = get_predict(ticker, models, start_test_date, end_test_date,period)

        for key, value in predicted_data.items():
            predicted_data[key] = np.array(value).tolist()
        
        predictions[ticker] = predicted_data
        
    return predictions

@app.route('/get_ticker_predictions')
def get_ticker_predictions():
    ticker = request.args.get('ticker')
    
    predictions = {}

    period = 7
    models = []
    for start, end in zip(start_time_line, end_time_line):
        
        # 모델 훈련
        trained_model = model(ticker, start, end)

        # 예측 실행
        models.append(trained_model)
    
    # ticker, models, start_test_date, end_test_date, pre
    predicted_data = get_predict(ticker, models, start_test_date, end_test_date,period)

    for key, value in predicted_data.items():
        predicted_data[key] = np.array(value).tolist()
    
    predictions[ticker] = predicted_data

    return predictions

if __name__ == "__main__":
    app.run(debug=True,port=5001)