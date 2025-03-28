from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
# from dataGathering import get_stock_data, preprocess_data, data_inverse_transform
from models.dataGathering import get_stock_data, preprocess_data, data_inverse_transform
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import numpy as np

features = ['MA50', 'MA200', 'RSI', 'MACD', 'MACD_Signal','SMA_10', 'EMA_5', 'Close_Lag5', 'Volume']

def model(ticker, start_date, end_date):
    data = get_stock_data(ticker, start_date, end_date)
    data = preprocess_data(data)

    # features = ['MA50', 'MA200', 'RSI', 'MACD', 'MACD_Signal','SMA_5', 'SMA_10', 'EMA_5']
    X = data[features]
    y = data['Target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    

    model = xgb.XGBRegressor(
        objective='reg:squarederror',  # 회귀 문제이므로 squared error 사용
        n_estimators=100,  # 트리 개수
        learning_rate=0.1,  # 학습률
        max_depth=5,  # 트리의 최대 깊이
        subsample=0.8,  # 학습 데이터 일부 샘플링 (과적합 방지)
        colsample_bytree=0.8  # 트리마다 사용할 변수 비율
    )
    model.fit(X_train, y_train)

    # 예측 수행
    y_pred = model.predict(X_test)

    # RMSE (Root Mean Squared Error) 계산
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    return [model,rmse]

def test(ticker, model, start_test_date, end_test_date,pre):

    test_model = model

    data = get_stock_data(ticker,start_test_date, end_test_date)
    data = preprocess_data(data)

    # features = ['MA50', 'MA200', 'RSI', 'MACD', 'MACD_Signal','SMA_5', 'SMA_10', 'EMA_5']
    X_test = data[features]
    y_test = data['Target']

    y_pred = test_model.predict(X_test)

    data = get_stock_data(ticker, start_test_date, end_test_date)

    predicted_prices = data_inverse_transform(ticker, start_test_date, end_test_date, y_pred)

    # 날짜를 pandas datetime 형식으로 변환
    dates = pd.to_datetime(data.index)

    actual_prices = data['Close'].to_numpy().flatten()

    future_dates = [dates[-1] + timedelta(days=i) for i in range(1, 1+pre)]
    future_dates = np.array(future_dates, dtype='datetime64[ns]')
    comparison_prices = np.append( predicted_prices, [np.nan] * pre)
    dates = np.append(dates, future_dates)
    actual_prices = np.append(actual_prices, [np.nan] * pre)
    final_prediction = np.append([np.nan] * pre, predicted_prices)
    
    # 그래프 그리기
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_prices, marker='o', linestyle='-', label="Actual Prices", color='blue')
    plt.plot(dates, final_prediction, marker='s', linestyle='--', label="Predicted Prices", color='red')
    plt.plot(dates, comparison_prices, marker='^', linestyle='--', label="Predicted Prices", color='green')

    # 그래프 세부 설정
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.title(ticker + " Actual vs Predicted Stock Prices")
    plt.legend()
    plt.xticks(dates[::7], rotation=45)  # 날짜 기울여서 보기 좋게
    plt.grid(True)

    # 그래프 출력
    plt.show()
    return

def get_predict(ticker, models, start_test_date, end_test_date, pre):
    data = get_stock_data(ticker,start_test_date, end_test_date)
    processed_data = preprocess_data(data)

    model_predicted_prices = []

    # features = ['MA50', 'MA200', 'RSI', 'MACD', 'MACD_Signal','SMA_5', 'SMA_10', 'EMA_5']
    X_test = processed_data[features]
    y_test = processed_data['Target']

    model_weights = []
    weights = 0  # 총 가중치 합

    for model in models:    
        y_pred = model[0].predict(X_test)
        predicted_prices = data_inverse_transform(ticker, start_test_date, end_test_date, y_pred)
        model_predicted_prices.append(predicted_prices)
        weight = 1 / model[1]
        model_weights.append(weight)
        weights += weight  # 전체 가중치 합

    # 가중치 정규화
    model_weights = [w / weights for w in model_weights]

    # 날짜를 pandas datetime 형식으로 변환
    dates = pd.to_datetime(data.index)
    actual_prices =  data['Close'].to_numpy().flatten()

    final_prediction = np.zeros(len(actual_prices))

    for model, weight in zip(model_predicted_prices, model_weights): 
        final_prediction = [i + j for i, j in zip(final_prediction, (model * weight))]
    
    future_dates = [dates[-1] + timedelta(days=i) for i in range(1, 1+pre)]
    future_dates = np.array(future_dates, dtype='datetime64[ns]')
    comparison_prices = np.append( final_prediction, [np.nan] * pre)
    dates = np.append(dates, future_dates)
    actual_prices = np.append(actual_prices, [np.nan] * pre)
    final_prediction = np.append([np.nan] * pre, final_prediction)

    return {'dates': dates.astype('datetime64[D]').astype(str), 'actual_prices': actual_prices, 'final_prediction': final_prediction, 'comparison_prices': comparison_prices}

def show_chart(ticker, models, start_test_date, end_test_date, pre):

    data = get_stock_data(ticker,start_test_date, end_test_date)
    processed_data = preprocess_data(data)

    model_predicted_prices = []

    # features = ['MA50', 'MA200', 'RSI', 'MACD', 'MACD_Signal','SMA_5', 'SMA_10', 'EMA_5']
    X_test = processed_data[features]
    y_test = processed_data['Target']

    model_weights = []
    weights = 0  # 총 가중치 합

    for model in models:    
        y_pred = model[0].predict(X_test)
        predicted_prices = data_inverse_transform(ticker, start_test_date, end_test_date, y_pred)
        model_predicted_prices.append(predicted_prices)
        weight = 1 / model[1]
        model_weights.append(weight)
        weights += weight  # 전체 가중치 합

    # 가중치 정규화
    model_weights = [w / weights for w in model_weights]

    # 날짜를 pandas datetime 형식으로 변환
    dates = pd.to_datetime(data.index)
    actual_prices =  data['Close'].to_numpy().flatten()

    final_prediction = np.zeros(len(actual_prices))

    for model, weight in zip(model_predicted_prices, model_weights): 
        final_prediction = [i + j for i, j in zip(final_prediction, (model * weight))]
    
    future_dates = [dates[-1] + timedelta(days=i) for i in range(1, 1+pre)]
    future_dates = np.array(future_dates, dtype='datetime64[ns]')
    comparison_prices = np.append( final_prediction, [np.nan] * pre)
    dates = np.append(dates, future_dates)
    actual_prices = np.append(actual_prices, [np.nan] * pre)
    final_prediction = np.append([np.nan] * pre, final_prediction)
    
    # 그래프 그리기
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_prices, marker='o', linestyle='-', label="Actual Prices", color='blue')
    plt.plot(dates, final_prediction, marker='s', linestyle='--', label="Comparison Prices", color='red')
    plt.plot(dates, comparison_prices, marker='^', linestyle='--', label="Predicted Prices", color='green')

    # 그래프 세부 설정
    plt.xlabel("Date")
    plt.ylabel("Stock Price (USD)")
    plt.title(ticker + " Actual vs Predicted Stock Prices")
    plt.legend()
    plt.xticks(dates[::7], rotation=45)  # 날짜 기울여서 보기 좋게
    plt.grid(True)

    # 그래프 출력
    plt.show()
    return

# 뉴욕 시간대 지정
ny_tz = pytz.timezone('America/New_York')
kr_tz = pytz.timezone('Asia/Seoul')
# 현재 시간(뉴욕 시간 기준)
now = datetime.now(kr_tz)

start_time_line = ["2017-01-20","2017-01-20", "2021-01-20", "2022-01-20"]
end_time_line = ["2021-01-20","2021-01-20", "2025-01-20", "2025-01-20"]

start_test_date = str((now - timedelta(weeks=12)).date())
end_test_date = str((now).date())

stock_list = ["NVDA", "TSLA", "BABA", "ADBE", "QQQ", "NKE"]
crpyto_list =["BTC-USD", "XRP-USD", 'DOGE-USD','ETH-USD','SOL-USD','ADA-USD']

new_stock_list = ["PLTR", "DASH"]


# test_model = model('TSLA',start_time_line[1],end_time_line[1])
# test('TSLA', test_model, start_test_date, end_test_date)
# show_chart('TSLA', models, start_test_date,end_test_date) 


# for stock in stock_list:
#     models = []
#     for i in range(len(start_time_line)):
#         models.append(model(stock,start_time_line[i],end_time_line[i]))
#     show_chart(stock, models, start_test_date,end_test_date,7)


# PATH_model = model("PATH", '2022-03-20', '2025-01-20')
# test("PATH", PATH_model[0],start_test_date,end_test_date,7)
# for model in models:
#     test(stock,model[0],start_test_date,end_test_date) 


