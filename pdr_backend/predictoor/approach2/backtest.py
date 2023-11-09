from pathlib import Path
import pandas as pd
import ccxt
from dotenv import load_dotenv
import datetime
import pytz
import ta
import numpy as np
import os
import sys
from sklearn.metrics import accuracy_score as acc

from pdr_backend.util.env import getenv_or_exit

load_dotenv()

model_dir: str = getenv_or_exit("MODELDIR")
trained_models_dir = os.path.join(model_dir, "trained_models")
sys.path.append(model_dir)
from model import OceanModel

start_date = "2023-11-09 00:00:00+00:00"
start_dt = datetime.datetime.fromisoformat(start_date)

end_date = "2023-11-10 00:00:00+00:00"
end_dt = datetime.datetime.fromisoformat(end_date)

exchange_id = "binance"
pair = "BTC/USDT"
timeframe = "5m"

models = [
    OceanModel(exchange_id, pair, timeframe),
]

results_path = "results"
if not os.path.exists(results_path):
    os.makedirs(results_path)

results_csv_name = (
    "./"
    + results_path
    + "/backtest_"
    + str(start_dt.date())
    + "_"
    + str(end_dt.date())
    + "_"
    + exchange_id
    + "_"
    + pair.replace("/", "-").lower()
    + "_"
    + timeframe
    + ".csv"
)

exchange_class = getattr(ccxt, exchange_id)
exchange_ccxt = exchange_class({"timeout": 30000})

start_dt_ts = pd.Timestamp(start_date, tz="UTC")
end_dt_ts = pd.Timestamp(end_date, tz="UTC")

current_datetime = datetime.datetime.now().astimezone(pytz.utc)
end_dt = min(end_dt, current_datetime.astimezone(pytz.utc))
end_dt_ts = pd.Timestamp(end_dt)



start_time = int(start_dt_ts.timestamp() * 1000)
end_time = int(end_dt_ts.timestamp() * 1000)

flag = True
while flag:
    candles = exchange_ccxt.fetch_ohlcv(
        symbol=pair,
        timeframe=timeframe,
        since=start_time,
        limit=1000,
        params={"until": end_time},
    )
            
    main_pd = pd.DataFrame(
        candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    main_pd["datetime"] = pd.to_datetime(main_pd["timestamp"], unit="ms", utc=True)
    start_time = int(
        (main_pd.datetime.iloc[-1] + pd.DateOffset(minutes=5)).timestamp() * 1000
    )       
    
    flag = main_pd.datetime.iloc[-1] <= end_dt_ts - datetime.timedelta(minutes=15)
    print('Current Date: ', main_pd.datetime.iloc[-1])
    print("End Date: ", end_dt_ts - datetime.timedelta(minutes=15))

df = main_pd

df = ta.add_all_ta_features(
    main_pd,
    open="open",
    high="high",
    low="low",
    close="close",
    volume="volume",
    fillna=True,
)

df['volume_adi_diff'] = df['volume_adi'].diff() 
df['volume_obv_diff'] = df['volume_obv'].diff() 
df['volume_nvi_diff'] = df['volume_nvi'].pct_change() 

df = df.assign(sin_month=np.zeros(len(df)), cos_month=np.zeros(len(df)), sin_day=np.zeros(len(df)), cos_day=np.zeros(len(df)), sin_hour=np.zeros(len(df)), cos_hour=np.zeros(len(df)), sin_minute=np.zeros(len(df)), cos_minute=np.zeros(len(df)),)

time_features = np.zeros((len(df),8))

for i in range(len(time_features)):
    datetime = pd.to_datetime(df['datetime'][i], utc=True)
    time_features[i,0] = (np.sin(2 * np.pi * datetime.month/12))
    time_features[i,1] = (np.cos(2 * np.pi * datetime.month/12))
    time_features[i,2] = (np.sin(2 * np.pi * datetime.day/31))
    time_features[i,3] = (np.cos(2 * np.pi * datetime.day/31))
    time_features[i,4] = (np.sin(2 * np.pi * datetime.hour/24))
    time_features[i,5] = (np.cos(2 * np.pi * datetime.hour/24))
    time_features[i,6] = (np.sin(2 * np.pi * datetime.minute/60))
    time_features[i,7] = (np.cos(2 * np.pi * datetime.minute/60))

df[['sin_month','cos_minute', 'sin_day', 'cos_day' , 'sin_hour', 'cos_hour', 'sin_minute', 'cos_minute']] = time_features

df = df.drop(['volume_adi', 'volume_obv', 'volume_nvi', 'others_cr'], axis=1)

timesteps = len(df)

df = df.dropna()
x = df.drop(["timestamp", "datetime"], axis=1)


r = df["close"].diff() 
y = np.sign(r)
y[y <= 0] = 0

model = models[0]
model.unpickle_model(trained_models_dir)

n_fold = model.n_fold

accs = np.zeros((n_fold,))
y_pred_list = np.zeros((n_fold, timesteps-1))
conf_list = np.zeros((n_fold, timesteps-1))
for split in range(n_fold):
    predict = model.model[split].predict(x.shift(1))
    y_pred_list[split,:] = predict.argmax(axis=1)
    conf_list[split] = predict[:,0] 
    accs[split] = acc(y[1:], y_pred_list[split,1:])

y_pred = np.median(y_pred_list, axis=0)
conf = np.median(conf_list, axis=0)  




accuracy = acc(y[1:], y_pred[1:])

results = df[['timestamp','datetime', 'close',]]
results['timestamp'] = (results['timestamp'] / 1000).astype(int)
results['Pred'] = y_pred
results['Conf'] = conf
results['GT'] = y 

results = results[1:] # remove first row because of NaN
results = results.set_index("timestamp")

results.to_csv(results_csv_name, index=False)

print(results.tail(15))
print("Mean Accuracy: ", accuracy)