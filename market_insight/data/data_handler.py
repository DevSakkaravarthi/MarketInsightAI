import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch

class DataHandler:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def get_historical_data(self, symbol):
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full"
        }
        response = requests.get(self.base_url, params=params)
        data = response.json()
        df = pd.DataFrame(data['Time Series (Daily)']).T.apply(pd.to_numeric)
        return df.iloc[::-1]

    def preprocess_data(self, data):    
        close_prices = data['4. close'].values.astype(float)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        close_prices_scaled = scaler.fit_transform(close_prices.reshape(-1, 1))
        close_prices_scaled = torch.FloatTensor(close_prices_scaled).view(-1)
        return close_prices_scaled, scaler
