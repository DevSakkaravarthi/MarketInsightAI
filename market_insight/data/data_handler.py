import requests
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from market_insight.logger import setup_console_logger

class AlphavantageDataHandler:
    def __init__(self, api_key):
        self.logger = setup_console_logger('data_handler')
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.logger.info('Initialized DataHandler')


    def get_historical_data(self, symbol):
        self.logger.info(f'Fetching historical data for {symbol}')
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full"
        }
        response = requests.get(self.base_url, params=params)
        data = response.json()
        print(response.content)
        df = pd.DataFrame(data['Time Series (Daily)']).T.apply(pd.to_numeric)
        self.logger.info('Data fetching completed successfully')

        return df.iloc[::-1]

    def preprocess_data(self, data):    
        close_prices = data['4. close'].values.astype(float)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        close_prices_scaled = scaler.fit_transform(close_prices.reshape(-1, 1))
        close_prices_scaled = torch.FloatTensor(close_prices_scaled).view(-1)
        return close_prices_scaled, scaler

class PolygonDataHandler:
    def __init__(self, api_key):
        self.logger = setup_console_logger('data_handler')
        self.api_key = api_key
        self.base_url = "https://api.polygon.io/v2/aggs/ticker"  # Polygon API endpoint
        self.logger.info('Initialized DataHandler with Polygon.io')

    def get_historical_data(self, symbol):
        self.logger.info(f'Fetching historical data for {symbol}')
        params = {
            "apiKey": self.api_key,
            "limit": 120  # Adjust as needed
        }
        url = f"{self.base_url}/{symbol}/range/1/minute/2020-01-01/2023-01-01?limit=50000"  # Adjust dates as needed
        response = requests.get(url, params=params)
        data = response.json()
        print(data)
        # Check for errors
        if 'results' not in data:
            self.logger.error(f"Error fetching data: {data.get('error', 'Unknown Error')}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data['results'])
        df['t'] = pd.to_datetime(df['t'], unit='ms')  # Convert timestamp to datetime
        df.set_index('t', inplace=True)
        df.rename(columns={'c': 'Close', 'o': 'Open', 'h': 'High', 'l': 'Low', 'v': 'Volume'}, inplace=True)
        self.logger.info('Data fetching completed successfully')

        return df

    def preprocess_data(self, data):    
        close_prices = data['Close'].values.astype(float)  # Adjust column name if different
        scaler = MinMaxScaler(feature_range=(-1, 1))
        close_prices_scaled = scaler.fit_transform(close_prices.reshape(-1, 1))
        close_prices_scaled = torch.FloatTensor(close_prices_scaled).view(-1)
        return close_prices_scaled, scaler