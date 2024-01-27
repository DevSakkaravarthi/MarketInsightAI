import torch
import numpy as np
from market_insight.data.data_handler import AlphavantageDataHandler, PolygonDataHandler
from market_insight.models.lstm_model import LSTM
from market_insight.training.trainer import Trainer
from market_insight.prediction.predictor import Predictor
from torch.utils.data import Dataset
import os

class StockDataset(Dataset):
    def __init__(self, series, sequence_length):
        self.data = series
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        sequence = self.data[index:index+self.sequence_length]
        label = self.data[index+self.sequence_length]
        return (sequence.unsqueeze(1), label)
    
def train_and_save_model_for_symbol(symbol):
    # alpha_vantage_api_key =os.environ.get('ALPHA_VANTAGE_API_KEY')
    # data_handler = AlphavantageDataHandler(api_key='alpha_vantage_api_key)
    poligon_api_key = os.environ.get('POLIGON_API_KEY')
    data_handler = PolygonDataHandler(api_key=poligon_api_key)
    data = data_handler.get_historical_data(symbol)
    scaled_data, scaler = data_handler.preprocess_data(data)

    # Extract the last sequence
    last_sequence = scaled_data[-60:]  # Assuming your model uses the last 60 data points

    model = LSTM()
    train_dataset = StockDataset(scaled_data, 60)
    trainer = Trainer(model, train_dataset)
    trainer.train_model()

    # Save Model, Scaler, and Last Sequence with symbol as part of the filename
    model_filename = f'model/{symbol}_lstm_model.pth'
    scaler_filename = f'model/{symbol}_scaler.pth'
    last_sequence_filename = f'model/{symbol}_last_sequence.pth'
    
    torch.save(model.state_dict(), model_filename)
    torch.save(scaler, scaler_filename)
    torch.save(last_sequence, last_sequence_filename)

    print(f"Model, scaler, and last sequence for {symbol} saved as {model_filename}, {scaler_filename}, and {last_sequence_filename}")

if __name__ == "__main__":
    stock_symbols = ['AAPL']  # List of stock symbols
    for symbol in stock_symbols:
        train_and_save_model_for_symbol(symbol)