# main.py
from market_insight.data.data_handler import DataHandler
from market_insight.models.lstm_model import LSTM
from market_insight.training.trainer import Trainer
from market_insight.prediction.predictor import Predictor
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np


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
    
# Initialize DataHandler
data_handler = DataHandler(api_key='')
data = data_handler.get_historical_data('')
scaled_data, scaler = data_handler.preprocess_data(data)

# Prepare the model and training
model = LSTM()
train_size = int(len(scaled_data) * 0.8)
train_dataset = StockDataset(scaled_data[:int(len(scaled_data) * 0.8)], 60)
trainer = Trainer(model, train_dataset)
trainer.train_model()

# Prediction
predictor = Predictor(model)
# Ensure start_sequence is a list and has enough data points

start_sequence = scaled_data[-60:].tolist()
predicted_prices = predictor.predict_next_days(start_sequence, 7)
predicted_prices_scaled = np.array(predicted_prices).reshape(-1, 1)
predicted_prices_original = scaler.inverse_transform(predicted_prices_scaled).reshape(-1)
print("Predicted Prices for the Next 7 Days:", predicted_prices_original)