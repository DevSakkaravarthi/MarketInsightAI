"""
MakeInsightAI with Pytourch and alphavantage API

"""
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

# Part 1: Fetching Data from Alpha Vantage
def get_historical_data(symbol, api_key):
    """
    Fetches historical daily stock data from Alpha Vantage.

    Args:
    - symbol (str): The stock symbol to fetch data for.
    - api_key (str):  Alpha Vantage API key.

    Returns:
    - DataFrame: Pandas DataFrame containing historical stock data.
    """
    base_url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": "full"
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    df = pd.DataFrame(data['Time Series (Daily)']).T.apply(pd.to_numeric)
    return df.iloc[::-1]

# Part 2: Dataset Class
class StockDataset(Dataset):
    """
    Custom PyTorch Dataset for stock data.

    Args:
    - series (Tensor): A tensor containing the stock data.
    - sequence_length (int): Length of the sequence to be used for training.
    """
    def __init__(self, series, sequence_length):
        self.data = series
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        sequence = self.data[index:index+self.sequence_length]
        label = self.data[index+self.sequence_length]
        return (sequence.unsqueeze(1), label)

# Part 3: LSTM Model
class LSTM(nn.Module):
    """
    LSTM model for time series prediction.

    Args:
    - input_size (int): Size of the input features. Default is 1.
    - hidden_layer_size (int): Size of the hidden layers.
    - output_size (int): Size of the output layer.
    """
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        batch_size, seq_len, _ = input_seq.size()
        hidden_state = torch.zeros(1, batch_size, self.hidden_layer_size).to(input_seq.device)
        cell_state = torch.zeros(1, batch_size, self.hidden_layer_size).to(input_seq.device)
        hidden_cell = (hidden_state, cell_state)

        lstm_out, hidden_cell = self.lstm(input_seq, hidden_cell)
        predictions = self.linear(lstm_out[:, -1, :])  # Get the last time step
        return predictions

# Part 4: Data Preprocessing and Loading
def preprocess_data(data):
    """
    Preprocesses the stock data for training.

    Args:
    - data (DataFrame): Pandas DataFrame containing the stock data.

    Returns:
    - Tensor: Preprocessed data as a PyTorch Tensor.
    - MinMaxScaler: Scaler used for the data transformation.
    """
    close_prices = data['4. close'].values.astype(float)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    close_prices_scaled = scaler.fit_transform(close_prices.reshape(-1, 1))
    close_prices_scaled = torch.FloatTensor(close_prices_scaled).view(-1)
    return close_prices_scaled, scaler

# Part 5: Training the Model
def train_model(model, train_loader, criterion, optimizer, epochs):
    """
    Trains the LSTM model.

    Args:
    - model (LSTM): The LSTM model to be trained.
    - train_loader (DataLoader): DataLoader for the training data.
    - criterion: Loss function.
    - optimizer: Optimization algorithm.
    - epochs (int): Number of training epochs.
    """
    for epoch in range(epochs):
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = criterion(y_pred, labels)
            single_loss.backward()
            optimizer.step()
        print(f'epoch: {epoch+1:3} loss: {single_loss.item():10.8f}')

# Main execution
api_key = ''  # Replace with your Alpha Vantage API key
symbol = ''  # Example stock symbol eg. AAPL(Apple)
data = get_historical_data(symbol, api_key)
print(data)
close_prices_scaled, scaler = preprocess_data(data)

sequence_length = 60


train_size = int(len(close_prices_scaled) * 0.8)

train_dataset = StockDataset(close_prices_scaled[:train_size], sequence_length)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

model = LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, epochs=5)

# Part 6: Predicting the Next 7 Days
def predict_next_days(model, start_sequence, days=7):
    """
    Predicts the next 'days' stock prices using the trained model.

    Args:
    - model (LSTM): The trained LSTM model.
    - start_sequence (list): The last known sequence of stock prices.
    - days (int): Number of days to predict.

    Returns:
    - list: Predicted stock prices for the next 'days'.
    """
    model.eval()  # Set the model to evaluation mode
    predicted_prices = []
    for _ in range(days):
        with torch.no_grad():
            seq = torch.FloatTensor(start_sequence[-sequence_length:]).unsqueeze(0).unsqueeze(-1)
            # Add an extra dimension for the number of features (which is 1 in this case)
            predicted_price = model(seq)
            predicted_prices.append(predicted_price.item())
            start_sequence.append(predicted_price.item())  # Update the sequence
    return predicted_prices

last_60_days = close_prices_scaled[-sequence_length:].tolist()
predicted_prices = predict_next_days(model, last_60_days, days=7)
predicted_prices_scaled = np.array(predicted_prices).reshape(-1, 1)
predicted_prices_original = scaler.inverse_transform(predicted_prices_scaled).reshape(-1)

print("Predicted Prices for the Next 7 Days:", predicted_prices_original)
