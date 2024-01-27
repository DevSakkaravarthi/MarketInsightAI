import streamlit as st
import torch
import os
import numpy as np
from market_insight.data.data_handler import PolygonDataHandler
from market_insight.models.lstm_model import LSTM
from market_insight.training.trainer import Trainer
from market_insight.prediction.predictor import Predictor
from torch.utils.data import Dataset
import datetime
import pandas as pd


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
    poligon_api_key = os.environ.get('POLIGON_API_KEY')
    data_handler = PolygonDataHandler(api_key=poligon_api_key)
    data = data_handler.get_historical_data(symbol)
    scaled_data, scaler = data_handler.preprocess_data(data)

    # Extract the last sequence
    # Assuming your model uses the last 60 data points
    last_sequence = scaled_data[-60:]

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

    print(
        f"Model, scaler, and last sequence for {symbol} saved as {model_filename}, {scaler_filename}, and {last_sequence_filename}")


def load_model(symbol):
    model_filename = f'model/{symbol}_lstm_model.pth'
    scaler_filename = f'model/{symbol}_scaler.pth'
    last_sequence_filename = f'model/{symbol}_last_sequence.pth'

    if os.path.exists(model_filename) and os.path.exists(scaler_filename) and os.path.exists(last_sequence_filename):
        model = LSTM()
        model.load_state_dict(torch.load(model_filename))
        scaler = torch.load(scaler_filename)
        last_sequence = torch.load(last_sequence_filename)
        return model, scaler, last_sequence
    else:
        return None, None, None


def predict_stock_price(model, scaler, last_sequence, day):
    # Use the last sequence for the symbol
    start_sequence = last_sequence[-60:].tolist()
    predictor = Predictor(model)
    predicted_prices = predictor.predict_next_days(start_sequence, day)
    predicted_prices_scaled = np.array(predicted_prices).reshape(-1, 1)
    predicted_prices_original = scaler.inverse_transform(
        predicted_prices_scaled).reshape(-1)
    # Generate response
    response = []
    today = datetime.date.today()
    for i in range(day):
        date = today + datetime.timedelta(days=i)
        response.append({
            "date": date.isoformat(),
            "predicted_high": predicted_prices_original[i],
            "predicted_low": '-'
        })
    return response
# Streamlit app


def main():
    st.title('Market Insight AI')

    symbol = st.text_input(
        "Enter the stock symbol (e.g., AAPL):", value='AAPL')
    day = st.text_input(
        "No of future days", value='10')

    model, scaler, last_sequence = None, None, None 

    if st.button('Train/Load Model'):
        with st.spinner('Processing...'):
            model, scaler, last_sequence = load_model(symbol)
            if model is None:
                st.info(
                    f"No pre-trained model found for {symbol}. Training a new model.")
                train_and_save_model_for_symbol(symbol)
                model, scaler, last_sequence = load_model(symbol)

            if model:
                model, scaler, last_sequence = load_model(symbol)
                st.success(f'Model ready for {symbol}!')

    if st.button('Predict'):
        with st.spinner('Making prediction...'):
            try:
                model, scaler, last_sequence = load_model(symbol)
                prediction = predict_stock_price(
                    model, scaler, last_sequence, int(day))

                # Convert predictions to DataFrame and display
                predictions_df = pd.DataFrame(prediction)
                st.success(
                    f'Predicted stock price for {symbol}!')
                st.dataframe(predictions_df)

     
            except Exception as e:
                st.error(f'An error occurred: {e}')


if __name__ == "__main__":
    main()
