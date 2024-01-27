from flask import Flask, request, jsonify
import torch
import numpy as np
from market_insight.models.lstm_model import LSTM
from market_insight.prediction.predictor import Predictor
import datetime

app = Flask(__name__)


# # Load the model and scalers
# model = LSTM()
# model.load_state_dict(torch.load('lstm_model.pth'))
# model.eval()
# scalers = torch.load('scalers.pth')
# last_sequences = torch.load('last_sequences.pth')

def load_model_scaler_and_last_sequence(symbol):
    model_filename = f'model/{symbol}_lstm_model.pth'
    scaler_filename = f'model/{symbol}_scaler.pth'
    last_sequence_filename = f'model/{symbol}_last_sequence.pth'

    model = LSTM()
    model.load_state_dict(torch.load(model_filename))
    model.eval()
    scaler = torch.load(scaler_filename)
    last_sequence = torch.load(last_sequence_filename)

    return model, scaler, last_sequence

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    symbol = data['symbol']
    days = data.get('days', 7)  # Default to 7 days if not specified
    model, scaler, last_sequence = load_model_scaler_and_last_sequence(symbol)

    # Use the last sequence for the symbol
    start_sequence = last_sequence[-60:].tolist() 

    # Ensure 'days' is within a reasonable range
    if not 1 <= days <= 30:  # Example: limit predictions to between 1 and 30 days
        return jsonify({"error": "Number of days must be between 1 and 30"}), 400



    # Make prediction
    predictor = Predictor(model)
    predicted_prices = predictor.predict_next_days(start_sequence, days)
    predicted_prices_scaled = np.array(predicted_prices).reshape(-1, 1)
    predicted_prices_original = scaler.inverse_transform(predicted_prices_scaled).reshape(-1)


    # Generate response
    response = []
    today = datetime.date.today()
    for i in range(days):
        date = today + datetime.timedelta(days=i)
        response.append({
            "date": date.isoformat(),
            "predicted_high": predicted_prices_original[i],
            "predicted_low": '-'
    })

    return jsonify(response)
if __name__ == "__main__":
    app.run(debug=True)
