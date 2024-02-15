import torch


class Predictor:
    def __init__(self, model, sequence_length=60):
        self.model = model
        self.sequence_length = sequence_length

    def predict_next_days(self, start_sequence, days=7):
        self.model.eval()
        predicted_highs = []
        predicted_lows = []

        for _ in range(days):
            with torch.no_grad():
                if isinstance(start_sequence, list):  # Ensure it's a list
                    seq = torch.FloatTensor(start_sequence[-self.sequence_length:]).unsqueeze(0).unsqueeze(-1)
                else:
                    raise TypeError("start_sequence must be a list")

                predicted_prices = self.model(seq)
                high, low = predicted_prices.view(-1).tolist()  # Extract high and low values
                predicted_highs.append(high)
                predicted_lows.append(low)
                # Update the sequence with both high and low values
                start_sequence.extend([high, low])  

        return predicted_highs, predicted_lows

        