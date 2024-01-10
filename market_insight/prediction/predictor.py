import torch


class Predictor:
    def __init__(self, model, sequence_length=60):
        self.model = model
        self.sequence_length = sequence_length

    def predict_next_days(self, start_sequence, days=7):
        self.model.eval()
        predicted_prices = []

        for _ in range(days):
            with torch.no_grad():
                if isinstance(start_sequence, list):  # Ensure it's a list
                    seq = torch.FloatTensor(start_sequence[-self.sequence_length:]).unsqueeze(0).unsqueeze(-1)
                else:
                    raise TypeError("start_sequence must be a list")

                predicted_price = self.model(seq)
                predicted_prices.append(predicted_price.item())
                start_sequence.append(predicted_price.item())  # Update the sequence

        return predicted_prices

        