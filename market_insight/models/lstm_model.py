import torch
import torch.nn as nn
from market_insight.logger import setup_console_logger

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.logger = setup_console_logger('LSTMLogger')
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.logger.info(f"LSTM model initialized with input_size={input_size}, hidden_layer_size={hidden_layer_size}, output_size={output_size}")

    def forward(self, input_seq):
        self.logger.debug("Starting forward pass")
        batch_size, seq_len, _ = input_seq.size()
        self.logger.debug(f"Input sequence size: batch_size={batch_size}, seq_len={seq_len}")

        hidden_state = torch.zeros(1, batch_size, self.hidden_layer_size).to(input_seq.device)
        cell_state = torch.zeros(1, batch_size, self.hidden_layer_size).to(input_seq.device)
        hidden_cell = (hidden_state, cell_state)

        lstm_out, hidden_cell = self.lstm(input_seq, hidden_cell)
        predictions = self.linear(lstm_out[:, -1, :])  # Get the last time step

        self.logger.debug("Forward pass completed")
        return predictions
