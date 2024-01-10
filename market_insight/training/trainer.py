import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class Trainer:
    def __init__(self, model, train_dataset, learning_rate=0.001, epochs=5):
        self.model = model
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.epochs = epochs

    def train_model(self):
        
        for epoch in range(self.epochs):
            for seq, labels in self.train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(seq)
                single_loss = self.criterion(y_pred, labels)
                single_loss.backward()
                self.optimizer.step()
            print(f'epoch: {epoch+1:3} loss: {single_loss.item():10.8f}')