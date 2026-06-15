"""
Train PyTorch neural network model for WNBA predictions
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import os

class WNBAPredictor(nn.Module):
    """
    Neural network for WNBA predictions
    """
    def __init__(self, input_size):
        super(WNBAPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_pytorch_model():
    """
    Train PyTorch model on WNBA data
    """
    print("Training PyTorch model...")
    
    # Load data
    if not os.path.exists('data/wnba_data.csv'):
        print("Data file not found. Please run fetch_wnba_data.py first.")
        return None
    
    data = pd.read_csv('data/wnba_data.csv')
    
    # TODO: Feature engineering
    X = data.drop(['date', 'team', 'player'], axis=1, errors='ignore').values.astype(np.float32)
    y = data.get('points', np.random.rand(len(data))).values.astype(np.float32).reshape(-1, 1)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WNBAPredictor(X.shape[1]).to(device)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    epochs = 50
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/pytorch_model.pth')
    print("Model saved to models/pytorch_model.pth")
    
    return model

if __name__ == "__main__":
    train_pytorch_model()