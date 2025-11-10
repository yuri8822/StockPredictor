"""
PyTorch model definitions for Hugging Face upload
These are the exact model architectures used in the stock forecasting application
"""

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    """
    LSTM model for stock price forecasting
    Used in the Stock Forecasting Application for CS4063 NLP Assignment
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output

class GRUModel(nn.Module):
    """
    GRU model for stock price forecasting
    Used in the Stock Forecasting Application for CS4063 NLP Assignment
    """
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                         batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out[:, -1, :])
        return output

# Model configurations for easy loading
LSTM_CONFIG = {
    'input_size': 1,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'lookback': 10
}

GRU_CONFIG = {
    'input_size': 1,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'lookback': 10
}

def load_lstm_model(model_path, device='cpu'):
    """Load a saved LSTM model"""
    checkpoint = torch.load(model_path, map_location=device)
    model = LSTMModel(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def load_gru_model(model_path, device='cpu'):
    """Load a saved GRU model"""
    checkpoint = torch.load(model_path, map_location=device)
    model = GRUModel(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model