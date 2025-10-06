---
tags:
- pytorch
- time-series
- stock-forecasting
- lstm
license: mit
---

# Stock Price Forecasting LSTM Model

This is a LSTM (Long Short-Term Memory) neural network trained for stock price forecasting.

## Model Details
- **Model Type**: LSTM
- **Architecture**: 2-layer LSTM with 64 hidden units
- **Input**: Normalized stock price sequences (lookback window: 10)
- **Output**: Next stock price prediction
- **Training**: 30 epochs with Adam optimizer
- **Loss Function**: Mean Squared Error (MSE)

## Usage

```python
import torch
import pickle
from your_model_file import LSTMModel

# Load model
checkpoint = torch.load('lstm_stock_forecaster.pth')
model = LSTMModel(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])

# Load scaler
with open('lstm_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Make predictions
model.eval()
# ... your prediction code here
```

## Training Data
- Stock price data from various tickers
- Technical indicators included
- Data preprocessed with MinMaxScaler

## Performance
This model is part of an ensemble forecasting system and shows competitive performance on stock price prediction tasks.
