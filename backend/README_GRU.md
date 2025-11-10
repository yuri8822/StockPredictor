---
tags:
- pytorch
- time-series
- stock-forecasting
- gru
license: mit
---

# Stock Price Forecasting GRU Model

This is a GRU (Gated Recurrent Unit) neural network trained for stock price forecasting.

## Model Details
- **Model Type**: GRU
- **Architecture**: 2-layer GRU with 64 hidden units
- **Input**: Normalized stock price sequences (lookback window: 10)
- **Output**: Next stock price prediction
- **Training**: 30 epochs with Adam optimizer
- **Loss Function**: Mean Squared Error (MSE)

## Usage

```python
import torch
import pickle
from your_model_file import GRUModel

# Load model
checkpoint = torch.load('gru_stock_forecaster.pth')
model = GRUModel(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])

# Load scaler
with open('gru_scaler.pkl', 'rb') as f:
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
