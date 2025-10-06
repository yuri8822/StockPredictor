---
tags:
- pytorch
- time-series
- stock-forecasting
- lstm
- gru
- ensemble
license: mit
datasets:
- financial
- stocks
language:
- en
library_name: pytorch
---

# Stock Price Forecasting Models Collection

This repository contains PyTorch models trained for stock price forecasting as part of a CS4063 Natural Language Processing assignment.

## Models Included

### 1. LSTM Model (`lstm_stock_forecaster.pth`)
- **Architecture**: 2-layer LSTM with 64 hidden units
- **Input**: Normalized stock price sequences (10-day lookback window)
- **Output**: Next stock price prediction
- **Dropout**: 0.2 for regularization

### 2. GRU Model (`gru_stock_forecaster.pth`)
- **Architecture**: 2-layer GRU with 64 hidden units
- **Input**: Normalized stock price sequences (10-day lookback window)
- **Output**: Next stock price prediction
- **Dropout**: 0.2 for regularization

## Training Details

- **Training Data**: Stock price data with technical indicators
- **Preprocessing**: MinMaxScaler normalization
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Training Epochs**: 30
- **Batch Size**: 32 (or dataset size if smaller)

## Usage

```python
import torch
import pickle
from model_definitions import LSTMModel, GRUModel

# Load LSTM model
lstm_checkpoint = torch.load('lstm_stock_forecaster.pth', map_location='cpu')
lstm_model = LSTMModel(**lstm_checkpoint['model_config'])
lstm_model.load_state_dict(lstm_checkpoint['model_state_dict'])

# Load GRU model
gru_checkpoint = torch.load('gru_stock_forecaster.pth', map_location='cpu')
gru_model = GRUModel(**gru_checkpoint['model_config'])
gru_model.load_state_dict(gru_checkpoint['model_state_dict'])

# Load scalers
with open('lstm_scaler.pkl', 'rb') as f:
    lstm_scaler = pickle.load(f)

with open('gru_scaler.pkl', 'rb') as f:
    gru_scaler = pickle.load(f)

# Make predictions
lstm_model.eval()
gru_model.eval()

# Example prediction (replace with your data)
import numpy as np
sample_data = np.random.randn(1, 10, 1)  # [batch, sequence, features]
sample_tensor = torch.FloatTensor(sample_data)

lstm_prediction = lstm_model(sample_tensor)
gru_prediction = gru_model(sample_tensor)
```

## Files Structure

```
trained_models/
├── lstm_stock_forecaster.pth     # LSTM model weights + config
├── gru_stock_forecaster.pth      # GRU model weights + config
├── lstm_scaler.pkl               # LSTM data scaler
├── gru_scaler.pkl                # GRU data scaler
├── model_definitions.py          # PyTorch model classes
├── README.md                     # This file
└── requirements.txt              # Dependencies
```

## Performance

These models are part of an ensemble forecasting system that combines:
- Traditional time series models (ARIMA, Moving Averages, VAR)
- Neural networks (LSTM, GRU)
- Weighted ensemble predictions

Typical performance metrics on stock data:
- LSTM: RMSE ~$2.18, MAE ~$1.65, MAPE ~1.56%
- GRU: Similar performance to LSTM
- Ensemble: Improved performance through model combination

## Application Context

These models were developed for a complete stock forecasting web application featuring:
- React frontend with interactive candlestick charts
- Flask backend with REST API
- MongoDB database for data persistence
- Real-time data collection with sentiment analysis
- Professional visualization with Plotly

## Citation

If you use these models, please cite:
```
@misc{stock_forecasting_models_2025,
  title={Stock Price Forecasting with LSTM and GRU Neural Networks},
  author={[Your Name]},
  year={2025},
  howpublished={CS4063 NLP Assignment, [Your University]}
}
```

## License

MIT License - See LICENSE file for details.