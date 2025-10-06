"""
Example usage script for the trained stock forecasting models
"""

import torch
import pickle
import numpy as np
from model_definitions import LSTMModel, GRUModel, load_lstm_model, load_gru_model

def load_models():
    """Load both LSTM and GRU models with their scalers"""
    
    # Load models using the convenience functions
    lstm_model = load_lstm_model('lstm_stock_forecaster.pth')
    gru_model = load_gru_model('gru_stock_forecaster.pth')
    
    # Load scalers
    with open('lstm_scaler.pkl', 'rb') as f:
        lstm_scaler = pickle.load(f)
    
    with open('gru_scaler.pkl', 'rb') as f:
        gru_scaler = pickle.load(f)
    
    return lstm_model, gru_model, lstm_scaler, gru_scaler

def make_predictions(models, scalers, input_data, steps=5):
    """
    Make stock price predictions using both models
    
    Args:
        models: tuple of (lstm_model, gru_model)
        scalers: tuple of (lstm_scaler, gru_scaler)
        input_data: numpy array of shape (lookback_window,) with historical prices
        steps: number of future steps to predict
    
    Returns:
        dict with predictions from both models
    """
    lstm_model, gru_model = models
    lstm_scaler, gru_scaler = scalers
    
    # Set models to evaluation mode
    lstm_model.eval()
    gru_model.eval()
    
    predictions = {}
    
    # LSTM predictions
    lstm_scaled = lstm_scaler.transform(input_data.reshape(-1, 1)).flatten()
    lstm_sequence = list(lstm_scaled[-10:])  # Last 10 values for lookback
    lstm_preds = []
    
    with torch.no_grad():
        for _ in range(steps):
            x = torch.FloatTensor(lstm_sequence[-10:]).unsqueeze(0).unsqueeze(-1)
            pred = lstm_model(x).item()
            lstm_preds.append(pred)
            lstm_sequence.append(pred)
    
    # Rescale LSTM predictions
    lstm_preds_rescaled = lstm_scaler.inverse_transform(
        np.array(lstm_preds).reshape(-1, 1)
    ).flatten()
    predictions['lstm'] = lstm_preds_rescaled
    
    # GRU predictions
    gru_scaled = gru_scaler.transform(input_data.reshape(-1, 1)).flatten()
    gru_sequence = list(gru_scaled[-10:])  # Last 10 values for lookback
    gru_preds = []
    
    with torch.no_grad():
        for _ in range(steps):
            x = torch.FloatTensor(gru_sequence[-10:]).unsqueeze(0).unsqueeze(-1)
            pred = gru_model(x).item()
            gru_preds.append(pred)
            gru_sequence.append(pred)
    
    # Rescale GRU predictions
    gru_preds_rescaled = gru_scaler.inverse_transform(
        np.array(gru_preds).reshape(-1, 1)
    ).flatten()
    predictions['gru'] = gru_preds_rescaled
    
    # Ensemble prediction (simple average)
    predictions['ensemble'] = (predictions['lstm'] + predictions['gru']) / 2
    
    return predictions

def example_usage():
    """Example of how to use the models"""
    print("Loading trained models...")
    
    try:
        models = load_models()
        print("✅ Models loaded successfully!")
        
        # Example input: 30 days of stock prices (you would replace this with real data)
        example_prices = np.array([
            150.0, 151.2, 149.8, 152.1, 153.0, 151.5, 150.9, 152.7, 154.1, 153.8,
            155.2, 154.6, 156.0, 157.3, 156.8, 158.1, 157.4, 159.2, 160.1, 158.9,
            160.5, 161.2, 159.8, 162.1, 163.0, 161.5, 160.9, 162.7, 164.1, 163.8
        ])
        
        print(f"\nInput data shape: {example_prices.shape}")
        print(f"Input price range: ${example_prices.min():.2f} - ${example_prices.max():.2f}")
        
        # Make predictions
        predictions = make_predictions(models[:2], models[2:], example_prices, steps=5)
        
        print("\n🔮 Predictions for next 5 days:")
        for model_name, preds in predictions.items():
            print(f"{model_name.upper()}: {[f'${p:.2f}' for p in preds]}")
            
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure the model files exist in the current directory:")
        print("- lstm_stock_forecaster.pth")
        print("- gru_stock_forecaster.pth") 
        print("- lstm_scaler.pkl")
        print("- gru_scaler.pkl")

if __name__ == "__main__":
    example_usage()