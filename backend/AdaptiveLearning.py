"""
Adaptive and Continuous Learning Module
Implements online learning, incremental updates, fine-tuning, and model versioning
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')


class ModelVersion:
    """Tracks model versions and their performance"""
    
    def __init__(self, model_dir='./model_versions'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.version_log_path = os.path.join(model_dir, 'version_log.json')
        self.version_history = self._load_version_history()
    
    def _load_version_history(self) -> List[Dict]:
        """Load version history from disk"""
        if os.path.exists(self.version_log_path):
            with open(self.version_log_path, 'r') as f:
                return json.load(f)
        return []
    
    def _save_version_history(self):
        """Save version history to disk"""
        with open(self.version_log_path, 'w') as f:
            json.dump(self.version_history, f, indent=2)
    
    def save_model_version(self, model, ticker: str, model_type: str, 
                          metrics: Dict, config: Dict) -> str:
        """
        Save a new model version with metadata
        
        Args:
            model: PyTorch model or scikit-learn model
            ticker: Stock ticker
            model_type: Type of model (LSTM, GRU, ARIMA, etc.)
            metrics: Performance metrics
            config: Model configuration
        
        Returns:
            version_id: Unique version identifier
        """
        timestamp = datetime.now()
        version_id = f"{ticker}_{model_type}_v{timestamp.strftime('%Y%m%d_%H%M%S')}"
        version_dir = os.path.join(self.model_dir, version_id)
        os.makedirs(version_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(version_dir, 'model.pth')
        if isinstance(model, nn.Module):
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': config
            }, model_path)
        else:
            # For traditional models, save with pickle or joblib
            import pickle
            with open(model_path.replace('.pth', '.pkl'), 'wb') as f:
                pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'version_id': version_id,
            'ticker': ticker,
            'model_type': model_type,
            'timestamp': timestamp.isoformat(),
            'metrics': metrics,
            'config': config
        }
        
        metadata_path = os.path.join(version_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update version history
        self.version_history.append(metadata)
        self._save_version_history()
        
        print(f"[ModelVersion] Saved {version_id} with MAE: {metrics.get('mae', 'N/A')}")
        
        return version_id
    
    def load_model_version(self, version_id: str, device='cpu'):
        """Load a specific model version"""
        version_dir = os.path.join(self.model_dir, version_id)
        
        # Load metadata
        metadata_path = os.path.join(version_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Load model
        model_path = os.path.join(version_dir, 'model.pth')
        if os.path.exists(model_path):
            # PyTorch model
            checkpoint = torch.load(model_path, map_location=device)
            # Model reconstruction would need model class
            return checkpoint, metadata
        else:
            # Traditional model
            import pickle
            model_path = model_path.replace('.pth', '.pkl')
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model, metadata
    
    def get_best_version(self, ticker: str, model_type: str, metric='mae') -> Optional[str]:
        """Get the best performing model version"""
        relevant_versions = [
            v for v in self.version_history 
            if v['ticker'] == ticker and v['model_type'] == model_type
        ]
        
        if not relevant_versions:
            return None
        
        # Sort by metric (lower is better for MAE/RMSE/MAPE)
        best_version = min(relevant_versions, 
                          key=lambda v: v['metrics'].get(metric, float('inf')))
        
        return best_version['version_id']
    
    def get_version_history(self, ticker: str = None, model_type: str = None) -> List[Dict]:
        """Get version history with optional filtering"""
        history = self.version_history
        
        if ticker:
            history = [v for v in history if v['ticker'] == ticker]
        if model_type:
            history = [v for v in history if v['model_type'] == model_type]
        
        return sorted(history, key=lambda v: v['timestamp'], reverse=True)


class OnlineLearningDataset(Dataset):
    """Dataset for online learning with sliding window"""
    
    def __init__(self, data: np.ndarray, lookback: int):
        self.data = data
        self.lookback = lookback
    
    def __len__(self):
        return len(self.data) - self.lookback
    
    def __getitem__(self, idx):
        X = self.data[idx:idx + self.lookback]
        y = self.data[idx + self.lookback]
        return torch.FloatTensor(X), torch.FloatTensor([y])


class AdaptiveLearningManager:
    """
    Manages adaptive and continuous learning for forecasting models
    Implements online learning, fine-tuning, and model ensemble
    """
    
    def __init__(self, model_version_manager: ModelVersion, device='cpu'):
        self.version_manager = model_version_manager
        self.device = device
        self.scaler = MinMaxScaler()
        self.update_threshold = 0.05  # 5% increase in error triggers update
        self.lookback = 10
        
        # Ensemble weights
        self.ensemble_weights = {}
        
    def incremental_update(self, model: nn.Module, new_data: pd.DataFrame, 
                          ticker: str, model_type: str, config: Dict,
                          epochs: int = 10, lr: float = 0.001) -> Tuple[nn.Module, Dict]:
        """
        Incrementally update model with new data using online learning
        
        Args:
            model: Current PyTorch model
            new_data: New data for incremental training
            ticker: Stock ticker
            model_type: Model type (LSTM, GRU)
            config: Model configuration
            epochs: Number of training epochs for new data
            lr: Learning rate
        
        Returns:
            updated_model: Updated model
            metrics: Performance metrics on new data
        """
        print(f"[AdaptiveLearning] Starting incremental update for {ticker} {model_type}")
        
        # Prepare data
        close_prices = new_data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(close_prices)
        
        # Create dataset
        dataset = OnlineLearningDataset(scaled_data.flatten(), self.lookback)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
        
        # Set model to training mode
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Incremental training
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.unsqueeze(-1).to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"[AdaptiveLearning] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Evaluate on new data
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.unsqueeze(-1).to(self.device)
                outputs = model(X_batch)
                predictions.extend(outputs.cpu().numpy().flatten())
                actuals.extend(y_batch.numpy().flatten())
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        metrics = {
            'mae': float(mean_absolute_error(actuals, predictions)),
            'rmse': float(np.sqrt(mean_squared_error(actuals, predictions))),
            'mape': float(mean_absolute_percentage_error(actuals, predictions) * 100)
        }
        
        print(f"[AdaptiveLearning] Updated model - MAE: {metrics['mae']:.6f}, "
              f"RMSE: {metrics['rmse']:.6f}, MAPE: {metrics['mape']:.2f}%")
        
        # Save updated model version
        version_id = self.version_manager.save_model_version(
            model, ticker, model_type, metrics, config
        )
        
        return model, metrics
    
    def fine_tune_model(self, model: nn.Module, recent_data: pd.DataFrame,
                       ticker: str, model_type: str, config: Dict,
                       epochs: int = 20, lr: float = 0.0001) -> Tuple[nn.Module, Dict]:
        """
        Fine-tune model on recent data with lower learning rate
        
        Similar to incremental_update but with:
        - Lower learning rate for fine-tuning
        - More epochs
        - Focus on recent patterns
        """
        print(f"[AdaptiveLearning] Fine-tuning {model_type} for {ticker}")
        
        return self.incremental_update(
            model, recent_data, ticker, model_type, config, 
            epochs=epochs, lr=lr
        )
    
    def rolling_window_retrain(self, model_class, data: pd.DataFrame, 
                               ticker: str, model_type: str, config: Dict,
                               window_size: int = 180, 
                               retrain_frequency: int = 30) -> List[Dict]:
        """
        Retrain model using rolling window approach
        
        Args:
            model_class: Model class to instantiate
            data: Full historical data
            ticker: Stock ticker
            model_type: Model type
            config: Model configuration
            window_size: Size of training window (in days)
            retrain_frequency: How often to retrain (in days)
        
        Returns:
            training_history: List of training metrics over time
        """
        print(f"[AdaptiveLearning] Rolling window retraining for {ticker}")
        
        training_history = []
        close_prices = data['Close'].values
        
        # Rolling window training
        start_idx = 0
        while start_idx + window_size < len(close_prices):
            end_idx = start_idx + window_size
            window_data = data.iloc[start_idx:end_idx]
            
            # Create and train model
            model = model_class(**config)
            model.to(self.device)
            
            # Train on window
            window_close = window_data['Close'].values.reshape(-1, 1)
            scaled_data = self.scaler.fit_transform(window_close)
            
            dataset = OnlineLearningDataset(scaled_data.flatten(), self.lookback)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Train for few epochs
            model.train()
            for epoch in range(10):
                for X_batch, y_batch in dataloader:
                    X_batch = X_batch.unsqueeze(-1).to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate
            model.eval()
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for X_batch, y_batch in dataloader:
                    X_batch = X_batch.unsqueeze(-1).to(self.device)
                    outputs = model(X_batch)
                    predictions.extend(outputs.cpu().numpy().flatten())
                    actuals.extend(y_batch.numpy().flatten())
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            metrics = {
                'window_start': int(start_idx),
                'window_end': int(end_idx),
                'mae': float(mean_absolute_error(actuals, predictions)),
                'rmse': float(np.sqrt(mean_squared_error(actuals, predictions))),
                'timestamp': datetime.now().isoformat()
            }
            
            training_history.append(metrics)
            
            # Save model version
            self.version_manager.save_model_version(
                model, ticker, f"{model_type}_rolling", metrics, config
            )
            
            # Move window
            start_idx += retrain_frequency
        
        print(f"[AdaptiveLearning] Completed {len(training_history)} rolling window retrains")
        return training_history
    
    def should_update_model(self, current_metrics: Dict, 
                           historical_metrics: List[Dict]) -> bool:
        """
        Determine if model should be updated based on performance degradation
        
        Args:
            current_metrics: Latest performance metrics
            historical_metrics: Historical metrics for comparison
        
        Returns:
            should_update: Boolean indicating if update is needed
        """
        if not historical_metrics:
            return False
        
        # Get average MAE from last 5 evaluations
        recent_mae = [m['mae'] for m in historical_metrics[-5:]]
        avg_mae = np.mean(recent_mae)
        
        current_mae = current_metrics.get('mae', float('inf'))
        
        # Update if current MAE is significantly worse
        if current_mae > avg_mae * (1 + self.update_threshold):
            print(f"[AdaptiveLearning] Performance degradation detected: "
                  f"Current MAE {current_mae:.6f} vs Avg MAE {avg_mae:.6f}")
            return True
        
        return False
    
    def adaptive_ensemble(self, models: Dict[str, nn.Module], 
                         recent_data: pd.DataFrame,
                         ticker: str) -> Tuple[np.ndarray, Dict]:
        """
        Create adaptive ensemble that weights models based on recent performance
        
        Args:
            models: Dictionary of {model_type: model}
            recent_data: Recent data for evaluation
            ticker: Stock ticker
        
        Returns:
            ensemble_predictions: Weighted predictions
            weights: Model weights used
        """
        print(f"[AdaptiveLearning] Creating adaptive ensemble for {ticker}")
        
        # Evaluate each model on recent data
        model_predictions = {}
        model_errors = {}
        
        close_prices = recent_data['Close'].values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(close_prices)
        
        dataset = OnlineLearningDataset(scaled_data.flatten(), self.lookback)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        for model_type, model in models.items():
            model.eval()
            predictions = []
            actuals = []
            
            with torch.no_grad():
                for X_batch, y_batch in dataloader:
                    X_batch = X_batch.unsqueeze(-1).to(self.device)
                    outputs = model(X_batch)
                    predictions.extend(outputs.cpu().numpy().flatten())
                    actuals.extend(y_batch.numpy().flatten())
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            mae = mean_absolute_error(actuals, predictions)
            model_predictions[model_type] = predictions
            model_errors[model_type] = mae
        
        # Calculate inverse error weights
        total_inverse_error = sum(1/e if e > 0 else 1.0 for e in model_errors.values())
        weights = {
            model_type: (1/error) / total_inverse_error if error > 0 else 1.0
            for model_type, error in model_errors.items()
        }
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        # Create ensemble predictions
        ensemble_predictions = np.zeros_like(list(model_predictions.values())[0])
        for model_type, preds in model_predictions.items():
            ensemble_predictions += weights[model_type] * preds
        
        # Store weights
        self.ensemble_weights[ticker] = {
            'weights': weights,
            'errors': model_errors,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"[AdaptiveLearning] Ensemble weights: {weights}")
        
        return ensemble_predictions, weights


# Example usage functions
def example_incremental_update():
    """Example of using incremental update"""
    from trained_models.model_definitions import LSTMModel
    
    # Initialize
    version_manager = ModelVersion()
    adaptive_manager = AdaptiveLearningManager(version_manager)
    
    # Load existing model
    model = LSTMModel(input_size=1, hidden_size=64, num_layers=2)
    
    # Simulate new data
    new_data = pd.DataFrame({
        'Date': pd.date_range(start='2025-01-01', periods=100),
        'Close': np.random.randn(100).cumsum() + 100
    })
    
    # Incremental update
    config = {'input_size': 1, 'hidden_size': 64, 'num_layers': 2}
    updated_model, metrics = adaptive_manager.incremental_update(
        model, new_data, 'AAPL', 'LSTM', config
    )
    
    print(f"Update complete. Metrics: {metrics}")


if __name__ == '__main__':
    print("Adaptive Learning Module loaded successfully")
    print("Available classes: ModelVersion, AdaptiveLearningManager")
