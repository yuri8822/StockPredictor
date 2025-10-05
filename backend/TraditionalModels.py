"""
Additional Traditional Forecasting Models
As required by the assignment: ARIMA, Moving Averages, VAR, etc.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings

warnings.filterwarnings('ignore')

class MovingAverageForecaster:
    """
    Traditional Moving Average forecasting model
    Implements Simple Moving Average (SMA) and Exponential Moving Average (EMA)
    """
    
    def __init__(self, window=10, method='sma', alpha=0.3):
        self.window = window
        self.method = method  # 'sma' or 'ema'
        self.alpha = alpha  # For EMA
        self.fitted_values = None
    
    def fit(self, train_data: np.ndarray):
        """Fit the moving average model"""
        self.train_data = train_data
        
        if self.method == 'sma':
            # Simple Moving Average
            self.fitted_values = pd.Series(train_data).rolling(window=self.window).mean().values
        elif self.method == 'ema':
            # Exponential Moving Average
            self.fitted_values = pd.Series(train_data).ewm(alpha=self.alpha).mean().values
        else:
            raise ValueError("Method must be 'sma' or 'ema'")
    
    def predict(self, steps: int) -> np.ndarray:
        """Generate forecasts using moving average"""
        if self.fitted_values is None:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        if self.method == 'sma':
            # For SMA, use the mean of the last window values
            last_values = self.train_data[-self.window:]
            for _ in range(steps):
                next_pred = np.mean(last_values)
                predictions.append(next_pred)
                # Update the window (shift and add prediction)
                last_values = np.append(last_values[1:], next_pred)
        
        elif self.method == 'ema':
            # For EMA, continue the exponential smoothing
            last_ema = self.fitted_values[-1]
            last_actual = self.train_data[-1]
            
            for _ in range(steps):
                # EMA prediction uses the last actual value
                next_pred = self.alpha * last_actual + (1 - self.alpha) * last_ema
                predictions.append(next_pred)
                last_ema = next_pred
                last_actual = next_pred  # Use prediction as next "actual" value
        
        return np.array(predictions)

class VARForecaster:
    """
    Vector Autoregression (VAR) model for multivariate time series
    Uses multiple features like Close, Volume, MA5, MA10 for forecasting
    """
    
    def __init__(self, maxlags=5):
        self.maxlags = maxlags
        self.model = None
        self.fitted_model = None
        self.feature_names = None
        self.target_index = None
    
    def fit(self, data: pd.DataFrame, target_col='Close'):
        """
        Fit VAR model using multiple features
        
        Args:
            data: DataFrame with multiple time series columns
            target_col: Column to forecast (default: 'Close')
        """
        # Select numeric columns for VAR model
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        # Use common financial indicators if available
        preferred_cols = ['Close', 'Volume', 'MA5', 'MA10', 'Return', 'Volatility_5d']
        available_cols = [col for col in preferred_cols if col in numeric_cols]
        
        # If preferred columns not available, use all numeric columns
        if len(available_cols) < 2:
            available_cols = numeric_cols.tolist()[:4]  # Limit to 4 features for stability
        
        # Ensure target column is included
        if target_col not in available_cols:
            available_cols.append(target_col)
        
        # Prepare data
        var_data = data[available_cols].dropna()
        
        if len(var_data) < self.maxlags + 10:  # Need sufficient data
            raise ValueError(f"Insufficient data for VAR model. Need at least {self.maxlags + 10} observations")
        
        self.feature_names = available_cols
        self.target_index = available_cols.index(target_col)
        
        # Fit VAR model
        self.model = VAR(var_data)
        self.fitted_model = self.model.fit(maxlags=self.maxlags)
        
        print(f"[VAR] Using features: {self.feature_names}")
        print(f"[VAR] Target variable: {target_col} (index {self.target_index})")
    
    def predict(self, steps: int) -> np.ndarray:
        """Generate VAR forecasts"""
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Generate forecasts for all variables
        forecast_result = self.fitted_model.forecast(
            self.fitted_model.endog[-self.fitted_model.k_ar:], 
            steps=steps
        )
        
        # Return only the target variable predictions
        target_predictions = forecast_result[:, self.target_index]
        return target_predictions

class LinearTrendForecaster:
    """
    Simple Linear Trend forecasting model
    Another traditional technique mentioned in assignments
    """
    
    def __init__(self):
        self.slope = None
        self.intercept = None
        self.trend_line = None
    
    def fit(self, train_data: np.ndarray):
        """Fit linear trend to the data"""
        x = np.arange(len(train_data))
        # Use numpy's polyfit for linear regression
        coefficients = np.polyfit(x, train_data, 1)
        self.slope = coefficients[0]
        self.intercept = coefficients[1]
        
        # Calculate trend line for fitted values
        self.trend_line = self.slope * x + self.intercept
    
    def predict(self, steps: int) -> np.ndarray:
        """Generate linear trend forecasts"""
        if self.slope is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Continue the linear trend
        start_x = len(self.trend_line)  # Where training data ended
        future_x = np.arange(start_x, start_x + steps)
        predictions = self.slope * future_x + self.intercept
        
        return predictions

class EnhancedEnsembleForecaster:
    """
    Enhanced Ensemble combining multiple traditional and neural models
    As required by assignment: "encouraged to build ensembles"
    """
    
    def __init__(self):
        # Import the original models
        from ForecastPredictor import ARIMAForecaster, LSTMForecaster
        
        self.arima = ARIMAForecaster()
        self.lstm = LSTMForecaster(lookback=10)
        self.sma = MovingAverageForecaster(window=10, method='sma')
        self.ema = MovingAverageForecaster(window=10, method='ema', alpha=0.3)
        self.linear_trend = LinearTrendForecaster()
        self.var = None  # Will be initialized if multivariate data is available
        
        # Enhanced weights for more models
        self.weights = {
            'arima': 0.25,
            'lstm': 0.30,
            'sma': 0.15,
            'ema': 0.15,
            'linear_trend': 0.10,
            'var': 0.05
        }
    
    def fit(self, train_data: np.ndarray, full_data: pd.DataFrame = None):
        """
        Fit all models in the enhanced ensemble
        
        Args:
            train_data: 1D array of target variable (Close prices)
            full_data: DataFrame with multiple features for VAR model
        """
        print("[Enhanced Ensemble] Training multiple models...")
        
        # Fit traditional models
        print("  - Training ARIMA model...")
        self.arima.fit(train_data)
        
        print("  - Training LSTM model...")
        self.lstm.fit(train_data, epochs=30)
        
        print("  - Training Simple Moving Average...")
        self.sma.fit(train_data)
        
        print("  - Training Exponential Moving Average...")
        self.ema.fit(train_data)
        
        print("  - Training Linear Trend model...")
        self.linear_trend.fit(train_data)
        
        # Fit VAR model if multivariate data is available
        if full_data is not None:
            try:
                print("  - Training VAR model...")
                self.var = VARForecaster()
                self.var.fit(full_data)
                print("  - VAR model trained successfully")
            except Exception as e:
                print(f"  - VAR model training failed: {e}")
                self.var = None
                # Redistribute VAR weight to other models
                self.weights['arima'] += 0.01
                self.weights['lstm'] += 0.02
                self.weights['sma'] += 0.01
                self.weights['ema'] += 0.01
    
    def predict(self, steps: int, seed_data: np.ndarray) -> dict:
        """Generate predictions from all models"""
        predictions = {}
        
        # Get predictions from each model
        predictions['arima'] = self.arima.predict(steps)
        predictions['lstm'] = self.lstm.predict(steps, seed_data)
        predictions['sma'] = self.sma.predict(steps)
        predictions['ema'] = self.ema.predict(steps)
        predictions['linear_trend'] = self.linear_trend.predict(steps)
        
        if self.var is not None:
            try:
                predictions['var'] = self.var.predict(steps)
            except Exception as e:
                print(f"[Warning] VAR prediction failed: {e}")
                predictions['var'] = predictions['sma']  # Fallback to SMA
        else:
            predictions['var'] = predictions['sma']  # Use SMA as placeholder
        
        # Calculate weighted ensemble
        ensemble_pred = np.zeros(steps)
        for model_name, pred_values in predictions.items():
            if model_name in self.weights:
                ensemble_pred += self.weights[model_name] * np.array(pred_values)
        
        predictions['enhanced_ensemble'] = ensemble_pred
        
        return predictions

# Additional utility functions for traditional models

def calculate_traditional_metrics(y_true: np.ndarray, predictions: dict) -> dict:
    """Calculate metrics for all traditional models"""
    metrics = {}
    
    for model_name, y_pred in predictions.items():
        if len(y_pred) <= len(y_true):
            y_true_subset = y_true[:len(y_pred)]
        else:
            y_true_subset = y_true
            y_pred = y_pred[:len(y_true)]
        
        if len(y_true_subset) > 0 and len(y_pred) > 0:
            rmse = np.sqrt(mean_squared_error(y_true_subset, y_pred))
            mae = mean_absolute_error(y_true_subset, y_pred)
            
            # Handle MAPE calculation safely
            try:
                mape = mean_absolute_percentage_error(y_true_subset, y_pred) * 100
            except:
                # Calculate MAPE manually if function fails
                mape = np.mean(np.abs((y_true_subset - y_pred) / y_true_subset)) * 100
            
            metrics[model_name] = {
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape)
            }
    
    return metrics