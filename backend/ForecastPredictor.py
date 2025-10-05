import os
import json
import subprocess
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import glob

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Neural network imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ===========================
# Configuration
# ===========================
class Config:
    MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
    DB_NAME = 'stock_forecasting'
    COLLECTION_HISTORICAL = 'historical_data'
    COLLECTION_PREDICTIONS = 'predictions'
    FORECAST_HORIZONS = {'1hr': 1, '3hrs': 3, '24hrs': 24, '72hrs': 72}
    LOOKBACK_DAYS = 90
    SECRET_KEY = 'your-secret-key-here'
    DATA_COLLECTOR_SCRIPT = 'StockDataCollector.py'  # Path to your first assignment script
    CURATED_DATA_DIR = '.'  # Directory where curated CSV files are stored

# ===========================
# Database Layer
# ===========================
class DatabaseManager:
    def __init__(self):
        self.client = MongoClient(Config.MONGO_URI)
        self.db = self.client[Config.DB_NAME]
        self.historical = self.db[Config.COLLECTION_HISTORICAL]
        self.predictions = self.db[Config.COLLECTION_PREDICTIONS]
        self._setup_indexes()
    
    def _setup_indexes(self):
        self.historical.create_index([('ticker', 1), ('date', -1)])
        self.predictions.create_index([('ticker', 1), ('created_at', -1)])
    
    def store_historical_data(self, ticker: str, data: pd.DataFrame):
        """Store curated dataset from StockDataCollector.py"""
        records = data.to_dict('records')
        for record in records:
            record['ticker'] = ticker
            record['date'] = pd.to_datetime(record['Date']).isoformat()
        self.historical.delete_many({'ticker': ticker})
        if records:
            self.historical.insert_many(records)
        print(f"[DB] Stored {len(records)} records for {ticker}")
    
    def get_historical_data(self, ticker: str) -> pd.DataFrame:
        cursor = self.historical.find({'ticker': ticker}).sort('date', 1)
        df = pd.DataFrame(list(cursor))
        if not df.empty:
            df['Date'] = pd.to_datetime(df['date'])
            df = df.drop(['_id', 'ticker', 'date'], axis=1, errors='ignore')
        return df
    
    def store_prediction(self, ticker: str, horizon: str, predictions: List[float], metrics: Dict):
        doc = {
            'ticker': ticker,
            'horizon': horizon,
            'predictions': predictions,
            'metrics': metrics,
            'created_at': datetime.now().isoformat()
        }
        self.predictions.insert_one(doc)
    
    def get_latest_prediction(self, ticker: str, horizon: str):
        return self.predictions.find_one(
            {'ticker': ticker, 'horizon': horizon},
            sort=[('created_at', -1)]
        )

# ===========================
# Data Loader - Integrates with StockDataCollector.py
# ===========================
class CuratedDataLoader:
    @staticmethod
    def run_data_collector(ticker: str, days: int = 90) -> str:
        """
        Run the StockDataCollector.py script to generate curated dataset
        Returns: Path to the generated CSV file
        """
        if not os.path.exists(Config.DATA_COLLECTOR_SCRIPT):
            raise FileNotFoundError(
                f"StockDataCollector.py not found at {Config.DATA_COLLECTOR_SCRIPT}. "
                f"Please ensure it's in the same directory as this script."
            )
        
        print(f"[INFO] Running StockDataCollector.py for {ticker}...")
        
        # Run the data collector script
        cmd = [
            'python', Config.DATA_COLLECTOR_SCRIPT,
            '--ticker', ticker,
            '--days', str(days)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Data collection failed: {result.stderr}")
        
        print(result.stdout)
        
        # Find the generated CSV file
        csv_file = f"{ticker}_curated_dataset.csv"
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"Expected curated dataset file {csv_file} not found")
        
        return csv_file
    
    @staticmethod
    def load_curated_dataset(ticker: str, days: int = 90) -> pd.DataFrame:
        """
        Load curated dataset for a ticker. 
        If not exists, run StockDataCollector.py to generate it.
        """
        csv_file = f"{ticker}_curated_dataset.csv"
        
        # Check if curated dataset exists and is recent (less than 1 day old)
        if os.path.exists(csv_file):
            file_age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(csv_file))
            if file_age.days < 1:
                print(f"[INFO] Loading existing curated dataset: {csv_file}")
                df = pd.read_csv(csv_file)
                df['Date'] = pd.to_datetime(df['Date'])
                return df
        
        # Generate new curated dataset
        csv_file = CuratedDataLoader.run_data_collector(ticker, days)
        df = pd.read_csv(csv_file)
        df['Date'] = pd.to_datetime(df['Date'])
        
        print(f"[INFO] Loaded curated dataset with {len(df)} rows")
        print(f"[INFO] Columns: {list(df.columns)}")
        print(f"[INFO] Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return df
    
    @staticmethod
    def prepare_for_forecasting(df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare curated dataset for forecasting by ensuring all required columns exist
        """
        # The curated dataset from StockDataCollector.py has:
        # Date, Adj Close, Close, High, Low, Open, Volume, Return, 
        # Volatility_5d, MA5, MA10, Headline, Publisher, Sentiment
        
        # Ensure we have OHLC data for candlestick charts
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in curated dataset")
        
        # Drop rows with NaN in critical columns
        df = df.dropna(subset=['Close', 'Adj Close'])
        
        return df

# ===========================
# Traditional Model: ARIMA
# ===========================
class ARIMAForecaster:
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model = None
        self.fitted_model = None
    
    def fit(self, train_data: np.ndarray):
        self.model = ARIMA(train_data, order=self.order)
        self.fitted_model = self.model.fit()
    
    def predict(self, steps: int) -> np.ndarray:
        forecast = self.fitted_model.forecast(steps=steps)
        # Handle different statsmodels return types
        if hasattr(forecast, 'values'):
            return forecast.values
        elif isinstance(forecast, np.ndarray):
            return forecast
        else:
            return np.array(forecast)

# ===========================
# Neural Model: LSTM
# ===========================
class StockDataset(Dataset):
    def __init__(self, data, lookback=10):
        self.data = data
        self.lookback = lookback
    
    def __len__(self):
        return len(self.data) - self.lookback
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.lookback]
        y = self.data[idx + self.lookback]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class LSTMModel(nn.Module):
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

class LSTMForecaster:
    def __init__(self, lookback=10, hidden_size=64, num_layers=2):
        self.lookback = lookback
        self.model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def fit(self, train_data: np.ndarray, epochs=50, lr=0.001):
        scaled_data = self.scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
        dataset = StockDataset(scaled_data, self.lookback)
        
        if len(dataset) == 0:
            raise ValueError("Insufficient data for LSTM training")
        
        dataloader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.unsqueeze(-1).to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}")
    
    def predict(self, steps: int, seed_data: np.ndarray) -> np.ndarray:
        self.model.eval()
        scaled_seed = self.scaler.transform(seed_data.reshape(-1, 1)).flatten()
        current_sequence = list(scaled_seed[-self.lookback:])
        predictions = []
        
        with torch.no_grad():
            for _ in range(steps):
                x = torch.FloatTensor(current_sequence[-self.lookback:]).unsqueeze(0).unsqueeze(-1).to(self.device)
                pred = self.model(x).item()
                predictions.append(pred)
                current_sequence.append(pred)
        
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_rescaled = self.scaler.inverse_transform(predictions_array).flatten()
        return predictions_rescaled

class GRUForecaster:
    def __init__(self, lookback=10, hidden_size=64, num_layers=2):
        self.lookback = lookback
        self.model = GRUModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers)
        self.scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def fit(self, train_data: np.ndarray, epochs=50, lr=0.001):
        scaled_data = self.scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
        dataset = StockDataset(scaled_data, self.lookback)
        
        if len(dataset) == 0:
            raise ValueError("Insufficient data for GRU training")
        
        dataloader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.unsqueeze(-1).to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"  GRU Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(dataloader):.6f}")
    
    def predict(self, steps: int, seed_data: np.ndarray) -> np.ndarray:
        self.model.eval()
        scaled_seed = self.scaler.transform(seed_data.reshape(-1, 1)).flatten()
        current_sequence = list(scaled_seed[-self.lookback:])
        predictions = []
        
        with torch.no_grad():
            for _ in range(steps):
                x = torch.FloatTensor(current_sequence[-self.lookback:]).unsqueeze(0).unsqueeze(-1).to(self.device)
                pred = self.model(x).item()
                predictions.append(pred)
                current_sequence.append(pred)
        
        predictions_array = np.array(predictions).reshape(-1, 1)
        predictions_rescaled = self.scaler.inverse_transform(predictions_array).flatten()
        return predictions_rescaled

# ===========================
# Ensemble Model
# ===========================
class EnsembleForecaster:
    def __init__(self):
        self.arima_forecaster = ARIMAForecaster()
        self.lstm_forecaster = LSTMForecaster(lookback=10)
        self.gru_forecaster = GRUForecaster(lookback=10)
        self.weights = {'arima': 0.3, 'lstm': 0.35, 'gru': 0.35}
    
    def fit(self, train_data: np.ndarray):
        print("[INFO] Training ARIMA model...")
        self.arima_forecaster.fit(train_data)
        
        print("[INFO] Training LSTM model...")
        self.lstm_forecaster.fit(train_data, epochs=30)
        
        print("[INFO] Training GRU model...")
        self.gru_forecaster.fit(train_data, epochs=30)
    
    def predict(self, steps: int, seed_data: np.ndarray) -> Dict[str, np.ndarray]:
        arima_pred = self.arima_forecaster.predict(steps)
        lstm_pred = self.lstm_forecaster.predict(steps, seed_data)
        gru_pred = self.gru_forecaster.predict(steps, seed_data)
        
        ensemble_pred = (self.weights['arima'] * arima_pred + 
                        self.weights['lstm'] * lstm_pred +
                        self.weights['gru'] * gru_pred)
        
        return {
            'arima': arima_pred,
            'lstm': lstm_pred,
            'gru': gru_pred,
            'ensemble': ensemble_pred
        }

# ===========================
# Model Evaluator
# ===========================
class ModelEvaluator:
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape)
        }

# ===========================
# Visualization
# ===========================
class ChartGenerator:
    @staticmethod
    def create_candlestick_with_forecast(df: pd.DataFrame, predictions: Dict[str, np.ndarray], 
                                        horizon: str, ticker: str) -> Dict:
        """
        Create candlestick chart data for React Plotly component
        """
        # Clean data and handle NaN values
        df_clean = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Prepare historical data with safety checks
        historical_dates = df_clean['Date'].dt.strftime('%Y-%m-%d').tolist()
        
        # Generate future dates for predictions (daily basis)
        last_date = df_clean['Date'].iloc[-1]
        future_dates = pd.date_range(start=last_date, periods=len(predictions['ensemble'])+1, freq='D')[1:]
        future_dates_str = future_dates.strftime('%Y-%m-%d').tolist()
        
        # Helper function to clean numeric data
        def clean_numeric_list(data):
            """Convert numpy arrays to clean Python lists, handling NaN values"""
            if hasattr(data, 'tolist'):
                data = data.tolist()
            return [float(x) if not (pd.isna(x) or np.isinf(x)) else None for x in data]
        
        # Prepare chart data structure
        chart_data = {
            'data': [],
            'layout': {
                'title': f'{ticker} Stock Price Forecast - {horizon} Horizon',
                'xaxis': {'title': 'Date', 'rangeslider': {'visible': False}},
                'yaxis': {'title': 'Price ($)', 'domain': [0.3, 1]},
                'yaxis2': {'title': 'Volume', 'domain': [0, 0.25]},
                'height': 800,
                'showlegend': True,
                'hovermode': 'x unified'
            }
        }
        
        # Add candlestick trace for historical data
        chart_data['data'].append({
            'type': 'candlestick',
            'x': historical_dates,
            'open': clean_numeric_list(df_clean['Open']),
            'high': clean_numeric_list(df_clean['High']),
            'low': clean_numeric_list(df_clean['Low']),
            'close': clean_numeric_list(df_clean['Close']),
            'name': 'Historical Price',
            'yaxis': 'y'
        })
        
        # Add moving averages if available
        if 'MA5' in df_clean.columns:
            chart_data['data'].append({
                'type': 'scatter',
                'x': historical_dates,
                'y': clean_numeric_list(df_clean['MA5']),
                'mode': 'lines',
                'name': 'MA5',
                'line': {'color': 'rgba(255, 165, 0, 0.5)', 'width': 1},
                'yaxis': 'y'
            })
        
        if 'MA10' in df_clean.columns:
            chart_data['data'].append({
                'type': 'scatter',
                'x': historical_dates,
                'y': clean_numeric_list(df_clean['MA10']),
                'mode': 'lines',
                'name': 'MA10',
                'line': {'color': 'rgba(0, 128, 255, 0.5)', 'width': 1},
                'yaxis': 'y'
            })
        
        # Add prediction lines
        colors = {'arima': 'orange', 'lstm': 'green', 'gru': 'blue', 'ensemble': 'red'}
        for model_name, pred_values in predictions.items():
            chart_data['data'].append({
                'type': 'scatter',
                'x': future_dates_str,
                'y': clean_numeric_list(pred_values),
                'mode': 'lines+markers',
                'name': f'{model_name.upper()} Forecast',
                'line': {'color': colors[model_name], 'width': 2, 'dash': 'dash'},
                'yaxis': 'y'
            })
        
        # Add volume bars
        chart_data['data'].append({
            'type': 'bar',
            'x': historical_dates,
            'y': clean_numeric_list(df_clean['Volume']),
            'name': 'Volume',
            'marker': {'color': 'lightblue'},
            'yaxis': 'y2'
        })
        
        return chart_data

# ===========================
# Flask Application
# ===========================
app = Flask(__name__)
app.config['SECRET_KEY'] = Config.SECRET_KEY

# Enable CORS for React frontend
CORS(app, origins=["http://localhost:3000"])

db_manager = DatabaseManager()

# HTML template removed - now using React frontend

@app.route('/api/health')
def health_check():
    return jsonify({'status': 'ok', 'message': 'Stock Forecasting API is running'})

@app.route('/api/forecast', methods=['POST'])
def forecast():
    try:
        data = request.get_json()
        ticker = data.get('ticker', 'AAPL').upper()
        horizon = data.get('horizon', '24hrs')
        days = data.get('days', 90)
        steps = Config.FORECAST_HORIZONS[horizon]
        
        print(f"\n{'='*60}")
        print(f"[INFO] Generating forecast for {ticker}")
        print(f"[INFO] Horizon: {horizon} ({steps} steps)")
        print(f"[INFO] Historical data: {days} days")
        print(f"{'='*60}\n")
        
        # Load curated dataset (will run StockDataCollector.py if needed)
        print("[INFO] Loading curated dataset from StockDataCollector.py...")
        df = CuratedDataLoader.load_curated_dataset(ticker, days)
        df = CuratedDataLoader.prepare_for_forecasting(df)
        
        if df.empty or len(df) < 30:
            return jsonify({'error': 'Insufficient data for forecasting (need at least 30 days)'}), 400
        
        # Store in database
        db_manager.store_historical_data(ticker, df)
        
        # Prepare training data
        train_size = int(len(df) * 0.8)
        train_data = df['Close'].values[:train_size]
        test_data = df['Close'].values[train_size:train_size + steps]
        
        print(f"[INFO] Training data: {len(train_data)} samples")
        print(f"[INFO] Test data: {len(test_data)} samples\n")
        
        # Train ensemble model
        ensemble = EnsembleForecaster()
        ensemble.fit(train_data)
        
        # Generate predictions
        print("\n[INFO] Generating predictions...")
        predictions = ensemble.predict(steps, train_data)
        
        # Calculate metrics - always provide metrics for all models
        metrics = {}
        print(f"\n[INFO] Calculating metrics...")
        print(f"  Test data length: {len(test_data)}, Steps: {steps}")
        
        for model_name, pred_values in predictions.items():
            if len(test_data) >= len(pred_values):
                metrics[model_name] = ModelEvaluator.calculate_metrics(
                    test_data[:len(pred_values)], pred_values
                )
                print(f"  {model_name.upper()} (test): RMSE=${metrics[model_name]['rmse']:.4f}, "
                      f"MAE=${metrics[model_name]['mae']:.4f}, "
                      f"MAPE={metrics[model_name]['mape']:.2f}%")
            else:
                # Use last known values for validation
                last_vals = train_data[-steps:]
                if len(pred_values) > len(last_vals):
                    # If predictions are longer, use the first part
                    pred_subset = pred_values[:len(last_vals)]
                else:
                    # If last_vals are longer, use the last part
                    last_vals = last_vals[-len(pred_values):]
                    pred_subset = pred_values
                
                metrics[model_name] = ModelEvaluator.calculate_metrics(last_vals, pred_subset)
                print(f"  {model_name.upper()} (train): RMSE=${metrics[model_name]['rmse']:.4f}, "
                      f"MAE=${metrics[model_name]['mae']:.4f}, "
                      f"MAPE={metrics[model_name]['mape']:.2f}%")
        
        # Store predictions
        db_manager.store_prediction(ticker, horizon, predictions['ensemble'].tolist(), metrics)
        
        # Generate chart
        print("\n[INFO] Generating visualization...")
        try:
            chart_data = ChartGenerator.create_candlestick_with_forecast(
                df.tail(100), predictions, horizon, ticker
            )
            print(f"[DEBUG] Chart generation successful: {len(chart_data['data'])} traces")
        except Exception as e:
            print(f"[ERROR] Chart generation failed: {str(e)}")
            chart_data = None
        
        # Dataset info
        dataset_info = {
            'num_rows': len(df),
            'date_range': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}",
            'features': [col for col in df.columns if col not in ['Date']]
        }
        
        # Prepare response
        response_data = {
            'chart': chart_data,
            'metrics': metrics,
            'predictions': {k: v.tolist() for k, v in predictions.items()},
            'dataset_info': dataset_info
        }
        
        print(f"\n[SUCCESS] Forecast generated successfully!")
        print(f"[DEBUG] Response contains:")
        print(f"  - Chart data: {'✅' if chart_data else '❌'}")
        print(f"  - Chart traces: {len(chart_data['data']) if chart_data else 0}")
        print(f"  - Metrics: {'✅' if metrics else '❌'}")
        print(f"  - Metrics for models: {list(metrics.keys()) if metrics else 'None'}")
        print(f"  - Predictions: {'✅' if predictions else '❌'}")
        print(f"  - Dataset info: {'✅' if dataset_info else '❌'}")
        
        # Debug the actual JSON structure
        import json
        try:
            json_str = json.dumps(response_data, indent=2)
            print(f"[DEBUG] JSON serialization successful: {len(json_str)} characters")
        except Exception as e:
            print(f"[ERROR] JSON serialization failed: {str(e)}")
            # Create a minimal response if serialization fails
            response_data = {
                'error': f'Serialization error: {str(e)}',
                'metrics': metrics,
                'chart': None,
                'dataset_info': dataset_info
            }
        
        print(f"{'='*60}\n")
        
        return jsonify(response_data)
        
    except FileNotFoundError as e:
        print(f"[ERROR] {str(e)}")
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Stock Forecasting API Server Starting...")
    print("=" * 60)
    print("\n📋 Prerequisites:")
    print("  1. Ensure 'StockDataCollector.py' is in the same directory")
    print("  2. MongoDB should be running on localhost:27017")
    print("     (Optional - app works without DB but won't persist data)")
    print("  3. React frontend should be running on http://localhost:3000")
    print("\n🌐 API Server: http://localhost:5000")
    print("🌐 Frontend: http://localhost:3000 (run 'npm start' in another terminal)")
    print("\n📊 API Endpoints:")
    print("  - GET  /api/health - Health check")
    print("  - POST /api/forecast - Generate stock forecast")
    print("\n🚀 To start the complete application:")
    print("  1. Run this script: python ForecastPredictor.py")
    print("  2. In another terminal: npm start")
    print("  3. Open http://localhost:3000 in your browser")
    print("\n💡 Tip: First forecast takes longer as it collects fresh data")
    print("       Subsequent runs for the same ticker use cached data")
    print("\nPress Ctrl+C to stop the API server\n")
    print("=" * 60)
    
    # Check if StockDataCollector.py exists
    if not os.path.exists(Config.DATA_COLLECTOR_SCRIPT):
        print(f"\n⚠️  WARNING: '{Config.DATA_COLLECTOR_SCRIPT}' not found!")
        print("   Please ensure StockDataCollector.py is in the same directory.")
        print("   The API will not work without it.\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)