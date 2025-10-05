"""
Unit tests for the Stock Forecasting Application

Tests critical components as required by the assignment:
- Data loading and processing
- Model training and prediction
- API endpoints
- Database operations
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from ForecastPredictor import (
    ARIMAForecaster, 
    LSTMForecaster, 
    EnsembleForecaster,
    ModelEvaluator,
    CuratedDataLoader,
    DatabaseManager
)

class TestARIMAForecaster(unittest.TestCase):
    """Test ARIMA forecasting model"""
    
    def setUp(self):
        self.arima = ARIMAForecaster()
        # Create synthetic stock price data
        np.random.seed(42)
        self.train_data = np.cumsum(np.random.randn(100)) + 100
    
    def test_arima_fit_predict(self):
        """Test ARIMA model fitting and prediction"""
        self.arima.fit(self.train_data)
        predictions = self.arima.predict(steps=5)
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(isinstance(p, (int, float, np.number)) for p in predictions))
        self.assertIsNotNone(self.arima.fitted_model)

class TestLSTMForecaster(unittest.TestCase):
    """Test LSTM forecasting model"""
    
    def setUp(self):
        self.lstm = LSTMForecaster(lookback=10, hidden_size=32, num_layers=1)
        # Create synthetic stock price data
        np.random.seed(42)
        self.train_data = np.cumsum(np.random.randn(50)) + 100
    
    def test_lstm_fit_predict(self):
        """Test LSTM model fitting and prediction"""
        if len(self.train_data) > self.lstm.lookback:
            self.lstm.fit(self.train_data, epochs=5)  # Quick training for testing
            predictions = self.lstm.predict(steps=3, seed_data=self.train_data)
            
            self.assertEqual(len(predictions), 3)
            self.assertTrue(all(isinstance(p, (int, float, np.number)) for p in predictions))

class TestEnsembleForecaster(unittest.TestCase):
    """Test Ensemble forecasting model"""
    
    def setUp(self):
        self.ensemble = EnsembleForecaster()
        # Create synthetic stock price data
        np.random.seed(42)
        self.train_data = np.cumsum(np.random.randn(60)) + 100
    
    def test_ensemble_fit_predict(self):
        """Test ensemble model fitting and prediction"""
        self.ensemble.fit(self.train_data)
        predictions = self.ensemble.predict(steps=3, seed_data=self.train_data)
        
        # Check that all model predictions are returned
        self.assertIn('arima', predictions)
        self.assertIn('lstm', predictions)
        self.assertIn('ensemble', predictions)
        
        # Check prediction lengths
        for model_name, pred_values in predictions.items():
            self.assertEqual(len(pred_values), 3)
            self.assertTrue(all(isinstance(p, (int, float, np.number)) for p in pred_values))

class TestModelEvaluator(unittest.TestCase):
    """Test model evaluation metrics"""
    
    def test_calculate_metrics(self):
        """Test RMSE, MAE, and MAPE calculations"""
        y_true = np.array([100, 101, 102, 103, 104])
        y_pred = np.array([99, 102, 101, 104, 105])
        
        metrics = ModelEvaluator.calculate_metrics(y_true, y_pred)
        
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mape', metrics)
        
        # Check that metrics are positive numbers
        self.assertGreater(metrics['rmse'], 0)
        self.assertGreater(metrics['mae'], 0)
        self.assertGreater(metrics['mape'], 0)

class TestCuratedDataLoader(unittest.TestCase):
    """Test data loading functionality"""
    
    @patch('subprocess.run')
    @patch('os.path.exists')
    def test_run_data_collector(self, mock_exists, mock_run):
        """Test running the StockDataCollector.py script"""
        # Mock that the script exists and runs successfully
        mock_exists.side_effect = lambda path: path in ['StockDataCollector.py', 'AAPL_curated_dataset.csv']
        mock_run.return_value = MagicMock(returncode=0, stdout="Data collected successfully")
        
        result = CuratedDataLoader.run_data_collector('AAPL', 30)
        
        self.assertEqual(result, 'AAPL_curated_dataset.csv')
        mock_run.assert_called_once()
    
    def test_prepare_for_forecasting(self):
        """Test data preparation for forecasting"""
        # Create sample dataset similar to what StockDataCollector produces
        data = {
            'Date': pd.date_range('2024-01-01', periods=10),
            'Open': np.random.uniform(100, 110, 10),
            'High': np.random.uniform(110, 120, 10),
            'Low': np.random.uniform(90, 100, 10),
            'Close': np.random.uniform(100, 110, 10),
            'Adj Close': np.random.uniform(100, 110, 10),
            'Volume': np.random.randint(1000000, 10000000, 10),
            'MA5': np.random.uniform(100, 110, 10),
            'MA10': np.random.uniform(100, 110, 10),
            'Sentiment': np.random.uniform(-1, 1, 10)
        }
        df = pd.DataFrame(data)
        
        prepared_df = CuratedDataLoader.prepare_for_forecasting(df)
        
        # Check that required columns exist
        required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in required_cols:
            self.assertIn(col, prepared_df.columns)
        
        # Check that no NaN values in critical columns
        self.assertFalse(prepared_df['Close'].isna().any())
        self.assertFalse(prepared_df['Adj Close'].isna().any())

class TestDatabaseManager(unittest.TestCase):
    """Test database operations"""
    
    @patch('pymongo.MongoClient')
    def setUp(self, mock_client):
        """Set up test database manager with mocked MongoDB"""
        self.mock_client = mock_client
        self.mock_db = MagicMock()
        self.mock_collection = MagicMock()
        
        mock_client.return_value = MagicMock()
        mock_client.return_value.__getitem__.return_value = self.mock_db
        self.mock_db.__getitem__.return_value = self.mock_collection
        
        self.db_manager = DatabaseManager()
    
    def test_store_historical_data(self):
        """Test storing historical data in database"""
        # Create sample data
        data = pd.DataFrame({
            'Date': pd.date_range('2024-01-01', periods=5),
            'Close': [100, 101, 102, 103, 104],
            'Volume': [1000000, 1100000, 1200000, 1300000, 1400000]
        })
        
        self.db_manager.store_historical_data('AAPL', data)
        
        # Verify that delete_many and insert_many were called
        self.mock_collection.delete_many.assert_called_once()
        self.mock_collection.insert_many.assert_called_once()
    
    def test_store_prediction(self):
        """Test storing prediction results"""
        predictions = [105, 106, 107]
        metrics = {'rmse': 1.5, 'mae': 1.2, 'mape': 2.1}
        
        self.db_manager.store_prediction('AAPL', '24hrs', predictions, metrics)
        
        # Verify that insert_one was called
        self.mock_collection.insert_one.assert_called_once()

if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)