"""
Unit tests for the Adaptive Learning Module
Tests model versioning, incremental updates, and ensemble learning
"""

import unittest
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from AdaptiveLearning import ModelVersion, AdaptiveLearningManager, OnlineLearningDataset


class SimpleTestModel(nn.Module):
    """Simple model for testing"""
    def __init__(self, input_size=1, hidden_size=10):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.out(x)


class TestModelVersion(unittest.TestCase):
    """Test ModelVersion class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.model_version = ModelVersion(model_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_save_model_version(self):
        """Test saving a model version"""
        model = SimpleTestModel()
        config = {'input_size': 1, 'hidden_size': 10}
        metrics = {'mae': 0.05, 'rmse': 0.07, 'mape': 2.5}
        
        version_id = self.model_version.save_model_version(
            model, 'AAPL', 'TestModel', metrics, config
        )
        
        self.assertIsNotNone(version_id)
        self.assertIn('AAPL', version_id)
        self.assertIn('TestModel', version_id)
        
        # Check if version directory was created
        version_dir = os.path.join(self.test_dir, version_id)
        self.assertTrue(os.path.exists(version_dir))
        
        # Check if metadata file exists
        metadata_path = os.path.join(version_dir, 'metadata.json')
        self.assertTrue(os.path.exists(metadata_path))
    
    def test_get_version_history(self):
        """Test retrieving version history"""
        model = SimpleTestModel()
        config = {'input_size': 1, 'hidden_size': 10}
        
        # Save multiple versions
        for i in range(3):
            metrics = {'mae': 0.05 + i * 0.01, 'rmse': 0.07, 'mape': 2.5}
            self.model_version.save_model_version(
                model, 'AAPL', 'TestModel', metrics, config
            )
        
        history = self.model_version.get_version_history('AAPL', 'TestModel')
        
        self.assertEqual(len(history), 3)
        self.assertEqual(history[0]['ticker'], 'AAPL')
    
    def test_get_best_version(self):
        """Test finding the best model version"""
        model = SimpleTestModel()
        config = {'input_size': 1, 'hidden_size': 10}
        
        # Save versions with different MAE
        metrics_list = [
            {'mae': 0.10, 'rmse': 0.15, 'mape': 5.0},
            {'mae': 0.05, 'rmse': 0.08, 'mape': 2.5},  # Best
            {'mae': 0.12, 'rmse': 0.18, 'mape': 6.0},
        ]
        
        for metrics in metrics_list:
            self.model_version.save_model_version(
                model, 'AAPL', 'TestModel', metrics, config
            )
        
        best_version_id = self.model_version.get_best_version('AAPL', 'TestModel', metric='mae')
        
        self.assertIsNotNone(best_version_id)
        
        # Verify it's the version with lowest MAE
        history = self.model_version.get_version_history('AAPL', 'TestModel')
        best_version = [v for v in history if v['version_id'] == best_version_id][0]
        self.assertEqual(best_version['metrics']['mae'], 0.05)


class TestOnlineLearningDataset(unittest.TestCase):
    """Test OnlineLearningDataset class"""
    
    def test_dataset_creation(self):
        """Test creating an online learning dataset"""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        lookback = 3
        
        dataset = OnlineLearningDataset(data, lookback)
        
        # Dataset length should be len(data) - lookback
        self.assertEqual(len(dataset), 7)
        
        # Test first sample
        X, y = dataset[0]
        self.assertEqual(len(X), 3)
        self.assertEqual(y.item(), 4)
        
        # Test last sample
        X, y = dataset[-1]
        self.assertEqual(y.item(), 10)


class TestAdaptiveLearningManager(unittest.TestCase):
    """Test AdaptiveLearningManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.model_version = ModelVersion(model_dir=self.test_dir)
        self.adaptive_manager = AdaptiveLearningManager(self.model_version)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_should_update_model(self):
        """Test model update decision logic"""
        current_metrics = {'mae': 0.10, 'rmse': 0.15}
        historical_metrics = [
            {'mae': 0.05, 'rmse': 0.08},
            {'mae': 0.06, 'rmse': 0.09},
            {'mae': 0.05, 'rmse': 0.07},
        ]
        
        # Should update because current MAE is much higher
        should_update = self.adaptive_manager.should_update_model(
            current_metrics, historical_metrics
        )
        
        self.assertTrue(should_update)
        
        # Should not update if current MAE is similar
        current_metrics = {'mae': 0.055, 'rmse': 0.08}
        should_update = self.adaptive_manager.should_update_model(
            current_metrics, historical_metrics
        )
        
        self.assertFalse(should_update)
    
    def test_incremental_update(self):
        """Test incremental model update"""
        # Create simple test data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        prices = np.random.randn(100).cumsum() + 100
        df = pd.DataFrame({'Date': dates, 'Close': prices})
        
        # Create a simple model
        model = SimpleTestModel(input_size=1, hidden_size=10)
        config = {'input_size': 1, 'hidden_size': 10}
        
        # Perform incremental update
        updated_model, metrics = self.adaptive_manager.incremental_update(
            model, df, 'TEST', 'TestModel', config, epochs=5, lr=0.01
        )
        
        # Check that metrics were calculated
        self.assertIn('mae', metrics)
        self.assertIn('rmse', metrics)
        self.assertIn('mape', metrics)
        
        # Check that model was updated (not None)
        self.assertIsNotNone(updated_model)
        
        # Check that version was saved
        history = self.model_version.get_version_history('TEST', 'TestModel')
        self.assertGreater(len(history), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for the adaptive learning system"""
    
    def test_full_workflow(self):
        """Test the complete adaptive learning workflow"""
        # Setup
        test_dir = tempfile.mkdtemp()
        model_version = ModelVersion(model_dir=test_dir)
        adaptive_manager = AdaptiveLearningManager(model_version)
        
        try:
            # Create test data
            dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
            prices = np.random.randn(200).cumsum() + 100
            df = pd.DataFrame({'Date': dates, 'Close': prices})
            
            # Initial training
            model = SimpleTestModel(input_size=1, hidden_size=20)
            config = {'input_size': 1, 'hidden_size': 20}
            
            # First update
            model1, metrics1 = adaptive_manager.incremental_update(
                model, df.iloc[:100], 'WORKFLOW', 'TestModel', config, epochs=5
            )
            
            # Second update with new data
            model2, metrics2 = adaptive_manager.incremental_update(
                model1, df.iloc[100:150], 'WORKFLOW', 'TestModel', config, epochs=5
            )
            
            # Third update
            model3, metrics3 = adaptive_manager.incremental_update(
                model2, df.iloc[150:], 'WORKFLOW', 'TestModel', config, epochs=5
            )
            
            # Verify version history
            history = model_version.get_version_history('WORKFLOW', 'TestModel')
            self.assertEqual(len(history), 3)
            
            # Verify timestamps are in order
            timestamps = [v['timestamp'] for v in history]
            self.assertEqual(timestamps, sorted(timestamps, reverse=True))
            
            # Get best version
            best_version_id = model_version.get_best_version('WORKFLOW', 'TestModel')
            self.assertIsNotNone(best_version_id)
            
        finally:
            # Cleanup
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)


if __name__ == '__main__':
    print("Running Adaptive Learning Module Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModelVersion))
    suite.addTests(loader.loadTestsFromTestCase(TestOnlineLearningDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestAdaptiveLearningManager))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.wasSuccessful()}")
    print("=" * 60)
