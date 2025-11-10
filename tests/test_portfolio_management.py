"""
Unit tests for Portfolio Management Module
Tests trading strategies, position management, and performance metrics
"""

import unittest
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from PortfolioManager import (
    Position, Trade, PortfolioManager,
    SimpleThresholdStrategy, MomentumStrategy, MeanReversionStrategy
)


class TestPosition(unittest.TestCase):
    """Test Position class"""
    
    def test_position_creation(self):
        """Test creating a position"""
        pos = Position('AAPL', 100, 150.0, datetime.now())
        
        self.assertEqual(pos.ticker, 'AAPL')
        self.assertEqual(pos.quantity, 100)
        self.assertEqual(pos.entry_price, 150.0)
        self.assertEqual(pos.get_value(), 15000.0)
    
    def test_position_update_price(self):
        """Test updating position price"""
        pos = Position('AAPL', 100, 150.0, datetime.now())
        pos.update_price(160.0)
        
        self.assertEqual(pos.current_price, 160.0)
        self.assertEqual(pos.unrealized_pnl, 1000.0)  # (160-150) * 100
        self.assertEqual(pos.get_return(), (10.0/150.0) * 100)
    
    def test_position_to_dict(self):
        """Test converting position to dictionary"""
        pos = Position('AAPL', 100, 150.0, datetime.now())
        pos.update_price(160.0)
        
        pos_dict = pos.to_dict()
        
        self.assertIn('ticker', pos_dict)
        self.assertIn('quantity', pos_dict)
        self.assertIn('unrealized_pnl', pos_dict)
        self.assertEqual(pos_dict['ticker'], 'AAPL')


class TestTrade(unittest.TestCase):
    """Test Trade class"""
    
    def test_trade_creation(self):
        """Test creating a trade"""
        trade = Trade('AAPL', 'buy', 100, 150.0, datetime.now(), commission=1.5)
        
        self.assertEqual(trade.ticker, 'AAPL')
        self.assertEqual(trade.action, 'buy')
        self.assertEqual(trade.quantity, 100)
        self.assertEqual(trade.total_value, 15001.5)  # 150*100 + 1.5
    
    def test_trade_to_dict(self):
        """Test converting trade to dictionary"""
        trade = Trade('AAPL', 'sell', 50, 160.0, datetime.now())
        
        trade_dict = trade.to_dict()
        
        self.assertIn('ticker', trade_dict)
        self.assertIn('action', trade_dict)
        self.assertEqual(trade_dict['action'], 'sell')


class TestTradingStrategies(unittest.TestCase):
    """Test trading strategy classes"""
    
    def setUp(self):
        """Set up test data"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        prices = np.random.randn(50).cumsum() + 100
        self.historical_data = pd.DataFrame({
            'Date': dates,
            'Close': prices
        })
    
    def test_simple_threshold_strategy(self):
        """Test SimpleThresholdStrategy"""
        strategy = SimpleThresholdStrategy(buy_threshold=0.02, sell_threshold=-0.02)
        
        # Test buy signal
        current_price = 100.0
        prediction = 103.0  # 3% increase
        signal = strategy.generate_signal(prediction, current_price, self.historical_data)
        self.assertEqual(signal, 'buy')
        
        # Test sell signal
        prediction = 97.0  # 3% decrease
        signal = strategy.generate_signal(prediction, current_price, self.historical_data)
        self.assertEqual(signal, 'sell')
        
        # Test hold signal
        prediction = 100.5  # 0.5% increase
        signal = strategy.generate_signal(prediction, current_price, self.historical_data)
        self.assertEqual(signal, 'hold')
    
    def test_momentum_strategy(self):
        """Test MomentumStrategy"""
        strategy = MomentumStrategy()
        
        current_price = 100.0
        prediction = 102.0
        
        signal = strategy.generate_signal(prediction, current_price, self.historical_data)
        
        # Signal should be one of: buy, sell, hold
        self.assertIn(signal, ['buy', 'sell', 'hold'])
    
    def test_mean_reversion_strategy(self):
        """Test MeanReversionStrategy"""
        strategy = MeanReversionStrategy(ma_period=20)
        
        current_price = 100.0
        prediction = 102.0
        
        signal = strategy.generate_signal(prediction, current_price, self.historical_data)
        
        # Signal should be one of: buy, sell, hold
        self.assertIn(signal, ['buy', 'sell', 'hold'])


class TestPortfolioManager(unittest.TestCase):
    """Test PortfolioManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.portfolio = PortfolioManager(
            initial_capital=100000.0,
            commission_rate=0.001,
            portfolio_dir=self.test_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_initial_state(self):
        """Test initial portfolio state"""
        self.assertEqual(self.portfolio.initial_capital, 100000.0)
        self.assertEqual(self.portfolio.cash, 100000.0)
        self.assertEqual(len(self.portfolio.positions), 0)
        self.assertEqual(len(self.portfolio.trade_history), 0)
    
    def test_buy_trade(self):
        """Test buying shares"""
        success = self.portfolio.execute_trade('AAPL', 'buy', 100, 150.0, datetime.now())
        
        self.assertTrue(success)
        self.assertIn('AAPL', self.portfolio.positions)
        self.assertEqual(self.portfolio.positions['AAPL'].quantity, 100)
        self.assertLess(self.portfolio.cash, 100000.0)  # Cash should decrease
        self.assertEqual(len(self.portfolio.trade_history), 1)
    
    def test_sell_trade(self):
        """Test selling shares"""
        # First buy
        self.portfolio.execute_trade('AAPL', 'buy', 100, 150.0, datetime.now())
        
        # Then sell
        success = self.portfolio.execute_trade('AAPL', 'sell', 50, 160.0, datetime.now())
        
        self.assertTrue(success)
        self.assertEqual(self.portfolio.positions['AAPL'].quantity, 50)  # 50 shares left
        self.assertEqual(len(self.portfolio.trade_history), 2)
    
    def test_sell_without_position(self):
        """Test selling shares without having a position"""
        success = self.portfolio.execute_trade('AAPL', 'sell', 100, 150.0, datetime.now())
        
        self.assertFalse(success)  # Should fail
    
    def test_insufficient_funds(self):
        """Test buying with insufficient funds"""
        success = self.portfolio.execute_trade('AAPL', 'buy', 10000, 150.0, datetime.now())
        
        self.assertFalse(success)  # Should fail
    
    def test_portfolio_value(self):
        """Test calculating portfolio value"""
        # Buy shares
        self.portfolio.execute_trade('AAPL', 'buy', 100, 150.0, datetime.now())
        self.portfolio.execute_trade('GOOGL', 'buy', 50, 200.0, datetime.now())
        
        # Update prices
        current_prices = {'AAPL': 160.0, 'GOOGL': 210.0}
        total_value = self.portfolio.get_portfolio_value(current_prices)
        
        # Should be cash + (100*160) + (50*210)
        expected_value = self.portfolio.cash + 16000 + 10500
        self.assertAlmostEqual(total_value, expected_value, places=2)
    
    def test_calculate_metrics(self):
        """Test calculating portfolio metrics"""
        # Buy and sell to create trade history
        self.portfolio.execute_trade('AAPL', 'buy', 100, 150.0, datetime.now())
        self.portfolio.execute_trade('AAPL', 'sell', 100, 160.0, datetime.now())
        
        # Log some portfolio states
        self.portfolio.log_portfolio_state({'AAPL': 160.0})
        
        # Calculate metrics
        current_prices = {}
        metrics = self.portfolio.calculate_metrics(current_prices)
        
        self.assertIn('total_value', metrics)
        self.assertIn('total_return_pct', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('win_rate_pct', metrics)
        self.assertIn('num_trades', metrics)
    
    def test_generate_and_execute_signal(self):
        """Test generating and executing trading signal"""
        # Create historical data
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        prices = np.random.randn(50).cumsum() + 150
        historical_data = pd.DataFrame({'Date': dates, 'Close': prices})
        
        # Generate signal
        action = self.portfolio.generate_and_execute_signal(
            'AAPL',
            prediction=155.0,  # Predicted price
            current_price=150.0,
            current_date=datetime.now(),
            historical_data=historical_data,
            position_size_pct=0.1
        )
        
        # Action should be one of: buy, sell, hold
        self.assertIn(action, ['buy', 'sell', 'hold'])
        
        # If buy signal was generated, should have a position
        if action == 'buy':
            self.assertIn('AAPL', self.portfolio.positions)
    
    def test_persistence(self):
        """Test saving and loading portfolio state"""
        # Create portfolio and make trades
        portfolio1 = PortfolioManager(
            initial_capital=100000.0,
            portfolio_dir=self.test_dir
        )
        portfolio1.execute_trade('AAPL', 'buy', 100, 150.0, datetime.now())
        portfolio1.execute_trade('GOOGL', 'buy', 50, 200.0, datetime.now())
        
        # Create new portfolio instance (should load saved state)
        portfolio2 = PortfolioManager(portfolio_dir=self.test_dir)
        
        # Check that state was loaded
        self.assertEqual(len(portfolio2.positions), 2)
        self.assertIn('AAPL', portfolio2.positions)
        self.assertIn('GOOGL', portfolio2.positions)
        self.assertEqual(len(portfolio2.trade_history), 2)


class TestIntegration(unittest.TestCase):
    """Integration tests for portfolio management"""
    
    def test_trading_workflow(self):
        """Test complete trading workflow"""
        test_dir = tempfile.mkdtemp()
        
        try:
            portfolio = PortfolioManager(
                initial_capital=100000.0,
                portfolio_dir=test_dir
            )
            
            # Set strategy
            portfolio.set_strategy(SimpleThresholdStrategy())
            
            # Create historical data
            dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
            prices = np.random.randn(100).cumsum() + 150
            historical_data = pd.DataFrame({'Date': dates, 'Close': prices})
            
            # Simulate trading over time
            for i in range(10):
                current_price = prices[i * 10]
                prediction = current_price * (1 + np.random.uniform(-0.05, 0.05))
                
                action = portfolio.generate_and_execute_signal(
                    'TEST',
                    prediction,
                    current_price,
                    dates[i * 10],
                    historical_data.iloc[:i * 10 + 1],
                    position_size_pct=0.1
                )
                
                # Log portfolio state
                portfolio.log_portfolio_state({'TEST': current_price})
            
            # Calculate final metrics
            final_prices = {'TEST': prices[-1]}
            metrics = portfolio.calculate_metrics(final_prices)
            
            # Verify metrics were calculated
            self.assertIn('total_value', metrics)
            self.assertGreaterEqual(metrics['num_trades'], 0)
            
            # Verify portfolio history was logged
            self.assertGreater(len(portfolio.portfolio_history), 0)
            
        finally:
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)


if __name__ == '__main__':
    print("Running Portfolio Management Tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPosition))
    suite.addTests(loader.loadTestsFromTestCase(TestTrade))
    suite.addTests(loader.loadTestsFromTestCase(TestTradingStrategies))
    suite.addTests(loader.loadTestsFromTestCase(TestPortfolioManager))
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
