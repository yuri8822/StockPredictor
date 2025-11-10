"""
Portfolio Management Module
Manages simulated financial portfolio with trading actions based on predictions
Tracks performance metrics including returns, volatility, and Sharpe ratio
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class Position:
    """Represents a position in a stock"""
    
    def __init__(self, ticker: str, quantity: float, entry_price: float, 
                 entry_date: datetime):
        self.ticker = ticker
        self.quantity = quantity
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.current_price = entry_price
        self.unrealized_pnl = 0.0
    
    def update_price(self, current_price: float):
        """Update current price and calculate unrealized P&L"""
        self.current_price = current_price
        self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
    
    def get_value(self) -> float:
        """Get current value of position"""
        return self.current_price * self.quantity
    
    def get_return(self) -> float:
        """Get return percentage"""
        return ((self.current_price - self.entry_price) / self.entry_price) * 100
    
    def to_dict(self) -> Dict:
        """Convert position to dictionary"""
        return {
            'ticker': self.ticker,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date.isoformat() if isinstance(self.entry_date, datetime) else self.entry_date,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'value': self.get_value(),
            'return_pct': self.get_return()
        }


class Trade:
    """Represents a completed trade"""
    
    def __init__(self, ticker: str, action: str, quantity: float, 
                 price: float, date: datetime, commission: float = 0.0):
        self.ticker = ticker
        self.action = action  # 'buy' or 'sell'
        self.quantity = quantity
        self.price = price
        self.date = date
        self.commission = commission
        self.total_value = price * quantity + commission
    
    def to_dict(self) -> Dict:
        """Convert trade to dictionary"""
        return {
            'ticker': self.ticker,
            'action': self.action,
            'quantity': self.quantity,
            'price': self.price,
            'date': self.date.isoformat() if isinstance(self.date, datetime) else self.date,
            'commission': self.commission,
            'total_value': self.total_value
        }


class TradingStrategy:
    """Base class for trading strategies"""
    
    def generate_signal(self, prediction: float, current_price: float, 
                       historical_data: pd.DataFrame) -> str:
        """
        Generate trading signal based on prediction
        
        Args:
            prediction: Predicted price
            current_price: Current price
            historical_data: Historical price data
        
        Returns:
            signal: 'buy', 'sell', or 'hold'
        """
        raise NotImplementedError


class SimpleThresholdStrategy(TradingStrategy):
    """
    Simple threshold-based strategy
    Buy if predicted price is significantly higher than current
    Sell if predicted price is significantly lower than current
    """
    
    def __init__(self, buy_threshold: float = 0.02, sell_threshold: float = -0.02):
        """
        Args:
            buy_threshold: Buy if predicted return > this threshold (e.g., 0.02 = 2%)
            sell_threshold: Sell if predicted return < this threshold (e.g., -0.02 = -2%)
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
    
    def generate_signal(self, prediction: float, current_price: float,
                       historical_data: pd.DataFrame = None) -> str:
        """Generate signal based on predicted return"""
        predicted_return = (prediction - current_price) / current_price
        
        if predicted_return > self.buy_threshold:
            return 'buy'
        elif predicted_return < self.sell_threshold:
            return 'sell'
        else:
            return 'hold'


class MomentumStrategy(TradingStrategy):
    """
    Momentum-based strategy
    Considers both prediction and recent price momentum
    """
    
    def __init__(self, prediction_weight: float = 0.6, momentum_weight: float = 0.4,
                 threshold: float = 0.015):
        self.prediction_weight = prediction_weight
        self.momentum_weight = momentum_weight
        self.threshold = threshold
    
    def generate_signal(self, prediction: float, current_price: float,
                       historical_data: pd.DataFrame) -> str:
        """Generate signal combining prediction and momentum"""
        # Calculate predicted return
        predicted_return = (prediction - current_price) / current_price
        
        # Calculate momentum (recent trend)
        if len(historical_data) >= 5:
            recent_prices = historical_data['Close'].tail(5).values
            momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        else:
            momentum = 0.0
        
        # Combined signal
        combined_signal = (self.prediction_weight * predicted_return + 
                          self.momentum_weight * momentum)
        
        if combined_signal > self.threshold:
            return 'buy'
        elif combined_signal < -self.threshold:
            return 'sell'
        else:
            return 'hold'


class MeanReversionStrategy(TradingStrategy):
    """
    Mean reversion strategy
    Buy when price is below moving average and predicted to rise
    Sell when price is above moving average and predicted to fall
    """
    
    def __init__(self, ma_period: int = 20, threshold: float = 0.01):
        self.ma_period = ma_period
        self.threshold = threshold
    
    def generate_signal(self, prediction: float, current_price: float,
                       historical_data: pd.DataFrame) -> str:
        """Generate signal based on mean reversion"""
        if len(historical_data) < self.ma_period:
            return 'hold'
        
        # Calculate moving average
        ma = historical_data['Close'].tail(self.ma_period).mean()
        
        # Price deviation from MA
        deviation = (current_price - ma) / ma
        
        # Predicted return
        predicted_return = (prediction - current_price) / current_price
        
        # Buy if below MA and predicted to rise
        if deviation < -self.threshold and predicted_return > self.threshold:
            return 'buy'
        # Sell if above MA and predicted to fall
        elif deviation > self.threshold and predicted_return < -self.threshold:
            return 'sell'
        else:
            return 'hold'


class PortfolioManager:
    """
    Manages a simulated financial portfolio
    Tracks positions, executes trades, and calculates performance metrics
    """
    
    def __init__(self, initial_capital: float = 100000.0, 
                 commission_rate: float = 0.001,
                 portfolio_dir: str = './portfolio_data'):
        """
        Initialize portfolio manager
        
        Args:
            initial_capital: Starting capital in dollars
            commission_rate: Commission rate per trade (e.g., 0.001 = 0.1%)
            portfolio_dir: Directory to store portfolio data
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.portfolio_dir = portfolio_dir
        os.makedirs(portfolio_dir, exist_ok=True)
        
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Trade] = []
        self.portfolio_history: List[Dict] = []
        
        self.strategy = SimpleThresholdStrategy()
        
        # Load existing portfolio if available
        self._load_portfolio()
    
    def set_strategy(self, strategy: TradingStrategy):
        """Set trading strategy"""
        self.strategy = strategy
        print(f"[PortfolioManager] Strategy set to {strategy.__class__.__name__}")
    
    def execute_trade(self, ticker: str, action: str, quantity: float,
                     price: float, date: datetime = None) -> bool:
        """
        Execute a trade (buy or sell)
        
        Args:
            ticker: Stock ticker
            action: 'buy' or 'sell'
            quantity: Number of shares
            price: Price per share
            date: Trade date (defaults to now)
        
        Returns:
            success: Whether trade was executed
        """
        if date is None:
            date = datetime.now()
        
        commission = price * quantity * self.commission_rate
        
        if action == 'buy':
            total_cost = price * quantity + commission
            
            if total_cost > self.cash:
                print(f"[PortfolioManager] Insufficient funds to buy {quantity} {ticker}")
                return False
            
            # Execute buy
            self.cash -= total_cost
            
            if ticker in self.positions:
                # Add to existing position (average price)
                old_pos = self.positions[ticker]
                total_quantity = old_pos.quantity + quantity
                avg_price = ((old_pos.entry_price * old_pos.quantity + price * quantity) /
                           total_quantity)
                self.positions[ticker] = Position(ticker, total_quantity, avg_price, old_pos.entry_date)
            else:
                # Create new position
                self.positions[ticker] = Position(ticker, quantity, price, date)
            
            trade = Trade(ticker, action, quantity, price, date, commission)
            self.trade_history.append(trade)
            
            print(f"[PortfolioManager] BUY {quantity:.2f} {ticker} @ ${price:.2f} "
                  f"(Commission: ${commission:.2f})")
            
            return True
        
        elif action == 'sell':
            if ticker not in self.positions:
                print(f"[PortfolioManager] No position in {ticker} to sell")
                return False
            
            if self.positions[ticker].quantity < quantity:
                print(f"[PortfolioManager] Insufficient shares to sell {quantity} {ticker}")
                return False
            
            # Execute sell
            total_proceeds = price * quantity - commission
            self.cash += total_proceeds
            
            # Update or remove position
            if self.positions[ticker].quantity == quantity:
                del self.positions[ticker]
            else:
                self.positions[ticker].quantity -= quantity
            
            trade = Trade(ticker, action, quantity, price, date, commission)
            self.trade_history.append(trade)
            
            print(f"[PortfolioManager] SELL {quantity:.2f} {ticker} @ ${price:.2f} "
                  f"(Commission: ${commission:.2f})")
            
            return True
        
        return False
    
    def generate_and_execute_signal(self, ticker: str, prediction: float,
                                   current_price: float, current_date: datetime,
                                   historical_data: pd.DataFrame,
                                   position_size_pct: float = 0.1) -> str:
        """
        Generate trading signal and execute if appropriate
        
        Args:
            ticker: Stock ticker
            prediction: Predicted price
            current_price: Current market price
            current_date: Current date
            historical_data: Historical price data
            position_size_pct: Percentage of portfolio to allocate (e.g., 0.1 = 10%)
        
        Returns:
            action_taken: 'buy', 'sell', or 'hold'
        """
        # Generate signal
        signal = self.strategy.generate_signal(prediction, current_price, historical_data)
        
        if signal == 'buy':
            # Calculate position size
            portfolio_value = self.get_portfolio_value({ticker: current_price})
            position_value = portfolio_value * position_size_pct
            quantity = position_value / current_price
            
            # Execute buy
            if quantity > 0:
                self.execute_trade(ticker, 'buy', quantity, current_price, current_date)
                return 'buy'
        
        elif signal == 'sell':
            # Sell entire position
            if ticker in self.positions:
                quantity = self.positions[ticker].quantity
                self.execute_trade(ticker, 'sell', quantity, current_price, current_date)
                return 'sell'
        
        return 'hold'
    
    def update_positions(self, current_prices: Dict[str, float]):
        """Update all positions with current prices"""
        for ticker, price in current_prices.items():
            if ticker in self.positions:
                self.positions[ticker].update_price(price)
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        self.update_positions(current_prices)
        
        positions_value = sum(pos.get_value() for pos in self.positions.values())
        total_value = self.cash + positions_value
        
        return total_value
    
    def get_positions_summary(self) -> List[Dict]:
        """Get summary of current positions"""
        return [pos.to_dict() for pos in self.positions.values()]
    
    def get_trade_history(self, limit: int = None) -> List[Dict]:
        """Get trade history"""
        history = [trade.to_dict() for trade in self.trade_history]
        if limit:
            history = history[-limit:]
        return history
    
    def calculate_metrics(self, current_prices: Dict[str, float],
                         time_period_days: int = 30) -> Dict:
        """
        Calculate portfolio performance metrics
        
        Args:
            current_prices: Current prices for all holdings
            time_period_days: Time period for calculations
        
        Returns:
            metrics: Dictionary of performance metrics
        """
        current_value = self.get_portfolio_value(current_prices)
        
        # Total return
        total_return = ((current_value - self.initial_capital) / self.initial_capital) * 100
        
        # Calculate returns from portfolio history
        if len(self.portfolio_history) >= 2:
            returns = []
            for i in range(1, len(self.portfolio_history)):
                prev_value = self.portfolio_history[i-1]['total_value']
                curr_value = self.portfolio_history[i]['total_value']
                if prev_value > 0:
                    daily_return = (curr_value - prev_value) / prev_value
                    returns.append(daily_return)
            
            if returns:
                returns_array = np.array(returns)
                
                # Volatility (annualized)
                volatility = np.std(returns_array) * np.sqrt(252)  # 252 trading days
                
                # Sharpe Ratio (assuming 0% risk-free rate for simplicity)
                avg_return = np.mean(returns_array)
                sharpe_ratio = (avg_return * 252) / volatility if volatility > 0 else 0.0
                
                # Maximum Drawdown
                cumulative_returns = (1 + returns_array).cumprod()
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = np.min(drawdown) * 100
            else:
                volatility = 0.0
                sharpe_ratio = 0.0
                max_drawdown = 0.0
        else:
            volatility = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
        
        # Win rate from trades
        if len(self.trade_history) >= 2:
            # Match buy/sell pairs to calculate wins/losses
            profitable_trades = 0
            total_trades = 0
            
            # Simple approach: compare sequential buy-sell pairs
            i = 0
            while i < len(self.trade_history) - 1:
                if (self.trade_history[i].action == 'buy' and 
                    self.trade_history[i+1].action == 'sell' and
                    self.trade_history[i].ticker == self.trade_history[i+1].ticker):
                    
                    profit = (self.trade_history[i+1].price - 
                             self.trade_history[i].price) * self.trade_history[i].quantity
                    
                    if profit > 0:
                        profitable_trades += 1
                    total_trades += 1
                    i += 2
                else:
                    i += 1
            
            win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0.0
        else:
            win_rate = 0.0
        
        metrics = {
            'total_value': current_value,
            'cash': self.cash,
            'positions_value': current_value - self.cash,
            'total_return_pct': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'win_rate_pct': win_rate,
            'num_trades': len(self.trade_history),
            'num_positions': len(self.positions)
        }
        
        return metrics
    
    def log_portfolio_state(self, current_prices: Dict[str, float]):
        """Log current portfolio state for historical tracking"""
        current_value = self.get_portfolio_value(current_prices)
        
        state = {
            'timestamp': datetime.now().isoformat(),
            'total_value': current_value,
            'cash': self.cash,
            'positions': self.get_positions_summary(),
            'num_trades': len(self.trade_history)
        }
        
        self.portfolio_history.append(state)
        
        # Keep last 1000 entries
        if len(self.portfolio_history) > 1000:
            self.portfolio_history = self.portfolio_history[-1000:]
        
        # Save to disk
        self._save_portfolio()
    
    def _save_portfolio(self):
        """Save portfolio state to disk"""
        portfolio_file = os.path.join(self.portfolio_dir, 'portfolio_state.json')
        
        state = {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'positions': [pos.to_dict() for pos in self.positions.values()],
            'trade_history': [trade.to_dict() for trade in self.trade_history],
            'portfolio_history': self.portfolio_history[-100:]  # Last 100 entries
        }
        
        with open(portfolio_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_portfolio(self):
        """Load portfolio state from disk"""
        portfolio_file = os.path.join(self.portfolio_dir, 'portfolio_state.json')
        
        if not os.path.exists(portfolio_file):
            return
        
        try:
            with open(portfolio_file, 'r') as f:
                state = json.load(f)
            
            self.cash = state['cash']
            
            # Reconstruct positions
            for pos_data in state.get('positions', []):
                pos = Position(
                    pos_data['ticker'],
                    pos_data['quantity'],
                    pos_data['entry_price'],
                    datetime.fromisoformat(pos_data['entry_date'])
                )
                pos.current_price = pos_data['current_price']
                pos.unrealized_pnl = pos_data['unrealized_pnl']
                self.positions[pos.ticker] = pos
            
            # Reconstruct trade history
            for trade_data in state.get('trade_history', []):
                trade = Trade(
                    trade_data['ticker'],
                    trade_data['action'],
                    trade_data['quantity'],
                    trade_data['price'],
                    datetime.fromisoformat(trade_data['date']),
                    trade_data['commission']
                )
                self.trade_history.append(trade)
            
            self.portfolio_history = state.get('portfolio_history', [])
            
            print(f"[PortfolioManager] Loaded portfolio: ${self.cash:.2f} cash, "
                  f"{len(self.positions)} positions, {len(self.trade_history)} trades")
        
        except Exception as e:
            print(f"[PortfolioManager] Error loading portfolio: {e}")


if __name__ == '__main__':
    print("Portfolio Management Module loaded successfully")
    print("Available classes: PortfolioManager, TradingStrategy, SimpleThresholdStrategy")
