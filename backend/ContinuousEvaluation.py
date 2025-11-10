"""
Continuous Evaluation and Monitoring Module
Tracks model performance over time, logs metrics, and provides dashboard data
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')


class MetricsLogger:
    """Logs and stores evaluation metrics over time"""
    
    def __init__(self, log_dir='./evaluation_logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_session = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def log_metrics(self, ticker: str, model_type: str, horizon: str,
                   predictions: np.ndarray, actuals: np.ndarray,
                   metadata: Optional[Dict] = None):
        """
        Log evaluation metrics for a prediction
        
        Args:
            ticker: Stock ticker
            model_type: Type of model
            horizon: Forecast horizon
            predictions: Predicted values
            actuals: Actual values
            metadata: Additional metadata to store
        """
        # Calculate metrics
        mae = float(mean_absolute_error(actuals, predictions))
        rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
        mape = float(mean_absolute_percentage_error(actuals, predictions) * 100)
        
        # Calculate additional metrics
        errors = predictions - actuals
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))
        max_error = float(np.max(np.abs(errors)))
        
        # Calculate directional accuracy (did we predict up/down correctly?)
        if len(actuals) > 1:
            actual_direction = np.sign(np.diff(actuals))
            pred_direction = np.sign(np.diff(predictions))
            directional_accuracy = float(np.mean(actual_direction == pred_direction) * 100)
        else:
            directional_accuracy = None
        
        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'model_type': model_type,
            'horizon': horizon,
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'mean_error': mean_error,
                'std_error': std_error,
                'max_error': max_error,
                'directional_accuracy': directional_accuracy
            },
            'data_points': len(predictions),
            'metadata': metadata or {}
        }
        
        # Save to file
        log_file = os.path.join(
            self.log_dir, 
            f"{ticker}_{model_type}_{horizon}_metrics.jsonl"
        )
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        print(f"[MetricsLogger] Logged metrics for {ticker} {model_type} - "
              f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.2f}%")
        
        return log_entry
    
    def get_metrics_history(self, ticker: str, model_type: str = None, 
                           horizon: str = None, limit: int = None) -> List[Dict]:
        """
        Retrieve metrics history from logs
        
        Args:
            ticker: Stock ticker
            model_type: Optional filter by model type
            horizon: Optional filter by horizon
            limit: Optional limit number of results
        
        Returns:
            metrics_history: List of metric entries
        """
        # Find matching log files
        pattern = f"{ticker}_"
        if model_type:
            pattern += f"{model_type}_"
        else:
            pattern += "*_"
        if horizon:
            pattern += f"{horizon}_"
        else:
            pattern += "*_"
        pattern += "metrics.jsonl"
        
        import glob
        log_files = glob.glob(os.path.join(self.log_dir, pattern))
        
        # Read all entries
        all_entries = []
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        all_entries.append(entry)
            except Exception as e:
                print(f"[MetricsLogger] Error reading {log_file}: {e}")
        
        # Sort by timestamp
        all_entries.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Apply limit
        if limit:
            all_entries = all_entries[:limit]
        
        return all_entries
    
    def get_aggregated_metrics(self, ticker: str, model_type: str = None,
                              time_window: timedelta = timedelta(days=30)) -> Dict:
        """
        Get aggregated metrics over a time window
        
        Args:
            ticker: Stock ticker
            model_type: Optional filter by model type
            time_window: Time window for aggregation
        
        Returns:
            aggregated_metrics: Dictionary of aggregated metrics
        """
        # Get recent history
        history = self.get_metrics_history(ticker, model_type)
        
        # Filter by time window
        cutoff_time = datetime.now() - time_window
        recent_history = [
            entry for entry in history
            if datetime.fromisoformat(entry['timestamp']) >= cutoff_time
        ]
        
        if not recent_history:
            return {}
        
        # Aggregate metrics
        mae_values = [e['metrics']['mae'] for e in recent_history]
        rmse_values = [e['metrics']['rmse'] for e in recent_history]
        mape_values = [e['metrics']['mape'] for e in recent_history]
        
        aggregated = {
            'ticker': ticker,
            'model_type': model_type,
            'time_window_days': time_window.days,
            'num_evaluations': len(recent_history),
            'mae': {
                'mean': float(np.mean(mae_values)),
                'std': float(np.std(mae_values)),
                'min': float(np.min(mae_values)),
                'max': float(np.max(mae_values))
            },
            'rmse': {
                'mean': float(np.mean(rmse_values)),
                'std': float(np.std(rmse_values)),
                'min': float(np.min(rmse_values)),
                'max': float(np.max(rmse_values))
            },
            'mape': {
                'mean': float(np.mean(mape_values)),
                'std': float(np.std(mape_values)),
                'min': float(np.min(mape_values)),
                'max': float(np.max(mape_values))
            },
            'recent_trend': self._calculate_trend(mae_values)
        }
        
        return aggregated
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate if metrics are improving, degrading, or stable"""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Compare recent half vs older half
        mid = len(values) // 2
        recent_avg = np.mean(values[:mid])
        older_avg = np.mean(values[mid:])
        
        if recent_avg < older_avg * 0.95:
            return 'improving'
        elif recent_avg > older_avg * 1.05:
            return 'degrading'
        else:
            return 'stable'


class ContinuousEvaluator:
    """
    Manages continuous evaluation of forecasting models
    Automatically evaluates predictions as ground truth becomes available
    """
    
    def __init__(self, metrics_logger: MetricsLogger):
        self.logger = metrics_logger
        self.pending_evaluations = {}  # Store predictions waiting for ground truth
        self.evaluation_schedule = {}  # Track when to evaluate
    
    def register_prediction(self, ticker: str, model_type: str, horizon: str,
                          predictions: List[float], prediction_dates: List[str],
                          metadata: Optional[Dict] = None):
        """
        Register a prediction for future evaluation
        
        Args:
            ticker: Stock ticker
            model_type: Model type
            horizon: Forecast horizon
            predictions: List of predicted values
            prediction_dates: List of dates for predictions
            metadata: Additional metadata
        """
        prediction_id = f"{ticker}_{model_type}_{horizon}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        self.pending_evaluations[prediction_id] = {
            'ticker': ticker,
            'model_type': model_type,
            'horizon': horizon,
            'predictions': predictions,
            'prediction_dates': prediction_dates,
            'metadata': metadata or {},
            'registered_at': datetime.now().isoformat()
        }
        
        # Schedule evaluation for when ground truth should be available
        last_date = pd.to_datetime(prediction_dates[-1])
        evaluation_date = last_date + timedelta(days=1)
        
        if evaluation_date not in self.evaluation_schedule:
            self.evaluation_schedule[evaluation_date] = []
        
        self.evaluation_schedule[evaluation_date].append(prediction_id)
        
        print(f"[ContinuousEvaluator] Registered prediction {prediction_id} "
              f"for evaluation on {evaluation_date.date()}")
        
        return prediction_id
    
    def evaluate_pending(self, actual_data: pd.DataFrame) -> List[Dict]:
        """
        Evaluate pending predictions against actual data
        
        Args:
            actual_data: DataFrame with actual price data (must have Date and Close columns)
        
        Returns:
            evaluation_results: List of evaluation results
        """
        results = []
        current_date = datetime.now().date()
        
        # Check scheduled evaluations
        for eval_date, prediction_ids in list(self.evaluation_schedule.items()):
            if eval_date.date() <= current_date:
                for pred_id in prediction_ids:
                    if pred_id in self.pending_evaluations:
                        result = self._evaluate_single(pred_id, actual_data)
                        if result:
                            results.append(result)
                
                # Remove processed date
                del self.evaluation_schedule[eval_date]
        
        return results
    
    def _evaluate_single(self, prediction_id: str, actual_data: pd.DataFrame) -> Optional[Dict]:
        """Evaluate a single pending prediction"""
        pred_info = self.pending_evaluations[prediction_id]
        
        try:
            # Get actual values for prediction dates
            actual_data['Date'] = pd.to_datetime(actual_data['Date'])
            pred_dates = [pd.to_datetime(d) for d in pred_info['prediction_dates']]
            
            actuals = []
            matched_predictions = []
            
            for pred_date, pred_value in zip(pred_dates, pred_info['predictions']):
                actual_row = actual_data[actual_data['Date'] == pred_date]
                if not actual_row.empty:
                    actuals.append(actual_row['Close'].values[0])
                    matched_predictions.append(pred_value)
            
            if not actuals:
                print(f"[ContinuousEvaluator] No actual data available yet for {prediction_id}")
                return None
            
            # Convert to numpy arrays
            actuals = np.array(actuals)
            matched_predictions = np.array(matched_predictions)
            
            # Log metrics
            log_entry = self.logger.log_metrics(
                pred_info['ticker'],
                pred_info['model_type'],
                pred_info['horizon'],
                matched_predictions,
                actuals,
                pred_info['metadata']
            )
            
            # Remove from pending
            del self.pending_evaluations[prediction_id]
            
            print(f"[ContinuousEvaluator] Evaluated {prediction_id} - "
                  f"MAE: {log_entry['metrics']['mae']:.6f}")
            
            return log_entry
            
        except Exception as e:
            print(f"[ContinuousEvaluator] Error evaluating {prediction_id}: {e}")
            return None
    
    def get_evaluation_status(self) -> Dict:
        """Get status of pending evaluations"""
        return {
            'pending_evaluations': len(self.pending_evaluations),
            'scheduled_dates': len(self.evaluation_schedule),
            'next_evaluation': min(self.evaluation_schedule.keys()).isoformat() 
                               if self.evaluation_schedule else None
        }


class PerformanceMonitor:
    """
    High-level monitoring dashboard for model performance
    Provides aggregated views and alerts
    """
    
    def __init__(self, metrics_logger: MetricsLogger):
        self.logger = metrics_logger
        self.alert_thresholds = {
            'mae_increase': 0.10,  # 10% increase triggers alert
            'mape_threshold': 15.0,  # MAPE > 15% triggers alert
            'directional_accuracy_min': 50.0  # Below 50% triggers alert
        }
    
    def get_dashboard_data(self, ticker: str, time_window: timedelta = timedelta(days=30)) -> Dict:
        """
        Get comprehensive dashboard data for monitoring
        
        Args:
            ticker: Stock ticker
            time_window: Time window for analysis
        
        Returns:
            dashboard_data: Dictionary with all dashboard metrics
        """
        # Get history for all models
        history = self.logger.get_metrics_history(ticker, limit=1000)
        
        # Filter by time window
        cutoff_time = datetime.now() - time_window
        recent_history = [
            entry for entry in history
            if datetime.fromisoformat(entry['timestamp']) >= cutoff_time
        ]
        
        if not recent_history:
            return {'error': 'No data available', 'ticker': ticker}
        
        # Group by model type
        by_model = {}
        for entry in recent_history:
            model_type = entry['model_type']
            if model_type not in by_model:
                by_model[model_type] = []
            by_model[model_type].append(entry)
        
        # Aggregate metrics by model
        model_performance = {}
        for model_type, entries in by_model.items():
            mae_values = [e['metrics']['mae'] for e in entries]
            rmse_values = [e['metrics']['rmse'] for e in entries]
            mape_values = [e['metrics']['mape'] for e in entries]
            
            model_performance[model_type] = {
                'num_evaluations': len(entries),
                'mae_mean': float(np.mean(mae_values)),
                'rmse_mean': float(np.mean(rmse_values)),
                'mape_mean': float(np.mean(mape_values)),
                'mae_trend': [float(v) for v in mae_values[-20:]],  # Last 20 for chart
                'latest_metrics': entries[0]['metrics']
            }
        
        # Generate alerts
        alerts = self._generate_alerts(recent_history)
        
        # Time series data for charts
        time_series = []
        for entry in recent_history[:50]:  # Last 50 entries
            time_series.append({
                'timestamp': entry['timestamp'],
                'model_type': entry['model_type'],
                'mae': entry['metrics']['mae'],
                'rmse': entry['metrics']['rmse'],
                'mape': entry['metrics']['mape']
            })
        
        dashboard_data = {
            'ticker': ticker,
            'time_window_days': time_window.days,
            'last_update': recent_history[0]['timestamp'],
            'total_evaluations': len(recent_history),
            'model_performance': model_performance,
            'time_series': time_series,
            'alerts': alerts,
            'summary': {
                'best_model': self._get_best_model(model_performance),
                'overall_mae': float(np.mean([e['metrics']['mae'] for e in recent_history])),
                'overall_mape': float(np.mean([e['metrics']['mape'] for e in recent_history]))
            }
        }
        
        return dashboard_data
    
    def _generate_alerts(self, history: List[Dict]) -> List[Dict]:
        """Generate performance alerts based on thresholds"""
        alerts = []
        
        if len(history) < 2:
            return alerts
        
        # Group by model
        by_model = {}
        for entry in history:
            model_type = entry['model_type']
            if model_type not in by_model:
                by_model[model_type] = []
            by_model[model_type].append(entry)
        
        for model_type, entries in by_model.items():
            if len(entries) < 2:
                continue
            
            # Check for MAE increase
            recent_mae = entries[0]['metrics']['mae']
            older_mae = np.mean([e['metrics']['mae'] for e in entries[1:6]])
            
            if recent_mae > older_mae * (1 + self.alert_thresholds['mae_increase']):
                alerts.append({
                    'type': 'performance_degradation',
                    'severity': 'warning',
                    'model_type': model_type,
                    'message': f"MAE increased by {((recent_mae/older_mae - 1) * 100):.1f}%",
                    'timestamp': entries[0]['timestamp']
                })
            
            # Check MAPE threshold
            recent_mape = entries[0]['metrics']['mape']
            if recent_mape > self.alert_thresholds['mape_threshold']:
                alerts.append({
                    'type': 'high_error',
                    'severity': 'warning',
                    'model_type': model_type,
                    'message': f"MAPE is {recent_mape:.1f}% (threshold: {self.alert_thresholds['mape_threshold']}%)",
                    'timestamp': entries[0]['timestamp']
                })
            
            # Check directional accuracy
            recent_dir_acc = entries[0]['metrics'].get('directional_accuracy')
            if recent_dir_acc and recent_dir_acc < self.alert_thresholds['directional_accuracy_min']:
                alerts.append({
                    'type': 'low_directional_accuracy',
                    'severity': 'info',
                    'model_type': model_type,
                    'message': f"Directional accuracy is {recent_dir_acc:.1f}%",
                    'timestamp': entries[0]['timestamp']
                })
        
        return alerts
    
    def _get_best_model(self, model_performance: Dict) -> str:
        """Determine best performing model"""
        if not model_performance:
            return None
        
        # Rank by MAE
        best_model = min(model_performance.items(), 
                        key=lambda x: x[1]['mae_mean'])
        
        return best_model[0]
    
    def export_metrics_report(self, ticker: str, output_file: str,
                             time_window: timedelta = timedelta(days=90)):
        """
        Export comprehensive metrics report to file
        
        Args:
            ticker: Stock ticker
            output_file: Path to output file (JSON or CSV)
            time_window: Time window for report
        """
        dashboard_data = self.get_dashboard_data(ticker, time_window)
        
        if output_file.endswith('.json'):
            with open(output_file, 'w') as f:
                json.dump(dashboard_data, f, indent=2)
        elif output_file.endswith('.csv'):
            # Convert to DataFrame for CSV export
            df = pd.DataFrame(dashboard_data['time_series'])
            df.to_csv(output_file, index=False)
        
        print(f"[PerformanceMonitor] Exported metrics report to {output_file}")


if __name__ == '__main__':
    print("Continuous Evaluation Module loaded successfully")
    print("Available classes: MetricsLogger, ContinuousEvaluator, PerformanceMonitor")
