# System Architecture Documentation

## Assignment 3: Adaptive and Continuous Learning for Forecasting

This document provides a comprehensive overview of the system architecture, design decisions, and implementation details for the adaptive stock forecasting and portfolio management system.

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Component Design](#component-design)
3. [Data Flow](#data-flow)
4. [Module Descriptions](#module-descriptions)
5. [API Design](#api-design)
6. [Database Schema](#database-schema)
7. [Deployment Architecture](#deployment-architecture)
8. [Security Considerations](#security-considerations)
9. [Performance Optimizations](#performance-optimizations)
10. [Future Enhancements](#future-enhancements)

---

## 1. High-Level Architecture

The system follows a **three-tier microservice architecture**:

```
┌──────────────────────────────────────────────────────────────────┐
│                        Presentation Layer                         │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │               React Frontend (Port 3000)                    │  │
│  │  - Material-UI Components                                   │  │
│  │  - Plotly Charts                                            │  │
│  │  - State Management (React Hooks)                           │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              │
                        REST API (HTTP/JSON)
                              │
┌──────────────────────────────────────────────────────────────────┐
│                        Application Layer                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                 Flask API Server (Port 5000)                │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │  │
│  │  │  Forecasting │  │   Adaptive   │  │  Portfolio   │     │  │
│  │  │   Service    │  │   Learning   │  │  Management  │     │  │
│  │  │              │  │   Service    │  │   Service    │     │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘     │  │
│  │  ┌──────────────┐  ┌──────────────┐                        │  │
│  │  │ Continuous   │  │    Model     │                        │  │
│  │  │ Evaluation   │  │  Versioning  │                        │  │
│  │  │   Service    │  │   Service    │                        │  │
│  │  └──────────────┘  └──────────────┘                        │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
                              │
                      Database & Storage Layer
                              │
┌──────────────────────────────────────────────────────────────────┐
│                         Data Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   MongoDB    │  │  File System │  │   Volumes    │          │
│  │  (Port 27017)│  │  (Logs)      │  │  (Models)    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Design

### 2.1 Frontend Components

**Technology Stack:**
- React 18.3.1
- Material-UI 5.x
- Plotly.js for charts
- Axios for HTTP requests
- Vite for building

**Main Components:**
- `EnhancedStockForecasting.jsx`: Main dashboard with tabs
  - Tab 1: Candlestick charts with error overlays
  - Tab 2: Evaluation dashboard with metrics
  - Tab 3: Portfolio management interface
- `StockForecasting.jsx`: Legacy forecasting interface

### 2.2 Backend Services

**Technology Stack:**
- Flask 2.2.0
- PyTorch 1.13.0
- scikit-learn 1.2.0
- pandas 1.5.0
- MongoDB (PyMongo 4.3.0)

**Service Modules:**

#### Forecasting Service
- **File**: `ForecastPredictor.py`
- **Responsibilities:**
  - Generate predictions using ensemble models
  - Coordinate between different forecasting models
  - Create visualizations
- **Models**: LSTM, GRU, ARIMA, Moving Average, VAR

#### Adaptive Learning Service
- **File**: `AdaptiveLearning.py`
- **Responsibilities:**
  - Incremental model updates
  - Model versioning
  - Performance tracking
  - Ensemble management
- **Classes**: `ModelVersion`, `AdaptiveLearningManager`

#### Continuous Evaluation Service
- **File**: `ContinuousEvaluation.py`
- **Responsibilities:**
  - Metric logging
  - Performance monitoring
  - Alert generation
  - Dashboard data aggregation
- **Classes**: `MetricsLogger`, `ContinuousEvaluator`, `PerformanceMonitor`

#### Portfolio Management Service
- **File**: `PortfolioManager.py`
- **Responsibilities:**
  - Trade execution
  - Position management
  - Strategy implementation
  - Performance calculation
- **Classes**: `PortfolioManager`, `TradingStrategy`

### 2.3 Data Collection

**File**: `StockDataCollector.py`
- Fetches historical price data
- Performs feature engineering
- Creates curated datasets
- Integrates with financial APIs

---

## 3. Data Flow

### 3.1 Forecasting Workflow

```
User Request (Frontend)
    │
    ├─> POST /api/forecast
    │
    ▼
Backend receives request
    │
    ├─> Load/Generate curated dataset
    │   (StockDataCollector.py)
    │
    ├─> Train ensemble models
    │   (LSTM, GRU, ARIMA)
    │
    ├─> Generate predictions
    │
    ├─> Calculate metrics (MAE, RMSE, MAPE)
    │
    ├─> Create visualizations
    │   (Candlestick charts)
    │
    ├─> Store in MongoDB
    │
    ▼
Return response to frontend
    │
    ▼
Frontend displays results
```

### 3.2 Adaptive Learning Workflow

```
Trigger Update Request
    │
    ▼
Load recent data
    │
    ▼
Load current model version
    │
    ▼
Perform incremental training
    │
    ├─> Online learning with new data
    ├─> Fine-tuning with lower LR
    │
    ▼
Evaluate updated model
    │
    ▼
Save new model version
    │
    ├─> Model state dict
    ├─> Metadata (metrics, config)
    ├─> Version timestamp
    │
    ▼
Update version history
    │
    ▼
Return metrics to caller
```

### 3.3 Continuous Evaluation Workflow

```
New prediction generated
    │
    ▼
Register prediction
    │
    ├─> Store prediction values
    ├─> Store prediction dates
    ├─> Schedule evaluation
    │
    ▼
Wait for ground truth data
    │
    ▼
Ground truth becomes available
    │
    ▼
Evaluate pending predictions
    │
    ├─> Match predictions with actuals
    ├─> Calculate metrics
    ├─> Log to file system
    │
    ▼
Update dashboard data
    │
    ├─> Aggregate recent metrics
    ├─> Generate alerts
    ├─> Create visualizations
    │
    ▼
Display in monitoring dashboard
```

### 3.4 Portfolio Management Workflow

```
Generate trading signal
    │
    ├─> Get prediction
    ├─> Get current price
    ├─> Apply trading strategy
    │
    ▼
Signal: BUY / SELL / HOLD
    │
    ▼
If BUY or SELL:
    │
    ├─> Calculate position size
    ├─> Check funds/shares available
    ├─> Execute trade
    │   ├─> Update cash
    │   ├─> Update positions
    │   ├─> Record trade history
    │
    ▼
Update portfolio metrics
    │
    ├─> Calculate returns
    ├─> Calculate volatility
    ├─> Calculate Sharpe ratio
    ├─> Calculate win rate
    │
    ▼
Log portfolio state
    │
    ▼
Save to disk
```

---

## 4. Module Descriptions

### 4.1 AdaptiveLearning.py

**Purpose**: Implements adaptive and continuous learning mechanisms

**Classes:**

#### ModelVersion
```python
class ModelVersion:
    def __init__(self, model_dir='./model_versions')
    def save_model_version(model, ticker, model_type, metrics, config) -> str
    def load_model_version(version_id, device='cpu')
    def get_best_version(ticker, model_type, metric='mae') -> str
    def get_version_history(ticker, model_type) -> List[Dict]
```

**Storage Structure:**
```
model_versions/
├── AAPL_LSTM_v20251110_143022/
│   ├── model.pth
│   └── metadata.json
├── AAPL_LSTM_v20251110_150045/
│   ├── model.pth
│   └── metadata.json
└── version_log.json
```

#### AdaptiveLearningManager
```python
class AdaptiveLearningManager:
    def __init__(self, model_version_manager, device='cpu')
    def incremental_update(model, new_data, ticker, model_type, config) -> Tuple[model, metrics]
    def fine_tune_model(model, recent_data, ticker, model_type, config) -> Tuple[model, metrics]
    def rolling_window_retrain(model_class, data, ticker, model_type, config) -> List[Dict]
    def should_update_model(current_metrics, historical_metrics) -> bool
    def adaptive_ensemble(models, recent_data, ticker) -> Tuple[predictions, weights]
```

**Update Strategies:**
1. **Incremental Update**: Train on new data with normal learning rate
2. **Fine-Tuning**: Train on recent data with lower learning rate
3. **Rolling Window**: Retrain periodically with sliding window
4. **Adaptive Ensemble**: Weight models based on recent performance

### 4.2 ContinuousEvaluation.py

**Purpose**: Tracks model performance over time and provides monitoring

**Classes:**

#### MetricsLogger
```python
class MetricsLogger:
    def __init__(self, log_dir='./evaluation_logs')
    def log_metrics(ticker, model_type, horizon, predictions, actuals, metadata)
    def get_metrics_history(ticker, model_type, horizon, limit) -> List[Dict]
    def get_aggregated_metrics(ticker, model_type, time_window) -> Dict
```

**Log Format (JSONL):**
```json
{
  "timestamp": "2025-11-10T14:30:22",
  "ticker": "AAPL",
  "model_type": "LSTM",
  "horizon": "24hrs",
  "metrics": {
    "mae": 2.34,
    "rmse": 3.12,
    "mape": 1.56,
    "directional_accuracy": 65.5
  }
}
```

#### ContinuousEvaluator
```python
class ContinuousEvaluator:
    def __init__(self, metrics_logger)
    def register_prediction(ticker, model_type, horizon, predictions, prediction_dates) -> str
    def evaluate_pending(actual_data) -> List[Dict]
```

#### PerformanceMonitor
```python
class PerformanceMonitor:
    def __init__(self, metrics_logger)
    def get_dashboard_data(ticker, time_window) -> Dict
    def export_metrics_report(ticker, output_file, time_window)
```

### 4.3 PortfolioManager.py

**Purpose**: Manages simulated portfolio and trading

**Classes:**

#### Position
```python
class Position:
    def __init__(self, ticker, quantity, entry_price, entry_date)
    def update_price(current_price)
    def get_value() -> float
    def get_return() -> float
```

#### Trade
```python
class Trade:
    def __init__(self, ticker, action, quantity, price, date, commission)
```

#### TradingStrategy (Abstract Base)
```python
class TradingStrategy:
    def generate_signal(prediction, current_price, historical_data) -> str
```

**Concrete Strategies:**
- `SimpleThresholdStrategy`: Buy if prediction > threshold, sell if < threshold
- `MomentumStrategy`: Combine prediction with recent momentum
- `MeanReversionStrategy`: Buy when below MA and predicted to rise

#### PortfolioManager
```python
class PortfolioManager:
    def __init__(self, initial_capital, commission_rate, portfolio_dir)
    def execute_trade(ticker, action, quantity, price, date) -> bool
    def generate_and_execute_signal(ticker, prediction, current_price, current_date, historical_data) -> str
    def calculate_metrics(current_prices) -> Dict
    def log_portfolio_state(current_prices)
```

**Performance Metrics:**
- **Total Return**: (Current Value - Initial Capital) / Initial Capital * 100
- **Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: (Average Return * 252) / Volatility
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades

---

## 5. API Design

### 5.1 RESTful Principles

- **Resource-based URLs**: `/api/<resource>/<identifier>`
- **HTTP methods**: GET (read), POST (create/update)
- **Status codes**: 200 (success), 404 (not found), 500 (error)
- **JSON payloads**: All requests/responses use JSON

### 5.2 Endpoint Groups

#### Forecasting Endpoints
- `POST /api/forecast`: Generate forecast
- `GET /api/latest/<ticker>`: Get cached prediction
- `GET /api/candlestick/<ticker>`: Get chart with errors

#### Adaptive Learning Endpoints
- `POST /api/adaptive/trigger-update`: Trigger update
- `GET /api/adaptive/versions/<ticker>`: Get version history

#### Evaluation Endpoints
- `GET /api/evaluation/dashboard/<ticker>`: Get dashboard
- `GET /api/evaluation/metrics/<ticker>`: Get metrics history
- `POST /api/evaluation/register-prediction`: Register prediction
- `POST /api/evaluation/evaluate-pending`: Evaluate pending

#### Portfolio Endpoints
- `GET /api/portfolio/status`: Get portfolio status
- `POST /api/portfolio/trade`: Execute trade
- `POST /api/portfolio/signal`: Generate signal
- `GET /api/portfolio/history`: Get trade history
- `GET /api/portfolio/performance`: Get performance

---

## 6. Database Schema

### 6.1 MongoDB Collections

#### historical_data
```json
{
  "_id": ObjectId,
  "ticker": "AAPL",
  "date": "2025-11-10T00:00:00",
  "Open": 150.0,
  "High": 152.0,
  "Low": 149.0,
  "Close": 151.5,
  "Volume": 50000000,
  "MA5": 150.2,
  "MA10": 149.8,
  "Sentiment": 0.75
}
```

#### predictions
```json
{
  "_id": ObjectId,
  "ticker": "AAPL",
  "horizon": "24hrs",
  "predictions": [151.2, 152.0, 151.8, ...],
  "metrics": {
    "lstm": {"mae": 2.34, "rmse": 3.12, "mape": 1.56},
    "ensemble": {"mae": 2.20, "rmse": 2.95, "mape": 1.40}
  },
  "created_at": "2025-11-10T14:30:22"
}
```

### 6.2 File System Storage

#### Model Versions
```
model_versions/
├── <ticker>_<model_type>_v<timestamp>/
│   ├── model.pth (or .pkl)
│   └── metadata.json
└── version_log.json
```

#### Evaluation Logs
```
evaluation_logs/
└── <ticker>_<model_type>_<horizon>_metrics.jsonl
```

#### Portfolio Data
```
portfolio_data/
└── portfolio_state.json
```

---

## 7. Deployment Architecture

### 7.1 Docker Compose Services

```yaml
services:
  mongodb: MongoDB database (Port 27017)
  backend: Flask API server (Port 5000)
  frontend: React application (Port 3000)
```

### 7.2 Volumes

- `mongodb_data`: Database persistence
- `model_versions`: Model versioning
- `evaluation_logs`: Metric logs
- `portfolio_data`: Portfolio state

### 7.3 Networking

- All services connected via `stock_network` bridge
- Backend accessible to frontend via `http://backend:5000`
- Frontend accessible via `http://localhost:3000`

---

## 8. Security Considerations

1. **API Security**: CORS enabled for specific origins only
2. **Input Validation**: All user inputs validated before processing
3. **Error Handling**: Generic error messages to users, detailed logs internally
4. **Data Privacy**: No sensitive user data stored
5. **Rate Limiting**: Can be added via Flask middleware

---

## 9. Performance Optimizations

1. **Caching**: Predictions cached in-memory
2. **Background Jobs**: Scheduled updates don't block API
3. **Batch Processing**: Multiple predictions processed together
4. **Model Checkpointing**: Models saved incrementally
5. **Database Indexing**: Indexes on ticker and date fields
6. **Lazy Loading**: Frontend loads data on-demand

---

## 10. Future Enhancements

1. **Authentication**: User login and API keys
2. **Real-time Updates**: WebSocket for live data
3. **Advanced Strategies**: More sophisticated trading algorithms
4. **Multi-asset Support**: Extend to crypto, forex, commodities
5. **Cloud Deployment**: AWS/Azure/GCP deployment
6. **Automated Backtesting**: Historical strategy evaluation
7. **Risk Management**: Stop-loss, position sizing rules
8. **Alert System**: Email/SMS notifications

---

## Conclusion

This architecture provides a **scalable, maintainable, and production-ready** system for adaptive stock forecasting and portfolio management. The modular design allows for easy extension and customization while maintaining clean separation of concerns.

**Key Strengths:**
- ✅ Modular microservice architecture
- ✅ Comprehensive testing coverage
- ✅ Docker containerization
- ✅ RESTful API design
- ✅ Continuous learning and evaluation
- ✅ Real portfolio management simulation

---

**Document Version**: 1.0  
**Last Updated**: November 10, 2025  
**Author**: CS4063 Student
