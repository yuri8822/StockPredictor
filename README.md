# Stock Forecasting Application

A modern stock forecasting application with a React frontend and Flask API backend, using machine learning models (ARIMA + LSTM ensemble) to predict stock prices.

## Architecture

- **Frontend**: React + Material-UI (runs on port 3000)
- **Backend**: Flask API (runs on port 5000)  
- **Data Collection**: StockDataCollector.py (automatically called by backend)
- **Database**: MongoDB (optional, for persistence)

## Prerequisites

1. **Node.js & npm** (for React frontend)
2. **Python 3.8+** (for Flask API and ML models)
3. **MongoDB** (optional, for data persistence)

## Installation & Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install React Dependencies

```bash
npm install
```

## Running the Application

### Method 1: Start Both Services (Recommended)

1. **Start the Flask API Server:**
   ```bash
   python ForecastPredictor.py
   ```
   The API will run on http://localhost:5000

2. **Start the React Frontend** (in another terminal):
   ```bash
   npm start
   ```
   The app will open at http://localhost:3000

### Method 2: API Only (for testing)

```bash
python ForecastPredictor.py
```

Test the API endpoints:
- GET `/api/health` - Health check
- POST `/api/forecast` - Generate forecast

## Features

### 🔄 Data Integration
- Automatically runs `StockDataCollector.py` to fetch fresh data
- Integrates stock prices, technical indicators, and sentiment analysis
- Caches data to avoid unnecessary API calls

### 🤖 Machine Learning Models
- **ARIMA**: Traditional time series forecasting
- **LSTM**: Deep learning neural network  
- **GRU**: Gated Recurrent Unit neural network
- **Ensemble**: Weighted combination of all models

### 📊 Forecasting Options
- **Tickers**: Any valid stock symbol (AAPL, GOOGL, MSFT, etc.)
- **Horizons**: 1, 3, 24, or 72 days
- **Historical Data**: 30-365 days of training data

### 📈 Visualizations
- Interactive candlestick charts with Plotly
- Technical indicators (Moving averages)
- Prediction overlays for all models
- Volume analysis

### 📋 Performance Metrics
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error  
- **MAPE**: Mean Absolute Percentage Error

## Usage

1. Open http://localhost:3000 in your browser
2. Enter a stock ticker symbol (e.g., "AAPL")
3. Select forecast horizon and historical data period
4. Click "Generate Forecast"
5. View results including:
   - Dataset information
   - Model performance metrics
   - Interactive price charts with predictions

## API Endpoints

### Health Check
```http
GET /api/health
```

### Generate Forecast
```http
POST /api/forecast
Content-Type: application/json

{
  "ticker": "AAPL",
  "horizon": "24hrs", 
  "days": 90
}
```

## File Structure

```
├── package.json              # React dependencies
├── requirements.txt          # Python dependencies
├── ForecastPredictor.py      # Flask API server
├── StockDataCollector.py     # Data collection script
├── public/                   # React static files
├── src/
│   ├── App.js               # Main React app
│   ├── StockForecasting.js  # Main forecasting component
│   ├── App.css              # Styles
│   └── index.css            # Global styles
└── README.md                # This file
```

## Troubleshooting

### Common Issues

1. **CORS Errors**: Make sure both servers are running and the API has CORS enabled
2. **Missing Dependencies**: Run `pip install -r requirements.txt` and `npm install`
3. **Port Conflicts**: Ensure ports 3000 and 5000 are available
4. **Data Collection Fails**: Check that `StockDataCollector.py` is in the same directory

### Performance Notes

- First forecast for a ticker takes 1-2 minutes (data collection + model training)
- Subsequent forecasts use cached data and are much faster
- MongoDB is optional but recommended for data persistence
- GPU acceleration (CUDA) will speed up LSTM training if available

## Development

### Adding New Features

1. **Backend**: Modify `ForecastPredictor.py` and add new API endpoints
2. **Frontend**: Add new React components in the `src/` directory
3. **Data**: Extend `StockDataCollector.py` for additional data sources

### Environment Variables

- `MONGO_URI`: MongoDB connection string (default: mongodb://localhost:27017/)

## License

MIT License - feel free to use this for educational or commercial purposes.