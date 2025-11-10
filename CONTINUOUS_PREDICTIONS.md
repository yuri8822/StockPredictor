# Continuous Prediction Updates

## Overview
The Stock Forecasting application now includes automatic, continuous prediction updates while the program is running. This feature ensures that predictions stay fresh and reflect the latest market data.

## How It Works

### Backend (ForecastPredictor.py)
1. **Background Scheduler**: Uses APScheduler to run prediction updates in the background
2. **Update Interval**: Predictions are regenerated every **5 minutes** for active tickers
   - This aligns with typical stock market data update frequencies
   - During market hours, data typically updates every 1-5 minutes
3. **Prediction Cache**: Latest predictions are stored in memory for fast retrieval
4. **Thread Safety**: Uses locks to ensure thread-safe access to cached predictions

### Key Components

#### Background Task
```python
def update_predictions_for_ticker(ticker, horizon, days):
    # Fetches latest data from StockDataCollector.py
    # Trains models (ARIMA, LSTM, GRU, Ensemble)
    # Updates prediction cache
    # Runs every 5 minutes
```

#### API Endpoints
- **POST /api/forecast**: Initial forecast generation + schedules continuous updates
- **GET /api/latest/<ticker>**: Returns cached predictions without recomputing
- **GET /api/active-tickers**: Lists all tickers with active updates

### Frontend (StockForecasting.js)
1. **Auto-Refresh**: Polls for latest predictions every **60 seconds**
2. **Status Indicator**: Shows when predictions were last updated and next update time
3. **Visual Feedback**: Rotating refresh icon indicates active auto-updates

## Usage

### Starting the System
1. Run the backend server:
   ```bash
   python start_server.py
   ```

2. Generate an initial forecast for a ticker (e.g., AAPL)
3. The system will automatically:
   - Schedule background updates every 5 minutes
   - Display the latest predictions with timestamps
   - Continue updating as long as the server runs

### User Experience
- Submit a forecast request for any ticker
- See real-time status: "Auto-updating predictions for AAPL"
- View last update time and next scheduled update
- Frontend automatically refreshes display every minute
- No manual refresh needed!

## Configuration

### Adjust Update Frequency
In `ForecastPredictor.py`, modify the interval:

```python
# Change from 5 minutes to desired interval
scheduler.add_job(
    func=update_predictions_for_ticker,
    trigger=IntervalTrigger(minutes=5),  # Change this value
    ...
)
```

### Adjust Frontend Polling
In `StockForecasting.js`, modify the polling interval:

```javascript
// Change from 60 seconds to desired interval
intervalRef.current = setInterval(() => {
    fetchLatestPredictions(currentTickerRef.current);
}, 60000);  // Change this value (in milliseconds)
```

## Technical Details

### Memory Management
- Predictions are cached in memory (not persistent across restarts)
- Each ticker's cache includes: predictions, metrics, charts, and timestamps
- Cache is thread-safe using Python's `threading.Lock`

### Data Freshness
- Background tasks call `StockDataCollector.py` to fetch latest market data
- If cached data is less than 24 hours old, it's reused (faster)
- Otherwise, fresh data is fetched from Yahoo Finance

### Performance
- Initial forecast: 1-2 minutes (data collection + model training)
- Background updates: 30-60 seconds (reuses recent data)
- Frontend fetch: < 100ms (served from cache)

## Benefits
1. **Always Fresh**: Predictions continuously update with latest market data
2. **No Manual Refresh**: Set and forget - the system handles updates
3. **Efficient**: Uses caching to avoid redundant computations
4. **User-Friendly**: Clear indicators of update status and timing
5. **Scalable**: Can handle multiple tickers simultaneously

## Notes
- Predictions update every 5 minutes regardless of market hours
- Frontend shows the exact update time for transparency
- System continues running until server is stopped
- Each ticker update runs independently in the background
