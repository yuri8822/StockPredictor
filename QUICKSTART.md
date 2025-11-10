# 🚀 Quick Start Guide

Get your adaptive stock forecasting system running in under 5 minutes!

---

## Option 1: Docker (Easiest - Recommended)

### Step 1: Prerequisites
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
- Ensure Docker is running

### Step 2: Start the Application
```bash
# Navigate to project directory
cd "Assignment-1,2"

# Build and start all services
docker-compose up --build

# Wait for services to start (about 2-3 minutes)
```

### Step 3: Access the Application
- **Frontend**: Open http://localhost:3000 in your browser
- **Backend API**: http://localhost:5000
- **MongoDB**: mongodb://localhost:27017

### Step 4: Use the Application
1. Enter a stock ticker (e.g., `AAPL`, `GOOGL`, `MSFT`)
2. Click "Generate Forecast"
3. Explore the 3 tabs:
   - 📊 **Candlestick Chart**: Price predictions with error overlays
   - 📈 **Evaluation Dashboard**: Model performance monitoring
   - 💼 **Portfolio Management**: Trading and portfolio tracking

### Step 5: Stop the Application
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

---

## Option 2: Local Development

### Step 1: Prerequisites
- Python 3.10+ installed
- Node.js 18+ installed
- MongoDB 6.0+ installed (optional)

### Step 2: Backend Setup
```bash
# Open Terminal 1
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start backend server
python start_server.py
```

### Step 3: Frontend Setup
```bash
# Open Terminal 2
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Step 4: MongoDB (Optional)
```bash
# Open Terminal 3

# Windows:
net start MongoDB

# Linux:
sudo systemctl start mongod

# Mac:
brew services start mongodb-community
```

### Step 5: Access the Application
Open http://localhost:3000 in your browser

---

## First-Time Usage

### 1. Generate Your First Forecast
1. Open the application
2. Enter `AAPL` in the ticker field
3. Click "Generate Forecast"
4. Wait 30-60 seconds for the forecast to generate
5. View the results in all three tabs

### 2. Explore the Evaluation Dashboard
1. Click on the "Evaluation Dashboard" tab
2. View model performance metrics
3. Click "Update Model" to trigger adaptive learning
4. Refresh to see updated metrics

### 3. Try Portfolio Management
1. Click on the "Portfolio Management" tab
2. Generate a forecast first if you haven't
3. Click "Auto-Trade (Simple Strategy)" to execute a trade
4. View your portfolio performance and positions

---

## Testing the System

### Run All Tests
```bash
cd tests

# Test adaptive learning
python test_adaptive_learning.py

# Test portfolio management
python test_portfolio_management.py

# Test forecasting
python test_forecasting.py
```

---

## API Examples

### 1. Generate Forecast
```bash
curl -X POST http://localhost:5000/api/forecast \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "horizon": "24hrs", "days": 90}'
```

### 2. Get Portfolio Status
```bash
curl http://localhost:5000/api/portfolio/status
```

### 3. Trigger Adaptive Update
```bash
curl -X POST http://localhost:5000/api/adaptive/trigger-update \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "model_type": "LSTM", "days": 30}'
```

### 4. Get Evaluation Dashboard
```bash
curl http://localhost:5000/api/evaluation/dashboard/AAPL?days=30
```

---

## Troubleshooting

### Problem: Docker containers won't start
**Solution**: 
```bash
# Stop all containers
docker-compose down

# Remove volumes
docker-compose down -v

# Rebuild
docker-compose up --build
```

### Problem: Port 5000 is already in use
**Solution**:
```bash
# Windows: Find and kill the process
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:5000 | xargs kill -9
```

### Problem: MongoDB connection error
**Solution**:
- Ensure MongoDB is running
- Check connection string: `mongodb://localhost:27017/`
- For Docker: MongoDB should start automatically

### Problem: Frontend can't connect to backend
**Solution**:
- Verify backend is running on port 5000
- Check CORS settings in `ForecastPredictor.py`
- Ensure firewall isn't blocking connections

### Problem: Module import errors in Python
**Solution**:
```bash
# Ensure virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Reinstall dependencies
pip install -r requirements.txt
```

---

## Directory Structure

```
Assignment-1,2/
├── backend/                    # Flask API server
│   ├── AdaptiveLearning.py
│   ├── ContinuousEvaluation.py
│   ├── PortfolioManager.py
│   └── ForecastPredictor.py
├── frontend/                   # React application
│   └── src/
│       ├── App.jsx
│       └── EnhancedStockForecasting.jsx
├── tests/                      # Test suites
├── docker-compose.yml          # Docker orchestration
└── README_ADAPTIVE.md          # Full documentation
```

---

## Next Steps

1. **Read the Documentation**
   - `README_ADAPTIVE.md` - Complete user guide
   - `ARCHITECTURE.md` - System architecture
   - `IMPLEMENTATION_SUMMARY.md` - Implementation details

2. **Explore the Features**
   - Try different stock tickers
   - Generate forecasts with different horizons
   - Execute trades and build a portfolio
   - Monitor model performance over time

3. **Customize the System**
   - Add new trading strategies
   - Implement additional models
   - Customize the frontend UI
   - Add new performance metrics

4. **Run Tests**
   - Verify all functionality works
   - Understand code structure through tests
   - Add your own test cases

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review the comprehensive documentation
3. Check the API documentation in `README_ADAPTIVE.md`
4. Look at test files for usage examples

---

## System Requirements

### Minimum
- **CPU**: 2 cores
- **RAM**: 4 GB
- **Storage**: 5 GB free
- **OS**: Windows 10+, Ubuntu 20.04+, macOS 12+

### Recommended
- **CPU**: 4+ cores
- **RAM**: 8+ GB
- **Storage**: 10+ GB SSD
- **GPU**: CUDA-capable (optional, for faster training)

---

## Quick Commands Reference

```bash
# Docker
docker-compose up -d              # Start in background
docker-compose logs -f backend    # View backend logs
docker-compose restart backend    # Restart backend only
docker-compose down -v            # Stop and clean

# Python
pip install -r requirements.txt   # Install dependencies
python start_server.py            # Start backend
python tests/test_*.py            # Run tests

# Node.js
npm install                       # Install dependencies
npm run dev                       # Start development server
npm run build                     # Build for production
```

---

**Ready to go!** 🎉

Your adaptive stock forecasting system with portfolio management is now ready to use.

**Access the application at**: http://localhost:3000

---

**Last Updated**: November 10, 2025  
**Version**: 3.0.0
