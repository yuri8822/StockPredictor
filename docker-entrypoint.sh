#!/bin/bash

echo "Starting Stock Forecasting Application..."

# Start backend in background
echo "Starting Flask API backend on port 5000..."
cd /app/backend
python ForecastPredictor.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 5

# Start frontend
echo "Starting React frontend on port 3000..."
cd /app/frontend
serve -s build -l 3000 &
FRONTEND_PID=$!

echo "============================================"
echo "Stock Forecasting Application is ready!"
echo "- Backend API: http://localhost:5000"
echo "- Frontend UI: http://localhost:3000"
echo "============================================"

# Wait for any process to exit
wait $BACKEND_PID $FRONTEND_PID