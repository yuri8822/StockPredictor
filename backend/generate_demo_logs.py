"""
Generate demonstration evaluation logs
This script runs multiple forecasts to populate the evaluation_logs directory
"""

import requests
import time
import sys

API_URL = "http://localhost:5000/api"

# List of tickers to generate forecasts for
TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
HORIZONS = ['24hrs', '72hrs']

def generate_forecasts():
    """Generate forecasts for multiple tickers"""
    print("=" * 60)
    print("Generating Demo Evaluation Logs")
    print("=" * 60)
    print("\nThis will generate forecasts for multiple tickers")
    print("to populate the evaluation_logs/ directory.\n")
    
    # Check if server is running
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server is not responding properly")
            print("Please start the backend server first:")
            print("   python backend/start_server.py")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to the backend server")
        print("Please start the backend server first:")
        print("   python backend/start_server.py")
        sys.exit(1)
    
    print("✓ Server is running\n")
    
    total_requests = len(TICKERS) * len(HORIZONS)
    completed = 0
    
    for ticker in TICKERS:
        for horizon in HORIZONS:
            completed += 1
            print(f"\n[{completed}/{total_requests}] Generating forecast for {ticker} ({horizon})...")
            
            try:
                response = requests.post(
                    f"{API_URL}/forecast",
                    json={
                        'ticker': ticker,
                        'horizon': horizon,
                        'days': 90
                    },
                    timeout=300  # 5 minutes timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"   ✓ Forecast generated successfully")
                    
                    # Show metrics if available
                    if 'metrics' in data:
                        ensemble_metrics = data['metrics'].get('ensemble', {})
                        if ensemble_metrics:
                            print(f"   📊 Ensemble MAE: {ensemble_metrics.get('mae', 'N/A'):.6f}")
                            print(f"   📊 Ensemble RMSE: {ensemble_metrics.get('rmse', 'N/A'):.6f}")
                            print(f"   📊 Ensemble MAPE: {ensemble_metrics.get('mape', 'N/A'):.2f}%")
                else:
                    print(f"   ❌ Error: {response.status_code} - {response.text}")
                
            except requests.exceptions.Timeout:
                print(f"   ⚠️  Request timed out (this can happen for first run)")
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
            
            # Small delay between requests
            if completed < total_requests:
                print("   ⏳ Waiting 2 seconds before next request...")
                time.sleep(2)
    
    print("\n" + "=" * 60)
    print("✓ Demo log generation complete!")
    print("=" * 60)
    print("\nCheck the backend/evaluation_logs/ directory")
    print("for the generated metric log files (.jsonl)")
    print("\nYou can now view these metrics in the Evaluation Dashboard")
    print("tab of the frontend application.")

if __name__ == '__main__':
    try:
        generate_forecasts()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(0)
