"""
Production-ready Flask server launcher
Handles Windows socket issues and provides better error management
"""

import os
import sys
import signal
from werkzeug.serving import WSGIRequestHandler

def signal_handler(sig, frame):
    print('\n[INFO] Shutting down gracefully...')
    sys.exit(0)

def run_server():
    """Run Flask server with Windows-compatible settings"""
    
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, signal_handler)
    
    # Import after signal handler setup
    from ForecastPredictor import app
    
    print("=" * 60)
    print("🚀 Starting Stock Forecasting API Server...")
    print("=" * 60)
    
    # Windows-compatible server settings
    server_config = {
        'host': '0.0.0.0',
        'port': 5000,
        'threaded': True,
        'use_reloader': False,  # Disable to prevent Windows socket issues
        'debug': True
    }
    
    print(f"🌐 Server will start on http://localhost:{server_config['port']}")
    print("🔄 Auto-reload: DISABLED (Windows compatibility)")
    print("🧵 Threading: ENABLED")
    print("\n💡 To stop the server, press Ctrl+C")
    print("=" * 60)
    
    try:
        # Try to run with debug mode
        app.run(**server_config)
        
    except OSError as e:
        if "WinError 10038" in str(e) or "socket" in str(e).lower():
            print(f"\n⚠️  Windows socket error detected: {e}")
            print("🔄 Switching to production mode...")
            
            # Fallback to production mode
            server_config['debug'] = False
            app.run(**server_config)
        else:
            raise e
            
    except KeyboardInterrupt:
        print("\n[INFO] Server stopped by user")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("💡 Try restarting or check if port 5000 is already in use")
        sys.exit(1)

if __name__ == '__main__':
    run_server()