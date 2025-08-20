"""
Test script for Alpaca Historical Data Tool
"""

import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the parent directory to the path to import tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.alpaca_historical_data import create_alpaca_historical_data_tool

def test_historical_data_tool():
    """Test the Alpaca historical data tool with various scenarios"""
    
    # Load environment variables
    load_dotenv()
    
    # Check if API credentials are available
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        print("❌ Alpaca API credentials not found in .env file")
        print("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY to test historical data functionality")
        return
    
    print("🧪 Testing Alpaca Historical Data Tool")
    print("=" * 50)
    
    # Create the tool
    tool = create_alpaca_historical_data_tool()
    
    # Calculate test dates
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    
    # Test cases
    test_cases = [
        {
            "name": "Single Stock Daily Bars (7 days)",
            "action": "bars",
            "symbols": "AAPL",
            "timeframe": "1Day",
            "start_date": start_date,
            "end_date": end_date,
            "asset_class": "stocks",
            "feed": "iex"
        },
        {
            "name": "Single Stock Hourly Bars (7 days)",
            "action": "bars",
            "symbols": "TSLA",
            "timeframe": "1Hour",
            "start_date": start_date,
            "end_date": end_date,
            "limit": 50,
            "asset_class": "stocks",
            "feed": "iex"
        },
        {
            "name": "Multiple Stocks Daily Bars",
            "action": "bars",
            "symbols": "AAPL,GOOGL",
            "timeframe": "1Day",
            "start_date": start_date,
            "end_date": end_date,
            "asset_class": "stocks",
            "feed": "iex"
        },
        {
            "name": "Crypto Daily Bars",
            "action": "bars",
            "symbols": "BTC/USD",
            "timeframe": "1Day",
            "start_date": start_date,
            "end_date": end_date,
            "asset_class": "crypto"
        },
        {
            "name": "Stock Trades (1 day, limited)",
            "action": "trades",
            "symbols": "AAPL",
            "start_date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
            "end_date": end_date,
            "limit": 100,
            "asset_class": "stocks"
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            result = tool._run(
                action=test_case["action"],
                symbols=test_case["symbols"],
                timeframe=test_case.get("timeframe", "1Day"),
                start_date=test_case.get("start_date"),
                end_date=test_case.get("end_date"),
                limit=test_case.get("limit", 100),
                asset_class=test_case.get("asset_class", "stocks"),
                feed=test_case.get("feed", "iex")
            )
            
            print(result)
            print("✅ Test passed")
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🏁 Historical Data Tool Testing Complete")

if __name__ == "__main__":
    test_historical_data_tool()