"""
Test script for Alpaca Market Data Tool
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Add the parent directory to the path to import tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.alpaca_market_data import create_alpaca_market_data_tool

def test_market_data_tool():
    """Test the Alpaca market data tool with various scenarios"""
    
    # Load environment variables
    load_dotenv()
    
    # Check if API credentials are available
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        print("❌ Alpaca API credentials not found in .env file")
        print("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY to test market data functionality")
        return
    
    print("🧪 Testing Alpaca Market Data Tool")
    print("=" * 50)
    
    # Create the tool
    tool = create_alpaca_market_data_tool()
    
    # Test cases
    test_cases = [
        {
            "name": "Single Stock Quote",
            "action": "quote",
            "symbols": "AAPL",
            "asset_class": "stocks"
        },
        {
            "name": "Single Stock Trade",
            "action": "trade",
            "symbols": "TSLA",
            "asset_class": "stocks"
        },
        {
            "name": "Single Stock Bar",
            "action": "bar",
            "symbols": "NVDA",
            "asset_class": "stocks"
        },
        {
            "name": "Multiple Stock Quotes",
            "action": "multi_quote",
            "symbols": "AAPL,TSLA,NVDA",
            "asset_class": "stocks"
        },
        {
            "name": "Multiple Stock Bars",
            "action": "multi_bar",
            "symbols": "AAPL,GOOGL,MSFT",
            "asset_class": "stocks"
        },
        {
            "name": "Crypto Quote",
            "action": "quote",
            "symbols": "BTC/USD",
            "asset_class": "crypto"
        },
        {
            "name": "Multiple Crypto Quotes",
            "action": "multi_quote",
            "symbols": "BTC/USD,ETH/USD",
            "asset_class": "crypto"
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            result = tool._run(
                action=test_case["action"],
                symbols=test_case["symbols"],
                asset_class=test_case["asset_class"]
            )
            
            print(result)
            print("✅ Test passed")
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🏁 Market Data Tool Testing Complete")

if __name__ == "__main__":
    test_market_data_tool()