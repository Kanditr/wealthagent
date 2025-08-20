"""
Test script for Alpaca News Tool
"""

import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the parent directory to the path to import tools
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.alpaca_news import create_alpaca_news_tool

def test_news_tool():
    """Test the Alpaca news tool with various scenarios"""
    
    # Load environment variables
    load_dotenv()
    
    # Check if API credentials are available
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        print("❌ Alpaca API credentials not found in .env file")
        print("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY to test news functionality")
        return
    
    print("🧪 Testing Alpaca News Tool")
    print("=" * 50)
    
    # Create the tool
    tool = create_alpaca_news_tool()
    
    # Calculate test dates
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
    
    # Test cases
    test_cases = [
        {
            "name": "Latest General Market News",
            "action": "latest",
            "symbols": None,
            "limit": 5
        },
        {
            "name": "Latest Apple News",
            "action": "latest",
            "symbols": "AAPL",
            "limit": 5
        },
        {
            "name": "Latest Tech Stock News",
            "action": "latest",
            "symbols": "AAPL,GOOGL,MSFT",
            "limit": 8
        },
        {
            "name": "News Search (Last 3 days)",
            "action": "search",
            "symbols": None,
            "start_date": start_date,
            "end_date": end_date,
            "limit": 5
        },
        {
            "name": "Tesla News Search (Last 3 days)",
            "action": "search",
            "symbols": "TSLA",
            "start_date": start_date,
            "end_date": end_date,
            "limit": 5
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔍 Test {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            result = tool._run(
                action=test_case["action"],
                symbols=test_case.get("symbols"),
                limit=test_case.get("limit", 10),
                start_date=test_case.get("start_date"),
                end_date=test_case.get("end_date")
            )
            
            print(result)
            print("✅ Test passed")
            
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🏁 News Tool Testing Complete")

if __name__ == "__main__":
    test_news_tool()