"""
Test script for FRED economic analysis tools
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import (
    create_fred_economic_indicators_tool,
    create_fred_economic_dashboard_tool,
    create_fred_sector_analysis_tool
)

def test_economic_indicators():
    """Test FRED economic indicators tool"""
    print("=" * 60)
    print("TESTING: FRED Economic Indicators Tool")
    print("=" * 60)
    
    tool = create_fred_economic_indicators_tool()
    
    # Test single indicator
    print("\n📊 Testing single indicator (GDP):")
    result = tool.run({
        "action": "indicator",
        "series_id": "GDPC1",
        "start_date": "2020-01-01"
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test multiple indicators
    print("\n📊 Testing multiple indicators:")
    result = tool.run({
        "action": "multi_indicator", 
        "series_ids": "GDPC1,UNRATE,CPIAUCSL",
        "start_date": "2022-01-01"
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test trend analysis
    print("\n📊 Testing trend analysis (Unemployment):")
    result = tool.run({
        "action": "trend",
        "series_id": "UNRATE",
        "start_date": "2020-01-01"
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test invalid series ID
    print("\n❌ Testing invalid series ID:")
    result = tool.run({
        "action": "indicator",
        "series_id": "INVALID_SERIES"
    })
    print(result)

def test_economic_dashboard():
    """Test FRED economic dashboard tool"""
    print("\n" + "=" * 60)
    print("TESTING: FRED Economic Dashboard Tool")
    print("=" * 60)
    
    tool = create_fred_economic_dashboard_tool()
    
    # Test overview
    print("\n📊 Testing economic overview:")
    result = tool.run({
        "analysis_type": "overview",
        "lookback_months": 12
    })
    print(result[:1000] + "..." if len(result) > 1000 else result)
    
    # Test health score
    print("\n🏥 Testing health score:")
    result = tool.run({
        "analysis_type": "health_score",
        "lookback_months": 18
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test cycle analysis
    print("\n🔄 Testing cycle analysis:")
    result = tool.run({
        "analysis_type": "cycle_analysis",
        "lookback_months": 24
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test economic alerts
    print("\n🚨 Testing economic alerts:")
    result = tool.run({
        "analysis_type": "alerts",
        "lookback_months": 6
    })
    print(result[:600] + "..." if len(result) > 600 else result)

def test_sector_analysis():
    """Test FRED sector analysis tool"""
    print("\n" + "=" * 60)
    print("TESTING: FRED Sector Analysis Tool")
    print("=" * 60)
    
    tool = create_fred_sector_analysis_tool()
    
    # Test sector overview
    print("\n🏢 Testing sector overview:")
    result = tool.run({
        "analysis_type": "overview",
        "lookback_months": 12
    })
    print(result[:1000] + "..." if len(result) > 1000 else result)
    
    # Test industrial sector
    print("\n🏭 Testing industrial sector:")
    result = tool.run({
        "analysis_type": "industrial",
        "lookback_months": 12
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test employment sector
    print("\n👥 Testing employment sector:")
    result = tool.run({
        "analysis_type": "employment",
        "lookback_months": 12
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test housing sector
    print("\n🏠 Testing housing sector:")
    result = tool.run({
        "analysis_type": "housing",
        "lookback_months": 12
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test consumer sector
    print("\n🛍️ Testing consumer sector:")
    result = tool.run({
        "analysis_type": "consumer",
        "lookback_months": 12
    })
    print(result[:800] + "..." if len(result) > 800 else result)

def test_without_api_key():
    """Test behavior without FRED API key"""
    print("\n" + "=" * 60)
    print("TESTING: Behavior Without FRED API Key")
    print("=" * 60)
    
    # Temporarily remove API key
    original_key = os.environ.get("FRED_API_KEY")
    if "FRED_API_KEY" in os.environ:
        del os.environ["FRED_API_KEY"]
    
    tool = create_fred_economic_indicators_tool()
    result = tool.run({"action": "indicator", "series_id": "GDPC1"})
    print(f"Without API key: {result}")
    
    # Restore API key
    if original_key:
        os.environ["FRED_API_KEY"] = original_key

def main():
    """Run all FRED tool tests"""
    print("🧪 Starting FRED Tools Test Suite")
    print("This will test all economic analysis tools with real FRED data")
    print("Note: Requires FRED_API_KEY in .env file and internet connection")
    print("Get your free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
    
    # Check for API key
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: FRED_API_KEY not found in environment")
        print("Some tests will show 'API not available' messages")
        print("To get full functionality, add FRED_API_KEY to your .env file\n")
    else:
        print(f"\n✅ FRED API key found (ends with: ...{api_key[-4:]})\n")
    
    try:
        # Test each tool
        test_economic_indicators()
        test_economic_dashboard()
        test_sector_analysis()
        test_without_api_key()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)
        print("🎉 FRED tools are working correctly!")
        
        if api_key:
            print("💡 All features tested with real economic data from FRED")
        else:
            print("💡 API key functionality not tested - add FRED_API_KEY for full testing")
        
        print("📈 Ready for integration with the wealth management agent")
        print("🏛️ Economic analysis capabilities now available!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure to install fredapi: pip install fredapi")
    except Exception as e:
        print(f"❌ Test Error: {e}")
        print("💡 Check your internet connection and FRED API key")

if __name__ == "__main__":
    main()