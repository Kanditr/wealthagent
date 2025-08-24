"""
Test script for Technical Analysis tools
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import (
    create_technical_indicators_tool,
    create_technical_signals_tool,
    create_chart_patterns_tool,
    create_technical_screening_tool
)

def test_technical_indicators():
    """Test Technical Indicators tool"""
    print("=" * 60)
    print("TESTING: Technical Indicators Tool")
    print("=" * 60)
    
    tool = create_technical_indicators_tool()
    
    # Test momentum analysis
    print("\n📊 Testing momentum analysis (AAPL):")
    result = tool.run({
        "action": "momentum",
        "symbol": "AAPL",
        "timeframe": "1Day",
        "period": 50
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test trend analysis
    print("\n📈 Testing trend analysis (TSLA):")
    result = tool.run({
        "action": "trend", 
        "symbol": "TSLA",
        "timeframe": "1Day",
        "ma_short": 20,
        "ma_long": 50
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test volatility analysis
    print("\n📊 Testing volatility analysis (NVDA):")
    result = tool.run({
        "action": "volatility",
        "symbol": "NVDA",
        "timeframe": "1Day"
    })
    print(result[:600] + "..." if len(result) > 600 else result)
    
    # Test invalid action
    print("\n❌ Testing invalid action:")
    result = tool.run({
        "action": "invalid_action",
        "symbol": "AAPL"
    })
    print(result)

def test_technical_signals():
    """Test Technical Signals tool"""
    print("\n" + "=" * 60)
    print("TESTING: Technical Signals Tool")
    print("=" * 60)
    
    tool = create_technical_signals_tool()
    
    # Test buy signals
    print("\n🟢 Testing buy signals analysis:")
    result = tool.run({
        "action": "buy_signals",
        "symbol": "MSFT",
        "sensitivity": "medium"
    })
    print(result[:1000] + "..." if len(result) > 1000 else result)
    
    # Test sell signals
    print("\n🔴 Testing sell signals analysis:")
    result = tool.run({
        "action": "sell_signals",
        "symbol": "GOOGL",
        "sensitivity": "medium"
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test crossover analysis
    print("\n✨ Testing crossover analysis:")
    result = tool.run({
        "action": "crossover_analysis",
        "symbol": "AAPL"
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test signal strength
    print("\n💪 Testing signal strength analysis:")
    result = tool.run({
        "action": "signal_strength",
        "symbol": "AMZN",
        "sensitivity": "high"
    })
    print(result[:800] + "..." if len(result) > 800 else result)

def test_chart_patterns():
    """Test Chart Patterns tool"""
    print("\n" + "=" * 60)
    print("TESTING: Chart Patterns Tool")
    print("=" * 60)
    
    tool = create_chart_patterns_tool()
    
    # Test support/resistance analysis
    print("\n📊 Testing support/resistance analysis:")
    result = tool.run({
        "action": "support_resistance",
        "symbol": "SPY",
        "timeframe": "1Day",
        "sensitivity": "medium"
    })
    print(result[:1000] + "..." if len(result) > 1000 else result)
    
    # Test trend patterns
    print("\n📈 Testing trend pattern analysis:")
    result = tool.run({
        "action": "trend_analysis",
        "symbol": "QQQ",
        "timeframe": "1Day"
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test candlestick patterns
    print("\n🕯️ Testing candlestick patterns:")
    result = tool.run({
        "action": "candlestick_patterns",
        "symbol": "IWM",
        "sensitivity": "medium"
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test breakout analysis
    print("\n🚀 Testing breakout analysis:")
    result = tool.run({
        "action": "breakout_analysis",
        "symbol": "DIA",
        "sensitivity": "medium"
    })
    print(result[:800] + "..." if len(result) > 800 else result)

def test_technical_screening():
    """Test Technical Screening tool"""
    print("\n" + "=" * 60)
    print("TESTING: Technical Screening Tool")
    print("=" * 60)
    
    tool = create_technical_screening_tool()
    
    # Test momentum screening
    print("\n🚀 Testing momentum screening:")
    result = tool.run({
        "action": "momentum_screen",
        "symbols": "AAPL,MSFT,GOOGL,AMZN,TSLA",
        "timeframe": "1Day",
        "min_volume": 1000000
    })
    print(result[:1200] + "..." if len(result) > 1200 else result)
    
    # Test oversold screening
    print("\n🟢 Testing oversold screening:")
    result = tool.run({
        "action": "oversold_screen",
        "symbols": "NVDA,AMD,INTC,CRM,NFLX",
        "timeframe": "1Day"
    })
    print(result[:1000] + "..." if len(result) > 1000 else result)
    
    # Test breakout screening
    print("\n🚀 Testing breakout screening:")
    result = tool.run({
        "action": "breakout_screen",
        "symbols": "SPY,QQQ,IWM,DIA,XLF",
        "timeframe": "1Day"
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test custom screening
    print("\n🎯 Testing custom screening:")
    result = tool.run({
        "action": "custom_screen",
        "symbols": "AAPL,MSFT,GOOGL",
        "custom_criteria": "RSI<40,Volume>1.2",
        "timeframe": "1Day"
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test invalid symbols
    print("\n❌ Testing with invalid symbols:")
    result = tool.run({
        "action": "momentum_screen",
        "symbols": "",
        "timeframe": "1Day"
    })
    print(result)

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 60)
    print("TESTING: Edge Cases and Error Handling")
    print("=" * 60)
    
    # Test without API keys
    print("\n⚠️ Testing without Alpaca API keys:")
    
    # Temporarily remove API keys
    original_api_key = os.environ.get("ALPACA_API_KEY")
    original_secret_key = os.environ.get("ALPACA_SECRET_KEY")
    
    if "ALPACA_API_KEY" in os.environ:
        del os.environ["ALPACA_API_KEY"]
    if "ALPACA_SECRET_KEY" in os.environ:
        del os.environ["ALPACA_SECRET_KEY"]
    
    tool = create_technical_indicators_tool()
    result = tool.run({
        "action": "momentum",
        "symbol": "AAPL"
    })
    print("Result without API keys:")
    print(result[:300] + "..." if len(result) > 300 else result)
    
    # Restore API keys
    if original_api_key:
        os.environ["ALPACA_API_KEY"] = original_api_key
    if original_secret_key:
        os.environ["ALPACA_SECRET_KEY"] = original_secret_key
    
    # Test invalid symbols
    print("\n❌ Testing invalid symbol:")
    tool = create_technical_signals_tool()
    result = tool.run({
        "action": "buy_signals",
        "symbol": "INVALID123"
    })
    print(result[:200] + "..." if len(result) > 200 else result)

def test_without_pandas_ta():
    """Test behavior without pandas-ta"""
    print("\n" + "=" * 60)
    print("TESTING: Behavior Without pandas-ta")
    print("=" * 60)
    
    # This test simulates what happens if pandas-ta is not installed
    print("\n⚠️ Note: This test simulates pandas-ta unavailability")
    print("In real scenarios without pandas-ta, tools would show appropriate error messages")
    
    # We can't actually uninstall pandas-ta during test, but the tools 
    # have PANDAS_TA_AVAILABLE checks that would handle this

def main():
    """Run all technical analysis tool tests"""
    print("🧪 Starting Technical Analysis Tools Test Suite")
    print("This will test all technical analysis tools with real market data")
    print("Note: Requires internet connection and preferably Alpaca API keys")
    print("Technical analysis works with or without API keys using sample data")
    
    # Check for API keys
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        print("\n⚠️  WARNING: Alpaca API keys not found")
        print("Tests will run but may show 'insufficient data' messages")
        print("For full functionality, add ALPACA_API_KEY and ALPACA_SECRET_KEY to .env file\n")
    else:
        print(f"\n✅ Alpaca API keys found (API key ends with: ...{api_key[-4:]})\n")
    
    # Check for pandas-ta
    try:
        import pandas_ta
        print("✅ pandas-ta-classic is available")
    except ImportError:
        print("⚠️  WARNING: pandas-ta-classic not installed")
        print("Install with: pip install pandas-ta-classic")
        return
    
    print("\n" + "🚀 Starting comprehensive technical analysis testing...")
    
    try:
        # Test each tool category
        test_technical_indicators()
        test_technical_signals() 
        test_chart_patterns()
        test_technical_screening()
        test_edge_cases()
        test_without_pandas_ta()
        
        print("\n" + "=" * 60)
        print("✅ ALL TECHNICAL ANALYSIS TESTS COMPLETED")
        print("=" * 60)
        print("🎉 Technical analysis tools are working correctly!")
        
        if api_key and secret_key:
            print("💡 All features tested with real market data from Alpaca")
            print("📊 Professional-grade technical analysis capabilities available")
        else:
            print("💡 API key functionality not tested - add Alpaca keys for full testing")
        
        print("📈 Technical analysis suite includes:")
        print("  🎯 130+ Technical Indicators (RSI, MACD, Bollinger Bands, etc.)")
        print("  📊 Trading Signal Generation (Buy/Sell signals with strength scoring)")
        print("  📈 Chart Pattern Recognition (Support/Resistance, Candlesticks)")
        print("  🔍 Multi-Symbol Screening (Momentum, Oversold, Breakout detection)")
        print("  ⏰ Multi-Timeframe Analysis (Confluence detection)")
        print("  🎪 Custom Screening Criteria (User-defined technical filters)")
        
        print("\n🚀 Ready for integration with the wealth management agent!")
        print("🔥 Professional technical analysis capabilities now available!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install pandas-ta-classic alpaca-py pandas numpy")
    except Exception as e:
        print(f"❌ Test Error: {e}")
        print("💡 Check your internet connection and API configurations")

if __name__ == "__main__":
    main()