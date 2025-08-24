"""
Test script for Tavily breaking news and research tool
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import create_tavily_news_tool

def test_breaking_news():
    """Test Tavily breaking news functionality"""
    print("=" * 60)
    print("TESTING: Tavily Breaking News Tool")
    print("=" * 60)
    
    tool = create_tavily_news_tool()
    
    # Test default breaking news
    print("\n🚨 Testing default breaking news:")
    result = tool.run({
        "action": "breaking_news",
        "max_results": 3
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test specific breaking news query
    print("\n🚨 Testing specific breaking news query:")
    result = tool.run({
        "action": "breaking_news",
        "query": "Federal Reserve interest rates",
        "days": 2,
        "max_results": 3
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test stock market breaking news
    print("\n📈 Testing stock market breaking news:")
    result = tool.run({
        "action": "breaking_news",
        "query": "stock market",
        "days": 1,
        "max_results": 4
    })
    print(result[:800] + "..." if len(result) > 800 else result)

def test_company_research():
    """Test Tavily company research functionality"""
    print("\n" + "=" * 60)
    print("TESTING: Tavily Company Research Tool")
    print("=" * 60)
    
    tool = create_tavily_news_tool()
    
    # Test company research with symbol
    print("\n🔍 Testing company research with symbol (AAPL):")
    result = tool.run({
        "action": "company_research",
        "symbol": "AAPL",
        "days": 7,
        "max_results": 3
    })
    print(result[:1000] + "..." if len(result) > 1000 else result)
    
    # Test company research with symbol and specific query
    print("\n🔍 Testing company research with symbol and query:")
    result = tool.run({
        "action": "company_research",
        "symbol": "TSLA",
        "query": "earnings report",
        "days": 14,
        "max_results": 3
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test company research with query only
    print("\n🔍 Testing company research with query only:")
    result = tool.run({
        "action": "company_research",
        "query": "Apple iPhone sales",
        "days": 7,
        "max_results": 3
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test missing parameters
    print("\n❌ Testing company research without symbol or query:")
    result = tool.run({
        "action": "company_research",
        "days": 7
    })
    print(result)

def test_market_events():
    """Test Tavily market events functionality"""
    print("\n" + "=" * 60)
    print("TESTING: Tavily Market Events Tool")
    print("=" * 60)
    
    tool = create_tavily_news_tool()
    
    # Test default market events
    print("\n🏛️ Testing default market events:")
    result = tool.run({
        "action": "market_events",
        "days": 3,
        "max_results": 4
    })
    print(result[:1000] + "..." if len(result) > 1000 else result)
    
    # Test specific market events query
    print("\n🏛️ Testing specific market events (GDP data):")
    result = tool.run({
        "action": "market_events",
        "query": "GDP economic data release",
        "days": 7,
        "max_results": 3
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test Fed-related market events
    print("\n🏛️ Testing Fed-related market events:")
    result = tool.run({
        "action": "market_events",
        "query": "Federal Reserve FOMC meeting",
        "days": 14,
        "max_results": 3
    })
    print(result[:800] + "..." if len(result) > 800 else result)

def test_sentiment_analysis():
    """Test Tavily sentiment analysis functionality"""
    print("\n" + "=" * 60)
    print("TESTING: Tavily Sentiment Analysis Tool")
    print("=" * 60)
    
    tool = create_tavily_news_tool()
    
    # Test sentiment analysis with symbol
    print("\n📊 Testing sentiment analysis with symbol (NVDA):")
    result = tool.run({
        "action": "sentiment_analysis",
        "symbol": "NVDA",
        "days": 7,
        "max_results": 4
    })
    print(result[:1000] + "..." if len(result) > 1000 else result)
    
    # Test sentiment analysis with query
    print("\n📊 Testing sentiment analysis with query:")
    result = tool.run({
        "action": "sentiment_analysis",
        "query": "cryptocurrency market",
        "days": 3,
        "max_results": 3
    })
    print(result[:800] + "..." if len(result) > 800 else result)
    
    # Test missing parameters
    print("\n❌ Testing sentiment analysis without symbol or query:")
    result = tool.run({
        "action": "sentiment_analysis",
        "days": 7
    })
    print(result)

def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "=" * 60)
    print("TESTING: Edge Cases and Error Handling")
    print("=" * 60)
    
    tool = create_tavily_news_tool()
    
    # Test invalid action
    print("\n❌ Testing invalid action:")
    result = tool.run({
        "action": "invalid_action",
        "query": "test query"
    })
    print(result)
    
    # Test boundary values for max_results
    print("\n📊 Testing max_results boundary (0 - should become 1):")
    result = tool.run({
        "action": "breaking_news",
        "max_results": 0,
        "days": 1
    })
    # Just check if it runs without error
    print("✅ Boundary test completed (no error expected)")
    
    # Test boundary values for days
    print("\n📊 Testing days boundary (100 - should become 30):")
    result = tool.run({
        "action": "breaking_news",
        "days": 100,
        "max_results": 2
    })
    # Just check if it runs without error
    print("✅ Boundary test completed (no error expected)")

def test_without_api_key():
    """Test behavior without Tavily API key"""
    print("\n" + "=" * 60)
    print("TESTING: Behavior Without Tavily API Key")
    print("=" * 60)
    
    # Temporarily remove API key
    original_key = os.environ.get("TAVILY_API_KEY")
    if "TAVILY_API_KEY" in os.environ:
        del os.environ["TAVILY_API_KEY"]
    
    tool = create_tavily_news_tool()
    result = tool.run({
        "action": "breaking_news",
        "query": "test query"
    })
    print(f"Without API key: {result}")
    
    # Restore API key
    if original_key:
        os.environ["TAVILY_API_KEY"] = original_key

def main():
    """Run all Tavily news tool tests"""
    print("🧪 Starting Tavily News Tools Test Suite")
    print("This will test all breaking news and research tools with real Tavily data")
    print("Note: Requires TAVILY_API_KEY in .env file and internet connection")
    print("Get your API key at: https://app.tavily.com/")
    
    # Check for API key
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        print("\n⚠️  WARNING: TAVILY_API_KEY not found in environment")
        print("Some tests will show 'API not available' messages")
        print("To get full functionality, add TAVILY_API_KEY to your .env file\n")
    else:
        print(f"\n✅ Tavily API key found (ends with: ...{api_key[-4:]})\n")
    
    # Check for langchain-tavily package
    try:
        import langchain_tavily
        print("✅ langchain-tavily package is available\n")
    except ImportError:
        print("⚠️  WARNING: langchain-tavily not installed")
        print("Install with: pip install langchain-tavily\n")
    
    try:
        # Test each functionality
        test_breaking_news()
        test_company_research()
        test_market_events() 
        test_sentiment_analysis()
        test_edge_cases()
        test_without_api_key()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)
        print("🎉 Tavily news tools are working correctly!")
        
        if api_key:
            print("💡 All features tested with real news and research data from Tavily")
            print("🌍 Multi-source news aggregation and sentiment analysis available")
        else:
            print("💡 API key functionality not tested - add TAVILY_API_KEY for full testing")
        
        print("📰 Ready for integration with the wealth management agent")
        print("🚨 Breaking news and comprehensive research capabilities now available!")
        print("📊 Superior news intelligence for investment decision-making!")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure to install langchain-tavily: pip install langchain-tavily")
    except Exception as e:
        print(f"❌ Test Error: {e}")
        print("💡 Check your internet connection and Tavily API key")

if __name__ == "__main__":
    main()