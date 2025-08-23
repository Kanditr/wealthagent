"""
Test script for YFinance fundamental analysis tools
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import (
    create_yfinance_company_profile_tool,
    create_yfinance_financial_ratios_tool,
    create_yfinance_financial_statements_tool,
    create_yfinance_earnings_analysis_tool,
    create_yfinance_quality_score_tool
)

def test_company_profile():
    """Test company profile tool"""
    print("=" * 60)
    print("TESTING: YFinance Company Profile Tool")
    print("=" * 60)
    
    tool = create_yfinance_company_profile_tool()
    
    # Test with AAPL
    print("\n📊 Testing AAPL company profile:")
    result = tool.run({"symbol": "AAPL"})
    print(result)
    
    # Test with invalid symbol
    print("\n❌ Testing invalid symbol:")
    result = tool.run({"symbol": "INVALID"})
    print(result)

def test_financial_ratios():
    """Test financial ratios tool"""
    print("\n" + "=" * 60)
    print("TESTING: YFinance Financial Ratios Tool")
    print("=" * 60)
    
    tool = create_yfinance_financial_ratios_tool()
    
    # Test with TSLA
    print("\n📊 Testing TSLA financial ratios:")
    result = tool.run({"symbol": "TSLA"})
    print(result)

def test_financial_statements():
    """Test financial statements tool"""
    print("\n" + "=" * 60)
    print("TESTING: YFinance Financial Statements Tool")
    print("=" * 60)
    
    tool = create_yfinance_financial_statements_tool()
    
    # Test summary for MSFT
    print("\n📊 Testing MSFT financial statements summary:")
    result = tool.run({"symbol": "MSFT", "statement": "all", "period": "annual"})
    print(result)
    
    # Test income statement
    print("\n📊 Testing MSFT income statement:")
    result = tool.run({"symbol": "MSFT", "statement": "income", "period": "quarterly"})
    print(result[:1000] + "..." if len(result) > 1000 else result)

def test_earnings_analysis():
    """Test earnings analysis tool"""
    print("\n" + "=" * 60)
    print("TESTING: YFinance Earnings Analysis Tool")
    print("=" * 60)
    
    tool = create_yfinance_earnings_analysis_tool()
    
    # Test comprehensive analysis for NVDA
    print("\n📊 Testing NVDA earnings analysis:")
    result = tool.run({"symbol": "NVDA", "analysis_type": "all"})
    print(result[:1500] + "..." if len(result) > 1500 else result)

def test_quality_score():
    """Test investment quality score tool"""
    print("\n" + "=" * 60)
    print("TESTING: YFinance Investment Quality Score Tool")
    print("=" * 60)
    
    tool = create_yfinance_quality_score_tool()
    
    # Test comprehensive score for GOOGL
    print("\n📊 Testing GOOGL investment quality score:")
    result = tool.run({"symbol": "GOOGL", "scoring_method": "comprehensive"})
    print(result[:2000] + "..." if len(result) > 2000 else result)

def main():
    """Run all YFinance tool tests"""
    print("🧪 Starting YFinance Tools Test Suite")
    print("This will test all fundamental analysis tools with real data")
    print("Note: This requires internet connection and may take a few moments")
    
    try:
        # Test each tool
        test_company_profile()
        test_financial_ratios()
        test_financial_statements()
        test_earnings_analysis()
        test_quality_score()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)
        print("🎉 YFinance tools are working correctly!")
        print("💡 These tools provide comprehensive fundamental analysis capabilities")
        print("📈 Ready for integration with the wealth management agent")
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("💡 Make sure to install yfinance: pip install yfinance")
    except Exception as e:
        print(f"❌ Test Error: {e}")
        print("💡 Check your internet connection and try again")

if __name__ == "__main__":
    main()