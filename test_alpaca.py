#!/usr/bin/env python3
"""
Test script for Alpaca portfolio integration
"""

import os
from dotenv import load_dotenv
from alpaca_tool import create_alpaca_portfolio_tool

load_dotenv()

def test_alpaca_portfolio_tool():
    print("Testing Alpaca Portfolio Integration")
    print("=" * 50)
    
    # Create the tool
    tool = create_alpaca_portfolio_tool()
    
    print("📊 Tool created successfully")
    print(f"Tool name: {tool.name}")
    print(f"Tool description: {tool.description}")
    print()
    
    # Check API credentials
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    paper_trading = os.getenv("ALPACA_PAPER_TRADING", "true")
    
    print("🔑 API Configuration:")
    print(f"   API Key: {'✅ Set' if api_key else '❌ Missing'}")
    print(f"   Secret Key: {'✅ Set' if secret_key else '❌ Missing'}")
    print(f"   Paper Trading: {paper_trading}")
    print()
    
    if not api_key or not secret_key:
        print("❌ Alpaca API credentials not found!")
        print("Please add ALPACA_API_KEY and ALPACA_SECRET_KEY to your .env file")
        print()
        print("To get API keys:")
        print("1. Sign up at https://alpaca.markets/")
        print("2. Go to 'Paper Trading' section")
        print("3. Generate API key and secret")
        print("4. Add them to your .env file")
        return
    
    print("Testing portfolio tool functions...")
    print("-" * 30)
    
    # Test account info
    print("\n1. Testing account information:")
    try:
        result = tool._run("account")
        print(result)
    except Exception as e:
        print(f"❌ Account test failed: {e}")
    
    # Test positions
    print("\n2. Testing positions:")
    try:
        result = tool._run("positions")
        print(result)
    except Exception as e:
        print(f"❌ Positions test failed: {e}")
    
    # Test portfolio history
    print("\n3. Testing portfolio history:")
    try:
        result = tool._run("history")
        print(result)
    except Exception as e:
        print(f"❌ History test failed: {e}")
    
    # Test specific position (if any exists)
    print("\n4. Testing specific position (AAPL):")
    try:
        result = tool._run("positions", "AAPL")
        print(result)
    except Exception as e:
        print(f"❌ Specific position test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Alpaca integration test completed!")

def test_with_wealth_agent():
    print("\nTesting Alpaca Integration with Wealth Agent")
    print("=" * 50)
    
    try:
        from main import WealthAgentChat
        
        # Create agent
        agent = WealthAgentChat(user_id="alpaca_test_user")
        
        print("🤖 Wealth agent created with Alpaca integration")
        
        # Test portfolio questions
        test_queries = [
            "Can you show me my current portfolio positions?",
            "What's my account summary?",
            "How has my portfolio performed recently?",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Testing: {query}")
            print("-" * 30)
            try:
                response = agent.chat(query, "alpaca_test")
                print(f"Response: {response}")
            except Exception as e:
                print(f"❌ Query failed: {e}")
        
        # Clean up
        agent.close()
        
    except ImportError:
        print("❌ Could not import main.WealthAgentChat")
    except Exception as e:
        print(f"❌ Wealth agent test failed: {e}")

if __name__ == "__main__":
    try:
        test_alpaca_portfolio_tool()
        test_with_wealth_agent()
    except Exception as e:
        print(f"Test script failed: {e}")
        print("\nMake sure you have:")
        print("1. Added Alpaca API credentials to .env file")
        print("2. Installed all dependencies: pip install -r requirements.txt")