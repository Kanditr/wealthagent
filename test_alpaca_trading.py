#!/usr/bin/env python3
"""
Test script for Alpaca trading integration (stocks and crypto)
"""

import os
from dotenv import load_dotenv
from alpaca_trading_tool import create_alpaca_trading_tool

load_dotenv()

def test_alpaca_trading_tool():
    print("Testing Alpaca Trading Tool Integration")
    print("=" * 50)
    
    # Create the trading tool
    tool = create_alpaca_trading_tool()
    
    print("🔧 Trading tool created successfully")
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
        return
    
    print("Testing trading tool functions...")
    print("-" * 40)
    
    # Test 1: Get current orders (should work without any orders)
    print("\n1. Testing order retrieval:")
    try:
        result = tool._run("get_orders")
        print(result)
    except Exception as e:
        print(f"❌ Order retrieval test failed: {e}")
    
    # Test 2: Test invalid action
    print("\n2. Testing invalid action:")
    try:
        result = tool._run("invalid_action")
        print(result)
    except Exception as e:
        print(f"❌ Invalid action test failed: {e}")
    
    # Test 3: Test order placement validation (without actually placing orders)
    print("\n3. Testing order validation (no symbol):")
    try:
        result = tool._run("place_order")
        print(result)
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
    
    # Test 4: Test order placement validation (no quantity or notional)
    print("\n4. Testing order validation (no qty/notional):")
    try:
        result = tool._run("place_order", symbol="AAPL")
        print(result)
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Basic trading tool validation completed!")

def test_stock_order_simulation():
    """Test stock order placement in simulation mode"""
    print("\nTesting Stock Order Simulation")
    print("=" * 50)
    
    tool = create_alpaca_trading_tool()
    
    # Test market order for stocks
    print("\n📈 Testing stock market order simulation:")
    print("Command: Buy $100 worth of Apple stock (AAPL)")
    try:
        # Note: This will place a real paper trading order if credentials are valid
        result = tool._run(
            action="place_order",
            symbol="AAPL", 
            order_type="market",
            side="buy",
            notional=100,  # $100 worth
            time_in_force="gtc"
        )
        print(result)
    except Exception as e:
        print(f"❌ Stock market order test failed: {e}")
    
    # Test limit order for stocks  
    print("\n📊 Testing stock limit order simulation:")
    print("Command: Buy 10 shares of Tesla (TSLA) at $200 limit price")
    try:
        result = tool._run(
            action="place_order",
            symbol="TSLA",
            order_type="limit", 
            side="buy",
            qty=10,
            price=200.00,
            time_in_force="gtc"
        )
        print(result)
    except Exception as e:
        print(f"❌ Stock limit order test failed: {e}")

def test_crypto_order_simulation():
    """Test crypto order placement in simulation mode"""
    print("\nTesting Crypto Order Simulation") 
    print("=" * 50)
    
    tool = create_alpaca_trading_tool()
    
    # Test market order for Bitcoin
    print("\n₿ Testing Bitcoin market order simulation:")
    print("Command: Buy $50 worth of Bitcoin (BTC/USD)")
    try:
        result = tool._run(
            action="place_order",
            symbol="BTC/USD",
            order_type="market", 
            side="buy",
            notional=50,  # $50 worth of Bitcoin
            time_in_force="gtc"
        )
        print(result)
    except Exception as e:
        print(f"❌ Bitcoin market order test failed: {e}")
    
    # Test limit order for Bitcoin
    print("\n💰 Testing Bitcoin limit order simulation:")
    print("Command: Buy 0.001 BTC at $60000 limit price")
    try:
        result = tool._run(
            action="place_order",
            symbol="BTC/USD",
            order_type="limit",
            side="buy", 
            qty=0.001,  # 0.001 Bitcoin
            price=60000,
            time_in_force="gtc"
        )
        print(result)
    except Exception as e:
        print(f"❌ Bitcoin limit order test failed: {e}")

def test_with_wealth_agent():
    """Test trading integration with the full wealth agent"""
    print("\nTesting Trading Integration with Wealth Agent")
    print("=" * 50)
    
    try:
        from main import WealthAgentChat
        
        # Create agent
        agent = WealthAgentChat(user_id="trading_test_user")
        
        print("🤖 Wealth agent created with trading integration")
        
        # Test trading questions
        test_queries = [
            "Can you show me my current orders?",
            "I want to buy $10 worth of Bitcoin. Can you help me place a market order?",
            "How do I place a limit order for Apple stock?",
            "Can you show me my portfolio positions first, then help me decide on a trade?",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Testing: {query}")
            print("-" * 30)
            try:
                response = agent.chat(query, "trading_test")
                print(f"Response: {response}")
            except Exception as e:
                print(f"❌ Query failed: {e}")
        
        # Clean up
        agent.close()
        
    except ImportError:
        print("❌ Could not import main.WealthAgentChat")
    except Exception as e:
        print(f"❌ Wealth agent test failed: {e}")

def print_trading_examples():
    """Print examples of how to use the trading functionality"""
    print("\nTrading Tool Usage Examples")
    print("=" * 50)
    
    examples = [
        {
            "description": "Place a market order to buy $100 worth of Apple stock",
            "action": "place_order",
            "params": {
                "symbol": "AAPL",
                "order_type": "market", 
                "side": "buy",
                "notional": 100,
                "time_in_force": "gtc"
            }
        },
        {
            "description": "Place a limit order to buy 10 shares of Tesla at $200",
            "action": "place_order", 
            "params": {
                "symbol": "TSLA",
                "order_type": "limit",
                "side": "buy", 
                "qty": 10,
                "price": 200.00,
                "time_in_force": "gtc"
            }
        },
        {
            "description": "Buy $50 worth of Bitcoin at market price",
            "action": "place_order",
            "params": {
                "symbol": "BTC/USD", 
                "order_type": "market",
                "side": "buy",
                "notional": 50,
                "time_in_force": "gtc"
            }
        },
        {
            "description": "Place a limit order for 0.001 Bitcoin at $60,000",
            "action": "place_order",
            "params": {
                "symbol": "BTC/USD",
                "order_type": "limit", 
                "side": "buy",
                "qty": 0.001,
                "price": 60000,
                "time_in_force": "gtc"
            }
        },
        {
            "description": "Get all recent orders",
            "action": "get_orders", 
            "params": {}
        },
        {
            "description": "Cancel all open orders",
            "action": "cancel_all",
            "params": {}
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['description']}:")
        print(f"   Action: {example['action']}")
        for param, value in example['params'].items():
            print(f"   {param}: {value}")

if __name__ == "__main__":
    try:
        print("🚀 Starting Alpaca Trading Integration Tests\n")
        
        # Basic tool validation
        test_alpaca_trading_tool()
        
        # Stock trading tests (only if credentials are available)
        api_key = os.getenv("ALPACA_API_KEY")
        if api_key:
            print("\n" + "="*60)
            test_stock_order_simulation()
            
            print("\n" + "="*60) 
            test_crypto_order_simulation()
            
            print("\n" + "="*60)
            test_with_wealth_agent()
        else:
            print("\n⚠️  Skipping order placement tests - no API credentials found")
        
        print("\n" + "="*60)
        print_trading_examples()
        
        print(f"\n{'='*60}")
        print("✅ All trading integration tests completed!")
        
    except Exception as e:
        print(f"❌ Test script failed: {e}")
        print("\nMake sure you have:")
        print("1. Added Alpaca API credentials to .env file (for order tests)")
        print("2. Installed all dependencies: pip install -r requirements.txt") 
        print("3. Set ALPACA_PAPER_TRADING=true for safety")