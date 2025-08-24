import os
import sqlite3
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langmem import create_manage_memory_tool, create_search_memory_tool
from tools import (
    # Alpaca tools
    create_alpaca_portfolio_tool, 
    create_alpaca_trading_tool,
    create_alpaca_market_data_tool,
    create_alpaca_historical_data_tool,
    create_alpaca_news_tool,
    # YFinance tools
    create_yfinance_company_profile_tool,
    create_yfinance_financial_ratios_tool,
    create_yfinance_financial_statements_tool,
    create_yfinance_earnings_analysis_tool,
    create_yfinance_quality_score_tool,
    # FRED tools
    create_fred_economic_indicators_tool,
    create_fred_economic_dashboard_tool,
    create_fred_sector_analysis_tool
)

load_dotenv()

class WealthAgentChat:
    def __init__(self, user_id: str = "default_user"):
        self.user_id = user_id
        
        # SQLite for conversation persistence + InMemoryStore for speed
        self.sqlite_conn = sqlite3.connect("data/wealth_agent_memories.db", check_same_thread=False)
        self.memory = SqliteSaver(self.sqlite_conn)
        
        # Configure InMemoryStore with proper embedding settings for LangMem
        self.store = InMemoryStore(
            index={
                "dims": 1536,
                "embed": "openai:text-embedding-3-small"
            }
        )
        self.memory_namespace = ("memories", user_id)
        
        # Create agent with persistent memory, fast in-session storage, and comprehensive market tools
        self.agent = create_react_agent(
            ChatOpenAI(model="gpt-4o-mini", temperature=0.1),
            tools=[
                # Memory tools
                create_manage_memory_tool(namespace=self.memory_namespace),
                create_search_memory_tool(namespace=self.memory_namespace),
                # Alpaca tools
                create_alpaca_portfolio_tool(),
                create_alpaca_trading_tool(),
                create_alpaca_market_data_tool(),
                create_alpaca_historical_data_tool(),
                create_alpaca_news_tool(),
                # YFinance fundamental analysis tools
                create_yfinance_company_profile_tool(),
                create_yfinance_financial_ratios_tool(),
                create_yfinance_financial_statements_tool(),
                create_yfinance_earnings_analysis_tool(),
                create_yfinance_quality_score_tool(),
                # FRED economic analysis tools
                create_fred_economic_indicators_tool(),
                create_fred_economic_dashboard_tool(),
                create_fred_sector_analysis_tool(),
            ],
            store=self.store,
            checkpointer=self.memory
        )
    
    def chat(self, message: str, thread_id: str = "default"):
        config = {"configurable": {"thread_id": f"{self.user_id}_{thread_id}"}}
        
        # Create system message for wealth management context with all capabilities
        system_msg = SystemMessage(
            content="You are a helpful wealth management assistant with advanced capabilities. "
                    "IMPORTANT: Actively use your available tools:\n\n"
                    "📝 MEMORY TOOLS:\n"
                    "1. STORE important user information (name, preferences, goals, risk tolerance, etc.)\n"
                    "2. SEARCH your memory before responding to find relevant past conversations\n"
                    "3. When users share personal info, use the manage_memory tool to save it\n"
                    "4. When users ask questions, use search_memory tool to find relevant context\n\n"
                    "📊 PORTFOLIO TOOLS:\n"
                    "1. Use alpaca_portfolio with action='positions' to get current holdings\n"
                    "2. Use alpaca_portfolio with action='account' to get account summary\n"
                    "3. Use alpaca_portfolio with action='history' to get performance data\n"
                    "4. You can specify a 'symbol' parameter for specific position details\n\n"
                    "💰 TRADING TOOLS:\n"
                    "1. Use alpaca_trading with action='place_order' to execute trades (stocks & crypto)\n"
                    "2. Support for stocks (e.g., 'AAPL') and crypto (e.g., 'BTC/USD')\n"
                    "3. Order types: 'market', 'limit', 'stop' orders\n"
                    "4. Use 'qty' for quantity or 'notional' for dollar amounts\n"
                    "5. Use alpaca_trading with action='get_orders' to check order status\n"
                    "6. Use alpaca_trading with action='cancel_order' to cancel orders\n"
                    "7. Always confirm order details before execution\n\n"
                    "📈 MARKET DATA TOOLS:\n"
                    "1. Use alpaca_market_data for real-time quotes, trades, and bars\n"
                    "2. Actions: 'quote' (bid/ask), 'trade' (latest trade), 'bar' (OHLCV)\n"
                    "3. Multi-symbol support: 'multi_quote', 'multi_bar' for multiple symbols\n"
                    "4. Supports both stocks ('AAPL') and crypto ('BTC/USD')\n\n"
                    "📊 HISTORICAL DATA TOOLS:\n"
                    "1. Use alpaca_historical_data for historical market analysis\n"
                    "2. Actions: 'bars' (OHLCV history), 'trades' (trade history)\n"
                    "3. Timeframes: '1Min', '5Min', '15Min', '30Min', '1Hour', '1Day'\n"
                    "4. Date filtering: specify start_date and end_date (YYYY-MM-DD)\n"
                    "5. Feed options: 'iex' (free) or 'sip' (premium)\n\n"
                    "📰 NEWS TOOLS:\n"
                    "1. Use alpaca_news for market news and analysis\n"
                    "2. Actions: 'latest' (recent 24h news), 'search' (date-filtered news)\n"
                    "3. Symbol filtering: get news for specific stocks or general market\n"
                    "4. Includes sentiment analysis where available\n\n"
                    "🔍 YFINANCE FUNDAMENTAL ANALYSIS TOOLS:\n"
                    "1. Use yfinance_company_profile for company overview, sector, industry, executives\n"
                    "2. Use yfinance_financial_ratios for P/E, ROE, debt ratios, margins, valuation metrics\n"
                    "3. Use yfinance_financial_statements for income statement, balance sheet, cash flow (annual/quarterly)\n"
                    "4. Use yfinance_earnings_analysis for earnings history, analyst estimates, recommendations\n"
                    "5. Use yfinance_quality_score for comprehensive investment quality assessment and scoring\n\n"
                    "🏛️ FRED ECONOMIC ANALYSIS TOOLS:\n"
                    "1. Use fred_economic_indicators for key economic indicators (GDP, unemployment, inflation, rates)\n"
                    "   - Actions: 'indicator' (single), 'multi_indicator' (multiple), 'trend' (with analysis)\n"
                    "   - Key series: GDPC1 (GDP), UNRATE (unemployment), CPIAUCSL (CPI), FEDFUNDS (fed rate)\n"
                    "2. Use fred_economic_dashboard for comprehensive economic health overview\n"
                    "   - Analysis types: 'overview', 'health_score', 'cycle_analysis', 'alerts'\n"
                    "3. Use fred_sector_analysis for industry-specific economic data\n"
                    "   - Analysis types: 'industrial', 'employment', 'housing', 'consumer', 'overview'\n\n"
                    "🚨 IMPORTANT TRADING NOTES:\n"
                    "- Always verify order details with the user before placing trades\n"
                    "- Explain the risks and implications of any trade\n"
                    "- This is paper trading for safety unless otherwise configured\n"
                    "- Ask for confirmation on order type, quantity, and price\n\n"
                    "💡 ANALYSIS WORKFLOW SUGGESTIONS:\n"
                    "- For stock analysis: Start with company profile, then financial ratios, then earnings analysis\n"
                    "- Use quality score for comprehensive investment assessment\n"
                    "- Check financial statements for detailed fundamental analysis\n"
                    "- For complete analysis: Economic context (FRED) + Market data (Alpaca) + Fundamentals (YFinance)\n"
                    "- Economic analysis: Use dashboard overview first, then specific indicators or sectors\n"
                    "- Consider economic cycle and sector health when making investment recommendations\n\n"
                    "Always remember: You have these powerful tools - use them proactively!\n"
                    "When discussing investments, check their actual portfolio when relevant.\n"
                    "Use market data and news tools to provide informed analysis.\n"
                    "For trading requests, always confirm details before executing.\n"
                    "Provide clear, accurate financial guidance while noting this is for educational purposes only."
        )
        
        # Configure with higher recursion limit and better error handling
        config_with_limits = {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "recursion_limit": 15  # Reduced from default 25 to prevent infinite loops
            }
        }
        
        response = self.agent.invoke(
            {"messages": [system_msg, HumanMessage(content=message)]}, 
            config_with_limits
        )
        
        return response["messages"][-1].content
    
    def close(self):
        """Close the SQLite connection"""
        if hasattr(self, 'sqlite_conn'):
            self.sqlite_conn.close()

def main():
    # Get user ID for personalized memory
    user_id = input("Enter your user ID (or press Enter for 'demo_user'): ").strip()
    if not user_id:
        user_id = "demo_user"
    
    chat_agent = WealthAgentChat(user_id=user_id)
    
    print(f"Wealth Agent Chat (User: {user_id}) - Type 'quit' to exit")
    print("🧠 The agent will remember your preferences across conversations!")
    print("📊 Portfolio integration available (requires Alpaca API setup)")
    print("🔍 Fundamental analysis powered by YFinance (no API key needed)")
    print("🏛️ Economic analysis powered by FRED (free API key required)")
    print("-" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! Your preferences have been saved.")
            break
        
        if not user_input:
            continue
        
        try:
            response = chat_agent.chat(user_input)
            print(f"\nAgent: {response}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Clean up
    chat_agent.close()

if __name__ == "__main__":
    main()