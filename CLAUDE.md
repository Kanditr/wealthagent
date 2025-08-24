# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a wealth management agent chat application built with LangChain, LangGraph, LangMem, Alpaca API, and OpenAI ChatGPT. The agent provides personalized financial guidance using advanced memory capabilities, real portfolio data integration, order execution functionality, and comprehensive technical analysis. It remembers user preferences, analyzes actual investment holdings, can execute trades for both stocks and cryptocurrencies, and provides professional-grade technical analysis with 130+ indicators and 62 candlestick patterns.

## Development Setup

### Quick Setup
```bash
# Automated setup with virtual environment
./setup.sh

# Manual setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### Running the Application
```bash
# Start the chat interface (prompts for user ID)
python main.py

# Test memory functionality
python tests/test_memory.py

# Test trading functionality (stocks & crypto)
python tests/test_alpaca_trading.py

# Test technical analysis tools
python tests/test_technical_analysis.py
```

### Environment Configuration
Required in `.env` file:
- `OPENAI_API_KEY`: Your OpenAI API key
- `ALPACA_API_KEY` (optional): Your Alpaca API key for portfolio access and trading
- `ALPACA_SECRET_KEY` (optional): Your Alpaca secret key
- `ALPACA_PAPER_TRADING=true` (optional): Enable paper trading mode (recommended for safety)
- `LANGSMITH_TRACING=true` (optional): Enable tracing
- `LANGSMITH_API_KEY`: Your LangSmith API key (optional)
- `LANGSMITH_PROJECT`: Project name for LangSmith (optional)
- `FRED_API_KEY` (optional): Your FRED API key for economic analysis
- `TAVILY_API_KEY` (optional): Your Tavily API key for breaking news

## Architecture Considerations

When developing this wealth agent application, consider:

### Core Components
- **Chat Interface**: User interaction layer for financial queries and trading commands
- **LLM Integration**: LangChain-based conversation management with tool support
- **Memory System**: LangMem SDK for persistent user preferences and conversation history
- **Portfolio Integration**: Alpaca API for real-time portfolio data and positions
- **Trading Execution**: Unified order execution for stocks and cryptocurrencies
- **Order Management**: Real-time order status, history, and cancellation capabilities
- **Risk Assessment**: Portfolio analysis and risk evaluation modules
- **Recommendation Engine**: Investment advice based on actual holdings
- **Technical Analysis**: 130+ technical indicators, 62 candlestick patterns, and signal generation
- **Market Intelligence**: Breaking news integration, economic analysis, and multi-source research
- **User Authentication**: Secure user session management with isolated memory
- **Data Storage**: SQLite persistence for conversations, InMemoryStore for performance

### Security Requirements
- Never log or store sensitive financial information
- Implement proper API key management
- Use environment variables for configuration
- Validate all user inputs for financial data
- Implement rate limiting for API calls

### LangChain Integration Patterns
- **LangMem SDK**: Long-term memory for user preferences and conversation history
- **User namespacing**: Each user gets isolated memory space for privacy
- **Memory tools**: `create_manage_memory_tool` and `create_search_memory_tool` for automatic memory management
- **Portfolio tools**: `AlpacaPortfolioTool` for real-time portfolio data access
- **Trading tools**: `AlpacaTradingTool` for unified stock and crypto order execution
- **React agent**: Uses `create_react_agent` for comprehensive tool integration
- **Hybrid storage**: SQLite checkpointing + InMemoryStore for performance
- **Tool-based architecture**: Clean separation of concerns with specialized tools
- **Order execution**: Support for market, limit, and stop orders with validation
- **Multi-asset support**: Seamless trading for traditional stocks and cryptocurrencies
- **Technical Analysis Tools**: 
  - `TechnicalIndicatorsTool` for 130+ indicators (RSI, MACD, Bollinger Bands, etc.)
  - `TechnicalSignalsTool` for buy/sell signal generation with strength scoring
  - `ChartPatternsTool` for candlestick patterns and chart analysis
  - `TechnicalScreeningTool` for multi-symbol screening and filtering
- **Market Data Tools**:
  - `YFinanceFundamentalTools` for company analysis and financial ratios
  - `FREDEconomicTools` for economic indicators and sector analysis
  - `TavilyNewsTool` for breaking news and market research
- **Pandas-ta-classic**: Professional technical analysis with 62 candlestick patterns
- Implement message trimming for token management
- Create custom tools for financial calculations
- Use prompt templates for consistent financial advice formatting
- Implement streaming for real-time responses

## File Structure

Current organized structure:
```
/
├── main.py                   # Main application entry point
├── requirements.txt          # Dependencies
├── setup.sh                  # Automated setup script
├── .env.example             # Environment variables template
├── README.md                # Project documentation
├── CLAUDE.md                # Claude Code instructions
├── MEMORY.md                # Memory system documentation
├── tools/                   # LangChain tools package
│   ├── __init__.py
│   ├── alpaca_portfolio.py  # Portfolio data retrieval
│   ├── alpaca_trading.py    # Order execution
│   ├── alpaca_market_data.py # Real-time market data
│   ├── alpaca_historical_data.py # Historical bars and trades
│   ├── alpaca_news.py       # Market news and sentiment
│   ├── yfinance_company_profile.py # Company fundamentals
│   ├── yfinance_financial_ratios.py # Financial ratios
│   ├── yfinance_financial_statements.py # Financial statements
│   ├── yfinance_earnings_analysis.py # Earnings analysis
│   ├── yfinance_quality_score.py # Investment quality scoring
│   ├── fred_economic_indicators.py # Economic indicators
│   ├── fred_economic_dashboard.py # Economic health overview
│   ├── fred_sector_analysis.py # Sector-specific economic data
│   ├── tavily_news.py       # Breaking news and research
│   ├── technical_indicators.py # 130+ technical indicators
│   ├── technical_signals.py # Buy/sell signal generation
│   ├── chart_patterns.py    # Candlestick and chart patterns
│   └── technical_screening.py # Multi-symbol technical screening
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_memory.py
│   ├── test_persistence.py
│   ├── test_alpaca_portfolio.py
│   ├── test_alpaca_trading.py
│   ├── test_yfinance_tools.py
│   ├── test_fred_tools.py
│   ├── test_tavily_news.py
│   └── test_technical_analysis.py
├── data/                    # Generated data files
│   └── wealth_agent_memories.db
└── venv/                    # Virtual environment
```

## Financial Domain Considerations

- Implement disclaimer mechanisms for investment advice
- Ensure compliance with financial regulations
- Use proper financial calculation libraries
- Implement data validation for financial inputs
- Consider real-time vs. delayed market data requirements

## Technical Analysis Capabilities

### Professional Technical Analysis Suite
- **130+ Technical Indicators** via pandas-ta-classic
- **62 Candlestick Patterns** with TA-Lib integration
- **Multi-timeframe Analysis** across various timeframes
- **Signal Generation** with strength scoring and confidence levels
- **Pattern Recognition** for chart patterns and price action
- **Multi-symbol Screening** for opportunity identification

### Technical Tools Overview
1. **TechnicalIndicatorsTool**: Comprehensive indicator analysis
   - Actions: `momentum`, `trend`, `volatility`, `volume`, `oscillators`, `all_indicators`
   - Covers RSI, MACD, Bollinger Bands, ATR, Stochastic, Williams %R, and more

2. **TechnicalSignalsTool**: Trading signal generation
   - Actions: `buy_signals`, `sell_signals`, `crossover_analysis`, `divergence_detection`, `signal_strength`
   - Multi-indicator consensus scoring with confidence levels

3. **ChartPatternsTool**: Pattern recognition and analysis  
   - Actions: `support_resistance`, `trend_analysis`, `candlestick_patterns`, `breakout_analysis`, `pattern_scan`
   - Includes all major candlestick patterns and chart formations

4. **TechnicalScreeningTool**: Multi-symbol screening
   - Actions: `momentum_screen`, `oversold_screen`, `breakout_screen`, `custom_screen`, `multi_timeframe_screen`
   - Rank and filter stocks by technical criteria

### Dependencies
- `pandas-ta-classic>=1.0.0`: Main technical analysis library
- `TA-Lib`: Provides 62 candlestick patterns (auto-detected by pandas-ta-classic)
- `pandas`, `numpy`: Data manipulation and analysis
- Integration with Alpaca API for real-time and historical market data