# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a wealth management agent chat application built with LangChain, LangGraph, LangMem, Alpaca API, and OpenAI ChatGPT. The agent provides personalized financial guidance using advanced memory capabilities, real portfolio data integration, and order execution functionality. It remembers user preferences, analyzes actual investment holdings, and can execute trades for both stocks and cryptocurrencies.

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
│   └── alpaca_trading.py    # Order execution
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_memory.py
│   ├── test_persistence.py
│   ├── test_alpaca_portfolio.py
│   └── test_alpaca_trading.py
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