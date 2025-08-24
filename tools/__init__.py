"""
Tools package for Wealth Agent

This package contains all LangChain tools used by the wealth management agent.
"""

# Alpaca tools (trading, portfolio, market data)
from .alpaca_portfolio import create_alpaca_portfolio_tool
from .alpaca_trading import create_alpaca_trading_tool
from .alpaca_market_data import create_alpaca_market_data_tool
from .alpaca_historical_data import create_alpaca_historical_data_tool
from .alpaca_news import create_alpaca_news_tool

# YFinance tools (fundamental analysis)
from .yfinance_company_profile import create_yfinance_company_profile_tool
from .yfinance_financial_ratios import create_yfinance_financial_ratios_tool
from .yfinance_financial_statements import create_yfinance_financial_statements_tool
from .yfinance_earnings_analysis import create_yfinance_earnings_analysis_tool
from .yfinance_quality_score import create_yfinance_quality_score_tool

# FRED tools (economic analysis)
from .fred_economic_indicators import create_fred_economic_indicators_tool
from .fred_economic_dashboard import create_fred_economic_dashboard_tool
from .fred_sector_analysis import create_fred_sector_analysis_tool

# Tavily tools (breaking news and research)
from .tavily_news import create_tavily_news_tool

__all__ = [
    # Alpaca tools
    "create_alpaca_portfolio_tool",
    "create_alpaca_trading_tool",
    "create_alpaca_market_data_tool",
    "create_alpaca_historical_data_tool",
    "create_alpaca_news_tool",
    
    # YFinance tools
    "create_yfinance_company_profile_tool",
    "create_yfinance_financial_ratios_tool",
    "create_yfinance_financial_statements_tool",
    "create_yfinance_earnings_analysis_tool",
    "create_yfinance_quality_score_tool",
    
    # FRED tools
    "create_fred_economic_indicators_tool",
    "create_fred_economic_dashboard_tool",
    "create_fred_sector_analysis_tool",
    
    # Tavily tools
    "create_tavily_news_tool",
]