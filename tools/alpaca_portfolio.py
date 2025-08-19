"""
Alpaca Portfolio Tool for LangChain Integration
"""

import os
from typing import Optional, Type
from datetime import datetime, timedelta
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetPortfolioHistoryRequest
    from alpaca.common.exceptions import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca-py not installed. Portfolio features will be disabled.")


class AlpacaPortfolioInput(BaseModel):
    """Input schema for Alpaca portfolio tool"""
    action: str = Field(description="Action to perform: 'positions', 'account', 'history'")
    symbol: Optional[str] = Field(default=None, description="Stock symbol (optional, for specific position)")


class AlpacaPortfolioTool(BaseTool):
    """Tool for retrieving portfolio information from Alpaca"""
    
    name: str = "alpaca_portfolio"
    description: str = (
        "Get real portfolio information from Alpaca brokerage account. "
        "Actions: 'positions' (current holdings), 'account' (account summary), "
        "'history' (portfolio performance). Optionally specify 'symbol' for specific position."
    )
    args_schema: Type[BaseModel] = AlpacaPortfolioInput
    
    def _get_client(self):
        """Get Alpaca trading client"""
        if not ALPACA_AVAILABLE:
            return None
        
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        paper_trading = os.getenv("ALPACA_PAPER_TRADING", "true").lower() == "true"
        
        if not api_key or not secret_key:
            return None
        
        try:
            return TradingClient(
                api_key=api_key,
                secret_key=secret_key,
                paper=paper_trading
            )
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            return None
    
    def _run(self, action: str, symbol: Optional[str] = None) -> str:
        """Execute the portfolio tool"""
        client = self._get_client()
        if not client:
            return "❌ Alpaca portfolio access not available. Please check your API credentials in .env file."
        
        try:
            if action == "positions":
                return self._get_positions(client, symbol)
            elif action == "account":
                return self._get_account_info(client)
            elif action == "history":
                return self._get_portfolio_history(client)
            else:
                return f"❌ Unknown action: {action}. Available actions: positions, account, history"
        
        except APIError as e:
            logger.error(f"Alpaca API error: {e}")
            return f"❌ Error accessing Alpaca API: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return f"❌ Unexpected error: {str(e)}"
    
    def _get_positions(self, client, symbol: Optional[str] = None) -> str:
        """Get current portfolio positions"""
        try:
            if symbol:
                # Get specific position
                position = client.get_open_position(symbol.upper())
                return self._format_position(position)
            else:
                # Get all positions
                positions = client.get_all_positions()
                if not positions:
                    return "📊 No open positions in your portfolio."
                
                result = "📊 **Current Portfolio Positions:**\n\n"
                total_value = 0
                
                for position in positions:
                    market_value = float(position.market_value)
                    total_value += market_value
                    result += self._format_position(position) + "\n"
                
                result += f"\n💰 **Total Portfolio Value: ${total_value:,.2f}**"
                return result
        
        except APIError as e:
            if "position does not exist" in str(e).lower():
                return f"📊 No position found for {symbol.upper() if symbol else 'requested symbol'}"
            raise
    
    def _format_position(self, position) -> str:
        """Format a single position for display"""
        qty = float(position.qty)
        market_value = float(position.market_value)
        cost_basis = float(position.cost_basis) if position.cost_basis else 0
        unrealized_pl = float(position.unrealized_pl) if position.unrealized_pl else 0
        unrealized_plpc = float(position.unrealized_plpc) if position.unrealized_plpc else 0
        
        pl_emoji = "📈" if unrealized_pl >= 0 else "📉"
        
        return (
            f"🔸 **{position.symbol}**: {qty:,.0f} shares\n"
            f"   Market Value: ${market_value:,.2f}\n"
            f"   Cost Basis: ${cost_basis:,.2f}\n"
            f"   {pl_emoji} P&L: ${unrealized_pl:,.2f} ({unrealized_plpc:.2%})"
        )
    
    def _get_account_info(self, client) -> str:
        """Get account summary information"""
        account = client.get_account()
        
        equity = float(account.equity)
        buying_power = float(account.buying_power)
        cash = float(account.cash)
        portfolio_value = float(account.portfolio_value) if account.portfolio_value else equity
        
        return (
            f"💼 **Account Summary:**\n\n"
            f"💰 Portfolio Value: ${portfolio_value:,.2f}\n"
            f"💵 Cash: ${cash:,.2f}\n"
            f"⚡ Buying Power: ${buying_power:,.2f}\n"
            f"📊 Total Equity: ${equity:,.2f}\n"
            f"🔒 Account Status: {account.status}\n"
            f"📅 Account Type: {'Paper Trading' if account.trading_blocked else 'Live Trading'}"
        )
    
    def _get_portfolio_history(self, client, days: int = 30) -> str:
        """Get portfolio performance history"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        request = GetPortfolioHistoryRequest(
            start=start_date,
            end=end_date,
            timeframe="1D"
        )
        
        history = client.get_portfolio_history(request)
        
        if not history.equity or len(history.equity) < 2:
            return "📈 Insufficient portfolio history data available."
        
        # Calculate performance metrics
        start_value = history.equity[0]
        end_value = history.equity[-1]
        total_return = ((end_value - start_value) / start_value) * 100
        
        return_emoji = "📈" if total_return >= 0 else "📉"
        
        return (
            f"📈 **Portfolio Performance ({days} days):**\n\n"
            f"🏁 Starting Value: ${start_value:,.2f}\n"
            f"🎯 Current Value: ${end_value:,.2f}\n"
            f"{return_emoji} Total Return: {total_return:+.2f}%\n"
            f"📊 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )


def create_alpaca_portfolio_tool() -> AlpacaPortfolioTool:
    """Factory function to create Alpaca portfolio tool"""
    return AlpacaPortfolioTool()