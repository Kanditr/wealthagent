"""
Alpaca Historical Market Data Tool for LangChain Integration
"""

import os
from typing import Optional, Type, List
from datetime import datetime, timezone, timedelta
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import (
        StockBarsRequest, CryptoBarsRequest,
        StockTradesRequest, CryptoTradesRequest
    )
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.common.exceptions import APIError
    ALPACA_HISTORICAL_AVAILABLE = True
except ImportError:
    ALPACA_HISTORICAL_AVAILABLE = False
    logger.warning("Alpaca-py historical data modules not installed. Historical data features will be disabled.")


class AlpacaHistoricalDataInput(BaseModel):
    """Input schema for Alpaca historical data tool"""
    action: str = Field(description="Action to perform: 'bars', 'trades'")
    symbols: str = Field(description="Symbol(s) to query (e.g., 'AAPL' or 'AAPL,TSLA' for multiple)")
    timeframe: Optional[str] = Field(default="1Day", description="Timeframe: '1Min', '5Min', '15Min', '30Min', '1Hour', '1Day'")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD format)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD format)")
    limit: Optional[int] = Field(default=100, description="Number of bars to retrieve (max 10000)")
    asset_class: Optional[str] = Field(default="stocks", description="Asset class: 'stocks' or 'crypto'")
    feed: Optional[str] = Field(default="iex", description="Data feed: 'iex' (free) or 'sip' (premium)")


class AlpacaHistoricalDataTool(BaseTool):
    """Tool for retrieving historical market data from Alpaca"""
    
    name: str = "alpaca_historical_data"
    description: str = (
        "Get historical market data from Alpaca. "
        "Actions: 'bars' (OHLCV historical data), 'trades' (historical trades). "
        "Timeframes: 1Min, 5Min, 15Min, 30Min, 1Hour, 1Day. "
        "Supports stocks (e.g., 'AAPL') and crypto (e.g., 'BTC/USD'). "
        "Date format: YYYY-MM-DD. Uses IEX feed by default (free). Optional 'sip' feed for premium accounts."
    )
    args_schema: Type[BaseModel] = AlpacaHistoricalDataInput
    
    def _get_clients(self):
        """Get Alpaca historical data clients"""
        if not ALPACA_HISTORICAL_AVAILABLE:
            return None, None
        
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            return None, None
        
        try:
            stock_client = StockHistoricalDataClient(api_key, secret_key)
            crypto_client = CryptoHistoricalDataClient(api_key, secret_key)
            return stock_client, crypto_client
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca historical data clients: {e}")
            return None, None
    
    def _parse_symbols(self, symbols: str) -> List[str]:
        """Parse symbols string into list"""
        return [s.strip().upper() for s in symbols.split(',') if s.strip()]
    
    def _get_timeframe(self, timeframe_str: str) -> TimeFrame:
        """Convert string timeframe to Alpaca TimeFrame object"""
        timeframe_map = {
            "1min": TimeFrame(1, TimeFrameUnit.Minute),
            "5min": TimeFrame(5, TimeFrameUnit.Minute),
            "15min": TimeFrame(15, TimeFrameUnit.Minute),
            "30min": TimeFrame(30, TimeFrameUnit.Minute),
            "1hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1day": TimeFrame(1, TimeFrameUnit.Day),
        }
        return timeframe_map.get(timeframe_str.lower(), TimeFrame(1, TimeFrameUnit.Day))
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object"""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    
    def _format_crypto_symbol(self, symbol: str) -> str:
        """Format symbol for crypto (ensure it has /USD if not specified)"""
        if "/" not in symbol and symbol.upper() not in ["USD", "USDT", "USDC"]:
            return f"{symbol.upper()}/USD"
        return symbol.upper()
    
    def _run(
        self,
        action: str,
        symbols: str,
        timeframe: str = "1Day",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
        asset_class: str = "stocks",
        feed: str = "iex"
    ) -> str:
        """Execute the historical data tool"""
        stock_client, crypto_client = self._get_clients()
        if not stock_client or not crypto_client:
            return "❌ Alpaca historical data not available. Please check your API credentials in .env file."
        
        symbol_list = self._parse_symbols(symbols)
        if not symbol_list:
            return "❌ No valid symbols provided"
        
        # Set default date range if not provided (last 30 days)
        end_dt = self._parse_date(end_date) or datetime.now(timezone.utc)
        start_dt = self._parse_date(start_date) or (end_dt - timedelta(days=30))
        
        # If requesting a single day, extend the range to include potential trading days
        # This handles: 1) weekends and holidays, 2) IEX feed limitations with single-day requests
        if start_dt == end_dt:
            # Extend range: 5 days before to 1 day after to ensure we capture the target date
            start_dt = end_dt - timedelta(days=5)
            end_dt = end_dt + timedelta(days=1)
        
        try:
            if asset_class.lower() == "crypto":
                client = crypto_client
                # Convert symbols to crypto format if needed
                symbol_list = [self._format_crypto_symbol(s) for s in symbol_list]
            else:
                client = stock_client
                # Ensure symbols are uppercase for stocks
                symbol_list = [s.upper() for s in symbol_list]
            
            if action == "bars":
                return self._get_historical_bars(
                    client, symbol_list, timeframe, start_dt, end_dt, limit, asset_class, feed
                )
            elif action == "trades":
                return self._get_historical_trades(
                    client, symbol_list, start_dt, end_dt, limit, asset_class
                )
            else:
                return f"❌ Unknown action: {action}. Available actions: bars, trades"
        
        except APIError as e:
            logger.error(f"Alpaca API error: {e}")
            return f"❌ Alpaca API error: {str(e)}"
        except Exception as e:
            logger.error(f"Historical data error: {e}")
            return f"❌ Historical data error: {str(e)}"
    
    def _get_historical_bars(
        self,
        client,
        symbols: List[str],
        timeframe: str,
        start_dt: datetime,
        end_dt: datetime,
        limit: int,
        asset_class: str,
        feed: str
    ) -> str:
        """Get historical bars (OHLCV) data"""
        try:
            tf = self._get_timeframe(timeframe)
            
            if asset_class.lower() == "crypto":
                request = CryptoBarsRequest(
                    symbol_or_symbols=symbols,
                    timeframe=tf,
                    start=start_dt,
                    end=end_dt,
                    limit=limit
                )
                bars_data = client.get_crypto_bars(request)
            else:
                # Default to IEX feed for free access
                request = StockBarsRequest(
                    symbol_or_symbols=symbols,
                    timeframe=tf,
                    start=start_dt,
                    end=end_dt,
                    limit=limit,
                    feed="iex"
                )
                bars_data = client.get_stock_bars(request)
            
            return self._format_bars_response(bars_data, symbols, timeframe, asset_class, start_dt, end_dt)
        
        except Exception as e:
            return f"❌ Failed to get historical bars: {str(e)}"
    
    def _get_historical_trades(
        self,
        client,
        symbols: List[str],
        start_dt: datetime,
        end_dt: datetime,
        limit: int,
        asset_class: str
    ) -> str:
        """Get historical trades data"""
        try:
            # Limit trades queries to single symbol for performance
            if len(symbols) > 1:
                return "❌ Historical trades query supports only single symbol at a time"
            
            symbol = symbols[0]
            
            if asset_class.lower() == "crypto":
                request = CryptoTradesRequest(
                    symbol_or_symbols=symbol,
                    start=start_dt,
                    end=end_dt,
                    limit=min(limit, 1000)  # Limit for performance
                )
                trades_data = client.get_crypto_trades(request)
            else:
                request = StockTradesRequest(
                    symbol_or_symbols=symbol,
                    start=start_dt,
                    end=end_dt,
                    limit=min(limit, 1000),  # Limit for performance
                    feed="iex"  # Default to IEX feed for free access
                )
                trades_data = client.get_stock_trades(request)
            
            return self._format_trades_response(trades_data, symbol, asset_class, start_dt, end_dt)
        
        except Exception as e:
            return f"❌ Failed to get historical trades: {str(e)}"
    
    def _format_bars_response(
        self,
        bars_data,
        symbols: List[str],
        timeframe: str,
        asset_class: str,
        start_dt: datetime,
        end_dt: datetime
    ) -> str:
        """Format historical bars response"""
        asset_emoji = "₿" if asset_class.lower() == "crypto" else "📈"
        
        # Convert dates to Thailand timezone for display
        thailand_tz = timezone(timedelta(hours=7))
        start_thai = start_dt.astimezone(thailand_tz)
        end_thai = end_dt.astimezone(thailand_tz)
        
        result = (
            f"{asset_emoji} **Historical Bars** ({asset_class.title()}) - {timeframe}\n"
            f"📅 Period: {start_thai.strftime('%Y-%m-%d')} to {end_thai.strftime('%Y-%m-%d')} (Thailand time)\n\n"
        )
        
        # Extract data from BarSet/CryptoBarSet
        data_dict = bars_data.data if hasattr(bars_data, 'data') else bars_data
        
        for symbol in symbols:
            if symbol in data_dict:
                bars = data_dict[symbol]
                if bars:
                    result += f"## {symbol}\n"
                    
                    # Calculate summary statistics
                    prices = [float(bar.close) for bar in bars]
                    volumes = [float(bar.volume) for bar in bars if bar.volume]
                    
                    first_bar = bars[0]
                    last_bar = bars[-1]
                    
                    open_price = float(first_bar.open)
                    close_price = float(last_bar.close)
                    
                    # Get Thailand timezone for display
                    thailand_tz = timezone(timedelta(hours=7))
                    latest_date = last_bar.timestamp.astimezone(thailand_tz)
                    high_price = max(float(bar.high) for bar in bars)
                    low_price = min(float(bar.low) for bar in bars)
                    avg_volume = sum(volumes) / len(volumes) if volumes else 0
                    
                    period_change = close_price - open_price
                    period_change_pct = (period_change / open_price * 100) if open_price > 0 else 0
                    change_emoji = "📈" if period_change >= 0 else "📉"
                    
                    result += (
                        f"📊 **Period Summary:**\n"
                        f"🔓 Open: ${open_price:,.4f} | 🔒 Close: ${close_price:,.4f}\n"
                        f"⬆️ High: ${high_price:,.4f} | ⬇️ Low: ${low_price:,.4f}\n"
                        f"{change_emoji} Change: ${period_change:+.4f} ({period_change_pct:+.2f}%)\n"
                        f"📊 Avg Volume: {avg_volume:,.0f} | 📏 Bars: {len(bars)}\n"
                        f"📅 Latest Trading Day: {latest_date.strftime('%Y-%m-%d %A')}\n\n"
                    )
                    
                    # Show recent bars (last 5)
                    recent_bars = bars[-5:] if len(bars) > 5 else bars
                    result += "📋 **Recent Bars:**\n"
                    
                    for bar in recent_bars:
                        bar_time_thai = bar.timestamp.astimezone(thailand_tz)
                        bar_change = float(bar.close) - float(bar.open)
                        bar_change_pct = (bar_change / float(bar.open) * 100) if float(bar.open) > 0 else 0
                        bar_emoji = "🟢" if bar_change >= 0 else "🔴"
                        
                        result += (
                            f"{bar_emoji} {bar_time_thai.strftime('%m/%d %H:%M')}: "
                            f"${float(bar.close):,.4f} ({bar_change_pct:+.2f}%) "
                            f"Vol: {float(bar.volume):,.0f}\n"
                        )
                    
                    result += "\n"
                else:
                    result += f"❌ **{symbol}**: No data available for the specified period\n\n"
            else:
                result += f"❌ **{symbol}**: Symbol not found\n\n"
        
        return result.strip()
    
    def _format_trades_response(
        self,
        trades_data,
        symbol: str,
        asset_class: str,
        start_dt: datetime,
        end_dt: datetime
    ) -> str:
        """Format historical trades response"""
        asset_emoji = "₿" if asset_class.lower() == "crypto" else "📈"
        
        # Convert dates to Thailand timezone for display
        thailand_tz = timezone(timedelta(hours=7))
        start_thai = start_dt.astimezone(thailand_tz)
        end_thai = end_dt.astimezone(thailand_tz)
        
        result = (
            f"{asset_emoji} **Historical Trades** ({asset_class.title()}) - {symbol}\n"
            f"📅 Period: {start_thai.strftime('%Y-%m-%d %H:%M')} to {end_thai.strftime('%Y-%m-%d %H:%M')} (Thailand time)\n\n"
        )
        
        # Extract data from TradeSet
        data_dict = trades_data.data if hasattr(trades_data, 'data') else trades_data
        
        if symbol in data_dict:
            trades = data_dict[symbol]
            if trades:
                # Calculate summary statistics
                prices = [float(trade.price) for trade in trades]
                sizes = [float(trade.size) for trade in trades]
                
                total_volume = sum(sizes)
                avg_price = sum(price * size for price, size in zip(prices, sizes)) / total_volume if total_volume > 0 else 0
                min_price = min(prices)
                max_price = max(prices)
                total_value = sum(price * size for price, size in zip(prices, sizes))
                
                result += (
                    f"📊 **Trade Summary:**\n"
                    f"📈 Total Trades: {len(trades):,}\n"
                    f"📊 Total Volume: {total_volume:,.4f}\n"
                    f"💰 Volume Weighted Avg Price: ${avg_price:,.4f}\n"
                    f"⬆️ High: ${max_price:,.4f} | ⬇️ Low: ${min_price:,.4f}\n"
                    f"💵 Total Value: ${total_value:,.2f}\n\n"
                )
                
                # Show recent trades (last 10)
                recent_trades = trades[-10:] if len(trades) > 10 else trades
                result += "📋 **Recent Trades:**\n"
                
                for trade in recent_trades:
                    trade_time_thai = trade.timestamp.astimezone(thailand_tz)
                    trade_value = float(trade.price) * float(trade.size)
                    
                    result += (
                        f"💹 {trade_time_thai.strftime('%m/%d %H:%M:%S')}: "
                        f"${float(trade.price):,.4f} × {float(trade.size):,.4f} "
                        f"= ${trade_value:,.2f}\n"
                    )
            else:
                result += f"❌ No trades found for {symbol} in the specified period"
        else:
            result += f"❌ Symbol {symbol} not found"
        
        return result


def create_alpaca_historical_data_tool() -> AlpacaHistoricalDataTool:
    """Factory function to create Alpaca historical data tool"""
    return AlpacaHistoricalDataTool()