"""
Alpaca Real-Time Market Data Tool for LangChain Integration
"""

import os
from typing import Optional, Type, List
from datetime import timezone, timedelta
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import (
        StockLatestQuoteRequest, StockLatestBarRequest, StockLatestTradeRequest,
        CryptoLatestQuoteRequest, CryptoLatestBarRequest, CryptoLatestTradeRequest
    )
    from alpaca.common.exceptions import APIError
    ALPACA_DATA_AVAILABLE = True
except ImportError:
    ALPACA_DATA_AVAILABLE = False
    logger.warning("Alpaca-py data modules not installed. Market data features will be disabled.")


class AlpacaMarketDataInput(BaseModel):
    """Input schema for Alpaca market data tool"""
    action: str = Field(description="Action to perform: 'quote', 'trade', 'bar', 'multi_quote', 'multi_bar'")
    symbols: str = Field(description="Symbol(s) to query (e.g., 'AAPL' or 'AAPL,TSLA,NVDA' for multiple)")
    asset_class: Optional[str] = Field(default="stocks", description="Asset class: 'stocks' or 'crypto'")


class AlpacaMarketDataTool(BaseTool):
    """Tool for retrieving real-time market data from Alpaca"""
    
    name: str = "alpaca_market_data"
    description: str = (
        "Get real-time market data from Alpaca. "
        "Actions: 'quote' (current bid/ask/last), 'trade' (latest trade), 'bar' (latest OHLCV), "
        "'multi_quote' (quotes for multiple symbols), 'multi_bar' (bars for multiple symbols). "
        "Supports stocks (e.g., 'AAPL') and crypto (e.g., 'BTC/USD'). "
        "For multiple symbols, use comma-separated format: 'AAPL,TSLA,NVDA'"
    )
    args_schema: Type[BaseModel] = AlpacaMarketDataInput
    
    def _get_clients(self):
        """Get Alpaca data clients"""
        if not ALPACA_DATA_AVAILABLE:
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
            logger.error(f"Failed to initialize Alpaca data clients: {e}")
            return None, None
    
    def _parse_symbols(self, symbols: str) -> List[str]:
        """Parse symbols string into list"""
        return [s.strip().upper() for s in symbols.split(',') if s.strip()]
    
    def _run(self, action: str, symbols: str, asset_class: str = "stocks") -> str:
        """Execute the market data tool"""
        stock_client, crypto_client = self._get_clients()
        if not stock_client or not crypto_client:
            return "❌ Alpaca market data not available. Please check your API credentials in .env file."
        
        symbol_list = self._parse_symbols(symbols)
        if not symbol_list:
            return "❌ No valid symbols provided"
        
        try:
            if asset_class.lower() == "crypto":
                client = crypto_client
                # Convert symbols to crypto format if needed
                symbol_list = [self._format_crypto_symbol(s) for s in symbol_list]
            else:
                client = stock_client
            
            if action == "quote":
                return self._get_single_quote(client, symbol_list[0], asset_class)
            elif action == "trade":
                return self._get_single_trade(client, symbol_list[0], asset_class)
            elif action == "bar":
                return self._get_single_bar(client, symbol_list[0], asset_class)
            elif action == "multi_quote":
                return self._get_multi_quotes(client, symbol_list, asset_class)
            elif action == "multi_bar":
                return self._get_multi_bars(client, symbol_list, asset_class)
            else:
                return f"❌ Unknown action: {action}. Available actions: quote, trade, bar, multi_quote, multi_bar"
        
        except APIError as e:
            logger.error(f"Alpaca API error: {e}")
            return f"❌ Alpaca API error: {str(e)}"
        except Exception as e:
            logger.error(f"Market data error: {e}")
            return f"❌ Market data error: {str(e)}"
    
    def _format_crypto_symbol(self, symbol: str) -> str:
        """Format symbol for crypto (ensure it has /USD if not specified)"""
        if "/" not in symbol and symbol.upper() not in ["USD", "USDT", "USDC"]:
            return f"{symbol.upper()}/USD"
        return symbol.upper()
    
    def _get_single_quote(self, client, symbol: str, asset_class: str) -> str:
        """Get single symbol quote"""
        try:
            if asset_class.lower() == "crypto":
                request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
                quote_data = client.get_crypto_latest_quote(request)
                quote = quote_data[symbol]
            else:
                request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
                quote_data = client.get_stock_latest_quote(request)
                quote = quote_data[symbol]
            
            return self._format_quote(symbol, quote, asset_class)
        
        except Exception as e:
            return f"❌ Failed to get quote for {symbol}: {str(e)}"
    
    def _get_single_trade(self, client, symbol: str, asset_class: str) -> str:
        """Get single symbol latest trade"""
        try:
            if asset_class.lower() == "crypto":
                request = CryptoLatestTradeRequest(symbol_or_symbols=symbol)
                trade_data = client.get_crypto_latest_trade(request)
                trade = trade_data[symbol]
            else:
                request = StockLatestTradeRequest(symbol_or_symbols=symbol)
                trade_data = client.get_stock_latest_trade(request)
                trade = trade_data[symbol]
            
            return self._format_trade(symbol, trade, asset_class)
        
        except Exception as e:
            return f"❌ Failed to get trade for {symbol}: {str(e)}"
    
    def _get_single_bar(self, client, symbol: str, asset_class: str) -> str:
        """Get single symbol latest bar"""
        try:
            if asset_class.lower() == "crypto":
                request = CryptoLatestBarRequest(symbol_or_symbols=symbol)
                bar_data = client.get_crypto_latest_bar(request)
                bar = bar_data[symbol]
            else:
                request = StockLatestBarRequest(symbol_or_symbols=symbol)
                bar_data = client.get_stock_latest_bar(request)
                bar = bar_data[symbol]
            
            return self._format_bar(symbol, bar, asset_class)
        
        except Exception as e:
            return f"❌ Failed to get bar for {symbol}: {str(e)}"
    
    def _get_multi_quotes(self, client, symbols: List[str], asset_class: str) -> str:
        """Get quotes for multiple symbols"""
        try:
            if asset_class.lower() == "crypto":
                request = CryptoLatestQuoteRequest(symbol_or_symbols=symbols)
                quotes_data = client.get_crypto_latest_quote(request)
            else:
                request = StockLatestQuoteRequest(symbol_or_symbols=symbols)
                quotes_data = client.get_stock_latest_quote(request)
            
            result = f"📊 **Live Quotes** ({asset_class.title()}):\n\n"
            
            for symbol in symbols:
                if symbol in quotes_data:
                    quote = quotes_data[symbol]
                    result += self._format_quote_compact(symbol, quote) + "\n"
                else:
                    result += f"❌ **{symbol}**: No data available\n"
            
            return result.strip()
        
        except Exception as e:
            return f"❌ Failed to get multi quotes: {str(e)}"
    
    def _get_multi_bars(self, client, symbols: List[str], asset_class: str) -> str:
        """Get bars for multiple symbols"""
        try:
            if asset_class.lower() == "crypto":
                request = CryptoLatestBarRequest(symbol_or_symbols=symbols)
                bars_data = client.get_crypto_latest_bar(request)
            else:
                request = StockLatestBarRequest(symbol_or_symbols=symbols)
                bars_data = client.get_stock_latest_bar(request)
            
            result = f"📈 **Latest Bars** ({asset_class.title()}):\n\n"
            
            for symbol in symbols:
                if symbol in bars_data:
                    bar = bars_data[symbol]
                    result += self._format_bar_compact(symbol, bar) + "\n"
                else:
                    result += f"❌ **{symbol}**: No data available\n"
            
            return result.strip()
        
        except Exception as e:
            return f"❌ Failed to get multi bars: {str(e)}"
    
    def _format_quote(self, symbol: str, quote, asset_class: str) -> str:
        """Format single quote for display"""
        asset_emoji = "₿" if asset_class.lower() == "crypto" else "📈"
        
        bid = float(quote.bid_price) if quote.bid_price else 0
        ask = float(quote.ask_price) if quote.ask_price else 0
        bid_size = float(quote.bid_size) if quote.bid_size else 0
        ask_size = float(quote.ask_size) if quote.ask_size else 0
        
        # Convert timestamp to Thailand timezone
        thailand_tz = timezone(timedelta(hours=7))
        timestamp_thailand = quote.timestamp.astimezone(thailand_tz)
        
        spread = ask - bid if bid > 0 and ask > 0 else 0
        mid_price = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
        spread_pct = (spread/mid_price*100) if mid_price > 0 else 0
        
        return (
            f"{asset_emoji} **{symbol} Live Quote**\n\n"
            f"💰 **Bid**: ${bid:,.4f} (Size: {bid_size:,.0f})\n"
            f"💸 **Ask**: ${ask:,.4f} (Size: {ask_size:,.0f})\n"
            f"📊 **Mid Price**: ${mid_price:,.4f}\n"
            f"📏 **Spread**: ${spread:.4f} ({spread_pct:.2f}%)\n"
            f"🕒 **Time**: {timestamp_thailand.strftime('%Y-%m-%d %H:%M:%S')} (Thailand)"
        )
    
    def _format_trade(self, symbol: str, trade, asset_class: str) -> str:
        """Format single trade for display"""
        asset_emoji = "₿" if asset_class.lower() == "crypto" else "📈"
        
        price = float(trade.price) if trade.price else 0
        size = float(trade.size) if trade.size else 0
        
        # Convert timestamp to Thailand timezone
        thailand_tz = timezone(timedelta(hours=7))
        timestamp_thailand = trade.timestamp.astimezone(thailand_tz)
        
        return (
            f"{asset_emoji} **{symbol} Latest Trade**\n\n"
            f"💰 **Price**: ${price:,.4f}\n"
            f"📊 **Size**: {size:,.4f}\n"
            f"💵 **Value**: ${price * size:,.2f}\n"
            f"🕒 **Time**: {timestamp_thailand.strftime('%Y-%m-%d %H:%M:%S')} (Thailand)"
        )
    
    def _format_bar(self, symbol: str, bar, asset_class: str) -> str:
        """Format single bar for display"""
        asset_emoji = "₿" if asset_class.lower() == "crypto" else "📈"
        
        open_price = float(bar.open) if bar.open else 0
        high_price = float(bar.high) if bar.high else 0
        low_price = float(bar.low) if bar.low else 0
        close_price = float(bar.close) if bar.close else 0
        volume = float(bar.volume) if bar.volume else 0
        
        # Calculate change
        change = close_price - open_price
        change_pct = (change / open_price * 100) if open_price > 0 else 0
        change_emoji = "📈" if change >= 0 else "📉"
        
        # Convert timestamp to Thailand timezone
        thailand_tz = timezone(timedelta(hours=7))
        timestamp_thailand = bar.timestamp.astimezone(thailand_tz)
        
        return (
            f"{asset_emoji} **{symbol} Latest Bar**\n\n"
            f"🔓 **Open**: ${open_price:,.4f}\n"
            f"⬆️ **High**: ${high_price:,.4f}\n"
            f"⬇️ **Low**: ${low_price:,.4f}\n"
            f"🔒 **Close**: ${close_price:,.4f}\n"
            f"{change_emoji} **Change**: ${change:+.4f} ({change_pct:+.2f}%)\n"
            f"📊 **Volume**: {volume:,.0f}\n"
            f"🕒 **Time**: {timestamp_thailand.strftime('%Y-%m-%d %H:%M:%S')} (Thailand)"
        )
    
    def _format_quote_compact(self, symbol: str, quote) -> str:
        """Format quote in compact format for multi-symbol display"""
        bid = float(quote.bid_price) if quote.bid_price else 0
        ask = float(quote.ask_price) if quote.ask_price else 0
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
        spread = ask - bid if bid > 0 and ask > 0 else 0
        
        return f"🔸 **{symbol}**: ${mid:,.4f} | Bid: ${bid:,.4f} | Ask: ${ask:,.4f} | Spread: ${spread:.4f}"
    
    def _format_bar_compact(self, symbol: str, bar) -> str:
        """Format bar in compact format for multi-symbol display"""
        open_price = float(bar.open) if bar.open else 0
        close_price = float(bar.close) if bar.close else 0
        high_price = float(bar.high) if bar.high else 0
        low_price = float(bar.low) if bar.low else 0
        
        change = close_price - open_price
        change_pct = (change / open_price * 100) if open_price > 0 else 0
        change_emoji = "📈" if change >= 0 else "📉"
        
        return f"🔸 **{symbol}**: ${close_price:,.4f} {change_emoji} {change_pct:+.2f}% | H: ${high_price:,.4f} | L: ${low_price:,.4f}"


def create_alpaca_market_data_tool() -> AlpacaMarketDataTool:
    """Factory function to create Alpaca market data tool"""
    return AlpacaMarketDataTool()