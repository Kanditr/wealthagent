"""
Alpaca Trading Tool for LangChain Integration - Unified Stock & Crypto Order Execution
"""

import os
from typing import Optional, Type
from datetime import datetime, timezone, timedelta
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest, StopOrderRequest,
        GetOrdersRequest
    )
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.common.exceptions import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca-py not installed. Trading features will be disabled.")


class AlpacaTradingInput(BaseModel):
    """Input schema for Alpaca trading tool"""
    action: str = Field(
        description="Action to perform: 'place_order', 'get_orders', 'cancel_order', 'get_order', 'cancel_all'"
    )
    # Order placement parameters
    symbol: Optional[str] = Field(
        default=None, 
        description="Symbol to trade (e.g., 'AAPL' for stocks, 'BTC/USD' for crypto)"
    )
    order_type: Optional[str] = Field(
        default="market", 
        description="Order type: 'market', 'limit', 'stop'"
    )
    side: Optional[str] = Field(
        default="buy", 
        description="Order side: 'buy' or 'sell'"
    )
    qty: Optional[float] = Field(
        default=None, 
        description="Quantity to trade (shares for stocks, coins for crypto)"
    )
    notional: Optional[float] = Field(
        default=None, 
        description="Dollar amount to trade (alternative to qty)"
    )
    price: Optional[float] = Field(
        default=None, 
        description="Price for limit or stop orders"
    )
    time_in_force: Optional[str] = Field(
        default="gtc", 
        description="Time in force: 'gtc' (good till canceled), 'day', 'ioc' (immediate or cancel)"
    )
    # Order management parameters
    order_id: Optional[str] = Field(
        default=None, 
        description="Order ID for cancel/get specific order operations"
    )


class AlpacaTradingTool(BaseTool):
    """Tool for executing trades on Alpaca (stocks and crypto)"""
    
    name: str = "alpaca_trading"
    description: str = (
        "Execute trades and manage orders on Alpaca for stocks and crypto. "
        "Actions: 'place_order' (buy/sell stocks or crypto), 'get_orders' (order history), "
        "'cancel_order' (cancel specific order), 'get_order' (get order status), "
        "'cancel_all' (cancel all orders). "
        "Supports stocks (e.g., 'AAPL') and crypto (e.g., 'BTC/USD'). "
        "Order types: market, limit, stop. Can use 'qty' for quantity or 'notional' for dollar amount."
    )
    args_schema: Type[BaseModel] = AlpacaTradingInput
    
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
            logger.error(f"Failed to initialize Alpaca trading client: {e}")
            return None
    
    def _check_market_status(self, client):
        """Check current market status using Alpaca clock API"""
        try:
            clock = client.get_clock()
            
            # Convert times to Thailand timezone (GMT+7)
            thailand_tz = timezone(timedelta(hours=7))
            current_time_thailand = clock.timestamp.astimezone(thailand_tz)
            
            if clock.next_open:
                next_open_thailand = clock.next_open.astimezone(thailand_tz)
            else:
                next_open_thailand = None
                
            if clock.next_close:
                next_close_thailand = clock.next_close.astimezone(thailand_tz)
            else:
                next_close_thailand = None
            
            return {
                "is_open": clock.is_open,
                "current_time": current_time_thailand,
                "next_open": next_open_thailand,
                "next_close": next_close_thailand
            }
        except Exception as e:
            logger.warning(f"Could not get market status: {e}")
            return None
    
    def _run(
        self,
        action: str,
        symbol: Optional[str] = None,
        order_type: str = "market",
        side: str = "buy",
        qty: Optional[float] = None,
        notional: Optional[float] = None,
        price: Optional[float] = None,
        time_in_force: str = "gtc",
        order_id: Optional[str] = None
    ) -> str:
        """Execute the trading tool"""
        client = self._get_client()
        if not client:
            return "❌ Alpaca trading not available. Please check your API credentials in .env file."
        
        try:
            if action == "place_order":
                return self._place_order(
                    client, symbol, order_type, side, qty, notional, price, time_in_force
                )
            elif action == "get_orders":
                return self._get_orders(client)
            elif action == "cancel_order":
                return self._cancel_order(client, order_id)
            elif action == "get_order":
                return self._get_order_by_id(client, order_id)
            elif action == "cancel_all":
                return self._cancel_all_orders(client)
            else:
                return f"❌ Unknown action: {action}. Available: place_order, get_orders, cancel_order, get_order, cancel_all"
        
        except APIError as e:
            logger.error(f"Alpaca API error: {e}")
            return f"❌ Alpaca API error: {str(e)}"
        except Exception as e:
            logger.error(f"Trading error: {e}")
            return f"❌ Trading error: {str(e)}"
    
    def _place_order(
        self,
        client,
        symbol: Optional[str],
        order_type: str,
        side: str,
        qty: Optional[float],
        notional: Optional[float],
        price: Optional[float],
        time_in_force: str
    ) -> str:
        """Place a new order"""
        if not symbol:
            return "❌ Symbol is required for order placement"
        
        if not qty and not notional:
            return "❌ Either 'qty' (quantity) or 'notional' (dollar amount) is required"
        
        # Validate and convert parameters
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            
            # Determine if this is crypto or stock
            is_crypto = "/" in symbol
            asset_type = "crypto" if is_crypto else "stock"
            
            # Smart auto-adjustment for fractional stock orders
            auto_adjustment_msg = ""
            original_tif = time_in_force
            
            # Alpaca requires DAY orders for fractional stock trading (notional amounts or fractional qty)
            if not is_crypto and (notional or (qty and qty != int(qty))) and time_in_force.lower() == "gtc":
                time_in_force = "day"
                auto_adjustment_msg = "\n✅ **Auto-adjusted to DAY order** (required for fractional stock trading)"
            
            tif = self._get_time_in_force(time_in_force)
            
            # Place order based on type
            if order_type.lower() == "market":
                order_request = self._create_market_order(symbol, order_side, qty, notional, tif)
            elif order_type.lower() == "limit":
                if not price:
                    return "❌ Price is required for limit orders"
                order_request = self._create_limit_order(symbol, order_side, qty, notional, price, tif)
            elif order_type.lower() == "stop":
                if not price:
                    return "❌ Stop price is required for stop orders"
                order_request = self._create_stop_order(symbol, order_side, qty, notional, price, tif)
            else:
                return f"❌ Unsupported order type: {order_type}. Use 'market', 'limit', or 'stop'"
            
            # Get market status for enhanced user info
            market_status = self._check_market_status(client)
            
            # Submit order
            order = client.submit_order(order_request)
            
            return self._format_order_confirmation(order, asset_type, auto_adjustment_msg, market_status)
        
        except Exception as e:
            error_msg = str(e)
            
            # Enhanced error messages for common issues
            if "fractional orders must be DAY orders" in error_msg:
                return (
                    f"❌ Order placement failed: Fractional orders require DAY time-in-force\n"
                    f"💡 **Auto-fix suggestion**: The system should have handled this automatically. "
                    f"Please try again or contact support if the issue persists."
                )
            elif "insufficient buying power" in error_msg.lower():
                return (
                    f"❌ Insufficient buying power to place this order\n"
                    f"💰 Check your account balance and available buying power before placing orders."
                )
            elif "asset is not tradable" in error_msg.lower():
                return (
                    f"❌ Asset '{symbol}' is not tradable at this time\n"
                    f"🔍 This could be due to market restrictions, halted trading, or unsupported asset."
                )
            elif "market is closed" in error_msg.lower():
                return (
                    f"❌ Market is currently closed for this asset\n"
                    f"🕒 Consider placing a limit order for execution when markets reopen."
                )
            else:
                return f"❌ Order placement failed: {error_msg}\n💡 Please check your order details and try again."
    
    def _create_market_order(self, symbol: str, side: OrderSide, qty: Optional[float], notional: Optional[float], tif: TimeInForce):
        """Create market order request"""
        if qty:
            return MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=tif
            )
        else:
            return MarketOrderRequest(
                symbol=symbol,
                notional=notional,
                side=side,
                time_in_force=tif
            )
    
    def _create_limit_order(self, symbol: str, side: OrderSide, qty: Optional[float], notional: Optional[float], price: float, tif: TimeInForce):
        """Create limit order request"""
        if qty:
            return LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                limit_price=price,
                time_in_force=tif
            )
        else:
            return LimitOrderRequest(
                symbol=symbol,
                notional=notional,
                side=side,
                limit_price=price,
                time_in_force=tif
            )
    
    def _create_stop_order(self, symbol: str, side: OrderSide, qty: Optional[float], notional: Optional[float], price: float, tif: TimeInForce):
        """Create stop order request"""
        if qty:
            return StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                stop_price=price,
                time_in_force=tif
            )
        else:
            return StopOrderRequest(
                symbol=symbol,
                notional=notional,
                side=side,
                stop_price=price,
                time_in_force=tif
            )
    
    def _get_time_in_force(self, tif_str: str) -> TimeInForce:
        """Convert string to TimeInForce enum"""
        tif_map = {
            "gtc": TimeInForce.GTC,
            "day": TimeInForce.DAY,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK
        }
        return tif_map.get(tif_str.lower(), TimeInForce.GTC)
    
    def _format_order_confirmation(self, order, asset_type: str, auto_adjustment_msg: str = "", market_status: Optional[dict] = None) -> str:
        """Format order confirmation message with enhanced info"""
        side_emoji = "🟢" if order.side.value.lower() == "buy" else "🔴"
        asset_emoji = "₿" if asset_type == "crypto" else "📈"
        
        qty_info = f"{float(order.qty):,.4f}" if order.qty else f"${float(order.notional):,.2f} worth"
        price_info = f" at ${float(order.limit_price):,.2f}" if hasattr(order, 'limit_price') and order.limit_price else ""
        stop_info = f" (stop: ${float(order.stop_price):,.2f})" if hasattr(order, 'stop_price') and order.stop_price else ""
        
        # Market status info for Thailand timezone
        market_info = ""
        if market_status:
            if market_status["is_open"]:
                market_info = "\n🟢 **Markets are currently OPEN**"
            else:
                market_info = "\n🔴 **Markets are currently CLOSED**"
                if market_status["next_open"]:
                    market_info += f"\n🕒 Next market open: {market_status['next_open'].strftime('%Y-%m-%d %H:%M')} (Thailand time)"
        
        return (
            f"✅ **Order Placed Successfully** {asset_emoji}{auto_adjustment_msg}\n\n"
            f"{side_emoji} **{order.side.value.title()}** {qty_info} of **{order.symbol}**\n"
            f"📋 Order ID: `{order.id}`\n"
            f"⏰ Type: {order.order_type.value.title()}{price_info}{stop_info}\n"
            f"⚡ Time in Force: {order.time_in_force.value.upper()}\n"
            f"📊 Status: {order.status.value.title()}\n"
            f"🗓️ Submitted: {order.submitted_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
            f"{market_info}\n"
            f"\n💡 *This is a {'paper trading' if 'paper' in str(order.id) else 'live trading'} order*"
        )
    
    def _get_orders(self, client, limit: int = 50) -> str:
        """Get recent orders"""
        try:
            orders = client.get_orders(
                filter=GetOrdersRequest(limit=limit, status="all")
            )
            
            if not orders:
                return "📋 No orders found in your account."
            
            result = f"📋 **Recent Orders** (Last {len(orders)}):\n\n"
            
            for order in orders[:10]:  # Limit display to 10 most recent
                side_emoji = "🟢" if order.side.value.lower() == "buy" else "🔴"
                status_emoji = self._get_status_emoji(order.status.value)
                
                qty_info = f"{float(order.qty):,.4f}" if order.qty else f"${float(order.notional):,.2f}"
                filled_qty = f" (filled: {float(order.filled_qty):,.4f})" if hasattr(order, 'filled_qty') and order.filled_qty and float(order.filled_qty) > 0 else ""
                
                result += (
                    f"{side_emoji} **{order.symbol}** | {qty_info}{filled_qty}\n"
                    f"   {status_emoji} {order.status.value.title()} | {order.order_type.value.title()}\n"
                    f"   🆔 `{str(order.id)[:8]}...` | {order.submitted_at.strftime('%m/%d %H:%M')}\n\n"
                )
            
            if len(orders) > 10:
                result += f"... and {len(orders) - 10} more orders"
            
            return result
        
        except Exception as e:
            return f"❌ Failed to retrieve orders: {str(e)}"
    
    def _get_order_by_id(self, client, order_id: Optional[str]) -> str:
        """Get specific order by ID"""
        if not order_id:
            return "❌ Order ID is required"
        
        try:
            order = client.get_order_by_id(order_id)
            
            side_emoji = "🟢" if order.side.value.lower() == "buy" else "🔴"
            status_emoji = self._get_status_emoji(order.status.value)
            
            qty_info = f"{float(order.qty):,.4f}" if order.qty else f"${float(order.notional):,.2f} worth"
            filled_info = ""
            if hasattr(order, 'filled_qty') and order.filled_qty and float(order.filled_qty) > 0:
                filled_info = f"\n✅ Filled Quantity: {float(order.filled_qty):,.4f}"
                if hasattr(order, 'filled_avg_price') and order.filled_avg_price:
                    filled_info += f" at avg price ${float(order.filled_avg_price):,.2f}"
            
            price_info = ""
            if hasattr(order, 'limit_price') and order.limit_price:
                price_info += f"\n💰 Limit Price: ${float(order.limit_price):,.2f}"
            if hasattr(order, 'stop_price') and order.stop_price:
                price_info += f"\n🛑 Stop Price: ${float(order.stop_price):,.2f}"
            
            return (
                f"📋 **Order Details**\n\n"
                f"{side_emoji} **{order.side.value.title()}** {qty_info} of **{order.symbol}**\n"
                f"{status_emoji} Status: **{order.status.value.title()}**\n"
                f"🆔 Order ID: `{order.id}`\n"
                f"📈 Type: {order.order_type.value.title()}\n"
                f"⚡ Time in Force: {order.time_in_force.value.upper()}\n"
                f"🗓️ Submitted: {order.submitted_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                f"{price_info}{filled_info}"
            )
        
        except Exception as e:
            return f"❌ Failed to get order {order_id}: {str(e)}"
    
    def _cancel_order(self, client, order_id: Optional[str]) -> str:
        """Cancel specific order"""
        if not order_id:
            return "❌ Order ID is required for cancellation"
        
        try:
            client.cancel_order_by_id(order_id)
            return f"✅ Order `{order_id}` has been canceled successfully"
        except Exception as e:
            return f"❌ Failed to cancel order {order_id}: {str(e)}"
    
    def _cancel_all_orders(self, client) -> str:
        """Cancel all open orders"""
        try:
            canceled_orders = client.cancel_orders()
            if canceled_orders:
                return f"✅ Successfully canceled {len(canceled_orders)} open orders"
            else:
                return "📋 No open orders to cancel"
        except Exception as e:
            return f"❌ Failed to cancel all orders: {str(e)}"
    
    def _get_status_emoji(self, status: str) -> str:
        """Get emoji for order status"""
        status_emojis = {
            "new": "🆕",
            "partially_filled": "🔄",
            "filled": "✅",
            "canceled": "❌",
            "expired": "⏰",
            "rejected": "🚫"
        }
        return status_emojis.get(status.lower(), "📋")


def create_alpaca_trading_tool() -> AlpacaTradingTool:
    """Factory function to create Alpaca trading tool"""
    return AlpacaTradingTool()