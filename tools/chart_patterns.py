"""
Chart Pattern Analysis Tool for LangChain Integration
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Type, Dict, Any, List, Tuple
from datetime import datetime, timedelta
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import pandas_ta_classic as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logger.warning("pandas-ta-classic not installed. Chart patterns features will be disabled.")

try:
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca-py not available for data retrieval.")


class ChartPatternsInput(BaseModel):
    """Input schema for chart patterns tool"""
    action: str = Field(description="Action: 'support_resistance', 'trend_analysis', 'candlestick_patterns', 'breakout_analysis', 'pattern_scan'")
    symbol: str = Field(description="Stock symbol (e.g., 'AAPL', 'TSLA')")
    timeframe: Optional[str] = Field(default="1Day", description="Timeframe: '1Min', '5Min', '15Min', '30Min', '1Hour', '1Day'")
    period: Optional[int] = Field(default=60, description="Number of periods to analyze (30-200)")
    sensitivity: Optional[str] = Field(default="medium", description="Pattern sensitivity: 'low', 'medium', 'high'")


class ChartPatternsTool(BaseTool):
    """Tool for chart pattern analysis and recognition"""
    
    name: str = "chart_patterns"
    description: str = (
        "Analyze chart patterns and price action. "
        "Actions: 'support_resistance' (key levels), 'trend_analysis' (trend direction/strength), "
        "'candlestick_patterns' (reversal/continuation patterns), 'breakout_analysis' (breakout detection), "
        "'pattern_scan' (comprehensive pattern recognition)."
    )
    args_schema: Type[BaseModel] = ChartPatternsInput
    
    def _get_alpaca_data(self, symbol: str, timeframe: str, period: int) -> Optional[pd.DataFrame]:
        """Get historical data from Alpaca for pattern analysis"""
        if not ALPACA_AVAILABLE:
            return None
            
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            return None
        
        try:
            # Determine if it's crypto or stock
            is_crypto = "/" in symbol or symbol.endswith("USD") or symbol.endswith("BTC")
            
            if is_crypto:
                client = CryptoHistoricalDataClient(api_key, secret_key)
                request_class = CryptoBarsRequest
            else:
                client = StockHistoricalDataClient(api_key, secret_key)
                request_class = StockBarsRequest
            
            # Parse timeframe
            timeframe_mapping = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, TimeFrameUnit.Minute),
                "15Min": TimeFrame(15, TimeFrameUnit.Minute),
                "30Min": TimeFrame(30, TimeFrameUnit.Minute),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day
            }
            
            tf = timeframe_mapping.get(timeframe, TimeFrame.Day)
            
            # Calculate start date
            end_date = datetime.now().date()
            if timeframe in ["1Min", "5Min"]:
                start_date = end_date - timedelta(days=max(7, period // 100))
            elif timeframe in ["15Min", "30Min"]:
                start_date = end_date - timedelta(days=max(14, period // 20))
            elif timeframe == "1Hour":
                start_date = end_date - timedelta(days=max(21, period // 6))
            else:  # 1Day
                start_date = end_date - timedelta(days=max(60, period + 20))
            
            request = request_class(
                symbol_or_symbols=[symbol],
                timeframe=tf,
                start=start_date,
                end=end_date,
                feed="iex"
            )
            
            bars = client.get_stock_bars(request) if not is_crypto else client.get_crypto_bars(request)
            
            # Convert to DataFrame
            df = bars.df.reset_index()
            if 'symbol' in df.columns:
                df = df[df['symbol'] == symbol].copy()
            
            # Rename columns to standard format
            df.columns = [col.lower() for col in df.columns]
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            if all(col in df.columns for col in required_cols):
                df = df[required_cols].copy()
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                return df.tail(period + 10)  # Add small buffer
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting Alpaca data: {e}")
            return None
    
    def _run(
        self,
        action: str,
        symbol: str,
        timeframe: str = "1Day",
        period: int = 60,
        sensitivity: str = "medium"
    ) -> str:
        """Execute chart pattern analysis"""
        if not PANDAS_TA_AVAILABLE:
            return "❌ pandas-ta-classic not available. Please install pandas-ta-classic to use chart patterns analysis."
        
        # Validate inputs
        period = max(30, min(period, 200))
        symbol = symbol.upper().strip()
        sensitivity = sensitivity.lower()
        
        if sensitivity not in ["low", "medium", "high"]:
            sensitivity = "medium"
        
        try:
            # Get historical data
            df = self._get_alpaca_data(symbol, timeframe, period)
            
            if df is None or len(df) < 20:
                return f"❌ Insufficient historical data for {symbol}. Need at least 20 periods for pattern analysis."
            
            # Perform analysis based on action
            if action == "support_resistance":
                return self._analyze_support_resistance(df, symbol, timeframe, sensitivity)
            elif action == "trend_analysis":
                return self._analyze_trend_patterns(df, symbol, timeframe)
            elif action == "candlestick_patterns":
                return self._analyze_candlestick_patterns(df, symbol, timeframe, sensitivity)
            elif action == "breakout_analysis":
                return self._analyze_breakouts(df, symbol, timeframe, sensitivity)
            elif action == "pattern_scan":
                return self._comprehensive_pattern_scan(df, symbol, timeframe, sensitivity)
            else:
                return f"❌ Unknown action: {action}. Available actions: support_resistance, trend_analysis, candlestick_patterns, breakout_analysis, pattern_scan"
        
        except Exception as e:
            logger.error(f"Chart patterns error: {e}")
            return f"❌ Chart patterns error: {str(e)}"
    
    def _find_support_resistance(self, df: pd.DataFrame, window: int = 5) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels using pivot points"""
        highs = []
        lows = []
        
        # Find local highs and lows
        for i in range(window, len(df) - window):
            # Check for local high
            if df['High'].iloc[i] == df['High'].iloc[i-window:i+window+1].max():
                highs.append(df['High'].iloc[i])
            
            # Check for local low  
            if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window+1].min():
                lows.append(df['Low'].iloc[i])
        
        # Group nearby levels (within 1% of each other)
        def group_levels(levels, tolerance=0.01):
            if not levels:
                return []
            
            levels.sort()
            grouped = []
            current_group = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - current_group[-1]) / current_group[-1] <= tolerance:
                    current_group.append(level)
                else:
                    grouped.append(sum(current_group) / len(current_group))
                    current_group = [level]
            
            grouped.append(sum(current_group) / len(current_group))
            return grouped
        
        resistance_levels = group_levels(highs)
        support_levels = group_levels(lows)
        
        return support_levels, resistance_levels
    
    def _analyze_support_resistance(self, df: pd.DataFrame, symbol: str, timeframe: str, sensitivity: str) -> str:
        """Analyze support and resistance levels"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            current_price = df.iloc[-1]['Close']
            
            result = f"📊 **SUPPORT & RESISTANCE ANALYSIS** - {symbol} ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"💰 Current Price: ${current_price:.2f}\n"
            result += f"🎯 Sensitivity: {sensitivity.upper()}\n\n"
            
            # Adjust window based on sensitivity
            if sensitivity == "high":
                window = 3
            elif sensitivity == "low":
                window = 7
            else:  # medium
                window = 5
            
            support_levels, resistance_levels = self._find_support_resistance(df, window)
            
            # Filter levels that are relevant to current price
            current_price = df.iloc[-1]['Close']
            nearby_supports = [level for level in support_levels if level < current_price and current_price - level <= current_price * 0.1]
            nearby_resistances = [level for level in resistance_levels if level > current_price and level - current_price <= current_price * 0.1]
            
            # Sort by proximity to current price
            nearby_supports.sort(reverse=True)  # Highest support first
            nearby_resistances.sort()  # Lowest resistance first
            
            result += f"🟢 **SUPPORT LEVELS** (Below current price):\n\n"
            
            if nearby_supports:
                for i, level in enumerate(nearby_supports[:5], 1):  # Show top 5
                    distance = ((current_price - level) / current_price) * 100
                    if distance <= 2:
                        strength = "💪 STRONG"
                    elif distance <= 5:
                        strength = "📊 MODERATE"
                    else:
                        strength = "🟡 WEAK"
                    
                    result += f"{i}. ${level:.2f} - {strength}\n"
                    result += f"   Distance: -{distance:.1f}%\n\n"
            else:
                result += "No significant support levels found within 10% of current price.\n\n"
            
            result += f"🔴 **RESISTANCE LEVELS** (Above current price):\n\n"
            
            if nearby_resistances:
                for i, level in enumerate(nearby_resistances[:5], 1):  # Show top 5
                    distance = ((level - current_price) / current_price) * 100
                    if distance <= 2:
                        strength = "💪 STRONG"
                    elif distance <= 5:
                        strength = "📊 MODERATE"
                    else:
                        strength = "🟡 WEAK"
                    
                    result += f"{i}. ${level:.2f} - {strength}\n"
                    result += f"   Distance: +{distance:.1f}%\n\n"
            else:
                result += "No significant resistance levels found within 10% of current price.\n\n"
            
            # Key levels analysis
            immediate_support = nearby_supports[0] if nearby_supports else None
            immediate_resistance = nearby_resistances[0] if nearby_resistances else None
            
            result += "🎯 **KEY LEVELS ANALYSIS**:\n\n"
            
            if immediate_support:
                support_distance = ((current_price - immediate_support) / current_price) * 100
                result += f"📍 **Immediate Support**: ${immediate_support:.2f} ({support_distance:.1f}% below)\n"
                
                if support_distance <= 1:
                    result += "   ⚠️ Very close to support - Watch for bounce or breakdown\n"
                elif support_distance <= 3:
                    result += "   📊 Near support - Potential buying zone\n"
                else:
                    result += "   ✅ Well above support - Room to decline\n"
            else:
                result += "📍 **Immediate Support**: Not identified within 10%\n"
            
            if immediate_resistance:
                resistance_distance = ((immediate_resistance - current_price) / current_price) * 100
                result += f"📍 **Immediate Resistance**: ${immediate_resistance:.2f} ({resistance_distance:.1f}% above)\n"
                
                if resistance_distance <= 1:
                    result += "   ⚠️ Very close to resistance - Watch for rejection or breakout\n"
                elif resistance_distance <= 3:
                    result += "   📊 Near resistance - Potential selling zone\n"
                else:
                    result += "   ✅ Well below resistance - Room to advance\n"
            else:
                result += "📍 **Immediate Resistance**: Not identified within 10%\n"
            
            # Trading range analysis
            if immediate_support and immediate_resistance:
                range_size = ((immediate_resistance - immediate_support) / immediate_support) * 100
                current_position = ((current_price - immediate_support) / (immediate_resistance - immediate_support)) * 100
                
                result += f"\n📊 **TRADING RANGE ANALYSIS**:\n"
                result += f"   Range Size: {range_size:.1f}%\n"
                result += f"   Current Position: {current_position:.0f}% of range\n"
                
                if current_position <= 25:
                    range_analysis = "🟢 LOWER QUARTILE - Near support, potential bounce zone"
                elif current_position <= 50:
                    range_analysis = "🟡 LOWER HALF - Below midpoint"
                elif current_position <= 75:
                    range_analysis = "🟡 UPPER HALF - Above midpoint"
                else:
                    range_analysis = "🔴 UPPER QUARTILE - Near resistance, potential rejection zone"
                
                result += f"   Analysis: {range_analysis}\n"
            
            # Volume at levels
            result += f"\n📊 **VOLUME CONFIRMATION**:\n"
            current_volume = df.iloc[-1]['Volume']
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume != 0 else 1
            
            if volume_ratio >= 1.5:
                volume_status = "🔥 HIGH - Strong conviction"
            elif volume_ratio >= 1.2:
                volume_status = "📈 ABOVE AVERAGE - Good participation"
            elif volume_ratio >= 0.8:
                volume_status = "📊 AVERAGE - Normal activity"
            else:
                volume_status = "📉 LOW - Weak participation"
            
            result += f"Current Volume: {volume_ratio:.1f}x average - {volume_status}\n"
            
            result += "\n💡 **Trading Strategy**:\n"
            result += "- Watch for price reactions at identified support/resistance levels\n"
            result += "- Confirm breakouts/breakdowns with volume\n"
            result += "- Use levels for entry/exit points and stop-loss placement\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error analyzing support/resistance: {str(e)}"
    
    def _analyze_trend_patterns(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """Analyze trend patterns and channels"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            current_price = df.iloc[-1]['Close']
            
            result = f"📈 **TREND PATTERN ANALYSIS** - {symbol} ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"💰 Current Price: ${current_price:.2f}\n\n"
            
            # Calculate moving averages for trend identification
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['EMA_12'] = ta.ema(df['Close'], length=12)
            
            # Identify trend direction
            latest = df.iloc[-1]
            sma20 = latest['SMA_20']
            sma50 = latest['SMA_50']
            
            # Trend strength using ADX
            adx = ta.adx(df['High'], df['Low'], df['Close'])
            df = pd.concat([df, adx], axis=1)
            adx_value = latest.get('ADX_14', 25)
            
            # Determine primary trend
            if sma20 > sma50 and current_price > sma20:
                primary_trend = "🟢 BULLISH UPTREND"
                trend_color = "🟢"
            elif sma20 < sma50 and current_price < sma20:
                primary_trend = "🔴 BEARISH DOWNTREND"
                trend_color = "🔴"
            else:
                primary_trend = "🟡 SIDEWAYS/CONSOLIDATION"
                trend_color = "🟡"
            
            result += f"📊 **PRIMARY TREND**: {primary_trend}\n\n"
            
            # Trend strength analysis
            if adx_value >= 25:
                trend_strength = f"💪 STRONG ({adx_value:.1f})"
            elif adx_value >= 20:
                trend_strength = f"📊 MODERATE ({adx_value:.1f})"
            else:
                trend_strength = f"😴 WEAK ({adx_value:.1f})"
            
            result += f"🎯 **TREND STRENGTH**: {trend_strength}\n\n"
            
            # Higher highs and higher lows analysis (bullish trend)
            # Lower highs and lower lows analysis (bearish trend)
            recent_highs = []
            recent_lows = []
            
            # Get recent peaks and troughs
            for i in range(5, len(df)-1):
                # Local high
                if (df['High'].iloc[i] > df['High'].iloc[i-2:i].max() and 
                    df['High'].iloc[i] > df['High'].iloc[i+1:i+3].max()):
                    recent_highs.append((i, df['High'].iloc[i]))
                
                # Local low
                if (df['Low'].iloc[i] < df['Low'].iloc[i-2:i].min() and 
                    df['Low'].iloc[i] < df['Low'].iloc[i+1:i+3].min()):
                    recent_lows.append((i, df['Low'].iloc[i]))
            
            # Take last 3 highs and lows
            recent_highs = recent_highs[-3:] if len(recent_highs) >= 3 else recent_highs
            recent_lows = recent_lows[-3:] if len(recent_lows) >= 3 else recent_lows
            
            result += f"📊 **TREND PATTERN ANALYSIS**:\n\n"
            
            # Analyze pattern of highs
            if len(recent_highs) >= 2:
                high_trend = "Higher Highs" if recent_highs[-1][1] > recent_highs[-2][1] else "Lower Highs"
                result += f"📈 Recent Highs: {high_trend}\n"
                
                if len(recent_highs) == 3:
                    consistency = "Consistent" if (recent_highs[2][1] > recent_highs[1][1] > recent_highs[0][1]) or (recent_highs[2][1] < recent_highs[1][1] < recent_highs[0][1]) else "Mixed"
                    result += f"   Pattern: {consistency}\n"
            
            # Analyze pattern of lows
            if len(recent_lows) >= 2:
                low_trend = "Higher Lows" if recent_lows[-1][1] > recent_lows[-2][1] else "Lower Lows"
                result += f"📉 Recent Lows: {low_trend}\n"
                
                if len(recent_lows) == 3:
                    consistency = "Consistent" if (recent_lows[2][1] > recent_lows[1][1] > recent_lows[0][1]) or (recent_lows[2][1] < recent_lows[1][1] < recent_lows[0][1]) else "Mixed"
                    result += f"   Pattern: {consistency}\n"
            
            result += "\n"
            
            # Channel analysis
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                # Calculate channel width
                avg_high = sum([h[1] for h in recent_highs]) / len(recent_highs)
                avg_low = sum([l[1] for l in recent_lows]) / len(recent_lows)
                channel_width = ((avg_high - avg_low) / avg_low) * 100
                
                result += f"📊 **CHANNEL ANALYSIS**:\n"
                result += f"   Average High: ${avg_high:.2f}\n"
                result += f"   Average Low: ${avg_low:.2f}\n"
                result += f"   Channel Width: {channel_width:.1f}%\n"
                
                # Current position in channel
                if avg_high > avg_low:
                    position_pct = ((current_price - avg_low) / (avg_high - avg_low)) * 100
                    
                    if position_pct >= 75:
                        position_desc = "🔴 UPPER CHANNEL - Near resistance"
                    elif position_pct >= 50:
                        position_desc = "🟡 MID-CHANNEL - Neutral zone"
                    elif position_pct >= 25:
                        position_desc = "🟡 LOWER-MID CHANNEL"
                    else:
                        position_desc = "🟢 LOWER CHANNEL - Near support"
                    
                    result += f"   Current Position: {position_pct:.0f}% - {position_desc}\n\n"
            
            # Breakout potential
            recent_range_high = df['High'].tail(10).max()
            recent_range_low = df['Low'].tail(10).min()
            range_size = ((recent_range_high - recent_range_low) / recent_range_low) * 100
            
            result += f"🎯 **BREAKOUT ANALYSIS**:\n"
            result += f"   10-Period Range: ${recent_range_low:.2f} - ${recent_range_high:.2f}\n"
            result += f"   Range Size: {range_size:.1f}%\n"
            
            # Distance to range boundaries
            upside_to_high = ((recent_range_high - current_price) / current_price) * 100
            downside_to_low = ((current_price - recent_range_low) / current_price) * 100
            
            result += f"   Upside to High: +{upside_to_high:.1f}%\n"
            result += f"   Downside to Low: -{downside_to_low:.1f}%\n"
            
            if upside_to_high <= 2:
                result += "   ⚠️ Near recent high - Breakout potential\n"
            elif downside_to_low <= 2:
                result += "   ⚠️ Near recent low - Breakdown potential\n"
            else:
                result += "   📊 Mid-range - No immediate breakout signals\n"
            
            # Trend continuation signals
            result += f"\n💡 **TREND SIGNALS**:\n"
            
            signals = []
            if primary_trend.startswith("🟢") and adx_value >= 25:
                signals.append("Strong uptrend confirmed")
            if primary_trend.startswith("🔴") and adx_value >= 25:
                signals.append("Strong downtrend confirmed")
            
            if len(recent_highs) >= 2 and len(recent_lows) >= 2:
                if recent_highs[-1][1] > recent_highs[-2][1] and recent_lows[-1][1] > recent_lows[-2][1]:
                    signals.append("Higher highs and higher lows")
                elif recent_highs[-1][1] < recent_highs[-2][1] and recent_lows[-1][1] < recent_lows[-2][1]:
                    signals.append("Lower highs and lower lows")
            
            if upside_to_high <= 1 and adx_value >= 20:
                signals.append("Potential upside breakout")
            elif downside_to_low <= 1 and adx_value >= 20:
                signals.append("Potential downside breakdown")
            
            if signals:
                for signal in signals:
                    result += f"• {signal}\n"
            else:
                result += "• Mixed signals - No clear trend continuation pattern\n"
            
            result += "\n⚠️ **Key Points**:\n"
            result += "- Trend analysis works best on higher timeframes\n"
            result += "- Confirm patterns with volume and momentum indicators\n"
            result += "- Watch for trend line breaks and channel violations\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error analyzing trend patterns: {str(e)}"
    
    def _analyze_candlestick_patterns(self, df: pd.DataFrame, symbol: str, timeframe: str, sensitivity: str) -> str:
        """Analyze candlestick patterns"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            current_price = df.iloc[-1]['Close']
            
            result = f"🕯️ **CANDLESTICK PATTERNS ANALYSIS** - {symbol} ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"💰 Current Price: ${current_price:.2f}\n"
            result += f"🎯 Sensitivity: {sensitivity.upper()}\n\n"
            
            patterns = []
            
            # Get recent candles
            if len(df) < 3:
                return "❌ Insufficient data for candlestick pattern analysis (need at least 3 periods)."
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            prev2 = df.iloc[-3] if len(df) >= 3 else None
            
            # Calculate candle properties
            def candle_info(row):
                body_size = abs(row['Close'] - row['Open'])
                total_size = row['High'] - row['Low']
                upper_wick = row['High'] - max(row['Close'], row['Open'])
                lower_wick = min(row['Close'], row['Open']) - row['Low']
                is_bullish = row['Close'] > row['Open']
                
                return {
                    'body_size': body_size,
                    'total_size': total_size,
                    'upper_wick': upper_wick,
                    'lower_wick': lower_wick,
                    'is_bullish': is_bullish,
                    'body_pct': (body_size / total_size * 100) if total_size > 0 else 0
                }
            
            latest_info = candle_info(latest)
            prev_info = candle_info(prev)
            prev2_info = candle_info(prev2) if prev2 is not None else None
            
            # Set sensitivity thresholds
            if sensitivity == "high":
                min_body_pct = 30
                wick_threshold = 2.0
            elif sensitivity == "low":
                min_body_pct = 60
                wick_threshold = 3.0
            else:  # medium
                min_body_pct = 40
                wick_threshold = 2.5
            
            # 1. HAMMER / INVERTED HAMMER
            if (latest_info['lower_wick'] >= latest_info['body_size'] * wick_threshold and
                latest_info['upper_wick'] <= latest_info['body_size'] * 0.3 and
                latest_info['body_pct'] >= min_body_pct):
                
                if latest_info['is_bullish']:
                    patterns.append({
                        'name': 'Hammer (Bullish)',
                        'signal': '🟢 BULLISH REVERSAL',
                        'strength': 75,
                        'description': 'Long lower wick with small body - potential bounce'
                    })
                else:
                    patterns.append({
                        'name': 'Hammer (Bearish)',
                        'signal': '🟢 BULLISH REVERSAL',
                        'strength': 70,
                        'description': 'Long lower wick - potential reversal despite bearish close'
                    })
            
            # 2. INVERTED HAMMER / SHOOTING STAR
            if (latest_info['upper_wick'] >= latest_info['body_size'] * wick_threshold and
                latest_info['lower_wick'] <= latest_info['body_size'] * 0.3 and
                latest_info['body_pct'] >= min_body_pct):
                
                if latest_info['is_bullish']:
                    patterns.append({
                        'name': 'Inverted Hammer',
                        'signal': '🟡 REVERSAL SIGNAL',
                        'strength': 60,
                        'description': 'Long upper wick - potential reversal if confirmed'
                    })
                else:
                    patterns.append({
                        'name': 'Shooting Star',
                        'signal': '🔴 BEARISH REVERSAL',
                        'strength': 75,
                        'description': 'Long upper wick with bearish close - potential rejection'
                    })
            
            # 3. DOJI
            if latest_info['body_size'] <= latest_info['total_size'] * 0.1:
                patterns.append({
                    'name': 'Doji',
                    'signal': '🟡 INDECISION',
                    'strength': 65,
                    'description': 'Very small body - market indecision, potential reversal'
                })
            
            # 4. ENGULFING PATTERNS (need previous candle)
            if prev_info:
                # Bullish Engulfing
                if (not prev_info['is_bullish'] and latest_info['is_bullish'] and
                    latest['Open'] < prev['Close'] and latest['Close'] > prev['Open']):
                    patterns.append({
                        'name': 'Bullish Engulfing',
                        'signal': '🟢 BULLISH REVERSAL',
                        'strength': 85,
                        'description': 'Bullish candle completely engulfs previous bearish candle'
                    })
                
                # Bearish Engulfing
                elif (prev_info['is_bullish'] and not latest_info['is_bullish'] and
                      latest['Open'] > prev['Close'] and latest['Close'] < prev['Open']):
                    patterns.append({
                        'name': 'Bearish Engulfing',
                        'signal': '🔴 BEARISH REVERSAL',
                        'strength': 85,
                        'description': 'Bearish candle completely engulfs previous bullish candle'
                    })
            
            # 5. PIERCING LINE / DARK CLOUD COVER
            if prev_info:
                # Piercing Line
                if (not prev_info['is_bullish'] and latest_info['is_bullish'] and
                    latest['Open'] < prev['Low'] and 
                    latest['Close'] > (prev['Open'] + prev['Close']) / 2):
                    patterns.append({
                        'name': 'Piercing Line',
                        'signal': '🟢 BULLISH REVERSAL',
                        'strength': 80,
                        'description': 'Bullish candle pierces more than halfway into previous bearish candle'
                    })
                
                # Dark Cloud Cover
                elif (prev_info['is_bullish'] and not latest_info['is_bullish'] and
                      latest['Open'] > prev['High'] and
                      latest['Close'] < (prev['Open'] + prev['Close']) / 2):
                    patterns.append({
                        'name': 'Dark Cloud Cover',
                        'signal': '🔴 BEARISH REVERSAL',
                        'strength': 80,
                        'description': 'Bearish candle covers more than halfway into previous bullish candle'
                    })
            
            # 6. THREE-CANDLE PATTERNS
            if prev2_info:
                # Morning Star
                if (not prev2_info['is_bullish'] and 
                    prev_info['body_size'] <= prev2_info['body_size'] * 0.5 and
                    latest_info['is_bullish'] and
                    latest['Close'] > (prev2['Open'] + prev2['Close']) / 2):
                    patterns.append({
                        'name': 'Morning Star',
                        'signal': '🟢 STRONG BULLISH REVERSAL',
                        'strength': 90,
                        'description': 'Three-candle bullish reversal pattern'
                    })
                
                # Evening Star
                elif (prev2_info['is_bullish'] and
                      prev_info['body_size'] <= prev2_info['body_size'] * 0.5 and
                      not latest_info['is_bullish'] and
                      latest['Close'] < (prev2['Open'] + prev2['Close']) / 2):
                    patterns.append({
                        'name': 'Evening Star',
                        'signal': '🔴 STRONG BEARISH REVERSAL',
                        'strength': 90,
                        'description': 'Three-candle bearish reversal pattern'
                    })
            
            # 7. MARUBOZU (Long body with little to no wicks)
            if latest_info['body_pct'] >= 90:
                if latest_info['is_bullish']:
                    patterns.append({
                        'name': 'White Marubozu',
                        'signal': '🟢 STRONG BULLISH',
                        'strength': 75,
                        'description': 'Long bullish candle with no wicks - strong buying pressure'
                    })
                else:
                    patterns.append({
                        'name': 'Black Marubozu',
                        'signal': '🔴 STRONG BEARISH',
                        'strength': 75,
                        'description': 'Long bearish candle with no wicks - strong selling pressure'
                    })
            
            # Present results
            if patterns:
                result += f"✅ **DETECTED {len(patterns)} CANDLESTICK PATTERNS:**\n\n"
                
                # Sort by strength
                patterns.sort(key=lambda x: x['strength'], reverse=True)
                
                for i, pattern in enumerate(patterns, 1):
                    strength_desc = "🔥 VERY STRONG" if pattern['strength'] >= 85 else "💪 STRONG" if pattern['strength'] >= 75 else "📊 MODERATE" if pattern['strength'] >= 65 else "🟡 WEAK"
                    
                    result += f"{i}. **{pattern['name']}** - {pattern['signal']}\n"
                    result += f"   Strength: {pattern['strength']}/100 - {strength_desc}\n"
                    result += f"   Description: {pattern['description']}\n\n"
                
                # Overall assessment
                bullish_patterns = [p for p in patterns if 'BULLISH' in p['signal'] and 'BEARISH' not in p['signal']]
                bearish_patterns = [p for p in patterns if 'BEARISH' in p['signal']]
                neutral_patterns = [p for p in patterns if 'INDECISION' in p['signal'] or 'REVERSAL SIGNAL' in p['signal']]
                
                if len(bullish_patterns) > len(bearish_patterns):
                    overall = "🟢 **NET BULLISH SENTIMENT** from candlestick patterns"
                elif len(bearish_patterns) > len(bullish_patterns):
                    overall = "🔴 **NET BEARISH SENTIMENT** from candlestick patterns"
                else:
                    overall = "🟡 **MIXED SIGNALS** from candlestick patterns"
                
                result += f"🎯 **Pattern Assessment**: {overall}\n"
                
                # Highest strength pattern
                strongest = max(patterns, key=lambda x: x['strength'])
                result += f"🏆 **Strongest Pattern**: {strongest['name']} ({strongest['strength']}/100)\n\n"
                
            else:
                result += "❌ **NO SIGNIFICANT CANDLESTICK PATTERNS DETECTED**\n\n"
                result += f"Current sensitivity ({sensitivity}) may be too restrictive, or patterns may not be clear.\n\n"
            
            # Current candle analysis
            result += "🕯️ **CURRENT CANDLE ANALYSIS**:\n\n"
            
            candle_type = "🟢 Bullish" if latest_info['is_bullish'] else "🔴 Bearish"
            body_strength = "💪 Strong" if latest_info['body_pct'] >= 70 else "📊 Moderate" if latest_info['body_pct'] >= 40 else "🟡 Weak"
            
            result += f"Type: {candle_type}\n"
            result += f"Body Strength: {body_strength} ({latest_info['body_pct']:.1f}% of total range)\n"
            result += f"Upper Wick: ${latest_info['upper_wick']:.2f}\n"
            result += f"Lower Wick: ${latest_info['lower_wick']:.2f}\n"
            result += f"Total Range: ${latest_info['total_size']:.2f}\n\n"
            
            if latest_info['upper_wick'] > latest_info['body_size'] * 2:
                result += "⚠️ Long upper wick suggests selling pressure at higher levels\n"
            if latest_info['lower_wick'] > latest_info['body_size'] * 2:
                result += "✅ Long lower wick suggests buying support at lower levels\n"
            
            result += "\n💡 **Notes**:\n"
            result += "- Candlestick patterns work best when confirmed by volume\n"
            result += "- Reversal patterns are more significant at support/resistance levels\n"
            result += "- Consider the overall trend context when interpreting patterns\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error analyzing candlestick patterns: {str(e)}"
    
    def _analyze_breakouts(self, df: pd.DataFrame, symbol: str, timeframe: str, sensitivity: str) -> str:
        """Analyze breakout and breakdown patterns"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            current_price = df.iloc[-1]['Close']
            
            result = f"🚀 **BREAKOUT ANALYSIS** - {symbol} ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"💰 Current Price: ${current_price:.2f}\n"
            result += f"🎯 Sensitivity: {sensitivity.upper()}\n\n"
            
            # Calculate indicators for breakout analysis
            df['Volume_MA'] = ta.sma(df['Volume'], length=20)
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
            
            # Define lookback periods based on sensitivity
            if sensitivity == "high":
                lookback = 10
                volume_threshold = 1.2
            elif sensitivity == "low":
                lookback = 30
                volume_threshold = 1.8
            else:  # medium
                lookback = 20
                volume_threshold = 1.5
            
            # Find recent range
            recent_high = df['High'].tail(lookback).max()
            recent_low = df['Low'].tail(lookback).min()
            range_size = recent_high - recent_low
            range_pct = (range_size / recent_low) * 100
            
            # Current position analysis
            current_vol = df.iloc[-1]['Volume']
            avg_vol = df.iloc[-1]['Volume_MA']
            vol_ratio = current_vol / avg_vol if avg_vol != 0 else 1
            atr = df.iloc[-1]['ATR']
            
            result += f"📊 **RANGE ANALYSIS** ({lookback} periods):\n"
            result += f"   Recent High: ${recent_high:.2f}\n"
            result += f"   Recent Low: ${recent_low:.2f}\n"
            result += f"   Range Size: ${range_size:.2f} ({range_pct:.1f}%)\n\n"
            
            # Distance to breakout levels
            upside_distance = recent_high - current_price
            upside_pct = (upside_distance / current_price) * 100
            downside_distance = current_price - recent_low
            downside_pct = (downside_distance / current_price) * 100
            
            result += f"🎯 **BREAKOUT DISTANCES**:\n"
            result += f"   To High: ${upside_distance:.2f} (+{upside_pct:.1f}%)\n"
            result += f"   To Low: ${downside_distance:.2f} (-{downside_pct:.1f}%)\n\n"
            
            # Breakout/breakdown detection
            breakouts = []
            
            # Check for recent breakouts
            if current_price >= recent_high:
                if vol_ratio >= volume_threshold:
                    breakouts.append({
                        'type': 'Upside Breakout',
                        'signal': '🚀 BULLISH BREAKOUT',
                        'strength': 90,
                        'description': f'Price broke above ${recent_high:.2f} with {vol_ratio:.1f}x volume'
                    })
                else:
                    breakouts.append({
                        'type': 'Upside Breakout (Weak Volume)',
                        'signal': '🟡 POTENTIAL BREAKOUT',
                        'strength': 60,
                        'description': f'Price broke above ${recent_high:.2f} but volume only {vol_ratio:.1f}x'
                    })
            
            elif current_price <= recent_low:
                if vol_ratio >= volume_threshold:
                    breakouts.append({
                        'type': 'Downside Breakdown',
                        'signal': '💥 BEARISH BREAKDOWN',
                        'strength': 90,
                        'description': f'Price broke below ${recent_low:.2f} with {vol_ratio:.1f}x volume'
                    })
                else:
                    breakouts.append({
                        'type': 'Downside Breakdown (Weak Volume)',
                        'signal': '🟡 POTENTIAL BREAKDOWN',
                        'strength': 60,
                        'description': f'Price broke below ${recent_low:.2f} but volume only {vol_ratio:.1f}x'
                    })
            
            # Check for imminent breakouts
            elif upside_pct <= 1.0:
                breakouts.append({
                    'type': 'Approaching Resistance',
                    'signal': '⚠️ BREAKOUT SETUP',
                    'strength': 70,
                    'description': f'Within 1% of resistance at ${recent_high:.2f} - watch for breakout'
                })
            
            elif downside_pct <= 1.0:
                breakouts.append({
                    'type': 'Approaching Support',
                    'signal': '⚠️ BREAKDOWN SETUP',
                    'strength': 70,
                    'description': f'Within 1% of support at ${recent_low:.2f} - watch for breakdown'
                })
            
            # Volume analysis for breakout confirmation
            result += f"📊 **VOLUME ANALYSIS**:\n"
            result += f"   Current: {current_vol:,.0f}\n"
            result += f"   20-Day Average: {avg_vol:,.0f}\n"
            result += f"   Volume Ratio: {vol_ratio:.1f}x\n"
            
            if vol_ratio >= 2.0:
                vol_status = "🔥 VERY HIGH - Strong conviction"
            elif vol_ratio >= volume_threshold:
                vol_status = "📈 HIGH - Good for breakouts"
            elif vol_ratio >= 1.0:
                vol_status = "📊 AVERAGE - Normal activity"
            else:
                vol_status = "📉 LOW - Weak for breakouts"
            
            result += f"   Status: {vol_status}\n\n"
            
            # Present breakout analysis
            if breakouts:
                result += f"🚨 **BREAKOUT SIGNALS DETECTED:**\n\n"
                
                for i, breakout in enumerate(breakouts, 1):
                    strength_desc = "🔥 VERY STRONG" if breakout['strength'] >= 85 else "💪 STRONG" if breakout['strength'] >= 75 else "📊 MODERATE"
                    
                    result += f"{i}. **{breakout['type']}** - {breakout['signal']}\n"
                    result += f"   Strength: {breakout['strength']}/100 - {strength_desc}\n"
                    result += f"   Description: {breakout['description']}\n\n"
            
            else:
                result += f"📊 **NO IMMEDIATE BREAKOUT SIGNALS**\n\n"
                result += f"Price is trading within the established range.\n\n"
            
            # Consolidation analysis
            if not breakouts or (breakouts and breakouts[0]['strength'] < 80):
                # Check for consolidation pattern
                recent_prices = df['Close'].tail(10)
                price_std = recent_prices.std()
                price_range = recent_prices.max() - recent_prices.min()
                consolidation_pct = (price_range / recent_prices.mean()) * 100
                
                result += f"📊 **CONSOLIDATION ANALYSIS**:\n"
                result += f"   10-Day Price Range: {consolidation_pct:.1f}%\n"
                
                if consolidation_pct <= 3:
                    consol_type = "🤏 TIGHT CONSOLIDATION - High breakout potential"
                elif consolidation_pct <= 5:
                    consol_type = "📊 MODERATE CONSOLIDATION - Building energy"
                else:
                    consol_type = "📈 LOOSE CONSOLIDATION - Less explosive potential"
                
                result += f"   Pattern: {consol_type}\n\n"
            
            # Measured move targets
            if breakouts and any('BREAKOUT' in b['signal'] for b in breakouts):
                result += f"🎯 **MEASURED MOVE TARGETS**:\n"
                
                if any('BULLISH' in b['signal'] for b in breakouts):
                    target1 = recent_high + (range_size * 0.5)
                    target2 = recent_high + range_size
                    result += f"   Bullish Target 1: ${target1:.2f} (+{((target1-current_price)/current_price)*100:.1f}%)\n"
                    result += f"   Bullish Target 2: ${target2:.2f} (+{((target2-current_price)/current_price)*100:.1f}%)\n"
                
                if any('BEARISH' in b['signal'] for b in breakouts):
                    target1 = recent_low - (range_size * 0.5)
                    target2 = recent_low - range_size
                    result += f"   Bearish Target 1: ${target1:.2f} ({((target1-current_price)/current_price)*100:.1f}%)\n"
                    result += f"   Bearish Target 2: ${target2:.2f} ({((target2-current_price)/current_price)*100:.1f}%)\n"
                
                result += "\n"
            
            # Stop loss suggestions
            if breakouts:
                result += f"🛑 **STOP LOSS SUGGESTIONS**:\n"
                
                if any('BULLISH' in b['signal'] for b in breakouts):
                    stop_loss = recent_high - (atr * 1.5)
                    stop_pct = ((current_price - stop_loss) / current_price) * 100
                    result += f"   Bullish Trade Stop: ${stop_loss:.2f} (-{stop_pct:.1f}%)\n"
                
                if any('BEARISH' in b['signal'] for b in breakouts):
                    stop_loss = recent_low + (atr * 1.5)
                    stop_pct = ((stop_loss - current_price) / current_price) * 100
                    result += f"   Bearish Trade Stop: ${stop_loss:.2f} (+{stop_pct:.1f}%)\n"
                
                result += "\n"
            
            result += "💡 **Trading Notes**:\n"
            result += "- Volume confirmation is crucial for valid breakouts\n"
            result += "- False breakouts are common - wait for follow-through\n"
            result += "- Use tight stops on breakout trades due to higher volatility\n"
            result += "- Best breakouts often occur after tight consolidation\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error analyzing breakouts: {str(e)}"
    
    def _comprehensive_pattern_scan(self, df: pd.DataFrame, symbol: str, timeframe: str, sensitivity: str) -> str:
        """Comprehensive scan of all chart patterns"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            result = f"🔍 **COMPREHENSIVE PATTERN SCAN** - {symbol} ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"💰 Current Price: ${df.iloc[-1]['Close']:.2f}\n"
            result += f"🎯 Sensitivity: {sensitivity.upper()}\n\n"
            
            # Run all individual analyses
            support_resistance = self._analyze_support_resistance(df, symbol, timeframe, sensitivity)
            trend_analysis = self._analyze_trend_patterns(df, symbol, timeframe)
            candlestick_analysis = self._analyze_candlestick_patterns(df, symbol, timeframe, sensitivity)
            breakout_analysis = self._analyze_breakouts(df, symbol, timeframe, sensitivity)
            
            # Extract key findings from each analysis
            result += "📊 **PATTERN SUMMARY**:\n\n"
            
            # Count detected patterns
            sr_patterns = support_resistance.count("SUPPORT LEVELS") + support_resistance.count("RESISTANCE LEVELS")
            trend_patterns = 1 if ("UPTREND" in trend_analysis or "DOWNTREND" in trend_analysis) else 0
            candle_patterns = candlestick_analysis.count("DETECTED") 
            breakout_patterns = breakout_analysis.count("BREAKOUT") + breakout_analysis.count("BREAKDOWN")
            
            result += f"🎯 Support/Resistance Levels: Found key levels\n"
            result += f"📈 Trend Patterns: {'Identified' if trend_patterns > 0 else 'Mixed/Unclear'}\n"
            result += f"🕯️ Candlestick Patterns: {candle_patterns if 'DETECTED' in candlestick_analysis else 0} detected\n"
            result += f"🚀 Breakout Signals: {'Active' if 'BREAKOUT' in breakout_analysis else 'None detected'}\n\n"
            
            result += "="*60 + "\n\n"
            
            # Include all detailed analyses
            result += support_resistance + "\n\n" + "="*40 + "\n\n"
            result += trend_analysis + "\n\n" + "="*40 + "\n\n"
            result += candlestick_analysis + "\n\n" + "="*40 + "\n\n"
            result += breakout_analysis + "\n\n"
            
            # Overall pattern assessment
            result += "="*60 + "\n"
            result += "🎯 **OVERALL PATTERN ASSESSMENT**\n"
            result += "="*60 + "\n\n"
            
            # Collect all signals
            bullish_signals = (support_resistance.count("BULLISH") + trend_analysis.count("BULLISH") + 
                             candlestick_analysis.count("BULLISH") + breakout_analysis.count("BULLISH"))
            bearish_signals = (support_resistance.count("BEARISH") + trend_analysis.count("BEARISH") + 
                             candlestick_analysis.count("BEARISH") + breakout_analysis.count("BEARISH"))
            
            neutral_signals = (support_resistance.count("MIXED") + support_resistance.count("NEUTRAL") +
                             trend_analysis.count("MIXED") + trend_analysis.count("SIDEWAYS") +
                             candlestick_analysis.count("INDECISION") + breakout_analysis.count("POTENTIAL"))
            
            total_signals = bullish_signals + bearish_signals + neutral_signals
            
            if total_signals > 0:
                bullish_pct = (bullish_signals / total_signals) * 100
                bearish_pct = (bearish_signals / total_signals) * 100
                
                result += f"📊 **Signal Distribution**:\n"
                result += f"🟢 Bullish Patterns: {bullish_signals} ({bullish_pct:.0f}%)\n"
                result += f"🔴 Bearish Patterns: {bearish_signals} ({bearish_pct:.0f}%)\n"
                result += f"🟡 Neutral/Mixed: {neutral_signals} ({((neutral_signals/total_signals)*100):.0f}%)\n\n"
                
                if bullish_pct >= 60:
                    pattern_bias = "🟢 **BULLISH PATTERN BIAS** - Multiple bullish signals detected"
                elif bearish_pct >= 60:
                    pattern_bias = "🔴 **BEARISH PATTERN BIAS** - Multiple bearish signals detected"
                else:
                    pattern_bias = "🟡 **MIXED PATTERN SIGNALS** - No clear directional bias"
                
                result += f"🎯 **Pattern Consensus**: {pattern_bias}\n\n"
            
            # Trading recommendations based on pattern analysis
            result += "💡 **TRADING IMPLICATIONS**:\n\n"
            
            if "STRONG" in support_resistance and "BULLISH" in trend_analysis:
                result += "✅ **CONFLUENCE SETUP**: Support + Trend alignment suggests buying opportunity\n"
            elif "STRONG" in support_resistance and "BEARISH" in trend_analysis:
                result += "⚠️ **CONFLICT SETUP**: Support vs. Trend - Wait for resolution\n"
            
            if "BREAKOUT" in breakout_analysis and "BULLISH" in candlestick_analysis:
                result += "🚀 **MOMENTUM SETUP**: Breakout + Candlestick patterns align\n"
            elif "BREAKDOWN" in breakout_analysis and "BEARISH" in candlestick_analysis:
                result += "💥 **BREAKDOWN SETUP**: Breakdown + Bearish patterns align\n"
            
            if "CONSOLIDATION" in breakout_analysis:
                result += "🤏 **COILING SETUP**: Tight range suggests explosive move ahead\n"
            
            # Risk assessment
            result += f"\n⚠️ **RISK ASSESSMENT**:\n"
            
            volatility_mentions = (support_resistance.count("HIGH") + trend_analysis.count("HIGH") + 
                                 breakout_analysis.count("HIGH"))
            
            if volatility_mentions >= 2:
                result += "🔴 **HIGH VOLATILITY ENVIRONMENT** - Use smaller position sizes\n"
            else:
                result += "📊 **MODERATE RISK ENVIRONMENT** - Standard position sizing\n"
            
            if "MIXED" in pattern_bias:
                result += "🟡 **CONFLICTING SIGNALS** - Wait for clearer setup or use smaller sizes\n"
            
            result += f"\n📈 **Pattern Strength Score**: {min(100, (bullish_signals + bearish_signals) * 10)}/100\n"
            result += f"🎯 **Signal Clarity**: {'High' if abs(bullish_signals - bearish_signals) >= 3 else 'Moderate' if abs(bullish_signals - bearish_signals) >= 1 else 'Low'}\n"
            
            result += "\n⚠️ **Final Notes**:\n"
            result += "- Pattern analysis works best when multiple timeframes align\n"
            result += "- Always confirm patterns with volume and momentum indicators\n"
            result += "- Risk management is crucial when trading pattern setups\n"
            result += "- Patterns are probabilities, not guarantees\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error performing comprehensive pattern scan: {str(e)}"


def create_chart_patterns_tool() -> ChartPatternsTool:
    """Create and return the chart patterns tool instance"""
    return ChartPatternsTool()