"""
Technical Indicators Tool for LangChain Integration
"""

import os
import pandas as pd
import numpy as np
from typing import Optional, Type, Dict, Any, List
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
    logger.warning("pandas-ta-classic not installed. Technical indicators features will be disabled.")

try:
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.common.exceptions import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca-py not available for data retrieval.")


class TechnicalIndicatorsInput(BaseModel):
    """Input schema for technical indicators tool"""
    action: str = Field(description="Action: 'momentum', 'trend', 'volatility', 'volume', 'oscillators', 'all_indicators'")
    symbol: str = Field(description="Stock symbol (e.g., 'AAPL', 'TSLA')")
    timeframe: Optional[str] = Field(default="1Day", description="Timeframe: '1Min', '5Min', '15Min', '30Min', '1Hour', '1Day'")
    period: Optional[int] = Field(default=100, description="Number of periods to analyze (20-500)")
    rsi_length: Optional[int] = Field(default=14, description="RSI period length")
    ma_short: Optional[int] = Field(default=20, description="Short moving average period") 
    ma_long: Optional[int] = Field(default=50, description="Long moving average period")
    bb_length: Optional[int] = Field(default=20, description="Bollinger Bands period")
    bb_std: Optional[float] = Field(default=2.0, description="Bollinger Bands standard deviation")
    macd_fast: Optional[int] = Field(default=12, description="MACD fast EMA period")
    macd_slow: Optional[int] = Field(default=26, description="MACD slow EMA period")
    macd_signal: Optional[int] = Field(default=9, description="MACD signal line period")


class TechnicalIndicatorsTool(BaseTool):
    """Tool for comprehensive technical indicators analysis using pandas-ta"""
    
    name: str = "technical_indicators"
    description: str = (
        "Calculate comprehensive technical indicators using pandas-ta. "
        "Actions: 'momentum' (RSI, MACD, Stochastic), 'trend' (Moving Averages, ADX), "
        "'volatility' (Bollinger Bands, ATR), 'volume' (OBV, MFI), 'oscillators' (Williams %R, CCI), "
        "'all_indicators' (comprehensive analysis). Requires historical price data."
    )
    args_schema: Type[BaseModel] = TechnicalIndicatorsInput
    
    def _get_alpaca_data(self, symbol: str, timeframe: str, period: int) -> Optional[pd.DataFrame]:
        """Get historical data from Alpaca for technical analysis"""
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
            
            # Calculate start date based on period and timeframe
            end_date = datetime.now().date()
            if timeframe == "1Min":
                start_date = end_date - timedelta(days=max(7, period // 300))  # ~300 minutes per day
            elif timeframe == "5Min":
                start_date = end_date - timedelta(days=max(7, period // 60))   # ~60 5-min bars per day
            elif timeframe == "15Min":
                start_date = end_date - timedelta(days=max(7, period // 20))   # ~20 15-min bars per day
            elif timeframe == "30Min":
                start_date = end_date - timedelta(days=max(7, period // 10))   # ~10 30-min bars per day
            elif timeframe == "1Hour":
                start_date = end_date - timedelta(days=max(7, period // 6))    # ~6 hourly bars per day
            else:  # 1Day
                start_date = end_date - timedelta(days=max(30, period + 10))   # Add buffer for indicators
            
            request = request_class(
                symbol_or_symbols=[symbol],
                timeframe=tf,
                start=start_date,
                end=end_date,
                feed="iex"  # Use IEX feed as default
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
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']  # Capitalize for pandas-ta
                return df.tail(period + 50)  # Add buffer for indicator calculation
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting Alpaca data: {e}")
            return None
    
    def _run(
        self,
        action: str,
        symbol: str,
        timeframe: str = "1Day",
        period: int = 100,
        rsi_length: int = 14,
        ma_short: int = 20,
        ma_long: int = 50,
        bb_length: int = 20,
        bb_std: float = 2.0,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9
    ) -> str:
        """Execute technical indicators analysis"""
        if not PANDAS_TA_AVAILABLE:
            return "❌ pandas-ta-classic not available. Please install pandas-ta-classic to use technical indicators."
        
        # Validate inputs
        period = max(20, min(period, 500))
        symbol = symbol.upper().strip()
        
        try:
            # Get historical data
            df = self._get_alpaca_data(symbol, timeframe, period)
            
            if df is None or len(df) < 20:
                return f"❌ Insufficient historical data for {symbol}. Need at least 20 periods for technical analysis."
            
            # Perform analysis based on action
            if action == "momentum":
                return self._analyze_momentum(df, symbol, timeframe, rsi_length, macd_fast, macd_slow, macd_signal)
            elif action == "trend":
                return self._analyze_trend(df, symbol, timeframe, ma_short, ma_long)
            elif action == "volatility":
                return self._analyze_volatility(df, symbol, timeframe, bb_length, bb_std)
            elif action == "volume":
                return self._analyze_volume(df, symbol, timeframe)
            elif action == "oscillators":
                return self._analyze_oscillators(df, symbol, timeframe)
            elif action == "all_indicators":
                return self._analyze_all_indicators(df, symbol, timeframe, rsi_length, ma_short, ma_long, bb_length, bb_std)
            else:
                return f"❌ Unknown action: {action}. Available actions: momentum, trend, volatility, volume, oscillators, all_indicators"
        
        except Exception as e:
            logger.error(f"Technical indicators error: {e}")
            return f"❌ Technical indicators error: {str(e)}"
    
    def _analyze_momentum(self, df: pd.DataFrame, symbol: str, timeframe: str, rsi_length: int, macd_fast: int, macd_slow: int, macd_signal: int) -> str:
        """Analyze momentum indicators"""
        try:
            # Calculate momentum indicators
            df['RSI'] = ta.rsi(df['Close'], length=rsi_length)
            
            # MACD
            macd = ta.macd(df['Close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
            df = pd.concat([df, macd], axis=1)
            
            # Stochastic
            stoch = ta.stoch(df['High'], df['Low'], df['Close'])
            df = pd.concat([df, stoch], axis=1)
            
            # Williams %R
            df['WILLR'] = ta.willr(df['High'], df['Low'], df['Close'])
            
            # Get latest values
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            result = f"📈 **MOMENTUM ANALYSIS** - {symbol} ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"💰 Current Price: ${latest['Close']:.2f}\n\n"
            
            # RSI Analysis
            rsi_current = latest['RSI']
            rsi_prev = prev['RSI']
            rsi_trend = "↗️" if rsi_current > rsi_prev else "↘️"
            
            if rsi_current >= 70:
                rsi_signal = "🔴 OVERBOUGHT"
            elif rsi_current <= 30:
                rsi_signal = "🟢 OVERSOLD"
            else:
                rsi_signal = "🟡 NEUTRAL"
            
            result += f"🎯 **RSI ({rsi_length})**: {rsi_current:.1f} {rsi_trend} - {rsi_signal}\n"
            
            # MACD Analysis
            macd_line = latest.get('MACD_12_26_9', 0)
            macd_signal_line = latest.get('MACDs_12_26_9', 0)
            macd_histogram = latest.get('MACDh_12_26_9', 0)
            
            macd_trend = "🟢 BULLISH" if macd_line > macd_signal_line else "🔴 BEARISH"
            hist_trend = "↗️" if macd_histogram > 0 else "↘️"
            
            result += f"📊 **MACD ({macd_fast},{macd_slow},{macd_signal})**: {macd_line:.4f} - {macd_trend}\n"
            result += f"   Signal: {macd_signal_line:.4f} | Histogram: {macd_histogram:.4f} {hist_trend}\n"
            
            # Stochastic Analysis
            stoch_k = latest.get('STOCHk_14_3_3', 0)
            stoch_d = latest.get('STOCHd_14_3_3', 0)
            
            if stoch_k >= 80:
                stoch_signal = "🔴 OVERBOUGHT"
            elif stoch_k <= 20:
                stoch_signal = "🟢 OVERSOLD"
            else:
                stoch_signal = "🟡 NEUTRAL"
            
            result += f"🎢 **Stochastic**: %K: {stoch_k:.1f} | %D: {stoch_d:.1f} - {stoch_signal}\n"
            
            # Williams %R Analysis
            willr = latest['WILLR']
            if willr >= -20:
                willr_signal = "🔴 OVERBOUGHT"
            elif willr <= -80:
                willr_signal = "🟢 OVERSOLD"
            else:
                willr_signal = "🟡 NEUTRAL"
            
            result += f"📉 **Williams %R**: {willr:.1f} - {willr_signal}\n\n"
            
            # Overall momentum assessment
            momentum_signals = []
            if rsi_current <= 30: momentum_signals.append("RSI Oversold")
            if rsi_current >= 70: momentum_signals.append("RSI Overbought")
            if macd_line > macd_signal_line and macd_histogram > 0: momentum_signals.append("MACD Bullish")
            if macd_line < macd_signal_line and macd_histogram < 0: momentum_signals.append("MACD Bearish")
            if stoch_k <= 20: momentum_signals.append("Stoch Oversold")
            if stoch_k >= 80: momentum_signals.append("Stoch Overbought")
            
            if momentum_signals:
                result += f"🎯 **Key Signals**: {', '.join(momentum_signals)}\n\n"
            
            result += "💡 **Momentum Summary**:\n"
            bullish_count = sum([
                rsi_current <= 30,
                macd_line > macd_signal_line,
                stoch_k <= 20 and stoch_k > prev.get('STOCHk_14_3_3', 0)
            ])
            bearish_count = sum([
                rsi_current >= 70,
                macd_line < macd_signal_line,
                stoch_k >= 80
            ])
            
            if bullish_count >= 2:
                result += "🟢 **BULLISH MOMENTUM** - Multiple oversold signals\n"
            elif bearish_count >= 2:
                result += "🔴 **BEARISH MOMENTUM** - Multiple overbought signals\n"
            else:
                result += "🟡 **MIXED MOMENTUM** - No clear directional bias\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error calculating momentum indicators: {str(e)}"
    
    def _analyze_trend(self, df: pd.DataFrame, symbol: str, timeframe: str, ma_short: int, ma_long: int) -> str:
        """Analyze trend indicators"""
        try:
            # Calculate trend indicators
            df['SMA_Short'] = ta.sma(df['Close'], length=ma_short)
            df['SMA_Long'] = ta.sma(df['Close'], length=ma_long)
            df['EMA_Short'] = ta.ema(df['Close'], length=ma_short)
            df['EMA_Long'] = ta.ema(df['Close'], length=ma_long)
            
            # ADX for trend strength
            adx = ta.adx(df['High'], df['Low'], df['Close'])
            df = pd.concat([df, adx], axis=1)
            
            # Aroon for trend direction
            aroon = ta.aroon(df['High'], df['Low'])
            df = pd.concat([df, aroon], axis=1)
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            result = f"📈 **TREND ANALYSIS** - {symbol} ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"💰 Current Price: ${latest['Close']:.2f}\n\n"
            
            # Moving Average Analysis
            sma_short = latest['SMA_Short']
            sma_long = latest['SMA_Long']
            ema_short = latest['EMA_Short']
            ema_long = latest['EMA_Long']
            price = latest['Close']
            
            result += f"📊 **Moving Averages**:\n"
            result += f"   SMA({ma_short}): ${sma_short:.2f}\n"
            result += f"   SMA({ma_long}): ${sma_long:.2f}\n"
            result += f"   EMA({ma_short}): ${ema_short:.2f}\n"
            result += f"   EMA({ma_long}): ${ema_long:.2f}\n\n"
            
            # Trend Direction
            if sma_short > sma_long and ema_short > ema_long:
                trend_direction = "🟢 BULLISH"
                trend_desc = "Both short-term MAs above long-term MAs"
            elif sma_short < sma_long and ema_short < ema_long:
                trend_direction = "🔴 BEARISH"  
                trend_desc = "Both short-term MAs below long-term MAs"
            else:
                trend_direction = "🟡 MIXED"
                trend_desc = "Mixed signals from moving averages"
            
            result += f"🎯 **Trend Direction**: {trend_direction}\n"
            result += f"   {trend_desc}\n\n"
            
            # Price vs MA Analysis
            if price > sma_short > sma_long:
                ma_signal = "🟢 STRONG UPTREND"
            elif price < sma_short < sma_long:
                ma_signal = "🔴 STRONG DOWNTREND"
            elif price > sma_short and sma_short < sma_long:
                ma_signal = "🟡 POTENTIAL REVERSAL UP"
            elif price < sma_short and sma_short > sma_long:
                ma_signal = "🟡 POTENTIAL REVERSAL DOWN"
            else:
                ma_signal = "🟡 SIDEWAYS TREND"
            
            result += f"📈 **Price Position**: {ma_signal}\n\n"
            
            # ADX Analysis (Trend Strength)
            adx_value = latest.get('ADX_14', 0)
            if adx_value >= 25:
                adx_strength = "💪 STRONG"
            elif adx_value >= 20:
                adx_strength = "📈 MODERATE"
            else:
                adx_strength = "😴 WEAK"
            
            result += f"💪 **ADX (Trend Strength)**: {adx_value:.1f} - {adx_strength}\n"
            
            # Aroon Analysis
            aroon_up = latest.get('AROONU_14', 0)
            aroon_down = latest.get('AROOND_14', 0)
            
            if aroon_up > 70 and aroon_down < 30:
                aroon_signal = "🟢 STRONG UPTREND"
            elif aroon_down > 70 and aroon_up < 30:
                aroon_signal = "🔴 STRONG DOWNTREND"
            elif abs(aroon_up - aroon_down) < 20:
                aroon_signal = "🟡 CONSOLIDATION"
            else:
                aroon_signal = "🟡 MIXED"
            
            result += f"🎯 **Aroon**: Up: {aroon_up:.1f} | Down: {aroon_down:.1f} - {aroon_signal}\n\n"
            
            # Overall trend assessment
            result += "💡 **Trend Summary**:\n"
            
            trend_factors = []
            if sma_short > sma_long: trend_factors.append("SMA Bullish")
            if ema_short > ema_long: trend_factors.append("EMA Bullish")
            if adx_value >= 25: trend_factors.append("Strong Trend")
            if aroon_up > aroon_down: trend_factors.append("Aroon Bullish")
            
            bullish_factors = len([f for f in trend_factors if "Bullish" in f or "Strong" in f])
            
            if bullish_factors >= 3:
                result += "🟢 **CONFIRMED UPTREND** - Multiple bullish confirmations\n"
            elif bullish_factors <= 1:
                result += "🔴 **CONFIRMED DOWNTREND** - Limited bullish signals\n"
            else:
                result += "🟡 **UNCERTAIN TREND** - Mixed trend signals\n"
            
            if trend_factors:
                result += f"📊 Active Signals: {', '.join(trend_factors)}\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error calculating trend indicators: {str(e)}"
    
    def _analyze_volatility(self, df: pd.DataFrame, symbol: str, timeframe: str, bb_length: int, bb_std: float) -> str:
        """Analyze volatility indicators"""
        try:
            # Calculate volatility indicators
            bb = ta.bbands(df['Close'], length=bb_length, std=bb_std)
            df = pd.concat([df, bb], axis=1)
            
            # Average True Range
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
            
            # Keltner Channels
            kc = ta.kc(df['High'], df['Low'], df['Close'])
            df = pd.concat([df, kc], axis=1)
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            result = f"📊 **VOLATILITY ANALYSIS** - {symbol} ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"💰 Current Price: ${latest['Close']:.2f}\n\n"
            
            # Bollinger Bands Analysis
            bb_upper = latest.get(f'BBU_{bb_length}_{bb_std}', 0)
            bb_middle = latest.get(f'BBM_{bb_length}_{bb_std}', 0)
            bb_lower = latest.get(f'BBL_{bb_length}_{bb_std}', 0)
            bb_width = latest.get(f'BBB_{bb_length}_{bb_std}', 0)
            bb_percent = latest.get(f'BBP_{bb_length}_{bb_std}', 0)
            
            price = latest['Close']
            
            result += f"🎯 **Bollinger Bands ({bb_length}, {bb_std})**:\n"
            result += f"   Upper: ${bb_upper:.2f}\n"
            result += f"   Middle: ${bb_middle:.2f}\n"
            result += f"   Lower: ${bb_lower:.2f}\n"
            result += f"   Width: {bb_width:.4f}\n"
            result += f"   %B: {bb_percent:.2f}\n\n"
            
            # BB Position Analysis
            if bb_percent >= 1.0:
                bb_signal = "🔴 ABOVE UPPER BAND - Overbought"
            elif bb_percent <= 0.0:
                bb_signal = "🟢 BELOW LOWER BAND - Oversold"
            elif bb_percent >= 0.8:
                bb_signal = "🟡 NEAR UPPER BAND - High"
            elif bb_percent <= 0.2:
                bb_signal = "🟡 NEAR LOWER BAND - Low"
            else:
                bb_signal = "🟡 MIDDLE RANGE - Normal"
            
            result += f"📍 **BB Position**: {bb_signal}\n\n"
            
            # ATR Analysis
            atr_current = latest['ATR']
            atr_prev = prev['ATR']
            atr_change = ((atr_current - atr_prev) / atr_prev) * 100 if atr_prev != 0 else 0
            atr_trend = "↗️" if atr_change > 0 else "↘️"
            
            # ATR as percentage of price
            atr_percent = (atr_current / price) * 100
            
            if atr_percent >= 3.0:
                atr_level = "🔥 HIGH"
            elif atr_percent >= 1.5:
                atr_level = "📈 MODERATE"
            else:
                atr_level = "😴 LOW"
            
            result += f"📏 **Average True Range**: ${atr_current:.2f} {atr_trend}\n"
            result += f"   ATR as % of Price: {atr_percent:.1f}% - {atr_level} Volatility\n"
            result += f"   24h Change: {atr_change:+.1f}%\n\n"
            
            # Keltner Channels Analysis
            kc_upper = latest.get('KCUe_20_2', 0)
            kc_middle = latest.get('KCBe_20_2', 0)  
            kc_lower = latest.get('KCLe_20_2', 0)
            
            if price > kc_upper:
                kc_signal = "🔴 ABOVE KC UPPER - Strong Uptrend"
            elif price < kc_lower:
                kc_signal = "🟢 BELOW KC LOWER - Strong Downtrend"
            else:
                kc_signal = "🟡 WITHIN KC - Normal Range"
            
            result += f"🎯 **Keltner Channels**:\n"
            result += f"   Upper: ${kc_upper:.2f}\n"
            result += f"   Middle: ${kc_middle:.2f}\n"
            result += f"   Lower: ${kc_lower:.2f}\n"
            result += f"   Signal: {kc_signal}\n\n"
            
            # Squeeze Detection (BB inside KC)
            if bb_upper < kc_upper and bb_lower > kc_lower:
                squeeze_status = "🤏 SQUEEZE ACTIVE - Low volatility, potential breakout"
            else:
                squeeze_status = "📈 NO SQUEEZE - Normal volatility"
            
            result += f"🎪 **Volatility Squeeze**: {squeeze_status}\n\n"
            
            # Overall volatility assessment
            result += "💡 **Volatility Summary**:\n"
            
            vol_signals = []
            if atr_percent >= 2.5: vol_signals.append("High ATR")
            if bb_percent >= 0.8 or bb_percent <= 0.2: vol_signals.append("BB Extreme")
            if price > kc_upper or price < kc_lower: vol_signals.append("KC Breakout")
            if "SQUEEZE" in squeeze_status: vol_signals.append("Squeeze")
            
            if "High ATR" in vol_signals and ("BB Extreme" in vol_signals or "KC Breakout" in vol_signals):
                result += "🔥 **HIGH VOLATILITY ENVIRONMENT** - Increased risk/reward\n"
            elif "Squeeze" in vol_signals:
                result += "🤏 **LOW VOLATILITY SQUEEZE** - Potential explosive move ahead\n"
            else:
                result += "📊 **NORMAL VOLATILITY** - Standard trading conditions\n"
            
            if vol_signals:
                result += f"📊 Active Signals: {', '.join(vol_signals)}\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error calculating volatility indicators: {str(e)}"
    
    def _analyze_volume(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """Analyze volume indicators"""
        try:
            # Calculate volume indicators
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
            df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
            df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'])
            df['VWMA'] = ta.vwma(df['Close'], df['Volume'])
            df['Volume_SMA'] = ta.sma(df['Volume'], length=20)
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            result = f"📊 **VOLUME ANALYSIS** - {symbol} ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"💰 Current Price: ${latest['Close']:.2f}\n\n"
            
            # Volume Analysis
            current_volume = latest['Volume']
            avg_volume = latest['Volume_SMA']
            volume_ratio = current_volume / avg_volume if avg_volume != 0 else 1
            
            if volume_ratio >= 2.0:
                volume_signal = "🔥 VERY HIGH"
            elif volume_ratio >= 1.5:
                volume_signal = "📈 HIGH"
            elif volume_ratio >= 0.5:
                volume_signal = "📊 NORMAL"
            else:
                volume_signal = "📉 LOW"
            
            result += f"📈 **Volume Analysis**:\n"
            result += f"   Current: {current_volume:,.0f}\n"
            result += f"   20-Day Avg: {avg_volume:,.0f}\n"
            result += f"   Ratio: {volume_ratio:.1f}x - {volume_signal}\n\n"
            
            # OBV Analysis
            obv_current = latest['OBV']
            obv_prev = prev['OBV']
            obv_trend = "↗️" if obv_current > obv_prev else "↘️"
            
            result += f"📊 **On-Balance Volume (OBV)**:\n"
            result += f"   Current: {obv_current:,.0f} {obv_trend}\n"
            
            # Compare price trend vs OBV trend
            price_trend = "up" if latest['Close'] > prev['Close'] else "down"
            obv_trend_direction = "up" if obv_current > obv_prev else "down"
            
            if price_trend == obv_trend_direction:
                obv_signal = "🟢 CONFIRMED - Volume supports price"
            else:
                obv_signal = "🟡 DIVERGENCE - Volume conflicts with price"
            
            result += f"   Signal: {obv_signal}\n\n"
            
            # Money Flow Index Analysis
            mfi = latest['MFI']
            
            if mfi >= 80:
                mfi_signal = "🔴 OVERBOUGHT"
            elif mfi <= 20:
                mfi_signal = "🟢 OVERSOLD"
            else:
                mfi_signal = "🟡 NEUTRAL"
            
            result += f"💰 **Money Flow Index**: {mfi:.1f} - {mfi_signal}\n"
            
            # Chaikin Money Flow Analysis
            cmf = latest['CMF']
            
            if cmf > 0.1:
                cmf_signal = "🟢 BUYING PRESSURE"
            elif cmf < -0.1:
                cmf_signal = "🔴 SELLING PRESSURE"
            else:
                cmf_signal = "🟡 NEUTRAL"
            
            result += f"📊 **Chaikin Money Flow**: {cmf:.3f} - {cmf_signal}\n\n"
            
            # VWMA vs Price Analysis
            vwma = latest['VWMA']
            price = latest['Close']
            
            if price > vwma:
                vwma_signal = "🟢 ABOVE VWMA - Bullish volume profile"
            else:
                vwma_signal = "🔴 BELOW VWMA - Bearish volume profile"
            
            result += f"📈 **Volume Weighted MA**: ${vwma:.2f}\n"
            result += f"   Signal: {vwma_signal}\n\n"
            
            # Overall volume assessment
            result += "💡 **Volume Summary**:\n"
            
            volume_signals = []
            if volume_ratio >= 1.5: volume_signals.append("High Volume")
            if "CONFIRMED" in obv_signal: volume_signals.append("OBV Confirms")
            if mfi <= 20: volume_signals.append("MFI Oversold")
            if mfi >= 80: volume_signals.append("MFI Overbought")
            if cmf > 0.1: volume_signals.append("Buying Pressure")
            if cmf < -0.1: volume_signals.append("Selling Pressure")
            
            bullish_volume = sum([
                volume_ratio >= 1.5,
                "CONFIRMED" in obv_signal and price_trend == "up",
                mfi <= 20,
                cmf > 0.1,
                price > vwma
            ])
            
            if bullish_volume >= 3:
                result += "🟢 **BULLISH VOLUME PROFILE** - Strong buying interest\n"
            elif bullish_volume <= 1:
                result += "🔴 **BEARISH VOLUME PROFILE** - Selling pressure evident\n"
            else:
                result += "🟡 **MIXED VOLUME SIGNALS** - No clear volume direction\n"
            
            if volume_signals:
                result += f"📊 Active Signals: {', '.join(volume_signals)}\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error calculating volume indicators: {str(e)}"
    
    def _analyze_oscillators(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """Analyze oscillator indicators"""
        try:
            # Calculate oscillators
            df['WILLR'] = ta.willr(df['High'], df['Low'], df['Close'])
            df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'])
            df['ROC'] = ta.roc(df['Close'], length=12)
            df['TSI'] = ta.tsi(df['Close'])
            
            # Ultimate Oscillator
            df['UO'] = ta.uo(df['High'], df['Low'], df['Close'])
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            result = f"🎢 **OSCILLATOR ANALYSIS** - {symbol} ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"💰 Current Price: ${latest['Close']:.2f}\n\n"
            
            # Williams %R Analysis
            willr = latest['WILLR']
            if willr >= -20:
                willr_signal = "🔴 OVERBOUGHT"
            elif willr <= -80:
                willr_signal = "🟢 OVERSOLD"
            else:
                willr_signal = "🟡 NEUTRAL"
            
            result += f"📉 **Williams %R**: {willr:.1f} - {willr_signal}\n"
            
            # Commodity Channel Index Analysis
            cci = latest['CCI']
            if cci >= 100:
                cci_signal = "🔴 OVERBOUGHT"
            elif cci <= -100:
                cci_signal = "🟢 OVERSOLD"
            else:
                cci_signal = "🟡 NORMAL RANGE"
            
            result += f"📊 **CCI (20)**: {cci:.1f} - {cci_signal}\n"
            
            # Rate of Change Analysis
            roc = latest['ROC']
            roc_trend = "↗️" if roc > 0 else "↘️"
            
            if abs(roc) >= 10:
                roc_strength = "💪 STRONG"
            elif abs(roc) >= 5:
                roc_strength = "📈 MODERATE"
            else:
                roc_strength = "😴 WEAK"
            
            result += f"🎯 **Rate of Change (12)**: {roc:.2f}% {roc_trend} - {roc_strength}\n"
            
            # True Strength Index Analysis
            tsi = latest.get('TSI_25_13_25', 0)
            if tsi >= 25:
                tsi_signal = "🔴 OVERBOUGHT"
            elif tsi <= -25:
                tsi_signal = "🟢 OVERSOLD"
            else:
                tsi_signal = "🟡 NEUTRAL"
            
            result += f"📈 **True Strength Index**: {tsi:.2f} - {tsi_signal}\n"
            
            # Ultimate Oscillator Analysis
            uo = latest['UO_7_14_28']
            if uo >= 70:
                uo_signal = "🔴 OVERBOUGHT"
            elif uo <= 30:
                uo_signal = "🟢 OVERSOLD"
            else:
                uo_signal = "🟡 NEUTRAL"
            
            result += f"🎯 **Ultimate Oscillator**: {uo:.1f} - {uo_signal}\n\n"
            
            # Oscillator Consensus
            result += "💡 **Oscillator Consensus**:\n"
            
            oversold_count = sum([
                willr <= -80,
                cci <= -100,
                tsi <= -25,
                uo <= 30
            ])
            
            overbought_count = sum([
                willr >= -20,
                cci >= 100,
                tsi >= 25,
                uo >= 70
            ])
            
            if oversold_count >= 3:
                result += "🟢 **STRONG OVERSOLD CONSENSUS** - Potential buying opportunity\n"
            elif overbought_count >= 3:
                result += "🔴 **STRONG OVERBOUGHT CONSENSUS** - Potential selling opportunity\n"
            elif oversold_count >= 2:
                result += "🟢 **MODERATE OVERSOLD** - Some buying signals\n"
            elif overbought_count >= 2:
                result += "🔴 **MODERATE OVERBOUGHT** - Some selling signals\n"
            else:
                result += "🟡 **MIXED OSCILLATOR SIGNALS** - No clear consensus\n"
            
            # Momentum strength
            momentum_strength = abs(roc)
            if momentum_strength >= 10:
                result += f"💪 **STRONG MOMENTUM** - {roc:+.1f}% price change\n"
            elif momentum_strength >= 5:
                result += f"📈 **MODERATE MOMENTUM** - {roc:+.1f}% price change\n"
            else:
                result += f"😴 **LOW MOMENTUM** - {roc:+.1f}% price change\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error calculating oscillator indicators: {str(e)}"
    
    def _analyze_all_indicators(self, df: pd.DataFrame, symbol: str, timeframe: str, rsi_length: int, ma_short: int, ma_long: int, bb_length: int, bb_std: float) -> str:
        """Comprehensive analysis of all technical indicators"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            result = f"🎯 **COMPREHENSIVE TECHNICAL ANALYSIS** - {symbol} ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"💰 Current Price: ${df.iloc[-1]['Close']:.2f}\n\n"
            
            # Run all individual analyses
            momentum_analysis = self._analyze_momentum(df, symbol, timeframe, rsi_length, 12, 26, 9)
            trend_analysis = self._analyze_trend(df, symbol, timeframe, ma_short, ma_long)
            volatility_analysis = self._analyze_volatility(df, symbol, timeframe, bb_length, bb_std)
            volume_analysis = self._analyze_volume(df, symbol, timeframe)
            oscillator_analysis = self._analyze_oscillators(df, symbol, timeframe)
            
            # Extract summary lines from each analysis
            result += "📊 **TECHNICAL SUMMARY BY CATEGORY**:\n\n"
            
            # Extract key signals from each analysis
            momentum_summary = [line for line in momentum_analysis.split('\n') if 'MOMENTUM' in line and ('BULLISH' in line or 'BEARISH' in line or 'MIXED' in line)]
            trend_summary = [line for line in trend_analysis.split('\n') if 'UPTREND' in line or 'DOWNTREND' in line or 'UNCERTAIN' in line]
            volume_summary = [line for line in volume_analysis.split('\n') if 'VOLUME PROFILE' in line]
            
            if momentum_summary:
                result += f"📈 Momentum: {momentum_summary[0].split('**')[1]}\n"
            if trend_summary:
                result += f"📊 Trend: {trend_summary[0].split('**')[1]}\n"  
            if volume_summary:
                result += f"📊 Volume: {volume_summary[0].split('**')[1]}\n"
            
            result += "\n" + "="*60 + "\n\n"
            
            # Add detailed analyses
            result += momentum_analysis + "\n\n" + "="*40 + "\n\n"
            result += trend_analysis + "\n\n" + "="*40 + "\n\n" 
            result += volatility_analysis + "\n\n" + "="*40 + "\n\n"
            result += volume_analysis + "\n\n" + "="*40 + "\n\n"
            result += oscillator_analysis + "\n\n"
            
            # Overall technical score
            result += "="*60 + "\n"
            result += "🎯 **OVERALL TECHNICAL ASSESSMENT**\n"
            result += "="*60 + "\n\n"
            
            # Count bullish/bearish signals across all categories
            bullish_signals = momentum_analysis.count('BULLISH') + trend_analysis.count('BULLISH') + volume_analysis.count('BULLISH')
            bearish_signals = momentum_analysis.count('BEARISH') + trend_analysis.count('BEARISH') + volume_analysis.count('BEARISH')
            neutral_signals = momentum_analysis.count('MIXED') + momentum_analysis.count('NEUTRAL') + trend_analysis.count('MIXED') + trend_analysis.count('UNCERTAIN')
            
            total_signals = bullish_signals + bearish_signals + neutral_signals
            
            if total_signals > 0:
                bullish_pct = (bullish_signals / total_signals) * 100
                bearish_pct = (bearish_signals / total_signals) * 100
                
                result += f"📊 **Signal Distribution**:\n"
                result += f"   🟢 Bullish: {bullish_signals} signals ({bullish_pct:.0f}%)\n"
                result += f"   🔴 Bearish: {bearish_signals} signals ({bearish_pct:.0f}%)\n"
                result += f"   🟡 Neutral/Mixed: {neutral_signals} signals\n\n"
                
                if bullish_pct >= 60:
                    overall_bias = "🟢 **BULLISH BIAS** - Multiple positive technical signals"
                elif bearish_pct >= 60:
                    overall_bias = "🔴 **BEARISH BIAS** - Multiple negative technical signals"
                else:
                    overall_bias = "🟡 **MIXED SIGNALS** - No clear technical direction"
                
                result += f"🎯 **Technical Bias**: {overall_bias}\n\n"
            
            result += "⚠️ **Risk Disclaimer**: Technical analysis is for educational purposes only. "
            result += "Always consider fundamental analysis, market conditions, and risk management.\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error performing comprehensive analysis: {str(e)}"


def create_technical_indicators_tool() -> TechnicalIndicatorsTool:
    """Create and return the technical indicators tool instance"""
    return TechnicalIndicatorsTool()