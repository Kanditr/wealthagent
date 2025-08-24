"""
Technical Analysis Signals Tool for LangChain Integration
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
    logger.warning("pandas-ta-classic not installed. Technical signals features will be disabled.")

try:
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca-py not available for data retrieval.")


class TechnicalSignalsInput(BaseModel):
    """Input schema for technical signals tool"""
    action: str = Field(description="Action: 'buy_signals', 'sell_signals', 'crossover_analysis', 'divergence_detection', 'signal_strength'")
    symbol: str = Field(description="Stock symbol (e.g., 'AAPL', 'TSLA')")
    timeframe: Optional[str] = Field(default="1Day", description="Timeframe: '1Min', '5Min', '15Min', '30Min', '1Hour', '1Day'")
    period: Optional[int] = Field(default=100, description="Number of periods to analyze (50-200)")
    sensitivity: Optional[str] = Field(default="medium", description="Signal sensitivity: 'low', 'medium', 'high'")


class TechnicalSignalsTool(BaseTool):
    """Tool for generating technical analysis trading signals"""
    
    name: str = "technical_signals"
    description: str = (
        "Generate technical trading signals and analysis. "
        "Actions: 'buy_signals' (bullish setups), 'sell_signals' (bearish setups), "
        "'crossover_analysis' (MA/MACD crossovers), 'divergence_detection' (price/indicator divergences), "
        "'signal_strength' (multi-indicator consensus scoring)."
    )
    args_schema: Type[BaseModel] = TechnicalSignalsInput
    
    def _get_alpaca_data(self, symbol: str, timeframe: str, period: int) -> Optional[pd.DataFrame]:
        """Get historical data from Alpaca for signal analysis"""
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
                start_date = end_date - timedelta(days=max(10, period // 100))
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
                return df.tail(period + 30)  # Add buffer for indicators
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting Alpaca data: {e}")
            return None
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators needed for signal analysis"""
        try:
            # Moving Averages
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['EMA_12'] = ta.ema(df['Close'], length=12)
            df['EMA_26'] = ta.ema(df['Close'], length=26)
            
            # Momentum indicators
            df['RSI'] = ta.rsi(df['Close'], length=14)
            macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
            df = pd.concat([df, macd], axis=1)
            
            # Stochastic
            stoch = ta.stoch(df['High'], df['Low'], df['Close'])
            df = pd.concat([df, stoch], axis=1)
            
            # Bollinger Bands
            bb = ta.bbands(df['Close'], length=20, std=2.0)
            df = pd.concat([df, bb], axis=1)
            
            # Volume indicators
            df['OBV'] = ta.obv(df['Close'], df['Volume'])
            df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'])
            
            # Volatility
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
            
            # Trend strength
            adx = ta.adx(df['High'], df['Low'], df['Close'])
            df = pd.concat([df, adx], axis=1)
            
            # Additional oscillators
            df['WILLR'] = ta.willr(df['High'], df['Low'], df['Close'])
            df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def _run(
        self,
        action: str,
        symbol: str,
        timeframe: str = "1Day",
        period: int = 100,
        sensitivity: str = "medium"
    ) -> str:
        """Execute technical signals analysis"""
        if not PANDAS_TA_AVAILABLE:
            return "❌ pandas-ta-classic not available. Please install pandas-ta-classic to use technical signals."
        
        # Validate inputs
        period = max(50, min(period, 200))
        symbol = symbol.upper().strip()
        sensitivity = sensitivity.lower()
        
        if sensitivity not in ["low", "medium", "high"]:
            sensitivity = "medium"
        
        try:
            # Get historical data
            df = self._get_alpaca_data(symbol, timeframe, period)
            
            if df is None or len(df) < 30:
                return f"❌ Insufficient historical data for {symbol}. Need at least 30 periods for signal analysis."
            
            # Calculate all indicators
            df = self._calculate_indicators(df)
            
            # Perform analysis based on action
            if action == "buy_signals":
                return self._analyze_buy_signals(df, symbol, timeframe, sensitivity)
            elif action == "sell_signals":
                return self._analyze_sell_signals(df, symbol, timeframe, sensitivity)
            elif action == "crossover_analysis":
                return self._analyze_crossovers(df, symbol, timeframe)
            elif action == "divergence_detection":
                return self._analyze_divergences(df, symbol, timeframe)
            elif action == "signal_strength":
                return self._analyze_signal_strength(df, symbol, timeframe, sensitivity)
            else:
                return f"❌ Unknown action: {action}. Available actions: buy_signals, sell_signals, crossover_analysis, divergence_detection, signal_strength"
        
        except Exception as e:
            logger.error(f"Technical signals error: {e}")
            return f"❌ Technical signals error: {str(e)}"
    
    def _analyze_buy_signals(self, df: pd.DataFrame, symbol: str, timeframe: str, sensitivity: str) -> str:
        """Analyze bullish trading signals"""
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            prev2 = df.iloc[-3]
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            result = f"🟢 **BUY SIGNALS ANALYSIS** - {symbol} ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"💰 Current Price: ${latest['Close']:.2f}\n"
            result += f"🎯 Sensitivity: {sensitivity.upper()}\n\n"
            
            buy_signals = []
            signal_scores = []
            
            # Set sensitivity thresholds
            if sensitivity == "high":
                rsi_oversold = 35
                stoch_oversold = 25
                bb_threshold = 0.3
            elif sensitivity == "low":
                rsi_oversold = 25
                stoch_oversold = 15
                bb_threshold = 0.1
            else:  # medium
                rsi_oversold = 30
                stoch_oversold = 20
                bb_threshold = 0.2
            
            # 1. RSI Oversold Recovery
            rsi_current = latest['RSI']
            rsi_prev = prev['RSI']
            
            if rsi_current > rsi_prev and rsi_prev <= rsi_oversold:
                buy_signals.append("🎯 RSI Oversold Recovery")
                signal_scores.append(75)
            elif rsi_current <= rsi_oversold:
                buy_signals.append("🟡 RSI Oversold Zone")
                signal_scores.append(60)
            
            # 2. MACD Bullish Crossover
            macd_line = latest.get('MACD_12_26_9', 0)
            macd_signal = latest.get('MACDs_12_26_9', 0)
            macd_prev_line = prev.get('MACD_12_26_9', 0)
            macd_prev_signal = prev.get('MACDs_12_26_9', 0)
            
            if macd_line > macd_signal and macd_prev_line <= macd_prev_signal:
                buy_signals.append("🚀 MACD Bullish Crossover")
                signal_scores.append(85)
            elif macd_line > macd_signal:
                buy_signals.append("🟢 MACD Above Signal")
                signal_scores.append(65)
            
            # 3. Moving Average Crossovers
            sma20 = latest['SMA_20']
            sma50 = latest['SMA_50']
            prev_sma20 = prev['SMA_20']
            prev_sma50 = prev['SMA_50']
            
            if sma20 > sma50 and prev_sma20 <= prev_sma50:
                buy_signals.append("📈 Golden Cross (SMA 20/50)")
                signal_scores.append(80)
            elif latest['Close'] > sma20 and prev['Close'] <= prev_sma20:
                buy_signals.append("🎯 Price Above SMA20")
                signal_scores.append(70)
            
            # 4. Bollinger Band Bounce
            bb_percent = latest.get('BBP_20_2.0', 0.5)
            bb_prev = prev.get('BBP_20_2.0', 0.5)
            
            if bb_percent > bb_prev and bb_prev <= bb_threshold:
                buy_signals.append("🎪 Bollinger Band Bounce")
                signal_scores.append(75)
            
            # 5. Stochastic Oversold Recovery
            stoch_k = latest.get('STOCHk_14_3_3', 50)
            stoch_k_prev = prev.get('STOCHk_14_3_3', 50)
            
            if stoch_k > stoch_k_prev and stoch_k_prev <= stoch_oversold:
                buy_signals.append("🎢 Stochastic Recovery")
                signal_scores.append(70)
            
            # 6. Volume Confirmation
            current_volume = latest['Volume']
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume != 0 else 1
            
            if volume_ratio >= 1.2:
                buy_signals.append("📊 Volume Confirmation")
                signal_scores.append(65)
            
            # 7. OBV Bullish Divergence
            obv_current = latest['OBV']
            obv_prev = prev['OBV']
            price_change = (latest['Close'] - prev['Close']) / prev['Close']
            obv_change = (obv_current - obv_prev) / abs(obv_prev) if obv_prev != 0 else 0
            
            if price_change < 0 and obv_change > 0:
                buy_signals.append("📈 OBV Bullish Divergence")
                signal_scores.append(80)
            
            # 8. Money Flow Index Recovery
            mfi = latest['MFI']
            mfi_prev = prev['MFI']
            
            if mfi > mfi_prev and mfi_prev <= 30:
                buy_signals.append("💰 MFI Oversold Recovery")
                signal_scores.append(70)
            
            # 9. Williams %R Recovery
            willr = latest['WILLR']
            willr_prev = prev['WILLR']
            
            if willr > willr_prev and willr_prev <= -80:
                buy_signals.append("📉 Williams %R Recovery")
                signal_scores.append(65)
            
            # 10. Price Pattern Signals
            close_prices = df['Close'].tail(5)
            if len(close_prices) >= 3:
                # Higher lows pattern
                if (close_prices.iloc[-1] > close_prices.iloc[-3] and 
                    close_prices.iloc[-2] > close_prices.iloc[-4]):
                    buy_signals.append("📊 Higher Lows Pattern")
                    signal_scores.append(60)
            
            # Present results
            if buy_signals:
                result += f"✅ **DETECTED {len(buy_signals)} BUY SIGNALS:**\n\n"
                
                for i, (signal, score) in enumerate(zip(buy_signals, signal_scores), 1):
                    strength = "🔥 STRONG" if score >= 80 else "📈 MODERATE" if score >= 70 else "🟡 WEAK"
                    result += f"{i}. {signal}\n"
                    result += f"   Strength: {score}/100 - {strength}\n\n"
                
                # Calculate overall signal strength
                avg_score = sum(signal_scores) / len(signal_scores) if signal_scores else 0
                signal_count_bonus = min(len(buy_signals) * 5, 25)  # Bonus for multiple signals
                total_score = min(avg_score + signal_count_bonus, 100)
                
                if total_score >= 80:
                    overall_signal = "🔥 VERY STRONG BUY"
                elif total_score >= 70:
                    overall_signal = "🟢 STRONG BUY"
                elif total_score >= 60:
                    overall_signal = "📈 MODERATE BUY"
                else:
                    overall_signal = "🟡 WEAK BUY"
                
                result += f"🎯 **OVERALL BUY SIGNAL**: {overall_signal}\n"
                result += f"📊 **Signal Strength**: {total_score:.0f}/100\n"
                result += f"📈 **Active Signals**: {len(buy_signals)}\n\n"
                
            else:
                result += "❌ **NO BUY SIGNALS DETECTED**\n\n"
                result += f"🔍 Current market conditions do not meet {sensitivity} sensitivity criteria for buy signals.\n\n"
            
            # Risk assessment
            result += "⚠️ **RISK ASSESSMENT**:\n"
            atr_percent = (latest['ATR'] / latest['Close']) * 100
            
            if atr_percent >= 3.0:
                risk_level = "🔴 HIGH VOLATILITY - Consider smaller position size"
            elif atr_percent >= 1.5:
                risk_level = "🟡 MODERATE VOLATILITY - Normal position sizing"
            else:
                risk_level = "🟢 LOW VOLATILITY - Stable conditions"
            
            result += f"📊 Volatility Risk: {risk_level}\n"
            
            # Stop loss suggestion
            if buy_signals:
                stop_loss = latest['Close'] - (2 * latest['ATR'])
                stop_loss_pct = ((latest['Close'] - stop_loss) / latest['Close']) * 100
                result += f"🛑 Suggested Stop Loss: ${stop_loss:.2f} ({stop_loss_pct:.1f}% below current)\n"
            
            result += "\n💡 **Note**: Signals are for educational purposes. Always consider fundamental analysis and risk management.\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error analyzing buy signals: {str(e)}"
    
    def _analyze_sell_signals(self, df: pd.DataFrame, symbol: str, timeframe: str, sensitivity: str) -> str:
        """Analyze bearish trading signals"""
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            prev2 = df.iloc[-3]
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            result = f"🔴 **SELL SIGNALS ANALYSIS** - {symbol} ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"💰 Current Price: ${latest['Close']:.2f}\n"
            result += f"🎯 Sensitivity: {sensitivity.upper()}\n\n"
            
            sell_signals = []
            signal_scores = []
            
            # Set sensitivity thresholds
            if sensitivity == "high":
                rsi_overbought = 65
                stoch_overbought = 75
                bb_threshold = 0.7
            elif sensitivity == "low":
                rsi_overbought = 75
                stoch_overbought = 85
                bb_threshold = 0.9
            else:  # medium
                rsi_overbought = 70
                stoch_overbought = 80
                bb_threshold = 0.8
            
            # 1. RSI Overbought Breakdown
            rsi_current = latest['RSI']
            rsi_prev = prev['RSI']
            
            if rsi_current < rsi_prev and rsi_prev >= rsi_overbought:
                sell_signals.append("🎯 RSI Overbought Breakdown")
                signal_scores.append(75)
            elif rsi_current >= rsi_overbought:
                sell_signals.append("🟡 RSI Overbought Zone")
                signal_scores.append(60)
            
            # 2. MACD Bearish Crossover
            macd_line = latest.get('MACD_12_26_9', 0)
            macd_signal = latest.get('MACDs_12_26_9', 0)
            macd_prev_line = prev.get('MACD_12_26_9', 0)
            macd_prev_signal = prev.get('MACDs_12_26_9', 0)
            
            if macd_line < macd_signal and macd_prev_line >= macd_prev_signal:
                sell_signals.append("📉 MACD Bearish Crossover")
                signal_scores.append(85)
            elif macd_line < macd_signal:
                sell_signals.append("🔴 MACD Below Signal")
                signal_scores.append(65)
            
            # 3. Moving Average Death Cross
            sma20 = latest['SMA_20']
            sma50 = latest['SMA_50']
            prev_sma20 = prev['SMA_20']
            prev_sma50 = prev['SMA_50']
            
            if sma20 < sma50 and prev_sma20 >= prev_sma50:
                sell_signals.append("💀 Death Cross (SMA 20/50)")
                signal_scores.append(80)
            elif latest['Close'] < sma20 and prev['Close'] >= prev_sma20:
                sell_signals.append("🎯 Price Below SMA20")
                signal_scores.append(70)
            
            # 4. Bollinger Band Rejection
            bb_percent = latest.get('BBP_20_2.0', 0.5)
            bb_prev = prev.get('BBP_20_2.0', 0.5)
            
            if bb_percent < bb_prev and bb_prev >= bb_threshold:
                sell_signals.append("🎪 Bollinger Band Rejection")
                signal_scores.append(75)
            
            # 5. Stochastic Overbought Breakdown
            stoch_k = latest.get('STOCHk_14_3_3', 50)
            stoch_k_prev = prev.get('STOCHk_14_3_3', 50)
            
            if stoch_k < stoch_k_prev and stoch_k_prev >= stoch_overbought:
                sell_signals.append("🎢 Stochastic Breakdown")
                signal_scores.append(70)
            
            # 6. Volume Selling Pressure
            current_volume = latest['Volume']
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume != 0 else 1
            price_change = (latest['Close'] - prev['Close']) / prev['Close']
            
            if volume_ratio >= 1.2 and price_change < -0.01:  # High volume + price drop
                sell_signals.append("📊 Volume Selling Pressure")
                signal_scores.append(75)
            
            # 7. OBV Bearish Divergence
            obv_current = latest['OBV']
            obv_prev = prev['OBV']
            price_change = (latest['Close'] - prev['Close']) / prev['Close']
            obv_change = (obv_current - obv_prev) / abs(obv_prev) if obv_prev != 0 else 0
            
            if price_change > 0 and obv_change < 0:
                sell_signals.append("📉 OBV Bearish Divergence")
                signal_scores.append(80)
            
            # 8. Money Flow Index Breakdown
            mfi = latest['MFI']
            mfi_prev = prev['MFI']
            
            if mfi < mfi_prev and mfi_prev >= 70:
                sell_signals.append("💰 MFI Overbought Breakdown")
                signal_scores.append(70)
            
            # 9. Williams %R Rejection
            willr = latest['WILLR']
            willr_prev = prev['WILLR']
            
            if willr < willr_prev and willr_prev >= -20:
                sell_signals.append("📈 Williams %R Rejection")
                signal_scores.append(65)
            
            # 10. Lower Highs Pattern
            close_prices = df['Close'].tail(5)
            if len(close_prices) >= 3:
                if (close_prices.iloc[-1] < close_prices.iloc[-3] and 
                    close_prices.iloc[-2] < close_prices.iloc[-4]):
                    sell_signals.append("📊 Lower Highs Pattern")
                    signal_scores.append(60)
            
            # 11. ADX Trend Weakness
            adx = latest.get('ADX_14', 25)
            adx_prev = prev.get('ADX_14', 25)
            
            if adx < 20 and adx < adx_prev and latest['Close'] < sma20:
                sell_signals.append("😴 Weak Trend + Below MA")
                signal_scores.append(60)
            
            # Present results
            if sell_signals:
                result += f"⚠️ **DETECTED {len(sell_signals)} SELL SIGNALS:**\n\n"
                
                for i, (signal, score) in enumerate(zip(sell_signals, signal_scores), 1):
                    strength = "🔥 STRONG" if score >= 80 else "📈 MODERATE" if score >= 70 else "🟡 WEAK"
                    result += f"{i}. {signal}\n"
                    result += f"   Strength: {score}/100 - {strength}\n\n"
                
                # Calculate overall signal strength
                avg_score = sum(signal_scores) / len(signal_scores) if signal_scores else 0
                signal_count_bonus = min(len(sell_signals) * 5, 25)
                total_score = min(avg_score + signal_count_bonus, 100)
                
                if total_score >= 80:
                    overall_signal = "🔥 VERY STRONG SELL"
                elif total_score >= 70:
                    overall_signal = "🔴 STRONG SELL"
                elif total_score >= 60:
                    overall_signal = "📉 MODERATE SELL"
                else:
                    overall_signal = "🟡 WEAK SELL"
                
                result += f"🎯 **OVERALL SELL SIGNAL**: {overall_signal}\n"
                result += f"📊 **Signal Strength**: {total_score:.0f}/100\n"
                result += f"📉 **Active Signals**: {len(sell_signals)}\n\n"
                
            else:
                result += "✅ **NO SELL SIGNALS DETECTED**\n\n"
                result += f"🔍 Current market conditions do not meet {sensitivity} sensitivity criteria for sell signals.\n\n"
            
            # Support levels
            if sell_signals:
                result += "📊 **SUPPORT LEVELS TO WATCH**:\n"
                support1 = latest['Close'] - latest['ATR']
                support2 = latest['SMA_20']
                support3 = latest['SMA_50']
                
                result += f"   Near-term: ${support1:.2f} (Price - ATR)\n"
                result += f"   SMA20: ${support2:.2f}\n"
                result += f"   SMA50: ${support3:.2f}\n\n"
            
            result += "\n💡 **Note**: Signals are for educational purposes. Consider taking profits or tightening stops based on your risk tolerance.\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error analyzing sell signals: {str(e)}"
    
    def _analyze_crossovers(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """Analyze moving average and MACD crossovers"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            result = f"✨ **CROSSOVER ANALYSIS** - {symbol} ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"💰 Current Price: ${df.iloc[-1]['Close']:.2f}\n\n"
            
            crossovers = []
            
            # Check last 5 periods for crossovers
            for i in range(1, min(6, len(df))):
                current = df.iloc[-i]
                previous = df.iloc[-(i+1)]
                
                # MA Crossovers
                sma20_curr = current['SMA_20']
                sma50_curr = current['SMA_50']
                sma20_prev = previous['SMA_20']
                sma50_prev = previous['SMA_50']
                
                # Golden Cross
                if sma20_curr > sma50_curr and sma20_prev <= sma50_prev:
                    days_ago = i
                    crossovers.append({
                        'type': 'Golden Cross',
                        'signal': '🟢 BULLISH',
                        'description': 'SMA20 crossed above SMA50',
                        'days_ago': days_ago,
                        'price': current['Close'],
                        'strength': 85
                    })
                
                # Death Cross
                elif sma20_curr < sma50_curr and sma20_prev >= sma50_prev:
                    days_ago = i
                    crossovers.append({
                        'type': 'Death Cross',
                        'signal': '🔴 BEARISH',
                        'description': 'SMA20 crossed below SMA50',
                        'days_ago': days_ago,
                        'price': current['Close'],
                        'strength': 85
                    })
                
                # MACD Crossovers
                macd_curr = current.get('MACD_12_26_9', 0)
                macd_signal_curr = current.get('MACDs_12_26_9', 0)
                macd_prev = previous.get('MACD_12_26_9', 0)
                macd_signal_prev = previous.get('MACDs_12_26_9', 0)
                
                # MACD Bullish Crossover
                if macd_curr > macd_signal_curr and macd_prev <= macd_signal_prev:
                    days_ago = i
                    crossovers.append({
                        'type': 'MACD Bullish',
                        'signal': '🟢 BULLISH',
                        'description': 'MACD crossed above signal line',
                        'days_ago': days_ago,
                        'price': current['Close'],
                        'strength': 80
                    })
                
                # MACD Bearish Crossover
                elif macd_curr < macd_signal_curr and macd_prev >= macd_signal_prev:
                    days_ago = i
                    crossovers.append({
                        'type': 'MACD Bearish',
                        'signal': '🔴 BEARISH',
                        'description': 'MACD crossed below signal line',
                        'days_ago': days_ago,
                        'price': current['Close'],
                        'strength': 80
                    })
            
            # Present crossover results
            if crossovers:
                result += f"🎯 **DETECTED {len(crossovers)} RECENT CROSSOVERS:**\n\n"
                
                # Sort by recency (days_ago)
                crossovers.sort(key=lambda x: x['days_ago'])
                
                for i, cross in enumerate(crossovers, 1):
                    time_desc = "today" if cross['days_ago'] == 1 else f"{cross['days_ago']} days ago"
                    result += f"{i}. **{cross['type']}** - {cross['signal']}\n"
                    result += f"   📅 Occurred: {time_desc}\n"
                    result += f"   💰 Price at crossover: ${cross['price']:.2f}\n"
                    result += f"   📊 {cross['description']}\n"
                    result += f"   🎯 Signal strength: {cross['strength']}/100\n\n"
            
            # Current crossover status
            latest = df.iloc[-1]
            result += "📊 **CURRENT CROSSOVER STATUS:**\n\n"
            
            # MA Status
            sma20 = latest['SMA_20']
            sma50 = latest['SMA_50']
            price = latest['Close']
            
            if sma20 > sma50:
                ma_status = "🟢 BULLISH (SMA20 > SMA50)"
            else:
                ma_status = "🔴 BEARISH (SMA20 < SMA50)"
            
            result += f"📈 **Moving Average Alignment**: {ma_status}\n"
            result += f"   Current Price: ${price:.2f}\n"
            result += f"   SMA20: ${sma20:.2f}\n"
            result += f"   SMA50: ${sma50:.2f}\n\n"
            
            # MACD Status
            macd_line = latest.get('MACD_12_26_9', 0)
            macd_signal = latest.get('MACDs_12_26_9', 0)
            macd_hist = latest.get('MACDh_12_26_9', 0)
            
            if macd_line > macd_signal:
                macd_status = "🟢 BULLISH (MACD > Signal)"
            else:
                macd_status = "🔴 BEARISH (MACD < Signal)"
            
            result += f"📊 **MACD Alignment**: {macd_status}\n"
            result += f"   MACD Line: {macd_line:.4f}\n"
            result += f"   Signal Line: {macd_signal:.4f}\n"
            result += f"   Histogram: {macd_hist:.4f}\n\n"
            
            # Potential upcoming crossovers
            result += "🔮 **POTENTIAL UPCOMING CROSSOVERS:**\n\n"
            
            # Check convergence/divergence
            ma_gap = abs(sma20 - sma50) / sma50 * 100
            macd_gap = abs(macd_line - macd_signal)
            
            if ma_gap < 1.0:
                if sma20 > sma50:
                    result += "⚠️ MA lines converging - Death Cross possible if momentum shifts\n"
                else:
                    result += "⚠️ MA lines converging - Golden Cross possible if momentum shifts\n"
            
            if macd_gap < 0.001:
                if macd_line > macd_signal:
                    result += "⚠️ MACD lines converging - Bearish crossover possible\n"
                else:
                    result += "⚠️ MACD lines converging - Bullish crossover possible\n"
            
            if ma_gap >= 1.0 and macd_gap >= 0.001:
                result += "✅ No immediate crossovers expected - Current trends likely to continue\n"
            
            result += "\n💡 **Note**: Crossovers work best when confirmed by volume and other technical indicators.\n"
            
            if not crossovers:
                result += "\n📊 **No recent crossovers detected** in the last 5 periods.\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error analyzing crossovers: {str(e)}"
    
    def _analyze_divergences(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """Analyze price/indicator divergences"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            result = f"🔍 **DIVERGENCE ANALYSIS** - {symbol} ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"💰 Current Price: ${df.iloc[-1]['Close']:.2f}\n\n"
            
            divergences = []
            
            # Look for divergences in the last 20 periods
            period = min(20, len(df))
            recent_df = df.tail(period).copy()
            
            # Find local highs and lows
            recent_df['High_Peak'] = recent_df['High'].rolling(window=3, center=True).apply(
                lambda x: x.iloc[1] == x.max(), raw=False
            )
            recent_df['Low_Trough'] = recent_df['Low'].rolling(window=3, center=True).apply(
                lambda x: x.iloc[1] == x.min(), raw=False
            )
            
            # Get peaks and troughs
            peaks = recent_df[recent_df['High_Peak'] == True]
            troughs = recent_df[recent_df['Low_Trough'] == True]
            
            # RSI Divergence Analysis
            if len(peaks) >= 2:
                # Bearish divergence: Price makes higher high, RSI makes lower high
                last_peak = peaks.iloc[-1]
                second_last_peak = peaks.iloc[-2]
                
                price_higher = last_peak['High'] > second_last_peak['High']
                rsi_lower = last_peak['RSI'] < second_last_peak['RSI']
                
                if price_higher and rsi_lower:
                    divergences.append({
                        'type': 'RSI Bearish Divergence',
                        'signal': '🔴 BEARISH',
                        'description': 'Price higher high, RSI lower high',
                        'strength': 80,
                        'indicator': 'RSI'
                    })
            
            if len(troughs) >= 2:
                # Bullish divergence: Price makes lower low, RSI makes higher low
                last_trough = troughs.iloc[-1]
                second_last_trough = troughs.iloc[-2]
                
                price_lower = last_trough['Low'] < second_last_trough['Low']
                rsi_higher = last_trough['RSI'] > second_last_trough['RSI']
                
                if price_lower and rsi_higher:
                    divergences.append({
                        'type': 'RSI Bullish Divergence',
                        'signal': '🟢 BULLISH',
                        'description': 'Price lower low, RSI higher low',
                        'strength': 80,
                        'indicator': 'RSI'
                    })
            
            # MACD Divergence Analysis
            if len(peaks) >= 2:
                last_peak = peaks.iloc[-1]
                second_last_peak = peaks.iloc[-2]
                
                price_higher = last_peak['High'] > second_last_peak['High']
                macd_lower = last_peak.get('MACD_12_26_9', 0) < second_last_peak.get('MACD_12_26_9', 0)
                
                if price_higher and macd_lower:
                    divergences.append({
                        'type': 'MACD Bearish Divergence',
                        'signal': '🔴 BEARISH',
                        'description': 'Price higher high, MACD lower high',
                        'strength': 85,
                        'indicator': 'MACD'
                    })
            
            if len(troughs) >= 2:
                last_trough = troughs.iloc[-1]
                second_last_trough = troughs.iloc[-2]
                
                price_lower = last_trough['Low'] < second_last_trough['Low']
                macd_higher = last_trough.get('MACD_12_26_9', 0) > second_last_trough.get('MACD_12_26_9', 0)
                
                if price_lower and macd_higher:
                    divergences.append({
                        'type': 'MACD Bullish Divergence',
                        'signal': '🟢 BULLISH',
                        'description': 'Price lower low, MACD higher low',
                        'strength': 85,
                        'indicator': 'MACD'
                    })
            
            # OBV Divergence Analysis
            if len(peaks) >= 2:
                last_peak = peaks.iloc[-1]
                second_last_peak = peaks.iloc[-2]
                
                price_higher = last_peak['High'] > second_last_peak['High']
                obv_lower = last_peak['OBV'] < second_last_peak['OBV']
                
                if price_higher and obv_lower:
                    divergences.append({
                        'type': 'OBV Bearish Divergence',
                        'signal': '🔴 BEARISH',
                        'description': 'Price higher high, OBV lower high',
                        'strength': 75,
                        'indicator': 'Volume (OBV)'
                    })
            
            if len(troughs) >= 2:
                last_trough = troughs.iloc[-1]
                second_last_trough = troughs.iloc[-2]
                
                price_lower = last_trough['Low'] < second_last_trough['Low']
                obv_higher = last_trough['OBV'] > second_last_trough['OBV']
                
                if price_lower and obv_higher:
                    divergences.append({
                        'type': 'OBV Bullish Divergence',
                        'signal': '🟢 BULLISH',
                        'description': 'Price lower low, OBV higher low',
                        'strength': 75,
                        'indicator': 'Volume (OBV)'
                    })
            
            # Present divergence results
            if divergences:
                result += f"⚠️ **DETECTED {len(divergences)} DIVERGENCES:**\n\n"
                
                for i, div in enumerate(divergences, 1):
                    result += f"{i}. **{div['type']}** - {div['signal']}\n"
                    result += f"   📊 Indicator: {div['indicator']}\n"
                    result += f"   📝 Pattern: {div['description']}\n"
                    result += f"   🎯 Strength: {div['strength']}/100\n\n"
                
                # Overall divergence assessment
                bullish_divs = [d for d in divergences if 'BULLISH' in d['signal']]
                bearish_divs = [d for d in divergences if 'BEARISH' in d['signal']]
                
                if len(bullish_divs) > len(bearish_divs):
                    overall = "🟢 **NET BULLISH DIVERGENCE** - Potential upward reversal"
                elif len(bearish_divs) > len(bullish_divs):
                    overall = "🔴 **NET BEARISH DIVERGENCE** - Potential downward reversal"
                else:
                    overall = "🟡 **MIXED DIVERGENCE SIGNALS** - Conflicting signals"
                
                result += f"🎯 **Overall Assessment**: {overall}\n\n"
                
            else:
                result += "✅ **NO DIVERGENCES DETECTED**\n\n"
                result += "Price and indicators are moving in harmony - no reversal signals from divergence analysis.\n\n"
            
            # Current momentum analysis
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            result += "📊 **CURRENT MOMENTUM ALIGNMENT:**\n\n"
            
            price_direction = "↗️ UP" if latest['Close'] > prev['Close'] else "↘️ DOWN"
            rsi_direction = "↗️ UP" if latest['RSI'] > prev['RSI'] else "↘️ DOWN"
            macd_direction = "↗️ UP" if latest.get('MACD_12_26_9', 0) > prev.get('MACD_12_26_9', 0) else "↘️ DOWN"
            obv_direction = "↗️ UP" if latest['OBV'] > prev['OBV'] else "↘️ DOWN"
            
            result += f"💰 Price: {price_direction}\n"
            result += f"📊 RSI: {rsi_direction}\n"
            result += f"📈 MACD: {macd_direction}\n"
            result += f"📊 OBV: {obv_direction}\n\n"
            
            # Alignment check
            directions = [price_direction, rsi_direction, macd_direction, obv_direction]
            up_count = sum('UP' in d for d in directions)
            
            if up_count >= 3:
                result += "✅ **STRONG ALIGNMENT** - Most indicators confirm price direction\n"
            elif up_count <= 1:
                result += "✅ **STRONG ALIGNMENT** - Most indicators confirm price direction\n"
            else:
                result += "⚠️ **MIXED SIGNALS** - Indicators showing conflicting directions\n"
            
            result += "\n💡 **Note**: Divergences can signal potential reversals but should be confirmed with other technical analysis.\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error analyzing divergences: {str(e)}"
    
    def _analyze_signal_strength(self, df: pd.DataFrame, symbol: str, timeframe: str, sensitivity: str) -> str:
        """Analyze overall signal strength using multi-indicator consensus"""
        try:
            latest = df.iloc[-1]
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            result = f"💪 **SIGNAL STRENGTH ANALYSIS** - {symbol} ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"💰 Current Price: ${latest['Close']:.2f}\n"
            result += f"🎯 Sensitivity: {sensitivity.upper()}\n\n"
            
            # Initialize scoring system
            bullish_score = 0
            bearish_score = 0
            max_score = 0
            indicator_scores = {}
            
            # 1. RSI Analysis (Weight: 15 points)
            rsi = latest['RSI']
            if rsi <= 30:
                rsi_score = 15
                bullish_score += rsi_score
                indicator_scores['RSI'] = f"🟢 +{rsi_score} (Oversold: {rsi:.1f})"
            elif rsi >= 70:
                rsi_score = 15
                bearish_score += rsi_score
                indicator_scores['RSI'] = f"🔴 -{rsi_score} (Overbought: {rsi:.1f})"
            elif 30 < rsi < 50:
                rsi_score = 10
                bullish_score += rsi_score
                indicator_scores['RSI'] = f"🟡 +{rsi_score} (Bearish: {rsi:.1f})"
            elif 50 < rsi < 70:
                rsi_score = 10
                bearish_score += rsi_score
                indicator_scores['RSI'] = f"🟡 -{rsi_score} (Bullish: {rsi:.1f})"
            else:
                indicator_scores['RSI'] = f"⚪ 0 (Neutral: {rsi:.1f})"
            
            max_score += 15
            
            # 2. MACD Analysis (Weight: 20 points)
            macd_line = latest.get('MACD_12_26_9', 0)
            macd_signal = latest.get('MACDs_12_26_9', 0)
            macd_hist = latest.get('MACDh_12_26_9', 0)
            
            if macd_line > macd_signal and macd_hist > 0:
                macd_score = 20
                bullish_score += macd_score
                indicator_scores['MACD'] = f"🟢 +{macd_score} (Bullish crossover)"
            elif macd_line < macd_signal and macd_hist < 0:
                macd_score = 20
                bearish_score += macd_score
                indicator_scores['MACD'] = f"🔴 -{macd_score} (Bearish crossover)"
            elif macd_line > macd_signal:
                macd_score = 10
                bullish_score += macd_score
                indicator_scores['MACD'] = f"🟡 +{macd_score} (Above signal)"
            elif macd_line < macd_signal:
                macd_score = 10
                bearish_score += macd_score
                indicator_scores['MACD'] = f"🟡 -{macd_score} (Below signal)"
            else:
                indicator_scores['MACD'] = "⚪ 0 (Neutral)"
            
            max_score += 20
            
            # 3. Moving Average Analysis (Weight: 15 points)
            price = latest['Close']
            sma20 = latest['SMA_20']
            sma50 = latest['SMA_50']
            
            if price > sma20 > sma50:
                ma_score = 15
                bullish_score += ma_score
                indicator_scores['MA'] = f"🟢 +{ma_score} (Price > SMA20 > SMA50)"
            elif price < sma20 < sma50:
                ma_score = 15
                bearish_score += ma_score
                indicator_scores['MA'] = f"🔴 -{ma_score} (Price < SMA20 < SMA50)"
            elif price > sma20:
                ma_score = 8
                bullish_score += ma_score
                indicator_scores['MA'] = f"🟡 +{ma_score} (Price > SMA20)"
            elif price < sma20:
                ma_score = 8
                bearish_score += ma_score
                indicator_scores['MA'] = f"🟡 -{ma_score} (Price < SMA20)"
            else:
                indicator_scores['MA'] = "⚪ 0 (Neutral)"
            
            max_score += 15
            
            # 4. Bollinger Bands Analysis (Weight: 10 points)
            bb_percent = latest.get('BBP_20_2.0', 0.5)
            
            if bb_percent <= 0.1:
                bb_score = 10
                bullish_score += bb_score
                indicator_scores['BB'] = f"🟢 +{bb_score} (Oversold: {bb_percent:.2f})"
            elif bb_percent >= 0.9:
                bb_score = 10
                bearish_score += bb_score
                indicator_scores['BB'] = f"🔴 -{bb_score} (Overbought: {bb_percent:.2f})"
            elif bb_percent <= 0.3:
                bb_score = 5
                bullish_score += bb_score
                indicator_scores['BB'] = f"🟡 +{bb_score} (Low: {bb_percent:.2f})"
            elif bb_percent >= 0.7:
                bb_score = 5
                bearish_score += bb_score
                indicator_scores['BB'] = f"🟡 -{bb_score} (High: {bb_percent:.2f})"
            else:
                indicator_scores['BB'] = f"⚪ 0 (Neutral: {bb_percent:.2f})"
            
            max_score += 10
            
            # 5. Stochastic Analysis (Weight: 10 points)
            stoch_k = latest.get('STOCHk_14_3_3', 50)
            
            if stoch_k <= 20:
                stoch_score = 10
                bullish_score += stoch_score
                indicator_scores['Stoch'] = f"🟢 +{stoch_score} (Oversold: {stoch_k:.1f})"
            elif stoch_k >= 80:
                stoch_score = 10
                bearish_score += stoch_score
                indicator_scores['Stoch'] = f"🔴 -{stoch_score} (Overbought: {stoch_k:.1f})"
            else:
                indicator_scores['Stoch'] = f"⚪ 0 (Neutral: {stoch_k:.1f})"
            
            max_score += 10
            
            # 6. Volume Analysis (Weight: 10 points)
            current_volume = latest['Volume']
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume != 0 else 1
            
            if volume_ratio >= 1.5:
                vol_score = 10
                # Determine direction based on price change
                prev_close = df.iloc[-2]['Close']
                price_change = (price - prev_close) / prev_close
                
                if price_change > 0:
                    bullish_score += vol_score
                    indicator_scores['Volume'] = f"🟢 +{vol_score} (High vol + up: {volume_ratio:.1f}x)"
                else:
                    bearish_score += vol_score
                    indicator_scores['Volume'] = f"🔴 -{vol_score} (High vol + down: {volume_ratio:.1f}x)"
            else:
                indicator_scores['Volume'] = f"⚪ 0 (Normal: {volume_ratio:.1f}x)"
            
            max_score += 10
            
            # 7. ADX Trend Strength (Weight: 10 points)
            adx = latest.get('ADX_14', 25)
            
            if adx >= 25:
                adx_score = 10
                # Determine trend direction
                if sma20 > sma50:
                    bullish_score += adx_score
                    indicator_scores['ADX'] = f"🟢 +{adx_score} (Strong uptrend: {adx:.1f})"
                else:
                    bearish_score += adx_score
                    indicator_scores['ADX'] = f"🔴 -{adx_score} (Strong downtrend: {adx:.1f})"
            else:
                indicator_scores['ADX'] = f"⚪ 0 (Weak trend: {adx:.1f})"
            
            max_score += 10
            
            # 8. Williams %R Analysis (Weight: 5 points)
            willr = latest['WILLR']
            
            if willr <= -80:
                willr_score = 5
                bullish_score += willr_score
                indicator_scores['WillR'] = f"🟢 +{willr_score} (Oversold: {willr:.1f})"
            elif willr >= -20:
                willr_score = 5
                bearish_score += willr_score
                indicator_scores['WillR'] = f"🔴 -{willr_score} (Overbought: {willr:.1f})"
            else:
                indicator_scores['WillR'] = f"⚪ 0 (Neutral: {willr:.1f})"
            
            max_score += 5
            
            # 9. Money Flow Index (Weight: 5 points)
            mfi = latest['MFI']
            
            if mfi <= 20:
                mfi_score = 5
                bullish_score += mfi_score
                indicator_scores['MFI'] = f"🟢 +{mfi_score} (Oversold: {mfi:.1f})"
            elif mfi >= 80:
                mfi_score = 5
                bearish_score += mfi_score
                indicator_scores['MFI'] = f"🔴 -{mfi_score} (Overbought: {mfi:.1f})"
            else:
                indicator_scores['MFI'] = f"⚪ 0 (Neutral: {mfi:.1f})"
            
            max_score += 5
            
            # Calculate final scores
            total_bullish_pct = (bullish_score / max_score) * 100 if max_score > 0 else 0
            total_bearish_pct = (bearish_score / max_score) * 100 if max_score > 0 else 0
            net_score = bullish_score - bearish_score
            
            result += "📊 **INDIVIDUAL INDICATOR SCORES:**\n\n"
            
            for indicator, score_desc in indicator_scores.items():
                result += f"   {indicator}: {score_desc}\n"
            
            result += f"\n📊 **COMPOSITE SCORING:**\n\n"
            result += f"🟢 Total Bullish Score: {bullish_score}/{max_score} ({total_bullish_pct:.1f}%)\n"
            result += f"🔴 Total Bearish Score: {bearish_score}/{max_score} ({total_bearish_pct:.1f}%)\n"
            result += f"⚖️ Net Score: {net_score:+d}\n\n"
            
            # Determine overall signal strength
            if net_score >= 30:
                signal_strength = "🔥 VERY STRONG BULLISH"
                confidence = "High"
            elif net_score >= 15:
                signal_strength = "🟢 STRONG BULLISH"
                confidence = "Moderate-High"
            elif net_score >= 5:
                signal_strength = "📈 MODERATE BULLISH"
                confidence = "Moderate"
            elif net_score <= -30:
                signal_strength = "💀 VERY STRONG BEARISH"
                confidence = "High"
            elif net_score <= -15:
                signal_strength = "🔴 STRONG BEARISH"
                confidence = "Moderate-High"
            elif net_score <= -5:
                signal_strength = "📉 MODERATE BEARISH"
                confidence = "Moderate"
            else:
                signal_strength = "🟡 MIXED/NEUTRAL"
                confidence = "Low"
            
            result += f"🎯 **OVERALL SIGNAL STRENGTH**: {signal_strength}\n"
            result += f"🎲 **Confidence Level**: {confidence}\n\n"
            
            # Trading recommendation based on sensitivity
            result += "💡 **TRADING IMPLICATIONS:**\n\n"
            
            if sensitivity == "high" and abs(net_score) >= 10:
                if net_score > 0:
                    result += "✅ **HIGH SENSITIVITY**: Consider LONG position\n"
                else:
                    result += "⚠️ **HIGH SENSITIVITY**: Consider SHORT position or exit longs\n"
            elif sensitivity == "medium" and abs(net_score) >= 15:
                if net_score > 0:
                    result += "✅ **MEDIUM SENSITIVITY**: Strong case for LONG position\n"
                else:
                    result += "⚠️ **MEDIUM SENSITIVITY**: Strong case for SHORT position\n"
            elif sensitivity == "low" and abs(net_score) >= 25:
                if net_score > 0:
                    result += "✅ **LOW SENSITIVITY**: Very strong case for LONG position\n"
                else:
                    result += "⚠️ **LOW SENSITIVITY**: Very strong case for SHORT position\n"
            else:
                result += "🟡 **HOLD**: Current signal strength doesn't meet your sensitivity criteria\n"
            
            result += f"\n📊 **Signal Distribution**:\n"
            active_indicators = len([score for score in indicator_scores.values() if not score.startswith('⚪')])
            result += f"   Active Signals: {active_indicators}/{len(indicator_scores)}\n"
            result += f"   Consensus Strength: {abs(net_score)}/100\n"
            
            result += "\n⚠️ **Risk Management**: Use appropriate position sizing and stop-losses based on signal strength and market volatility.\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error analyzing signal strength: {str(e)}"


def create_technical_signals_tool() -> TechnicalSignalsTool:
    """Create and return the technical signals tool instance"""
    return TechnicalSignalsTool()