"""
Technical Screening Tool for LangChain Integration
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
    logger.warning("pandas-ta-classic not installed. Technical screening features will be disabled.")

try:
    from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    logger.warning("Alpaca-py not available for data retrieval.")


class TechnicalScreeningInput(BaseModel):
    """Input schema for technical screening tool"""
    action: str = Field(description="Action: 'momentum_screen', 'oversold_screen', 'breakout_screen', 'custom_screen', 'multi_timeframe_screen'")
    symbols: str = Field(description="Comma-separated symbols to screen (e.g., 'AAPL,MSFT,TSLA')")
    timeframe: Optional[str] = Field(default="1Day", description="Timeframe: '1Day', '1Hour' (recommended for screening)")
    min_volume: Optional[int] = Field(default=1000000, description="Minimum daily volume filter")
    custom_criteria: Optional[str] = Field(default=None, description="Custom screening criteria (e.g., 'RSI<30,MACD>0')")


class TechnicalScreeningTool(BaseTool):
    """Tool for screening multiple symbols using technical analysis criteria"""
    
    name: str = "technical_screening"
    description: str = (
        "Screen multiple stocks using technical analysis criteria. "
        "Actions: 'momentum_screen' (high momentum stocks), 'oversold_screen' (oversold opportunities), "
        "'breakout_screen' (breakout setups), 'custom_screen' (user-defined criteria), "
        "'multi_timeframe_screen' (cross-timeframe analysis). Ranks and scores opportunities."
    )
    args_schema: Type[BaseModel] = TechnicalScreeningInput
    
    def _get_symbol_data(self, symbol: str, timeframe: str, period: int = 60) -> Optional[pd.DataFrame]:
        """Get historical data for a single symbol"""
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
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day
            }
            
            tf = timeframe_mapping.get(timeframe, TimeFrame.Day)
            
            # Calculate start date
            end_date = datetime.now().date()
            if timeframe == "1Hour":
                start_date = end_date - timedelta(days=max(21, period // 6))
            else:  # 1Day
                start_date = end_date - timedelta(days=max(120, period + 30))
            
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
                return df.tail(period + 10)
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting data for {symbol}: {e}")
            return None
    
    def _calculate_screening_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key metrics for screening"""
        try:
            if len(df) < 20:
                return {}
            
            # Calculate indicators
            df['RSI'] = ta.rsi(df['Close'], length=14)
            df['SMA_20'] = ta.sma(df['Close'], length=20)
            df['SMA_50'] = ta.sma(df['Close'], length=50)
            df['Volume_SMA'] = ta.sma(df['Volume'], length=20)
            
            # MACD
            macd = ta.macd(df['Close'])
            df = pd.concat([df, macd], axis=1)
            
            # Bollinger Bands
            bb = ta.bbands(df['Close'], length=20)
            df = pd.concat([df, bb], axis=1)
            
            # ATR
            df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'])
            
            # Stochastic
            stoch = ta.stoch(df['High'], df['Low'], df['Close'])
            df = pd.concat([df, stoch], axis=1)
            
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Calculate metrics
            metrics = {
                'symbol': None,  # Will be set by caller
                'price': latest['Close'],
                'volume': latest['Volume'],
                'avg_volume': latest['Volume_SMA'],
                'volume_ratio': latest['Volume'] / latest['Volume_SMA'] if latest['Volume_SMA'] != 0 else 1,
                
                # Technical indicators
                'rsi': latest['RSI'],
                'macd': latest.get('MACD_12_26_9', 0),
                'macd_signal': latest.get('MACDs_12_26_9', 0),
                'macd_hist': latest.get('MACDh_12_26_9', 0),
                'bb_percent': latest.get('BBP_20_2.0', 0.5),
                'stoch_k': latest.get('STOCHk_14_3_3', 50),
                'atr': latest['ATR'],
                
                # Price position
                'price_vs_sma20': (latest['Close'] - latest['SMA_20']) / latest['SMA_20'] * 100 if latest['SMA_20'] != 0 else 0,
                'price_vs_sma50': (latest['Close'] - latest['SMA_50']) / latest['SMA_50'] * 100 if latest['SMA_50'] != 0 else 0,
                'sma20_vs_sma50': (latest['SMA_20'] - latest['SMA_50']) / latest['SMA_50'] * 100 if latest['SMA_50'] != 0 else 0,
                
                # Momentum
                'price_change_1d': (latest['Close'] - prev['Close']) / prev['Close'] * 100,
                'price_change_5d': (latest['Close'] - df.iloc[-6]['Close']) / df.iloc[-6]['Close'] * 100 if len(df) >= 6 else 0,
                'price_change_20d': (latest['Close'] - df.iloc[-21]['Close']) / df.iloc[-21]['Close'] * 100 if len(df) >= 21 else 0,
                
                # Volatility
                'atr_percent': (latest['ATR'] / latest['Close']) * 100,
                
                # Range position
                'high_20d': df['High'].tail(20).max(),
                'low_20d': df['Low'].tail(20).min(),
            }
            
            # Calculate position in 20-day range
            if metrics['high_20d'] != metrics['low_20d']:
                metrics['range_position'] = (latest['Close'] - metrics['low_20d']) / (metrics['high_20d'] - metrics['low_20d']) * 100
            else:
                metrics['range_position'] = 50
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _run(
        self,
        action: str,
        symbols: str,
        timeframe: str = "1Day",
        min_volume: int = 1000000,
        custom_criteria: Optional[str] = None
    ) -> str:
        """Execute technical screening analysis"""
        if not PANDAS_TA_AVAILABLE:
            return "❌ pandas-ta-classic not available. Please install pandas-ta-classic to use technical screening."
        
        # Parse symbols
        symbol_list = [s.strip().upper() for s in symbols.split(',') if s.strip()]
        
        if not symbol_list:
            return "❌ No valid symbols provided. Please provide comma-separated symbols (e.g., 'AAPL,MSFT,TSLA')."
        
        if len(symbol_list) > 20:
            return "❌ Too many symbols (max 20). Please reduce the number of symbols for optimal performance."
        
        try:
            # Perform screening based on action
            if action == "momentum_screen":
                return self._momentum_screen(symbol_list, timeframe, min_volume)
            elif action == "oversold_screen":
                return self._oversold_screen(symbol_list, timeframe, min_volume)
            elif action == "breakout_screen":
                return self._breakout_screen(symbol_list, timeframe, min_volume)
            elif action == "custom_screen":
                return self._custom_screen(symbol_list, timeframe, min_volume, custom_criteria)
            elif action == "multi_timeframe_screen":
                return self._multi_timeframe_screen(symbol_list, min_volume)
            else:
                return f"❌ Unknown action: {action}. Available actions: momentum_screen, oversold_screen, breakout_screen, custom_screen, multi_timeframe_screen"
        
        except Exception as e:
            logger.error(f"Technical screening error: {e}")
            return f"❌ Technical screening error: {str(e)}"
    
    def _momentum_screen(self, symbols: List[str], timeframe: str, min_volume: int) -> str:
        """Screen for high momentum opportunities"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            result = f"🚀 **MOMENTUM SCREENING RESULTS** ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"📊 Symbols Analyzed: {len(symbols)}\n"
            result += f"📈 Min Volume Filter: {min_volume:,}\n\n"
            
            candidates = []
            
            for symbol in symbols:
                df = self._get_symbol_data(symbol, timeframe)
                if df is None or len(df) < 20:
                    continue
                
                metrics = self._calculate_screening_metrics(df)
                if not metrics:
                    continue
                
                metrics['symbol'] = symbol
                
                # Volume filter
                if metrics['avg_volume'] < min_volume:
                    continue
                
                # Momentum scoring (0-100)
                momentum_score = 0
                
                # Price momentum (30 points max)
                if metrics['price_change_1d'] >= 3:
                    momentum_score += 10
                elif metrics['price_change_1d'] >= 1:
                    momentum_score += 5
                
                if metrics['price_change_5d'] >= 10:
                    momentum_score += 10
                elif metrics['price_change_5d'] >= 5:
                    momentum_score += 5
                
                if metrics['price_change_20d'] >= 20:
                    momentum_score += 10
                elif metrics['price_change_20d'] >= 10:
                    momentum_score += 5
                
                # Technical momentum (40 points max)
                if metrics['rsi'] >= 60 and metrics['rsi'] <= 80:
                    momentum_score += 15
                elif metrics['rsi'] >= 50:
                    momentum_score += 10
                
                if metrics['macd'] > metrics['macd_signal'] and metrics['macd_hist'] > 0:
                    momentum_score += 15
                elif metrics['macd'] > metrics['macd_signal']:
                    momentum_score += 10
                
                if metrics['sma20_vs_sma50'] > 0:
                    momentum_score += 10
                
                # Volume confirmation (20 points max)
                if metrics['volume_ratio'] >= 2.0:
                    momentum_score += 20
                elif metrics['volume_ratio'] >= 1.5:
                    momentum_score += 15
                elif metrics['volume_ratio'] >= 1.2:
                    momentum_score += 10
                
                # Position strength (10 points max)
                if metrics['range_position'] >= 80:
                    momentum_score += 10
                elif metrics['range_position'] >= 60:
                    momentum_score += 5
                
                metrics['momentum_score'] = momentum_score
                
                # Only include candidates with score >= 50
                if momentum_score >= 50:
                    candidates.append(metrics)
            
            # Sort by momentum score
            candidates.sort(key=lambda x: x['momentum_score'], reverse=True)
            
            if candidates:
                result += f"✅ **FOUND {len(candidates)} MOMENTUM CANDIDATES:**\n\n"
                
                for i, candidate in enumerate(candidates[:10], 1):  # Top 10
                    score_desc = "🔥 VERY HIGH" if candidate['momentum_score'] >= 80 else "🚀 HIGH" if candidate['momentum_score'] >= 70 else "📈 MODERATE"
                    
                    result += f"**{i}. {candidate['symbol']}** - Score: {candidate['momentum_score']}/100 ({score_desc})\n"
                    result += f"   💰 Price: ${candidate['price']:.2f}\n"
                    result += f"   📊 1D: {candidate['price_change_1d']:+.1f}% | 5D: {candidate['price_change_5d']:+.1f}% | 20D: {candidate['price_change_20d']:+.1f}%\n"
                    result += f"   🎯 RSI: {candidate['rsi']:.1f} | Volume: {candidate['volume_ratio']:.1f}x\n"
                    result += f"   📈 Range Position: {candidate['range_position']:.0f}%\n\n"
                
            else:
                result += "❌ **NO MOMENTUM CANDIDATES FOUND**\n\n"
                result += "No symbols met the momentum screening criteria (score >= 50).\n"
            
            result += "💡 **Momentum Screen Criteria**: Strong price gains + technical confirmation + volume support\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error in momentum screening: {str(e)}"
    
    def _oversold_screen(self, symbols: List[str], timeframe: str, min_volume: int) -> str:
        """Screen for oversold opportunities"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            result = f"🟢 **OVERSOLD SCREENING RESULTS** ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"📊 Symbols Analyzed: {len(symbols)}\n"
            result += f"📈 Min Volume Filter: {min_volume:,}\n\n"
            
            candidates = []
            
            for symbol in symbols:
                df = self._get_symbol_data(symbol, timeframe)
                if df is None or len(df) < 20:
                    continue
                
                metrics = self._calculate_screening_metrics(df)
                if not metrics:
                    continue
                
                metrics['symbol'] = symbol
                
                # Volume filter
                if metrics['avg_volume'] < min_volume:
                    continue
                
                # Oversold scoring (0-100)
                oversold_score = 0
                
                # RSI oversold (25 points max)
                if metrics['rsi'] <= 25:
                    oversold_score += 25
                elif metrics['rsi'] <= 30:
                    oversold_score += 20
                elif metrics['rsi'] <= 35:
                    oversold_score += 15
                elif metrics['rsi'] <= 40:
                    oversold_score += 10
                
                # Stochastic oversold (20 points max)
                if metrics['stoch_k'] <= 15:
                    oversold_score += 20
                elif metrics['stoch_k'] <= 20:
                    oversold_score += 15
                elif metrics['stoch_k'] <= 25:
                    oversold_score += 10
                
                # Bollinger Bands oversold (15 points max)
                if metrics['bb_percent'] <= 0.1:
                    oversold_score += 15
                elif metrics['bb_percent'] <= 0.2:
                    oversold_score += 10
                elif metrics['bb_percent'] <= 0.3:
                    oversold_score += 5
                
                # Price decline severity (20 points max)
                if metrics['price_change_5d'] <= -10:
                    oversold_score += 20
                elif metrics['price_change_5d'] <= -5:
                    oversold_score += 15
                elif metrics['price_change_5d'] <= -2:
                    oversold_score += 10
                
                # Range position (10 points max)
                if metrics['range_position'] <= 20:
                    oversold_score += 10
                elif metrics['range_position'] <= 30:
                    oversold_score += 5
                
                # Volume confirmation for selling climax (10 points max)
                if metrics['volume_ratio'] >= 1.5 and metrics['price_change_1d'] < -2:
                    oversold_score += 10
                elif metrics['volume_ratio'] >= 1.2 and metrics['price_change_1d'] < -1:
                    oversold_score += 5
                
                metrics['oversold_score'] = oversold_score
                
                # Only include candidates with score >= 40
                if oversold_score >= 40:
                    candidates.append(metrics)
            
            # Sort by oversold score
            candidates.sort(key=lambda x: x['oversold_score'], reverse=True)
            
            if candidates:
                result += f"✅ **FOUND {len(candidates)} OVERSOLD CANDIDATES:**\n\n"
                
                for i, candidate in enumerate(candidates[:10], 1):  # Top 10
                    score_desc = "💎 DEEP OVERSOLD" if candidate['oversold_score'] >= 80 else "🟢 STRONG OVERSOLD" if candidate['oversold_score'] >= 70 else "📉 MODERATE OVERSOLD"
                    
                    result += f"**{i}. {candidate['symbol']}** - Score: {candidate['oversold_score']}/100 ({score_desc})\n"
                    result += f"   💰 Price: ${candidate['price']:.2f}\n"
                    result += f"   📊 1D: {candidate['price_change_1d']:+.1f}% | 5D: {candidate['price_change_5d']:+.1f}% | 20D: {candidate['price_change_20d']:+.1f}%\n"
                    result += f"   🎯 RSI: {candidate['rsi']:.1f} | Stoch: {candidate['stoch_k']:.1f} | BB%: {candidate['bb_percent']:.2f}\n"
                    result += f"   📉 Range Position: {candidate['range_position']:.0f}%\n\n"
                
                result += "🎯 **OVERSOLD TRADE SETUP**:\n"
                result += "- Look for reversal candlestick patterns\n"
                result += "- Wait for RSI to turn up from oversold levels\n"
                result += "- Use tight stops below recent lows\n"
                result += "- Consider scaling into positions\n\n"
                
            else:
                result += "❌ **NO OVERSOLD CANDIDATES FOUND**\n\n"
                result += "No symbols met the oversold screening criteria (score >= 40).\n"
            
            result += "💡 **Oversold Screen Criteria**: Multiple oversold indicators + price decline + low range position\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error in oversold screening: {str(e)}"
    
    def _breakout_screen(self, symbols: List[str], timeframe: str, min_volume: int) -> str:
        """Screen for breakout opportunities"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            result = f"🚀 **BREAKOUT SCREENING RESULTS** ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"📊 Symbols Analyzed: {len(symbols)}\n"
            result += f"📈 Min Volume Filter: {min_volume:,}\n\n"
            
            candidates = []
            
            for symbol in symbols:
                df = self._get_symbol_data(symbol, timeframe)
                if df is None or len(df) < 30:
                    continue
                
                metrics = self._calculate_screening_metrics(df)
                if not metrics:
                    continue
                
                metrics['symbol'] = symbol
                
                # Volume filter
                if metrics['avg_volume'] < min_volume:
                    continue
                
                # Calculate additional breakout metrics
                recent_high = df['High'].tail(20).max()
                recent_low = df['Low'].tail(20).min()
                consolidation_range = (recent_high - recent_low) / recent_low * 100
                
                # Distance to breakout levels
                upside_to_high = (recent_high - metrics['price']) / metrics['price'] * 100
                downside_to_low = (metrics['price'] - recent_low) / metrics['price'] * 100
                
                # Breakout scoring (0-100)
                breakout_score = 0
                
                # Consolidation quality (30 points max)
                if consolidation_range <= 5:
                    breakout_score += 30  # Tight consolidation
                elif consolidation_range <= 8:
                    breakout_score += 20
                elif consolidation_range <= 12:
                    breakout_score += 10
                
                # Proximity to breakout (25 points max)
                if upside_to_high <= 1:
                    breakout_score += 25  # Very close to resistance
                elif upside_to_high <= 2:
                    breakout_score += 20
                elif upside_to_high <= 3:
                    breakout_score += 15
                elif upside_to_high <= 5:
                    breakout_score += 10
                
                # Volume buildup (20 points max)
                if metrics['volume_ratio'] >= 1.5:
                    breakout_score += 20
                elif metrics['volume_ratio'] >= 1.3:
                    breakout_score += 15
                elif metrics['volume_ratio'] >= 1.1:
                    breakout_score += 10
                
                # Technical setup (15 points max)
                if metrics['rsi'] >= 50 and metrics['rsi'] <= 70:
                    breakout_score += 10
                
                if metrics['macd'] > metrics['macd_signal']:
                    breakout_score += 5
                
                # Position in range (10 points max)
                if metrics['range_position'] >= 75:
                    breakout_score += 10
                elif metrics['range_position'] >= 60:
                    breakout_score += 5
                
                metrics['breakout_score'] = breakout_score
                metrics['consolidation_range'] = consolidation_range
                metrics['upside_to_high'] = upside_to_high
                metrics['recent_high'] = recent_high
                
                # Only include candidates with score >= 50
                if breakout_score >= 50:
                    candidates.append(metrics)
            
            # Sort by breakout score
            candidates.sort(key=lambda x: x['breakout_score'], reverse=True)
            
            if candidates:
                result += f"✅ **FOUND {len(candidates)} BREAKOUT CANDIDATES:**\n\n"
                
                for i, candidate in enumerate(candidates[:10], 1):  # Top 10
                    score_desc = "🔥 IMMINENT" if candidate['breakout_score'] >= 80 else "🚀 HIGH POTENTIAL" if candidate['breakout_score'] >= 70 else "📈 MODERATE SETUP"
                    
                    result += f"**{i}. {candidate['symbol']}** - Score: {candidate['breakout_score']}/100 ({score_desc})\n"
                    result += f"   💰 Current: ${candidate['price']:.2f} | Resistance: ${candidate['recent_high']:.2f}\n"
                    result += f"   🎯 Distance to Breakout: {candidate['upside_to_high']:.1f}%\n"
                    result += f"   📊 Consolidation Range: {candidate['consolidation_range']:.1f}%\n"
                    result += f"   📈 Volume: {candidate['volume_ratio']:.1f}x | RSI: {candidate['rsi']:.1f}\n"
                    result += f"   📍 Range Position: {candidate['range_position']:.0f}%\n\n"
                
                result += "🎯 **BREAKOUT TRADING SETUP**:\n"
                result += "- Enter on volume-confirmed break above resistance\n"
                result += "- Set stop loss below consolidation range\n"
                result += "- Target measured move (range size added to breakout level)\n"
                result += "- Watch for false breakouts in low volume\n\n"
                
            else:
                result += "❌ **NO BREAKOUT CANDIDATES FOUND**\n\n"
                result += "No symbols met the breakout screening criteria (score >= 50).\n"
            
            result += "💡 **Breakout Screen Criteria**: Tight consolidation + proximity to resistance + volume + technical setup\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error in breakout screening: {str(e)}"
    
    def _custom_screen(self, symbols: List[str], timeframe: str, min_volume: int, criteria: Optional[str]) -> str:
        """Custom screening based on user-defined criteria"""
        try:
            if not criteria:
                return "❌ Custom criteria required. Example: 'RSI<30,MACD>0,Volume>1.5x'"
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            result = f"🎯 **CUSTOM SCREENING RESULTS** ({timeframe})\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"📊 Symbols Analyzed: {len(symbols)}\n"
            result += f"🔍 Criteria: {criteria}\n"
            result += f"📈 Min Volume Filter: {min_volume:,}\n\n"
            
            # Parse criteria
            conditions = []
            for condition in criteria.split(','):
                condition = condition.strip()
                if condition:
                    conditions.append(condition)
            
            if not conditions:
                return "❌ No valid conditions found in criteria."
            
            candidates = []
            
            for symbol in symbols:
                df = self._get_symbol_data(symbol, timeframe)
                if df is None or len(df) < 20:
                    continue
                
                metrics = self._calculate_screening_metrics(df)
                if not metrics:
                    continue
                
                metrics['symbol'] = symbol
                
                # Volume filter
                if metrics['avg_volume'] < min_volume:
                    continue
                
                # Evaluate custom conditions
                matches = 0
                total_conditions = len(conditions)
                condition_results = []
                
                for condition in conditions:
                    try:
                        match = self._evaluate_condition(condition, metrics)
                        condition_results.append((condition, match))
                        if match:
                            matches += 1
                    except Exception as e:
                        condition_results.append((condition, f"Error: {e}"))
                
                # Calculate match percentage
                match_pct = (matches / total_conditions) * 100 if total_conditions > 0 else 0
                
                metrics['match_pct'] = match_pct
                metrics['matches'] = matches
                metrics['total_conditions'] = total_conditions
                metrics['condition_results'] = condition_results
                
                # Include candidates with 100% match or 80%+ with strong signals
                if match_pct >= 100 or (match_pct >= 80 and matches >= 2):
                    candidates.append(metrics)
            
            # Sort by match percentage, then by number of matches
            candidates.sort(key=lambda x: (x['match_pct'], x['matches']), reverse=True)
            
            if candidates:
                result += f"✅ **FOUND {len(candidates)} CUSTOM SCREEN MATCHES:**\n\n"
                
                for i, candidate in enumerate(candidates[:10], 1):  # Top 10
                    match_desc = "🎯 PERFECT MATCH" if candidate['match_pct'] >= 100 else "📊 PARTIAL MATCH"
                    
                    result += f"**{i}. {candidate['symbol']}** - {candidate['matches']}/{candidate['total_conditions']} conditions ({candidate['match_pct']:.0f}%)\n"
                    result += f"   Status: {match_desc}\n"
                    result += f"   💰 Price: ${candidate['price']:.2f}\n"
                    result += f"   📊 Volume: {candidate['volume_ratio']:.1f}x | RSI: {candidate['rsi']:.1f}\n"
                    
                    # Show condition results
                    result += "   🔍 Condition Results:\n"
                    for condition, match in candidate['condition_results']:
                        status = "✅" if match is True else "❌" if match is False else "⚠️"
                        result += f"      {status} {condition}\n"
                    
                    result += "\n"
                
            else:
                result += "❌ **NO MATCHES FOUND**\n\n"
                result += "No symbols met the custom screening criteria.\n"
                result += "Try adjusting the criteria or expanding the symbol list.\n\n"
            
            result += "💡 **Custom Screening Notes**:\n"
            result += "- Supported indicators: RSI, MACD, Volume, Price, BB%, Stoch\n"
            result += "- Operators: <, >, <=, >=, =\n"
            result += "- Example: 'RSI<30,MACD>MACDSignal,Volume>1.5'\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error in custom screening: {str(e)}"
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate a single screening condition"""
        try:
            # Simple condition parser
            operators = ['<=', '>=', '<', '>', '=']
            operator = None
            
            for op in operators:
                if op in condition:
                    operator = op
                    break
            
            if not operator:
                raise ValueError(f"No valid operator found in condition: {condition}")
            
            left, right = condition.split(operator, 1)
            left = left.strip().upper()
            right = right.strip()
            
            # Get left value
            left_value = self._get_metric_value(left, metrics)
            
            # Get right value
            right_value = self._get_metric_value(right, metrics)
            
            # Evaluate condition
            if operator == '<':
                return left_value < right_value
            elif operator == '>':
                return left_value > right_value
            elif operator == '<=':
                return left_value <= right_value
            elif operator == '>=':
                return left_value >= right_value
            elif operator == '=':
                return abs(left_value - right_value) < 0.01  # Allow small floating point errors
            
            return False
            
        except Exception as e:
            raise ValueError(f"Error evaluating condition '{condition}': {e}")
    
    def _get_metric_value(self, metric: str, metrics: Dict[str, Any]) -> float:
        """Get metric value from metrics dictionary"""
        metric = metric.upper().strip()
        
        # Direct metric mappings
        metric_map = {
            'RSI': 'rsi',
            'MACD': 'macd',
            'MACDSIGNAL': 'macd_signal',
            'MACD_SIGNAL': 'macd_signal',
            'VOLUME': 'volume_ratio',
            'PRICE': 'price',
            'BB%': 'bb_percent',
            'BBPERCENT': 'bb_percent',
            'STOCH': 'stoch_k',
            'STOCHK': 'stoch_k',
            'ATR%': 'atr_percent',
            'ATRPERCENT': 'atr_percent'
        }
        
        # Check if it's a number
        try:
            return float(metric)
        except ValueError:
            pass
        
        # Check if it's a mapped metric
        if metric in metric_map:
            return metrics.get(metric_map[metric], 0)
        
        # Check for special cases
        if metric.endswith('X'):  # Volume multiplier (e.g., "1.5X")
            try:
                return float(metric[:-1])
            except ValueError:
                pass
        
        raise ValueError(f"Unknown metric: {metric}")
    
    def _multi_timeframe_screen(self, symbols: List[str], min_volume: int) -> str:
        """Screen across multiple timeframes for confluence"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            result = f"⏰ **MULTI-TIMEFRAME SCREENING RESULTS**\n"
            result += f"🕒 Analysis Time: {current_time}\n"
            result += f"📊 Symbols Analyzed: {len(symbols)}\n"
            result += f"⏰ Timeframes: 1Day (Primary), 1Hour (Secondary)\n"
            result += f"📈 Min Volume Filter: {min_volume:,}\n\n"
            
            candidates = []
            
            for symbol in symbols:
                # Get data for both timeframes
                daily_df = self._get_symbol_data(symbol, "1Day", 60)
                hourly_df = self._get_symbol_data(symbol, "1Hour", 60)
                
                if daily_df is None or hourly_df is None or len(daily_df) < 20 or len(hourly_df) < 20:
                    continue
                
                daily_metrics = self._calculate_screening_metrics(daily_df)
                hourly_metrics = self._calculate_screening_metrics(hourly_df)
                
                if not daily_metrics or not hourly_metrics:
                    continue
                
                # Volume filter (daily)
                if daily_metrics['avg_volume'] < min_volume:
                    continue
                
                # Multi-timeframe confluence scoring
                confluence_score = 0
                signals = []
                
                # Trend alignment (25 points max)
                daily_trend = 1 if daily_metrics['sma20_vs_sma50'] > 0 else -1
                hourly_trend = 1 if hourly_metrics['sma20_vs_sma50'] > 0 else -1
                
                if daily_trend == hourly_trend:
                    confluence_score += 25
                    signals.append(f"Trend Aligned ({'Bullish' if daily_trend > 0 else 'Bearish'})")
                
                # RSI confluence (20 points max)
                daily_rsi_signal = 1 if daily_metrics['rsi'] >= 50 else -1
                hourly_rsi_signal = 1 if hourly_metrics['rsi'] >= 50 else -1
                
                if daily_rsi_signal == hourly_rsi_signal:
                    confluence_score += 20
                    signals.append(f"RSI Aligned ({'Bullish' if daily_rsi_signal > 0 else 'Bearish'})")
                
                # MACD confluence (20 points max)
                daily_macd_signal = 1 if daily_metrics['macd'] > daily_metrics['macd_signal'] else -1
                hourly_macd_signal = 1 if hourly_metrics['macd'] > hourly_metrics['macd_signal'] else -1
                
                if daily_macd_signal == hourly_macd_signal:
                    confluence_score += 20
                    signals.append(f"MACD Aligned ({'Bullish' if daily_macd_signal > 0 else 'Bearish'})")
                
                # Volume confirmation (15 points max)
                if daily_metrics['volume_ratio'] >= 1.2 or hourly_metrics['volume_ratio'] >= 1.2:
                    confluence_score += 15
                    signals.append("Volume Support")
                
                # Momentum alignment (20 points max)
                daily_momentum = 1 if daily_metrics['price_change_5d'] > 0 else -1
                hourly_momentum = 1 if hourly_metrics['price_change_5d'] > 0 else -1
                
                if daily_momentum == hourly_momentum:
                    confluence_score += 20
                    signals.append(f"Momentum Aligned ({'Bullish' if daily_momentum > 0 else 'Bearish'})")
                
                candidate = {
                    'symbol': symbol,
                    'confluence_score': confluence_score,
                    'signals': signals,
                    'daily_metrics': daily_metrics,
                    'hourly_metrics': hourly_metrics
                }
                
                # Only include candidates with score >= 60 (3+ confluences)
                if confluence_score >= 60:
                    candidates.append(candidate)
            
            # Sort by confluence score
            candidates.sort(key=lambda x: x['confluence_score'], reverse=True)
            
            if candidates:
                result += f"✅ **FOUND {len(candidates)} MULTI-TIMEFRAME CANDIDATES:**\n\n"
                
                for i, candidate in enumerate(candidates[:8], 1):  # Top 8
                    score_desc = "🎯 PERFECT" if candidate['confluence_score'] >= 90 else "🔥 STRONG" if candidate['confluence_score'] >= 80 else "📊 MODERATE"
                    
                    result += f"**{i}. {candidate['symbol']}** - Confluence: {candidate['confluence_score']}/100 ({score_desc})\n"
                    result += f"   💰 Price: ${candidate['daily_metrics']['price']:.2f}\n"
                    
                    # Daily timeframe summary
                    result += f"   📅 Daily: RSI {candidate['daily_metrics']['rsi']:.1f} | "
                    result += f"Vol {candidate['daily_metrics']['volume_ratio']:.1f}x | "
                    result += f"5D {candidate['daily_metrics']['price_change_5d']:+.1f}%\n"
                    
                    # Hourly timeframe summary
                    result += f"   ⏰ Hourly: RSI {candidate['hourly_metrics']['rsi']:.1f} | "
                    result += f"Vol {candidate['hourly_metrics']['volume_ratio']:.1f}x | "
                    result += f"5D {candidate['hourly_metrics']['price_change_5d']:+.1f}%\n"
                    
                    # Confluence signals
                    result += f"   🎯 Confluences: {', '.join(candidate['signals'])}\n\n"
                
            else:
                result += "❌ **NO MULTI-TIMEFRAME CANDIDATES FOUND**\n\n"
                result += "No symbols showed sufficient confluence across timeframes (score >= 60).\n"
            
            result += "💡 **Multi-Timeframe Analysis**: Higher confluence scores indicate stronger, more reliable setups\n"
            
            return result
            
        except Exception as e:
            return f"❌ Error in multi-timeframe screening: {str(e)}"


def create_technical_screening_tool() -> TechnicalScreeningTool:
    """Create and return the technical screening tool instance"""
    return TechnicalScreeningTool()