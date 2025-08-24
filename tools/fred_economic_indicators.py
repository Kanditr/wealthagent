"""
FRED Economic Indicators Tool for LangChain Integration
"""

import os
from typing import Optional, Type, List, Dict, ClassVar
from datetime import datetime, timedelta, timezone
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from fredapi import Fred
    import pandas as pd
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    logger.warning("fredapi not installed. FRED economic indicators features will be disabled.")


class FREDEconomicIndicatorsInput(BaseModel):
    """Input schema for FRED economic indicators tool"""
    action: str = Field(description="Action to perform: 'indicator', 'multi_indicator', 'trend'")
    series_id: Optional[str] = Field(
        default=None, 
        description="FRED series ID (e.g., 'GDPC1', 'UNRATE', 'CPIAUCSL'). Required for 'indicator' action."
    )
    series_ids: Optional[str] = Field(
        default=None,
        description="Multiple FRED series IDs separated by commas (e.g., 'GDPC1,UNRATE,CPIAUCSL'). Required for 'multi_indicator' action."
    )
    start_date: Optional[str] = Field(
        default=None,
        description="Start date in YYYY-MM-DD format. Defaults to 5 years ago."
    )
    end_date: Optional[str] = Field(
        default=None,
        description="End date in YYYY-MM-DD format. Defaults to latest available."
    )
    frequency: Optional[str] = Field(
        default=None,
        description="Data frequency: 'd' (daily), 'w' (weekly), 'm' (monthly), 'q' (quarterly), 'a' (annual)"
    )


class FREDEconomicIndicatorsTool(BaseTool):
    """Tool for retrieving economic indicators from FRED (Federal Reserve Economic Data)"""
    
    name: str = "fred_economic_indicators"
    description: str = (
        "Get key economic indicators from the Federal Reserve Economic Data (FRED). "
        "Actions: 'indicator' (single indicator), 'multi_indicator' (multiple indicators), "
        "'trend' (indicator with trend analysis). "
        "Key series: GDP (GDPC1), Unemployment (UNRATE), CPI (CPIAUCSL), "
        "Fed Funds Rate (FEDFUNDS), 10-Year Treasury (DGS10), PPI (PPIACO). "
        "Provides historical data, current values, and trend analysis for macroeconomic insights."
    )
    args_schema: Type[BaseModel] = FREDEconomicIndicatorsInput
    
    # Key economic indicators and their metadata
    INDICATORS: ClassVar[Dict] = {
        # GDP & Growth
        'GDPC1': {
            'name': 'Real Gross Domestic Product',
            'category': 'GDP',
            'unit': 'Billions of Chained 2017 Dollars',
            'frequency': 'Quarterly',
            'emoji': '📈'
        },
        'GDP': {
            'name': 'Gross Domestic Product',
            'category': 'GDP', 
            'unit': 'Billions of Dollars',
            'frequency': 'Quarterly',
            'emoji': '💰'
        },
        
        # Employment
        'UNRATE': {
            'name': 'Unemployment Rate',
            'category': 'Employment',
            'unit': 'Percent',
            'frequency': 'Monthly',
            'emoji': '👥'
        },
        'PAYEMS': {
            'name': 'Nonfarm Payrolls',
            'category': 'Employment',
            'unit': 'Thousands of Persons',
            'frequency': 'Monthly',
            'emoji': '💼'
        },
        
        # Inflation
        'CPIAUCSL': {
            'name': 'Consumer Price Index for All Urban Consumers: All Items',
            'category': 'Inflation',
            'unit': 'Index 1982-84=100',
            'frequency': 'Monthly',
            'emoji': '📊'
        },
        'CPILFESL': {
            'name': 'Core CPI (Less Food & Energy)',
            'category': 'Inflation',
            'unit': 'Index 1982-84=100',
            'frequency': 'Monthly',
            'emoji': '🎯'
        },
        'PPIACO': {
            'name': 'Producer Price Index: All Commodities',
            'category': 'Inflation',
            'unit': 'Index 1982=100',
            'frequency': 'Monthly',
            'emoji': '🏭'
        },
        
        # Interest Rates
        'FEDFUNDS': {
            'name': 'Federal Funds Effective Rate',
            'category': 'Interest Rates',
            'unit': 'Percent',
            'frequency': 'Monthly',
            'emoji': '🏦'
        },
        'DGS10': {
            'name': '10-Year Treasury Constant Maturity Rate',
            'category': 'Interest Rates',
            'unit': 'Percent',
            'frequency': 'Daily',
            'emoji': '📋'
        },
        'DGS2': {
            'name': '2-Year Treasury Constant Maturity Rate',
            'category': 'Interest Rates',
            'unit': 'Percent',
            'frequency': 'Daily',
            'emoji': '📄'
        },
        
        # Housing
        'HOUST': {
            'name': 'Housing Starts',
            'category': 'Housing',
            'unit': 'Thousands of Units',
            'frequency': 'Monthly',
            'emoji': '🏠'
        },
        
        # Consumer
        'UMCSENT': {
            'name': 'University of Michigan Consumer Sentiment',
            'category': 'Consumer',
            'unit': 'Index 1966:Q1=100',
            'frequency': 'Monthly',
            'emoji': '🛍️'
        }
    }
    
    def _get_fred_client(self):
        """Get FRED API client"""
        if not FRED_AVAILABLE:
            return None
        
        api_key = os.getenv("FRED_API_KEY")
        if not api_key:
            return None
        
        try:
            return Fred(api_key=api_key)
        except Exception as e:
            logger.error(f"Failed to initialize FRED client: {e}")
            return None
    
    def _run(
        self,
        action: str,
        series_id: Optional[str] = None,
        series_ids: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        frequency: Optional[str] = None
    ) -> str:
        """Execute the FRED economic indicators tool"""
        if not FRED_AVAILABLE:
            return "❌ fredapi not available. Please install fredapi to use FRED economic indicators features."
        
        fred_client = self._get_fred_client()
        if not fred_client:
            return "❌ FRED API not available. Please set FRED_API_KEY in your .env file. Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html"
        
        try:
            if action == "indicator":
                if not series_id:
                    return "❌ series_id required for 'indicator' action"
                return self._get_single_indicator(fred_client, series_id, start_date, end_date, frequency)
            elif action == "multi_indicator":
                if not series_ids:
                    return "❌ series_ids required for 'multi_indicator' action"
                return self._get_multiple_indicators(fred_client, series_ids, start_date, end_date, frequency)
            elif action == "trend":
                if not series_id:
                    return "❌ series_id required for 'trend' action"
                return self._get_indicator_trend(fred_client, series_id, start_date, end_date, frequency)
            else:
                return f"❌ Unknown action: {action}. Available actions: indicator, multi_indicator, trend"
        
        except Exception as e:
            logger.error(f"FRED API error: {e}")
            return f"❌ FRED API error: {str(e)}"
    
    def _get_single_indicator(
        self, 
        fred_client, 
        series_id: str, 
        start_date: Optional[str], 
        end_date: Optional[str],
        frequency: Optional[str]
    ) -> str:
        """Get single economic indicator"""
        try:
            # Get series info
            info = self.INDICATORS.get(series_id.upper(), {
                'name': series_id,
                'category': 'Economic Indicator',
                'unit': 'N/A',
                'frequency': 'N/A',
                'emoji': '📊'
            })
            
            # Set default date range (5 years)
            if not start_date:
                start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Fetch data
            try:
                # Build parameters for get_series - only include frequency if specified
                params = {
                    'start': start_date,
                    'end': end_date
                }
                if frequency:
                    params['frequency'] = frequency
                
                data = fred_client.get_series(series_id, **params)
            except Exception as e:
                return f"❌ Error retrieving data for {series_id}: {str(e)}. Check if the series ID is valid."
            
            if data.empty:
                return f"❌ No data found for series {series_id} in the specified date range."
            
            return self._format_single_indicator(series_id, info, data, start_date, end_date)
            
        except Exception as e:
            return f"❌ Error processing indicator {series_id}: {str(e)}"
    
    def _get_multiple_indicators(
        self,
        fred_client,
        series_ids: str,
        start_date: Optional[str],
        end_date: Optional[str],
        frequency: Optional[str]
    ) -> str:
        """Get multiple economic indicators"""
        try:
            series_list = [s.strip().upper() for s in series_ids.split(',') if s.strip()]
            
            if len(series_list) > 10:
                return "❌ Maximum 10 series allowed for multi_indicator action."
            
            # Set default date range (2 years for multiple indicators)
            if not start_date:
                start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            result = f"📊 **Multiple Economic Indicators** ({len(series_list)} series)\n"
            result += f"📅 Period: {start_date} to {end_date}\n\n"
            
            for series_id in series_list:
                info = self.INDICATORS.get(series_id, {
                    'name': series_id,
                    'category': 'Economic Indicator',
                    'unit': 'N/A',
                    'frequency': 'N/A',
                    'emoji': '📊'
                })
                
                try:
                    # Build parameters for get_series - only include frequency if specified
                    params = {
                        'start': start_date,
                        'end': end_date
                    }
                    if frequency:
                        params['frequency'] = frequency
                    
                    data = fred_client.get_series(series_id, **params)
                    
                    if not data.empty:
                        result += self._format_indicator_summary(series_id, info, data)
                    else:
                        result += f"❌ **{info['emoji']} {info['name']} ({series_id})**: No data available\n"
                
                except Exception as e:
                    result += f"❌ **{info['emoji']} {info['name']} ({series_id})**: Error - {str(e)}\n"
                
                result += "\n"
            
            return result.strip()
            
        except Exception as e:
            return f"❌ Error processing multiple indicators: {str(e)}"
    
    def _get_indicator_trend(
        self,
        fred_client,
        series_id: str,
        start_date: Optional[str],
        end_date: Optional[str],
        frequency: Optional[str]
    ) -> str:
        """Get economic indicator with trend analysis"""
        try:
            # Get series info
            info = self.INDICATORS.get(series_id.upper(), {
                'name': series_id,
                'category': 'Economic Indicator',
                'unit': 'N/A',
                'frequency': 'N/A',
                'emoji': '📊'
            })
            
            # Set default date range (3 years for trend analysis)
            if not start_date:
                start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Fetch data
            # Build parameters for get_series - only include frequency if specified
            params = {
                'start': start_date,
                'end': end_date
            }
            if frequency:
                params['frequency'] = frequency
            
            data = fred_client.get_series(series_id, **params)
            
            if data.empty:
                return f"❌ No data found for series {series_id} in the specified date range."
            
            return self._format_trend_analysis(series_id, info, data, start_date, end_date)
            
        except Exception as e:
            return f"❌ Error analyzing trend for {series_id}: {str(e)}"
    
    def _format_single_indicator(self, series_id: str, info: Dict, data: pd.Series, start_date: str, end_date: str) -> str:
        """Format single indicator for display"""
        latest_value = data.iloc[-1] if not data.empty else None
        latest_date = data.index[-1] if not data.empty else None
        
        result = f"{info['emoji']} **{info['name']} ({series_id})**\n\n"
        result += f"📅 **Period**: {start_date} to {end_date}\n"
        result += f"📊 **Category**: {info['category']}\n"
        result += f"📏 **Unit**: {info['unit']}\n"
        result += f"🔄 **Frequency**: {info['frequency']}\n\n"
        
        if latest_value is not None and latest_date is not None:
            result += f"💎 **Latest Value**: {latest_value:.2f}\n"
            result += f"📅 **As of**: {latest_date.strftime('%Y-%m-%d')}\n\n"
            
            # Calculate basic statistics
            if len(data) > 1:
                # Recent change (last vs previous)
                prev_value = data.iloc[-2] if len(data) > 1 else None
                if prev_value is not None:
                    change = latest_value - prev_value
                    change_pct = (change / prev_value * 100) if prev_value != 0 else 0
                    change_emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    result += f"📊 **Recent Change**: {change_emoji} {change:+.2f} ({change_pct:+.2f}%)\n"
                
                # Historical stats
                avg_value = data.mean()
                max_value = data.max()
                min_value = data.min()
                
                result += f"📈 **Average**: {avg_value:.2f}\n"
                result += f"⬆️ **Maximum**: {max_value:.2f} ({data.idxmax().strftime('%Y-%m-%d')})\n"
                result += f"⬇️ **Minimum**: {min_value:.2f} ({data.idxmin().strftime('%Y-%m-%d')})\n\n"
                
                # Current vs historical context
                if latest_value > avg_value:
                    context = f"above average by {((latest_value - avg_value) / avg_value * 100):.1f}%"
                    context_emoji = "📈"
                elif latest_value < avg_value:
                    context = f"below average by {((avg_value - latest_value) / avg_value * 100):.1f}%"
                    context_emoji = "📉"
                else:
                    context = "at historical average"
                    context_emoji = "⚖️"
                
                result += f"{context_emoji} **Historical Context**: Current value is {context}\n"
        
        # Economic interpretation
        result += f"\n💡 **Economic Significance**:\n"
        result += self._get_economic_interpretation(series_id, info, latest_value)
        
        return result.strip()
    
    def _format_indicator_summary(self, series_id: str, info: Dict, data: pd.Series) -> str:
        """Format indicator summary for multi-indicator display"""
        latest_value = data.iloc[-1] if not data.empty else None
        latest_date = data.index[-1] if not data.empty else None
        
        summary = f"**{info['emoji']} {info['name']} ({series_id})**\n"
        
        if latest_value is not None and latest_date is not None:
            summary += f"  Current: {latest_value:.2f} {info['unit']}\n"
            summary += f"  Date: {latest_date.strftime('%Y-%m-%d')}\n"
            
            # Recent trend (last 3 values if available)
            if len(data) >= 3:
                recent_data = data.tail(3)
                if recent_data.iloc[-1] > recent_data.iloc[0]:
                    trend = "📈 Rising"
                elif recent_data.iloc[-1] < recent_data.iloc[0]:
                    trend = "📉 Falling"
                else:
                    trend = "➡️ Stable"
                summary += f"  Trend: {trend}\n"
        else:
            summary += "  Status: ❌ No recent data\n"
        
        return summary
    
    def _format_trend_analysis(self, series_id: str, info: Dict, data: pd.Series, start_date: str, end_date: str) -> str:
        """Format trend analysis for display"""
        result = f"{info['emoji']} **{info['name']} ({series_id}) - Trend Analysis**\n\n"
        result += f"📅 **Analysis Period**: {start_date} to {end_date}\n"
        result += f"📊 **Data Points**: {len(data)}\n\n"
        
        if data.empty or len(data) < 2:
            return result + "❌ Insufficient data for trend analysis."
        
        latest_value = data.iloc[-1]
        earliest_value = data.iloc[0]
        latest_date = data.index[-1]
        earliest_date = data.index[0]
        
        # Overall trend
        total_change = latest_value - earliest_value
        total_change_pct = (total_change / earliest_value * 100) if earliest_value != 0 else 0
        
        result += f"📊 **Overall Trend**:\n"
        result += f"  Start: {earliest_value:.2f} ({earliest_date.strftime('%Y-%m-%d')})\n"
        result += f"  End: {latest_value:.2f} ({latest_date.strftime('%Y-%m-%d')})\n"
        
        if total_change > 0:
            result += f"  📈 **Change**: +{total_change:.2f} (+{total_change_pct:.2f}%)\n"
            result += f"  🎯 **Direction**: Upward trend\n"
        elif total_change < 0:
            result += f"  📉 **Change**: {total_change:.2f} ({total_change_pct:.2f}%)\n"
            result += f"  🎯 **Direction**: Downward trend\n"
        else:
            result += f"  ➡️ **Change**: No change (0.00%)\n"
            result += f"  🎯 **Direction**: Stable\n"
        
        result += "\n"
        
        # Recent trend (last 6 months or 25% of data, whichever is larger)
        recent_periods = max(6, len(data) // 4)
        if len(data) > recent_periods:
            recent_data = data.tail(recent_periods)
            recent_change = recent_data.iloc[-1] - recent_data.iloc[0]
            recent_change_pct = (recent_change / recent_data.iloc[0] * 100) if recent_data.iloc[0] != 0 else 0
            
            result += f"📊 **Recent Trend** (Last {recent_periods} periods):\n"
            if recent_change > 0:
                result += f"  📈 **Recent Direction**: Rising (+{recent_change:.2f}, +{recent_change_pct:.2f}%)\n"
            elif recent_change < 0:
                result += f"  📉 **Recent Direction**: Falling ({recent_change:.2f}, {recent_change_pct:.2f}%)\n"
            else:
                result += f"  ➡️ **Recent Direction**: Stable (0.00%)\n"
            
            result += "\n"
        
        # Statistical analysis
        mean_value = data.mean()
        std_value = data.std()
        max_value = data.max()
        min_value = data.min()
        
        result += f"📈 **Statistical Summary**:\n"
        result += f"  Average: {mean_value:.2f}\n"
        result += f"  Standard Deviation: {std_value:.2f}\n"
        result += f"  Range: {min_value:.2f} to {max_value:.2f}\n"
        result += f"  Current vs Average: {((latest_value - mean_value) / mean_value * 100):+.1f}%\n\n"
        
        # Volatility assessment
        cv = (std_value / mean_value) * 100 if mean_value != 0 else 0
        if cv < 5:
            volatility = "Very Low"
            vol_emoji = "🟢"
        elif cv < 15:
            volatility = "Low"
            vol_emoji = "🟡"
        elif cv < 30:
            volatility = "Moderate"
            vol_emoji = "🟠"
        else:
            volatility = "High"
            vol_emoji = "🔴"
        
        result += f"{vol_emoji} **Volatility**: {volatility} (CV: {cv:.1f}%)\n\n"
        
        # Economic context
        result += f"💡 **Economic Context**:\n"
        result += self._get_trend_interpretation(series_id, info, data, total_change_pct, recent_change_pct if 'recent_change_pct' in locals() else 0)
        
        return result.strip()
    
    def _get_economic_interpretation(self, series_id: str, info: Dict, latest_value: float) -> str:
        """Get economic interpretation for an indicator"""
        series_upper = series_id.upper()
        
        if series_upper == 'UNRATE':  # Unemployment Rate
            if latest_value <= 4.0:
                return "Low unemployment indicates a strong labor market and economic health."
            elif latest_value <= 6.0:
                return "Moderate unemployment suggests normal economic conditions."
            else:
                return "High unemployment indicates economic weakness and potential recession."
        
        elif series_upper == 'FEDFUNDS':  # Federal Funds Rate
            if latest_value <= 2.0:
                return "Low interest rates suggest accommodative monetary policy to stimulate growth."
            elif latest_value <= 5.0:
                return "Moderate interest rates indicate balanced monetary policy."
            else:
                return "High interest rates suggest restrictive monetary policy to combat inflation."
        
        elif series_upper in ['CPIAUCSL', 'CPILFESL']:  # CPI measures
            # Calculate YoY change if possible (approximate)
            return "CPI measures inflation - rising values indicate increasing prices and inflationary pressure."
        
        elif series_upper == 'GDPC1':  # Real GDP
            return "GDP growth indicates overall economic expansion and health of the economy."
        
        elif series_upper == 'DGS10':  # 10-Year Treasury
            if latest_value <= 2.0:
                return "Low long-term rates suggest economic uncertainty or deflationary pressures."
            elif latest_value <= 4.0:
                return "Moderate long-term rates indicate balanced economic expectations."
            else:
                return "High long-term rates suggest inflation expectations or strong economic growth."
        
        else:
            return f"Monitor this {info['category'].lower()} indicator for economic trends and policy implications."
    
    def _get_trend_interpretation(self, series_id: str, info: Dict, data: pd.Series, total_change_pct: float, recent_change_pct: float) -> str:
        """Get trend interpretation for an indicator"""
        series_upper = series_id.upper()
        interpretation = ""
        
        # Overall trend assessment
        if abs(total_change_pct) < 5:
            trend_strength = "stable"
        elif abs(total_change_pct) < 20:
            trend_strength = "moderate"
        else:
            trend_strength = "strong"
        
        if series_upper == 'UNRATE':
            if total_change_pct > 10:
                interpretation = f"Strong upward trend in unemployment ({trend_strength}) suggests economic deterioration."
            elif total_change_pct < -10:
                interpretation = f"Strong downward trend in unemployment ({trend_strength}) indicates economic improvement."
            else:
                interpretation = f"Unemployment trend is {trend_strength}, suggesting stable labor market conditions."
        
        elif series_upper == 'FEDFUNDS':
            if total_change_pct > 20:
                interpretation = f"Rising interest rates ({trend_strength} trend) indicate tightening monetary policy."
            elif total_change_pct < -20:
                interpretation = f"Falling interest rates ({trend_strength} trend) suggest accommodative monetary policy."
            else:
                interpretation = f"Interest rates show {trend_strength} trend, indicating stable monetary policy."
        
        elif series_upper in ['CPIAUCSL', 'CPILFESL']:
            if total_change_pct > 10:
                interpretation = f"Rising inflation trend ({trend_strength}) may prompt policy tightening."
            elif total_change_pct < 0:
                interpretation = f"Deflationary trend ({trend_strength}) suggests economic weakness."
            else:
                interpretation = f"Inflation trend is {trend_strength}, within acceptable ranges."
        
        else:
            if total_change_pct > 0:
                interpretation = f"Upward trend ({trend_strength}) in {info['name'].lower()}."
            elif total_change_pct < 0:
                interpretation = f"Downward trend ({trend_strength}) in {info['name'].lower()}."
            else:
                interpretation = f"Stable trend in {info['name'].lower()}."
        
        # Add recent trend context
        if abs(recent_change_pct) > abs(total_change_pct) * 0.5:
            if (recent_change_pct > 0) != (total_change_pct > 0):
                interpretation += " Recent data suggests a potential trend reversal."
            else:
                interpretation += " Recent trend confirms the overall direction."
        
        return interpretation


def create_fred_economic_indicators_tool() -> FREDEconomicIndicatorsTool:
    """Factory function to create FRED economic indicators tool"""
    return FREDEconomicIndicatorsTool()