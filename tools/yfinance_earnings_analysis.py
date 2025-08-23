"""
YFinance Earnings & Analyst Intelligence Tool for LangChain Integration
"""

import os
from typing import Optional, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Earnings analysis features will be disabled.")


class YFinanceEarningsAnalysisInput(BaseModel):
    """Input schema for YFinance earnings analysis tool"""
    symbol: str = Field(description="Stock symbol (e.g., 'AAPL', 'TSLA', 'GOOGL')")
    analysis_type: Optional[str] = Field(
        default="all",
        description="Analysis type: 'earnings', 'estimates', 'calendar', or 'all' for comprehensive analysis"
    )


class YFinanceEarningsAnalysisTool(BaseTool):
    """Tool for retrieving earnings and analyst data from YFinance"""
    
    name: str = "yfinance_earnings_analysis"
    description: str = (
        "Get comprehensive earnings and analyst intelligence from Yahoo Finance. "
        "Provides earnings history, analyst estimates, earnings calendar, upgrades/downgrades, "
        "and earnings surprises. Essential for earnings-based investment decisions and "
        "understanding analyst sentiment."
    )
    args_schema: Type[BaseModel] = YFinanceEarningsAnalysisInput
    
    def _run(self, symbol: str, analysis_type: str = "all") -> str:
        """Execute the earnings analysis tool"""
        if not YFINANCE_AVAILABLE:
            return "❌ yfinance not available. Please install yfinance to use earnings analysis features."
        
        try:
            ticker = yf.Ticker(symbol.upper())
            
            if analysis_type.lower() == "all":
                return self._get_comprehensive_analysis(ticker, symbol.upper())
            elif analysis_type.lower() == "earnings":
                return self._get_earnings_history(ticker, symbol.upper())
            elif analysis_type.lower() == "estimates":
                return self._get_analyst_estimates(ticker, symbol.upper())
            elif analysis_type.lower() == "calendar":
                return self._get_earnings_calendar(ticker, symbol.upper())
            else:
                return "❌ Invalid analysis type. Use 'earnings', 'estimates', 'calendar', or 'all'."
        
        except Exception as e:
            logger.error(f"YFinance error for {symbol}: {e}")
            return f"❌ Error retrieving earnings analysis for {symbol.upper()}: {str(e)}"
    
    def _get_comprehensive_analysis(self, ticker, symbol: str) -> str:
        """Get comprehensive earnings and analyst analysis"""
        try:
            result = f"📊 **{symbol} Comprehensive Earnings & Analyst Analysis**\n\n"
            
            # Get basic info for context
            info = ticker.info
            company_name = info.get('longName', symbol) if info else symbol
            result = f"📊 **{company_name} ({symbol}) - Earnings Analysis**\n\n"
            
            # Earnings Calendar & Next Earnings
            result += self._get_earnings_calendar_section(ticker, symbol)
            
            # Recent Earnings History
            result += self._get_recent_earnings_section(ticker, symbol)
            
            # Analyst Estimates & Recommendations
            result += self._get_analyst_section(ticker, symbol)
            
            # Earnings Trends & Growth
            result += self._get_earnings_trends_section(ticker, symbol)
            
            return result.strip()
        
        except Exception as e:
            return f"❌ Error generating comprehensive analysis: {str(e)}"
    
    def _get_earnings_calendar_section(self, ticker, symbol: str) -> str:
        """Get earnings calendar information"""
        section = "📅 **EARNINGS CALENDAR:**\n"
        
        try:
            # Get earnings dates from info
            info = ticker.info
            if info:
                earnings_date = info.get('earningsDate')
                if earnings_date:
                    # earningsDate is usually a list with date range
                    if isinstance(earnings_date, list) and len(earnings_date) > 0:
                        next_earnings = earnings_date[0]
                        if hasattr(next_earnings, 'strftime'):
                            section += f"📈 Next Earnings Date: {next_earnings.strftime('%Y-%m-%d')}\n"
                        else:
                            section += f"📈 Next Earnings Date: {next_earnings}\n"
                    else:
                        section += f"📈 Next Earnings Date: {earnings_date}\n"
                else:
                    section += "📈 Next Earnings Date: Not scheduled/available\n"
                
                # Earnings time if available
                earnings_quarterly_revenue_growth = info.get('earningsQuarterlyGrowth')
                if earnings_quarterly_revenue_growth:
                    section += f"📊 Recent Quarterly Earnings Growth: {earnings_quarterly_revenue_growth:.2%}\n"
            
            section += "\n"
            return section
            
        except Exception as e:
            section += f"❌ Error getting earnings calendar: {str(e)}\n\n"
            return section
    
    def _get_recent_earnings_section(self, ticker, symbol: str) -> str:
        """Get recent earnings history"""
        section = "📈 **RECENT EARNINGS HISTORY:**\n"
        
        try:
            # Get quarterly earnings
            quarterly_earnings = ticker.quarterly_earnings
            
            if not quarterly_earnings.empty:
                # Show last 4 quarters
                recent_earnings = quarterly_earnings.head(4)
                
                section += f"📊 **Last {len(recent_earnings)} Quarters:**\n"
                for date, row in recent_earnings.iterrows():
                    earnings = row.get('Earnings', 'N/A')
                    revenue = row.get('Revenue', 'N/A')
                    
                    earnings_str = f"${earnings/1e9:.2f}B" if earnings != 'N/A' and earnings > 1e9 else f"${earnings/1e6:.1f}M" if earnings != 'N/A' and earnings > 1e6 else f"${earnings:.2f}" if earnings != 'N/A' else "N/A"
                    revenue_str = f"${revenue/1e9:.2f}B" if revenue != 'N/A' and revenue > 1e9 else f"${revenue/1e6:.1f}M" if revenue != 'N/A' and revenue > 1e6 else f"${revenue:.2f}" if revenue != 'N/A' else "N/A"
                    
                    section += f"  📅 {date.strftime('%Y-%m-%d')}: EPS: {earnings_str} | Revenue: {revenue_str}\n"
            
            # Get annual earnings
            annual_earnings = ticker.earnings
            if not annual_earnings.empty:
                section += f"\n📊 **Annual Earnings (Last 3 Years):**\n"
                recent_annual = annual_earnings.tail(3)
                
                for date, row in recent_annual.iterrows():
                    earnings = row.get('Earnings', 'N/A')
                    revenue = row.get('Revenue', 'N/A')
                    
                    earnings_str = f"${earnings/1e9:.2f}B" if earnings != 'N/A' and earnings > 1e9 else f"${earnings/1e6:.1f}M" if earnings != 'N/A' and earnings > 1e6 else f"${earnings:.2f}" if earnings != 'N/A' else "N/A"
                    revenue_str = f"${revenue/1e9:.2f}B" if revenue != 'N/A' and revenue > 1e9 else f"${revenue/1e6:.1f}M" if revenue != 'N/A' and revenue > 1e6 else f"${revenue:.2f}" if revenue != 'N/A' else "N/A"
                    
                    section += f"  📅 {date}: EPS: {earnings_str} | Revenue: {revenue_str}\n"
            
            section += "\n"
            return section
            
        except Exception as e:
            section += f"❌ Error getting earnings history: {str(e)}\n\n"
            return section
    
    def _get_analyst_section(self, ticker, symbol: str) -> str:
        """Get analyst recommendations and estimates"""
        section = "👔 **ANALYST INTELLIGENCE:**\n"
        
        try:
            info = ticker.info
            if info:
                # Analyst recommendations
                recommendation = info.get('recommendationKey', 'N/A')
                num_analyst_opinions = info.get('numberOfAnalystOpinions', 'N/A')
                
                if recommendation != 'N/A':
                    recommendation_emoji = self._get_recommendation_emoji(recommendation)
                    section += f"{recommendation_emoji} **Current Recommendation**: {recommendation.title()}\n"
                
                if num_analyst_opinions != 'N/A':
                    section += f"👥 **Analyst Coverage**: {num_analyst_opinions} analysts\n"
                
                # Price targets
                target_high = info.get('targetHighPrice')
                target_low = info.get('targetLowPrice')
                target_mean = info.get('targetMeanPrice')
                current_price = info.get('currentPrice', info.get('regularMarketPrice'))
                
                if target_mean:
                    section += f"🎯 **Mean Target**: ${target_mean:.2f}"
                    if current_price:
                        upside = ((target_mean - current_price) / current_price) * 100
                        upside_emoji = "📈" if upside >= 0 else "📉"
                        section += f" ({upside_emoji} {upside:+.1f}% from current)\n"
                    else:
                        section += "\n"
                
                if target_high and target_low:
                    section += f"📊 **Target Range**: ${target_low:.2f} - ${target_high:.2f}\n"
                
                # EPS estimates
                forward_eps = info.get('forwardEps')
                trailing_eps = info.get('trailingEps')
                
                if forward_eps:
                    section += f"🔮 **Forward EPS Estimate**: ${forward_eps:.2f}\n"
                if trailing_eps:
                    section += f"📈 **Trailing EPS**: ${trailing_eps:.2f}\n"
                
                # Growth estimates
                peg_ratio = info.get('pegRatio')
                earnings_growth = info.get('earningsGrowth')
                revenue_growth = info.get('revenueGrowth')
                
                if earnings_growth:
                    section += f"📊 **Earnings Growth (YoY)**: {earnings_growth:.2%}\n"
                if revenue_growth:
                    section += f"💰 **Revenue Growth (YoY)**: {revenue_growth:.2%}\n"
                if peg_ratio:
                    peg_assessment = "Undervalued" if peg_ratio < 1.0 else "Fairly Valued" if peg_ratio < 1.5 else "Potentially Overvalued"
                    section += f"⚖️ **PEG Ratio**: {peg_ratio:.2f} ({peg_assessment})\n"
            
            # Try to get upgrades/downgrades
            try:
                upgrades = ticker.upgrades_downgrades
                if upgrades is not None and not upgrades.empty:
                    section += f"\n📊 **Recent Analyst Actions** (Last 5):\n"
                    recent_actions = upgrades.head(5)
                    
                    for date, row in recent_actions.iterrows():
                        firm = row.get('Firm', 'Unknown')
                        action = row.get('ToGrade', row.get('Action', 'N/A'))
                        from_grade = row.get('FromGrade', '')
                        
                        action_emoji = "📈" if any(word in action.lower() for word in ['buy', 'outperform', 'positive', 'upgrade']) else "📉" if any(word in action.lower() for word in ['sell', 'underperform', 'negative', 'downgrade']) else "📊"
                        
                        grade_change = f" (from {from_grade})" if from_grade else ""
                        section += f"  {action_emoji} {date.strftime('%m/%d')}: {firm} - {action}{grade_change}\n"
            except:
                pass  # Upgrades data may not be available
            
            section += "\n"
            return section
            
        except Exception as e:
            section += f"❌ Error getting analyst data: {str(e)}\n\n"
            return section
    
    def _get_earnings_trends_section(self, ticker, symbol: str) -> str:
        """Get earnings trends and surprise history"""
        section = "📊 **EARNINGS TRENDS & ANALYSIS:**\n"
        
        try:
            info = ticker.info
            
            if info:
                # Earnings quality metrics
                profit_margins = info.get('profitMargins')
                operating_margins = info.get('operatingMargins')
                return_on_equity = info.get('returnOnEquity')
                return_on_assets = info.get('returnOnAssets')
                
                if profit_margins:
                    margin_quality = "Excellent" if profit_margins > 0.20 else "Good" if profit_margins > 0.10 else "Average" if profit_margins > 0.05 else "Poor"
                    section += f"💰 **Net Profit Margin**: {profit_margins:.2%} ({margin_quality})\n"
                
                if operating_margins:
                    op_margin_quality = "Excellent" if operating_margins > 0.25 else "Good" if operating_margins > 0.15 else "Average" if operating_margins > 0.08 else "Poor"
                    section += f"⚙️ **Operating Margin**: {operating_margins:.2%} ({op_margin_quality})\n"
                
                if return_on_equity:
                    roe_quality = "Excellent" if return_on_equity > 0.20 else "Good" if return_on_equity > 0.15 else "Average" if return_on_equity > 0.10 else "Poor"
                    section += f"🔄 **Return on Equity**: {return_on_equity:.2%} ({roe_quality})\n"
                
                # Financial strength indicators
                current_ratio = info.get('currentRatio')
                debt_to_equity = info.get('debtToEquity')
                
                if current_ratio:
                    liquidity_strength = "Strong" if current_ratio > 2.0 else "Adequate" if current_ratio > 1.2 else "Weak"
                    section += f"💧 **Current Ratio**: {current_ratio:.2f} ({liquidity_strength} liquidity)\n"
                
                if debt_to_equity:
                    debt_ratio = debt_to_equity / 100
                    debt_strength = "Conservative" if debt_ratio < 0.3 else "Moderate" if debt_ratio < 0.6 else "High Leverage"
                    section += f"🏦 **Debt-to-Equity**: {debt_ratio:.2f} ({debt_strength})\n"
                
                # Earnings consistency (based on available growth metrics)
                quarterly_earnings_growth = info.get('earningsQuarterlyGrowth')
                quarterly_revenue_growth = info.get('revenueQuarterlyGrowth')
                
                section += f"\n📈 **Growth Consistency:**\n"
                if quarterly_earnings_growth:
                    earnings_trend = "Accelerating" if quarterly_earnings_growth > 0.15 else "Growing" if quarterly_earnings_growth > 0.05 else "Stable" if quarterly_earnings_growth > -0.05 else "Declining"
                    section += f"  📊 Quarterly Earnings Growth: {quarterly_earnings_growth:.2%} ({earnings_trend})\n"
                
                if quarterly_revenue_growth:
                    revenue_trend = "Strong" if quarterly_revenue_growth > 0.10 else "Moderate" if quarterly_revenue_growth > 0.03 else "Slow" if quarterly_revenue_growth > -0.03 else "Declining"
                    section += f"  💰 Quarterly Revenue Growth: {quarterly_revenue_growth:.2%} ({revenue_trend})\n"
                
                # Investment recommendation based on metrics
                section += f"\n🎯 **Earnings Quality Assessment:**\n"
                
                quality_score = 0
                if profit_margins and profit_margins > 0.10: quality_score += 1
                if operating_margins and operating_margins > 0.15: quality_score += 1
                if return_on_equity and return_on_equity > 0.15: quality_score += 1
                if current_ratio and current_ratio > 1.5: quality_score += 1
                if quarterly_earnings_growth and quarterly_earnings_growth > 0: quality_score += 1
                
                if quality_score >= 4:
                    section += "  🌟 **High Quality**: Strong fundamentals across multiple metrics\n"
                elif quality_score >= 2:
                    section += "  ⚖️ **Moderate Quality**: Mixed fundamentals, further analysis recommended\n"
                else:
                    section += "  ⚠️ **Lower Quality**: Weak fundamentals, consider risks carefully\n"
            
            section += "\n"
            return section
            
        except Exception as e:
            section += f"❌ Error analyzing earnings trends: {str(e)}\n\n"
            return section
    
    def _get_earnings_history(self, ticker, symbol: str) -> str:
        """Get detailed earnings history"""
        try:
            result = f"📈 **{symbol} Detailed Earnings History**\n\n"
            
            # Quarterly earnings
            quarterly_earnings = ticker.quarterly_earnings
            if not quarterly_earnings.empty:
                result += "📊 **Quarterly Earnings (Last 8 Quarters):**\n"
                recent_quarters = quarterly_earnings.head(8)
                
                for date, row in recent_quarters.iterrows():
                    earnings = row.get('Earnings', 'N/A')
                    revenue = row.get('Revenue', 'N/A')
                    
                    earnings_str = self._format_currency(earnings) if earnings != 'N/A' else "N/A"
                    revenue_str = self._format_currency(revenue) if revenue != 'N/A' else "N/A"
                    
                    result += f"📅 {date.strftime('%Y-Q%q')}: EPS: {earnings_str} | Revenue: {revenue_str}\n"
                
                result += "\n"
            
            # Annual earnings
            annual_earnings = ticker.earnings
            if not annual_earnings.empty:
                result += "📊 **Annual Earnings (Last 5 Years):**\n"
                recent_years = annual_earnings.tail(5)
                
                for date, row in recent_years.iterrows():
                    earnings = row.get('Earnings', 'N/A')
                    revenue = row.get('Revenue', 'N/A')
                    
                    earnings_str = self._format_currency(earnings) if earnings != 'N/A' else "N/A"
                    revenue_str = self._format_currency(revenue) if revenue != 'N/A' else "N/A"
                    
                    result += f"📅 {date}: EPS: {earnings_str} | Revenue: {revenue_str}\n"
            
            return result.strip()
            
        except Exception as e:
            return f"❌ Error retrieving earnings history: {str(e)}"
    
    def _get_analyst_estimates(self, ticker, symbol: str) -> str:
        """Get analyst estimates and recommendations"""
        try:
            result = f"👔 **{symbol} Analyst Estimates & Recommendations**\n\n"
            
            info = ticker.info
            if info:
                # Current recommendation
                recommendation = info.get('recommendationKey', 'N/A')
                num_analysts = info.get('numberOfAnalystOpinions', 'N/A')
                
                result += "📊 **Current Consensus:**\n"
                if recommendation != 'N/A':
                    rec_emoji = self._get_recommendation_emoji(recommendation)
                    result += f"{rec_emoji} Recommendation: {recommendation.title()}\n"
                if num_analysts != 'N/A':
                    result += f"👥 Analyst Coverage: {num_analysts} analysts\n"
                
                # Price targets
                result += "\n🎯 **Price Targets:**\n"
                target_mean = info.get('targetMeanPrice')
                target_high = info.get('targetHighPrice')
                target_low = info.get('targetLowPrice')
                current_price = info.get('currentPrice', info.get('regularMarketPrice'))
                
                if target_mean:
                    result += f"🎯 Mean Target: ${target_mean:.2f}\n"
                if target_high:
                    result += f"📈 High Target: ${target_high:.2f}\n"
                if target_low:
                    result += f"📉 Low Target: ${target_low:.2f}\n"
                
                if current_price and target_mean:
                    upside = ((target_mean - current_price) / current_price) * 100
                    upside_emoji = "📈" if upside >= 0 else "📉"
                    result += f"💡 Implied Upside: {upside_emoji} {upside:+.1f}%\n"
                
                # EPS estimates
                result += "\n📊 **Earnings Estimates:**\n"
                forward_eps = info.get('forwardEps')
                trailing_eps = info.get('trailingEps')
                forward_pe = info.get('forwardPE')
                trailing_pe = info.get('trailingPE')
                
                if trailing_eps:
                    result += f"📈 Trailing EPS: ${trailing_eps:.2f}\n"
                if forward_eps:
                    result += f"🔮 Forward EPS: ${forward_eps:.2f}\n"
                if trailing_pe:
                    result += f"📊 Trailing P/E: {trailing_pe:.2f}\n"
                if forward_pe:
                    result += f"🔮 Forward P/E: {forward_pe:.2f}\n"
            
            return result.strip()
            
        except Exception as e:
            return f"❌ Error retrieving analyst estimates: {str(e)}"
    
    def _get_earnings_calendar(self, ticker, symbol: str) -> str:
        """Get earnings calendar information"""
        try:
            result = f"📅 **{symbol} Earnings Calendar**\n\n"
            
            info = ticker.info
            if info:
                # Next earnings date
                earnings_date = info.get('earningsDate')
                if earnings_date:
                    result += "📈 **Upcoming Earnings:**\n"
                    if isinstance(earnings_date, list) and len(earnings_date) > 0:
                        next_earnings = earnings_date[0]
                        if hasattr(next_earnings, 'strftime'):
                            result += f"📅 Next Earnings: {next_earnings.strftime('%Y-%m-%d')}\n"
                        else:
                            result += f"📅 Next Earnings: {next_earnings}\n"
                    else:
                        result += f"📅 Next Earnings: {earnings_date}\n"
                else:
                    result += "📅 Next Earnings: Not scheduled/available\n"
                
                # Recent performance
                quarterly_earnings_growth = info.get('earningsQuarterlyGrowth')
                quarterly_revenue_growth = info.get('revenueQuarterlyGrowth')
                
                if quarterly_earnings_growth or quarterly_revenue_growth:
                    result += "\n📊 **Recent Performance:**\n"
                    if quarterly_earnings_growth:
                        growth_emoji = "📈" if quarterly_earnings_growth > 0 else "📉"
                        result += f"{growth_emoji} Quarterly Earnings Growth: {quarterly_earnings_growth:.2%}\n"
                    if quarterly_revenue_growth:
                        revenue_emoji = "📈" if quarterly_revenue_growth > 0 else "📉"
                        result += f"{revenue_emoji} Quarterly Revenue Growth: {quarterly_revenue_growth:.2%}\n"
            
            return result.strip()
            
        except Exception as e:
            return f"❌ Error retrieving earnings calendar: {str(e)}"
    
    def _get_recommendation_emoji(self, recommendation: str) -> str:
        """Get emoji for recommendation"""
        recommendation = recommendation.lower()
        if recommendation in ['strong_buy', 'buy']:
            return "🚀"
        elif recommendation in ['outperform', 'overweight']:
            return "📈"
        elif recommendation in ['hold', 'neutral']:
            return "⚖️"
        elif recommendation in ['underperform', 'underweight']:
            return "📉"
        elif recommendation in ['sell', 'strong_sell']:
            return "⚠️"
        else:
            return "📊"
    
    def _format_currency(self, value) -> str:
        """Format currency values"""
        if value is None:
            return "N/A"
        
        try:
            value = float(value)
            if abs(value) >= 1e12:
                return f"${value/1e12:.2f}T"
            elif abs(value) >= 1e9:
                return f"${value/1e9:.2f}B"
            elif abs(value) >= 1e6:
                return f"${value/1e6:.2f}M"
            elif abs(value) >= 1e3:
                return f"${value/1e3:.2f}K"
            else:
                return f"${value:,.2f}"
        except (ValueError, TypeError):
            return "N/A"


def create_yfinance_earnings_analysis_tool() -> YFinanceEarningsAnalysisTool:
    """Factory function to create YFinance earnings analysis tool"""
    return YFinanceEarningsAnalysisTool()