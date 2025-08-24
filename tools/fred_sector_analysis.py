"""
FRED Sector Analysis Tool for LangChain Integration
"""

import os
from typing import Optional, Type, Dict, List, ClassVar
from datetime import datetime, timedelta
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
    logger.warning("fredapi not installed. FRED sector analysis features will be disabled.")


class FREDSectorAnalysisInput(BaseModel):
    """Input schema for FRED sector analysis tool"""
    analysis_type: str = Field(description="Analysis type: 'industrial', 'employment', 'housing', 'consumer', 'overview'")
    lookback_months: Optional[int] = Field(
        default=12,
        description="Number of months to look back for analysis (default: 12 months)"
    )
    sector_focus: Optional[str] = Field(
        default=None,
        description="Specific sector focus for detailed analysis (e.g., 'manufacturing', 'services', 'construction')"
    )


class FREDSectorAnalysisTool(BaseTool):
    """Tool for sector-specific economic analysis from FRED data"""
    
    name: str = "fred_sector_analysis"
    description: str = (
        "Get sector-specific economic analysis from FRED data. "
        "Analysis types: 'industrial' (manufacturing, production), 'employment' (jobs by sector), "
        "'housing' (real estate, construction), 'consumer' (spending, retail), 'overview' (all sectors). "
        "Provides detailed sector performance, trends, and economic implications "
        "for investment and policy analysis."
    )
    args_schema: Type[BaseModel] = FREDSectorAnalysisInput
    
    # Sector-specific indicators
    SECTOR_INDICATORS: ClassVar[Dict] = {
        'industrial': {
            'INDPRO': {
                'name': 'Industrial Production Index',
                'category': 'Manufacturing',
                'unit': 'Index 2017=100',
                'emoji': '🏭'
            },
            'CAPUTL.B50001.A': {
                'name': 'Capacity Utilization: Manufacturing',
                'category': 'Manufacturing',
                'unit': 'Percent',
                'emoji': '⚙️'
            },
            'NEWORDER': {
                'name': 'Manufacturers New Orders',
                'category': 'Manufacturing',
                'unit': 'Millions of Dollars',
                'emoji': '📋'
            },
            'ISRATIO': {
                'name': 'Total Business Inventories to Sales Ratio',
                'category': 'Business',
                'unit': 'Ratio',
                'emoji': '📦'
            }
        },
        'employment': {
            'PAYEMS': {
                'name': 'Total Nonfarm Payrolls',
                'category': 'Employment',
                'unit': 'Thousands of Persons',
                'emoji': '👥'
            },
            'MANEMP': {
                'name': 'Manufacturing Employment',
                'category': 'Manufacturing Jobs',
                'unit': 'Thousands of Persons',
                'emoji': '🏭'
            },
            'SRVPRD': {
                'name': 'Service-Providing Employment',
                'category': 'Services Jobs',
                'unit': 'Thousands of Persons',
                'emoji': '💼'
            },
            'USCONS': {
                'name': 'Construction Employment',
                'category': 'Construction Jobs',
                'unit': 'Thousands of Persons',
                'emoji': '🏗️'
            },
            'USTRADE': {
                'name': 'Retail Trade Employment',
                'category': 'Retail Jobs',
                'unit': 'Thousands of Persons',
                'emoji': '🛍️'
            }
        },
        'housing': {
            'HOUST': {
                'name': 'Housing Starts',
                'category': 'Housing Construction',
                'unit': 'Thousands of Units',
                'emoji': '🏠'
            },
            'PERMIT': {
                'name': 'Building Permits',
                'category': 'Housing Construction',
                'unit': 'Thousands of Units',
                'emoji': '📄'
            },
            'EXHOSLUSM495S': {
                'name': 'Existing Home Sales',
                'category': 'Housing Sales',
                'unit': 'Millions',
                'emoji': '🏡'
            },
            'MSPUS': {
                'name': 'Median Sales Price of Houses Sold',
                'category': 'Housing Prices',
                'unit': 'Dollars',
                'emoji': '💰'
            },
            'MORTGAGE30US': {
                'name': '30-Year Fixed Rate Mortgage Average',
                'category': 'Housing Finance',
                'unit': 'Percent',
                'emoji': '🏦'
            }
        },
        'consumer': {
            'PCEC96': {
                'name': 'Real Personal Consumption Expenditures',
                'category': 'Consumer Spending',
                'unit': 'Billions of Chained 2017 Dollars',
                'emoji': '💳'
            },
            'RSXFS': {
                'name': 'Retail Sales',
                'category': 'Retail',
                'unit': 'Millions of Dollars',
                'emoji': '🛒'
            },
            'UMCSENT': {
                'name': 'Consumer Sentiment',
                'category': 'Consumer Confidence',
                'unit': 'Index 1966:Q1=100',
                'emoji': '😊'
            },
            'PSAVERT': {
                'name': 'Personal Saving Rate',
                'category': 'Consumer Behavior',
                'unit': 'Percent',
                'emoji': '💰'
            },
            'DSPIC96': {
                'name': 'Real Disposable Personal Income',
                'category': 'Consumer Income',
                'unit': 'Billions of Chained 2017 Dollars',
                'emoji': '💵'
            }
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
        analysis_type: str,
        lookback_months: int = 12,
        sector_focus: Optional[str] = None
    ) -> str:
        """Execute the FRED sector analysis tool"""
        if not FRED_AVAILABLE:
            return "❌ fredapi not available. Please install fredapi to use FRED sector analysis features."
        
        fred_client = self._get_fred_client()
        if not fred_client:
            return "❌ FRED API not available. Please set FRED_API_KEY in your .env file. Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html"
        
        try:
            if analysis_type == "overview":
                return self._get_sector_overview(fred_client, lookback_months)
            elif analysis_type in self.SECTOR_INDICATORS:
                return self._get_sector_analysis(fred_client, analysis_type, lookback_months)
            else:
                available_types = list(self.SECTOR_INDICATORS.keys()) + ['overview']
                return f"❌ Unknown analysis type: {analysis_type}. Available types: {', '.join(available_types)}"
        
        except Exception as e:
            logger.error(f"FRED Sector Analysis error: {e}")
            return f"❌ FRED Sector Analysis error: {str(e)}"
    
    def _get_sector_overview(self, fred_client, lookback_months: int) -> str:
        """Get overview of all sectors"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_months * 30)
        
        result = f"🏢 **Sector Economic Overview**\n"
        result += f"📅 Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n"
        
        sector_summaries = []
        
        for sector_name, indicators in self.SECTOR_INDICATORS.items():
            sector_health = self._analyze_sector_health(fred_client, sector_name, indicators, start_date, end_date)
            sector_summaries.append((sector_name, sector_health))
        
        # Display sector summaries
        for sector_name, health_info in sector_summaries:
            emoji = self._get_sector_emoji(sector_name)
            result += f"{emoji} **{sector_name.upper()} SECTOR:**\n"
            result += f"  Health Score: {health_info['score']}/100\n"
            result += f"  Status: {health_info['status']}\n"
            result += f"  Key Trend: {health_info['trend']}\n\n"
        
        # Overall economic sector assessment
        avg_score = sum(h['score'] for _, h in sector_summaries) / len(sector_summaries)
        result += f"🎯 **OVERALL SECTOR HEALTH: {avg_score:.0f}/100**\n\n"
        
        if avg_score >= 75:
            overall_assessment = "🟢 **Broad-Based Strength** - Most sectors showing healthy performance"
        elif avg_score >= 60:
            overall_assessment = "🟡 **Mixed Performance** - Sectors showing varied strength"
        elif avg_score >= 45:
            overall_assessment = "🟠 **Widespread Weakness** - Multiple sectors underperforming"
        else:
            overall_assessment = "🔴 **Broad-Based Weakness** - Most sectors showing poor performance"
        
        result += overall_assessment + "\n\n"
        
        # Investment implications
        result += "💡 **Investment Implications:**\n"
        result += self._get_sector_investment_implications(sector_summaries, avg_score)
        
        return result.strip()
    
    def _get_sector_analysis(self, fred_client, analysis_type: str, lookback_months: int) -> str:
        """Get detailed analysis for specific sector"""
        indicators = self.SECTOR_INDICATORS[analysis_type]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_months * 30)
        
        sector_emoji = self._get_sector_emoji(analysis_type)
        result = f"{sector_emoji} **{analysis_type.upper()} SECTOR ANALYSIS**\n"
        result += f"📅 Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n"
        
        # Collect indicator data
        indicator_data = {}
        for series_id, info in indicators.items():
            try:
                data = fred_client.get_series(
                    series_id,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
                indicator_data[series_id] = {
                    'data': data,
                    'info': info
                }
            except Exception as e:
                logger.warning(f"Could not fetch {series_id}: {e}")
                indicator_data[series_id] = {'data': pd.Series(), 'info': info}
        
        # Display individual indicators
        result += "📊 **KEY INDICATORS:**\n"
        for series_id, data_info in indicator_data.items():
            data = data_info['data']
            info = data_info['info']
            
            result += f"\n{info['emoji']} **{info['name']}:**\n"
            
            if not data.empty:
                latest_value = data.iloc[-1]
                latest_date = data.index[-1]
                
                result += f"  Current: {self._format_value(latest_value, info['unit'])}\n"
                result += f"  Date: {latest_date.strftime('%Y-%m-%d')}\n"
                
                # Calculate trend
                if len(data) > 1:
                    prev_value = data.iloc[-2] if len(data) > 1 else None
                    if prev_value is not None:
                        change = latest_value - prev_value
                        change_pct = (change / prev_value * 100) if prev_value != 0 else 0
                        change_emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                        result += f"  Change: {change_emoji} {change_pct:+.1f}%\n"
                
                # Year-over-year change if enough data
                if len(data) >= 12:
                    yoy_idx = min(12, len(data) - 1)
                    yoy_change_pct = ((latest_value / data.iloc[-yoy_idx-1]) - 1) * 100 if data.iloc[-yoy_idx-1] != 0 else 0
                    yoy_emoji = "📈" if yoy_change_pct > 0 else "📉" if yoy_change_pct < 0 else "➡️"
                    result += f"  YoY Change: {yoy_emoji} {yoy_change_pct:+.1f}%\n"
                
                # Trend assessment
                trend_assessment = self._assess_indicator_trend(data, info)
                result += f"  Assessment: {trend_assessment}\n"
            else:
                result += "  Status: ❌ Data not available\n"
        
        # Sector health assessment
        result += f"\n🏥 **SECTOR HEALTH ASSESSMENT:**\n"
        health_score = self._calculate_sector_health_score(indicator_data)
        
        if health_score >= 80:
            health_status = "🟢 Excellent"
        elif health_score >= 65:
            health_status = "🟡 Good"
        elif health_score >= 50:
            health_status = "🟠 Fair"
        else:
            health_status = "🔴 Poor"
        
        result += f"  Health Score: {health_score}/100\n"
        result += f"  Status: {health_status}\n\n"
        
        # Sector-specific insights
        result += f"💡 **SECTOR INSIGHTS:**\n"
        result += self._get_sector_specific_insights(analysis_type, indicator_data, health_score)
        
        return result.strip()
    
    def _analyze_sector_health(self, fred_client, sector_name: str, indicators: Dict, start_date: datetime, end_date: datetime) -> Dict:
        """Analyze overall health of a sector"""
        scores = []
        trends = []
        
        for series_id, info in indicators.items():
            try:
                data = fred_client.get_series(
                    series_id,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
                
                if not data.empty and len(data) > 1:
                    # Calculate simple trend score
                    recent_avg = data.tail(3).mean() if len(data) >= 3 else data.iloc[-1]
                    earlier_avg = data.head(3).mean() if len(data) >= 6 else data.iloc[0]
                    
                    trend_score = 50  # Neutral
                    if recent_avg > earlier_avg * 1.05:
                        trend_score = 75  # Improving
                        trend_direction = "Improving"
                    elif recent_avg < earlier_avg * 0.95:
                        trend_score = 25  # Declining
                        trend_direction = "Declining"
                    else:
                        trend_direction = "Stable"
                    
                    scores.append(trend_score)
                    trends.append(trend_direction)
            
            except Exception as e:
                logger.warning(f"Could not analyze {series_id}: {e}")
        
        # Calculate overall health
        if scores:
            avg_score = sum(scores) / len(scores)
            dominant_trend = max(set(trends), key=trends.count) if trends else "Unknown"
        else:
            avg_score = 50
            dominant_trend = "Unknown"
        
        # Status based on score
        if avg_score >= 70:
            status = "🟢 Strong"
        elif avg_score >= 55:
            status = "🟡 Stable" 
        elif avg_score >= 40:
            status = "🟠 Weak"
        else:
            status = "🔴 Poor"
        
        return {
            'score': avg_score,
            'status': status,
            'trend': dominant_trend
        }
    
    def _calculate_sector_health_score(self, indicator_data: Dict) -> int:
        """Calculate numerical health score for sector"""
        scores = []
        
        for series_id, data_info in indicator_data.items():
            data = data_info['data']
            if not data.empty and len(data) > 1:
                # Simple scoring based on recent trend
                if len(data) >= 6:
                    recent_avg = data.tail(3).mean()
                    earlier_avg = data.iloc[-6:-3].mean() if len(data) >= 6 else data.head(3).mean()
                    
                    if recent_avg > earlier_avg * 1.05:
                        scores.append(80)  # Strong positive trend
                    elif recent_avg > earlier_avg * 1.02:
                        scores.append(70)  # Moderate positive trend
                    elif recent_avg > earlier_avg * 0.98:
                        scores.append(60)  # Stable
                    elif recent_avg > earlier_avg * 0.95:
                        scores.append(40)  # Moderate decline
                    else:
                        scores.append(20)  # Strong decline
                else:
                    scores.append(50)  # Not enough data for trend
        
        return int(sum(scores) / len(scores)) if scores else 50
    
    def _assess_indicator_trend(self, data: pd.Series, info: Dict) -> str:
        """Assess trend for individual indicator"""
        if len(data) < 3:
            return "Insufficient data"
        
        recent_avg = data.tail(3).mean()
        earlier_avg = data.head(3).mean() if len(data) >= 6 else data.iloc[0:len(data)//2].mean()
        
        change_pct = ((recent_avg - earlier_avg) / earlier_avg * 100) if earlier_avg != 0 else 0
        
        if abs(change_pct) < 2:
            return "🟡 Stable trend"
        elif change_pct > 10:
            return "🟢 Strong upward trend"
        elif change_pct > 2:
            return "🟢 Moderate upward trend"
        elif change_pct < -10:
            return "🔴 Strong downward trend"
        else:
            return "🟠 Moderate downward trend"
    
    def _get_sector_emoji(self, sector_name: str) -> str:
        """Get emoji for sector"""
        emojis = {
            'industrial': '🏭',
            'employment': '👥',
            'housing': '🏠',
            'consumer': '🛍️'
        }
        return emojis.get(sector_name, '📊')
    
    def _format_value(self, value: float, unit: str) -> str:
        """Format value based on unit"""
        if 'Millions' in unit or 'Thousands' in unit:
            if abs(value) >= 1000:
                return f"{value:,.0f}"
            else:
                return f"{value:.1f}"
        elif 'Percent' in unit or 'Ratio' in unit:
            return f"{value:.2f}"
        elif 'Dollars' in unit:
            if abs(value) >= 1000000:
                return f"${value/1000000:.1f}M"
            elif abs(value) >= 1000:
                return f"${value/1000:.1f}K"
            else:
                return f"${value:.0f}"
        elif 'Index' in unit:
            return f"{value:.1f}"
        else:
            return f"{value:.2f}"
    
    def _get_sector_specific_insights(self, sector_name: str, indicator_data: Dict, health_score: int) -> str:
        """Get sector-specific insights"""
        insights = ""
        
        if sector_name == 'industrial':
            insights += "Manufacturing sector health impacts business investment and export competitiveness. "
            if health_score >= 70:
                insights += "Strong industrial production suggests robust business demand and export opportunities."
            elif health_score >= 50:
                insights += "Moderate industrial activity indicates stable but not accelerating business conditions."
            else:
                insights += "Weak industrial production may signal declining business investment and potential recession risks."
        
        elif sector_name == 'employment':
            insights += "Employment trends are leading indicators of consumer spending and economic health. "
            if health_score >= 70:
                insights += "Strong job growth supports consumer confidence and spending power."
            elif health_score >= 50:
                insights += "Steady employment suggests stable consumer conditions."
            else:
                insights += "Weak job growth indicates potential consumer spending weakness and economic slowdown."
        
        elif sector_name == 'housing':
            insights += "Housing sector performance affects construction, financial services, and consumer wealth. "
            if health_score >= 70:
                insights += "Strong housing market supports construction jobs and household wealth effects."
            elif health_score >= 50:
                insights += "Stable housing market provides steady economic support."
            else:
                insights += "Weak housing market may drag on economic growth and consumer spending."
        
        elif sector_name == 'consumer':
            insights += "Consumer sector drives ~70% of US economic activity and reflects household confidence. "
            if health_score >= 70:
                insights += "Strong consumer activity indicates healthy household finances and economic optimism."
            elif health_score >= 50:
                insights += "Moderate consumer activity suggests stable household conditions."
            else:
                insights += "Weak consumer activity signals potential economic contraction and household stress."
        
        return insights
    
    def _get_sector_investment_implications(self, sector_summaries: List, avg_score: float) -> str:
        """Get investment implications from sector analysis"""
        implications = ""
        
        # Find strongest and weakest sectors
        strongest_sector = max(sector_summaries, key=lambda x: x[1]['score'])
        weakest_sector = min(sector_summaries, key=lambda x: x[1]['score'])
        
        implications += f"**Strongest Sector**: {strongest_sector[0].title()} ({strongest_sector[1]['score']:.0f}/100)\n"
        implications += f"**Weakest Sector**: {weakest_sector[0].title()} ({weakest_sector[1]['score']:.0f}/100)\n\n"
        
        # Overall investment strategy
        if avg_score >= 70:
            implications += "**Strategy**: Broad market exposure with emphasis on cyclical sectors. "
            implications += "Consider growth-oriented investments as multiple sectors show strength."
        elif avg_score >= 55:
            implications += "**Strategy**: Selective sector allocation with focus on relative strength. "
            implications += "Balance between defensive and cyclical positions."
        else:
            implications += "**Strategy**: Defensive positioning with emphasis on quality and dividends. "
            implications += "Reduce cyclical exposure and increase defensive sectors."
        
        return implications


def create_fred_sector_analysis_tool() -> FREDSectorAnalysisTool:
    """Factory function to create FRED sector analysis tool"""
    return FREDSectorAnalysisTool()