"""
FRED Economic Dashboard Tool for LangChain Integration
"""

import os
from typing import Optional, Type, Dict, ClassVar
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
    logger.warning("fredapi not installed. FRED economic dashboard features will be disabled.")


class FREDEconomicDashboardInput(BaseModel):
    """Input schema for FRED economic dashboard tool"""
    analysis_type: Optional[str] = Field(
        default="overview",
        description="Analysis type: 'overview', 'health_score', 'cycle_analysis', 'alerts'"
    )
    lookback_months: Optional[int] = Field(
        default=24,
        description="Number of months to look back for analysis (default: 24 months)"
    )


class FREDEconomicDashboardTool(BaseTool):
    """Tool for economic health overview and dashboard from FRED data"""
    
    name: str = "fred_economic_dashboard"
    description: str = (
        "Get comprehensive economic health overview and dashboard from FRED data. "
        "Analysis types: 'overview' (key indicators summary), 'health_score' (economic health scoring), "
        "'cycle_analysis' (economic cycle assessment), 'alerts' (significant changes detection). "
        "Provides overall economic assessment, trend analysis, and policy implications "
        "based on multiple key economic indicators."
    )
    args_schema: Type[BaseModel] = FREDEconomicDashboardInput
    
    # Key indicators for economic health assessment
    DASHBOARD_INDICATORS: ClassVar[Dict] = {
        'GDPC1': {
            'name': 'Real GDP',
            'weight': 0.25,
            'category': 'Growth',
            'higher_is_better': True,
            'benchmark': 2.0  # 2% growth
        },
        'UNRATE': {
            'name': 'Unemployment Rate',
            'weight': 0.20,
            'category': 'Employment',
            'higher_is_better': False,
            'benchmark': 4.0  # 4% unemployment
        },
        'CPIAUCSL': {
            'name': 'Consumer Price Index',
            'weight': 0.15,
            'category': 'Inflation',
            'higher_is_better': None,  # Target around 2%
            'benchmark': 2.0  # 2% inflation YoY
        },
        'FEDFUNDS': {
            'name': 'Federal Funds Rate',
            'weight': 0.15,
            'category': 'Monetary Policy',
            'higher_is_better': None,  # Context dependent
            'benchmark': 2.5  # Neutral rate
        },
        'DGS10': {
            'name': '10-Year Treasury Yield',
            'weight': 0.10,
            'category': 'Financial Markets',
            'higher_is_better': None,
            'benchmark': 3.0
        },
        'UMCSENT': {
            'name': 'Consumer Sentiment',
            'weight': 0.15,
            'category': 'Consumer Confidence',
            'higher_is_better': True,
            'benchmark': 85.0  # Historical average ~85
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
        analysis_type: str = "overview",
        lookback_months: int = 24
    ) -> str:
        """Execute the FRED economic dashboard tool"""
        if not FRED_AVAILABLE:
            return "❌ fredapi not available. Please install fredapi to use FRED economic dashboard features."
        
        fred_client = self._get_fred_client()
        if not fred_client:
            return "❌ FRED API not available. Please set FRED_API_KEY in your .env file. Get a free API key at https://fred.stlouisfed.org/docs/api/api_key.html"
        
        try:
            if analysis_type == "overview":
                return self._get_economic_overview(fred_client, lookback_months)
            elif analysis_type == "health_score":
                return self._get_health_score(fred_client, lookback_months)
            elif analysis_type == "cycle_analysis":
                return self._get_cycle_analysis(fred_client, lookback_months)
            elif analysis_type == "alerts":
                return self._get_economic_alerts(fred_client, lookback_months)
            else:
                return f"❌ Unknown analysis type: {analysis_type}. Available types: overview, health_score, cycle_analysis, alerts"
        
        except Exception as e:
            logger.error(f"FRED Dashboard error: {e}")
            return f"❌ FRED Dashboard error: {str(e)}"
    
    def _get_economic_overview(self, fred_client, lookback_months: int) -> str:
        """Get comprehensive economic overview"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_months * 30)
        
        result = f"📊 **Economic Dashboard Overview**\n"
        result += f"📅 Analysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n\n"
        
        # Collect data for all indicators
        indicators_data = {}
        for series_id, info in self.DASHBOARD_INDICATORS.items():
            try:
                data = fred_client.get_series(
                    series_id,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
                indicators_data[series_id] = {
                    'data': data,
                    'info': info
                }
            except Exception as e:
                logger.warning(f"Could not fetch {series_id}: {e}")
                indicators_data[series_id] = {'data': pd.Series(), 'info': info}
        
        # Economic Growth
        result += "📈 **ECONOMIC GROWTH:**\n"
        gdp_data = indicators_data.get('GDPC1', {}).get('data', pd.Series())
        if not gdp_data.empty:
            latest_gdp = gdp_data.iloc[-1]
            if len(gdp_data) > 4:  # Need at least 4 quarters for YoY
                yoy_gdp = ((gdp_data.iloc[-1] / gdp_data.iloc[-5]) - 1) * 100 if len(gdp_data) >= 5 else 0
                growth_emoji = "📈" if yoy_gdp > 2 else "⚖️" if yoy_gdp > 0 else "📉"
                result += f"  {growth_emoji} Real GDP Growth: {yoy_gdp:.1f}% YoY\n"
                result += f"  📊 Level: ${latest_gdp:,.0f}B (Chained 2017$)\n"
            else:
                result += f"  📊 Current Level: ${latest_gdp:,.0f}B (Chained 2017$)\n"
        else:
            result += "  ❌ GDP data not available\n"
        result += "\n"
        
        # Employment
        result += "👥 **EMPLOYMENT:**\n"
        unrate_data = indicators_data.get('UNRATE', {}).get('data', pd.Series())
        if not unrate_data.empty:
            latest_unrate = unrate_data.iloc[-1]
            if len(unrate_data) > 1:
                change = unrate_data.iloc[-1] - unrate_data.iloc[-2]
                change_emoji = "📉" if change < 0 else "📈" if change > 0 else "➡️"
                result += f"  📊 Unemployment Rate: {latest_unrate:.1f}%\n"
                result += f"  {change_emoji} Monthly Change: {change:+.1f}pp\n"
                
                # Assessment
                if latest_unrate <= 4.0:
                    assessment = "🟢 Very Strong"
                elif latest_unrate <= 5.5:
                    assessment = "🟡 Good"
                elif latest_unrate <= 7.0:
                    assessment = "🟠 Concerning"
                else:
                    assessment = "🔴 Weak"
                result += f"  📋 Assessment: {assessment}\n"
            else:
                result += f"  📊 Current: {latest_unrate:.1f}%\n"
        else:
            result += "  ❌ Unemployment data not available\n"
        result += "\n"
        
        # Inflation
        result += "💰 **INFLATION:**\n"
        cpi_data = indicators_data.get('CPIAUCSL', {}).get('data', pd.Series())
        if not cpi_data.empty and len(cpi_data) > 12:
            latest_cpi = cpi_data.iloc[-1]
            yoy_inflation = ((cpi_data.iloc[-1] / cpi_data.iloc[-13]) - 1) * 100
            inflation_emoji = "🔴" if yoy_inflation > 4 else "🟠" if yoy_inflation > 3 else "🟡" if yoy_inflation > 2 else "🟢"
            result += f"  {inflation_emoji} CPI Inflation: {yoy_inflation:.1f}% YoY\n"
            result += f"  📊 CPI Level: {latest_cpi:.1f}\n"
            
            # Fed target assessment
            if 1.5 <= yoy_inflation <= 2.5:
                target_status = "🎯 Near Fed Target (2%)"
            elif yoy_inflation > 2.5:
                target_status = "⬆️ Above Fed Target"
            else:
                target_status = "⬇️ Below Fed Target"
            result += f"  📋 Target Status: {target_status}\n"
        else:
            result += "  ❌ Inflation data not available or insufficient\n"
        result += "\n"
        
        # Monetary Policy
        result += "🏦 **MONETARY POLICY:**\n"
        fedfunds_data = indicators_data.get('FEDFUNDS', {}).get('data', pd.Series())
        dgs10_data = indicators_data.get('DGS10', {}).get('data', pd.Series())
        
        if not fedfunds_data.empty:
            latest_fedfunds = fedfunds_data.iloc[-1]
            result += f"  📊 Federal Funds Rate: {latest_fedfunds:.2f}%\n"
            
            if not dgs10_data.empty:
                latest_10y = dgs10_data.iloc[-1]
                yield_spread = latest_10y - latest_fedfunds
                result += f"  📊 10-Year Treasury: {latest_10y:.2f}%\n"
                result += f"  📏 Yield Spread (10Y-FF): {yield_spread:.2f}%\n"
                
                # Yield curve assessment
                if yield_spread < 0:
                    curve_status = "🔴 Inverted (Recession Signal)"
                elif yield_spread < 1:
                    curve_status = "🟠 Flat"
                else:
                    curve_status = "🟢 Normal"
                result += f"  📋 Yield Curve: {curve_status}\n"
        else:
            result += "  ❌ Interest rate data not available\n"
        result += "\n"
        
        # Consumer Confidence
        result += "🛍️ **CONSUMER SENTIMENT:**\n"
        sentiment_data = indicators_data.get('UMCSENT', {}).get('data', pd.Series())
        if not sentiment_data.empty:
            latest_sentiment = sentiment_data.iloc[-1]
            if len(sentiment_data) > 1:
                change = sentiment_data.iloc[-1] - sentiment_data.iloc[-2]
                change_emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                result += f"  📊 Consumer Sentiment: {latest_sentiment:.1f}\n"
                result += f"  {change_emoji} Monthly Change: {change:+.1f}\n"
                
                # Assessment
                if latest_sentiment >= 90:
                    sentiment_status = "🟢 Very Optimistic"
                elif latest_sentiment >= 80:
                    sentiment_status = "🟡 Optimistic"
                elif latest_sentiment >= 70:
                    sentiment_status = "🟠 Cautious"
                else:
                    sentiment_status = "🔴 Pessimistic"
                result += f"  📋 Assessment: {sentiment_status}\n"
        else:
            result += "  ❌ Consumer sentiment data not available\n"
        
        # Overall economic assessment
        result += f"\n🎯 **OVERALL ECONOMIC ASSESSMENT:**\n"
        result += self._get_overall_assessment(indicators_data)
        
        return result.strip()
    
    def _get_health_score(self, fred_client, lookback_months: int) -> str:
        """Calculate economic health score"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_months * 30)
        
        result = f"🏥 **Economic Health Score**\n"
        result += f"📅 Analysis Period: Last {lookback_months} months\n\n"
        
        # Collect data and calculate scores
        total_score = 0
        total_weight = 0
        component_scores = {}
        
        for series_id, info in self.DASHBOARD_INDICATORS.items():
            try:
                data = fred_client.get_series(
                    series_id,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
                
                if not data.empty:
                    score = self._calculate_indicator_score(series_id, data, info)
                    component_scores[series_id] = {
                        'score': score,
                        'weight': info['weight'],
                        'name': info['name'],
                        'category': info['category']
                    }
                    total_score += score * info['weight']
                    total_weight += info['weight']
                
            except Exception as e:
                logger.warning(f"Could not calculate score for {series_id}: {e}")
        
        # Overall health score
        if total_weight > 0:
            overall_score = (total_score / total_weight) * 100
            result += f"🎯 **Overall Health Score: {overall_score:.1f}/100**\n\n"
            
            # Grade
            if overall_score >= 80:
                grade = "A"
                grade_emoji = "🟢"
                assessment = "Excellent"
            elif overall_score >= 70:
                grade = "B" 
                grade_emoji = "🟡"
                assessment = "Good"
            elif overall_score >= 60:
                grade = "C"
                grade_emoji = "🟠"
                assessment = "Fair"
            else:
                grade = "D"
                grade_emoji = "🔴"
                assessment = "Poor"
            
            result += f"{grade_emoji} **Economic Grade: {grade} ({assessment})**\n\n"
        else:
            result += "❌ Unable to calculate health score due to insufficient data\n\n"
            return result
        
        # Component breakdown
        result += "📊 **Component Scores:**\n"
        for series_id, component in component_scores.items():
            score_emoji = "🟢" if component['score'] >= 0.8 else "🟡" if component['score'] >= 0.6 else "🟠" if component['score'] >= 0.4 else "🔴"
            result += f"  {score_emoji} **{component['name']}**: {component['score']*100:.1f}/100 (Weight: {component['weight']*100:.0f}%)\n"
        
        result += f"\n💡 **Health Score Interpretation:**\n"
        result += self._get_health_interpretation(overall_score, component_scores)
        
        return result.strip()
    
    def _get_cycle_analysis(self, fred_client, lookback_months: int) -> str:
        """Analyze economic cycle phase"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_months * 30)
        
        result = f"🔄 **Economic Cycle Analysis**\n"
        result += f"📅 Analysis Period: Last {lookback_months} months\n\n"
        
        # Key indicators for cycle analysis
        cycle_indicators = ['GDPC1', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'DGS10', 'UMCSENT']
        trends = {}
        
        for series_id in cycle_indicators:
            try:
                data = fred_client.get_series(
                    series_id,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
                
                if not data.empty and len(data) > 1:
                    # Calculate trend
                    recent_avg = data.tail(6).mean() if len(data) >= 6 else data.mean()
                    earlier_avg = data.head(6).mean() if len(data) >= 12 else data.mean()
                    
                    if recent_avg > earlier_avg * 1.02:
                        trends[series_id] = 'rising'
                    elif recent_avg < earlier_avg * 0.98:
                        trends[series_id] = 'falling'
                    else:
                        trends[series_id] = 'stable'
                
            except Exception as e:
                logger.warning(f"Could not analyze trend for {series_id}: {e}")
        
        # Determine cycle phase
        cycle_phase, confidence = self._determine_cycle_phase(trends)
        
        result += f"🎯 **Current Economic Cycle Phase:**\n"
        result += f"  Phase: **{cycle_phase}**\n"
        result += f"  Confidence: {confidence}/5 ⭐\n\n"
        
        # Detailed trend analysis
        result += f"📊 **Indicator Trends:**\n"
        for series_id, trend in trends.items():
            indicator_name = self.DASHBOARD_INDICATORS.get(series_id, {}).get('name', series_id)
            trend_emoji = "📈" if trend == 'rising' else "📉" if trend == 'falling' else "➡️"
            result += f"  {trend_emoji} {indicator_name}: {trend.title()}\n"
        
        result += f"\n💡 **Cycle Implications:**\n"
        result += self._get_cycle_implications(cycle_phase, trends)
        
        return result.strip()
    
    def _get_economic_alerts(self, fred_client, lookback_months: int) -> str:
        """Detect significant economic changes and alerts"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_months * 30)
        
        result = f"🚨 **Economic Alerts & Significant Changes**\n"
        result += f"📅 Monitoring Period: Last {lookback_months} months\n\n"
        
        alerts = []
        
        for series_id, info in self.DASHBOARD_INDICATORS.items():
            try:
                data = fred_client.get_series(
                    series_id,
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d')
                )
                
                if not data.empty and len(data) > 1:
                    alert = self._check_for_alerts(series_id, data, info)
                    if alert:
                        alerts.append(alert)
                
            except Exception as e:
                logger.warning(f"Could not check alerts for {series_id}: {e}")
        
        if alerts:
            for alert in alerts:
                result += f"{alert}\n\n"
        else:
            result += "✅ No significant alerts detected in the monitored indicators.\n\n"
        
        result += f"💡 **Alert Criteria:**\n"
        result += "- Unemployment rate changes > 0.5% in a month\n"
        result += "- Inflation changes > 1% YoY in recent months\n"
        result += "- Interest rate changes > 0.5% in recent periods\n"
        result += "- GDP growth changes > 1% QoQ (annualized)\n"
        result += "- Consumer sentiment changes > 10 points\n"
        result += "- Yield curve inversions or significant steepening/flattening\n"
        
        return result.strip()
    
    def _calculate_indicator_score(self, series_id: str, data: pd.Series, info: Dict) -> float:
        """Calculate health score for individual indicator (0-1 scale)"""
        if data.empty:
            return 0.5  # Neutral score for missing data
        
        latest_value = data.iloc[-1]
        
        if series_id == 'GDPC1':  # Real GDP - look at growth rate
            if len(data) >= 4:
                yoy_growth = ((data.iloc[-1] / data.iloc[-5]) - 1) * 100 if len(data) >= 5 else 0
                if yoy_growth >= 3:
                    return 1.0
                elif yoy_growth >= 2:
                    return 0.8
                elif yoy_growth >= 1:
                    return 0.6
                elif yoy_growth >= 0:
                    return 0.4
                else:
                    return 0.2
            return 0.5
        
        elif series_id == 'UNRATE':  # Unemployment - lower is better
            if latest_value <= 3.5:
                return 1.0
            elif latest_value <= 5.0:
                return 0.8
            elif latest_value <= 6.5:
                return 0.6
            elif latest_value <= 8.0:
                return 0.4
            else:
                return 0.2
        
        elif series_id == 'CPIAUCSL':  # CPI - target around 2% YoY
            if len(data) >= 12:
                yoy_inflation = ((data.iloc[-1] / data.iloc[-13]) - 1) * 100
                deviation = abs(yoy_inflation - 2.0)
                if deviation <= 0.5:
                    return 1.0
                elif deviation <= 1.0:
                    return 0.8
                elif deviation <= 2.0:
                    return 0.6
                elif deviation <= 3.0:
                    return 0.4
                else:
                    return 0.2
            return 0.5
        
        elif series_id == 'FEDFUNDS':  # Fed Funds - context dependent
            # Score based on appropriateness given inflation environment
            # This is simplified - in reality would need more complex analysis
            if 2.0 <= latest_value <= 4.0:
                return 0.8
            elif 1.0 <= latest_value <= 5.0:
                return 0.6
            else:
                return 0.4
        
        elif series_id == 'DGS10':  # 10-Year Treasury
            if 2.5 <= latest_value <= 4.0:
                return 0.8
            elif 2.0 <= latest_value <= 5.0:
                return 0.6
            else:
                return 0.4
        
        elif series_id == 'UMCSENT':  # Consumer Sentiment
            if latest_value >= 90:
                return 1.0
            elif latest_value >= 80:
                return 0.8
            elif latest_value >= 70:
                return 0.6
            elif latest_value >= 60:
                return 0.4
            else:
                return 0.2
        
        return 0.5  # Default neutral score
    
    def _determine_cycle_phase(self, trends: Dict) -> tuple:
        """Determine economic cycle phase based on trends"""
        # Simplified cycle analysis based on key indicator trends
        rising_count = sum(1 for trend in trends.values() if trend == 'rising')
        falling_count = sum(1 for trend in trends.values() if trend == 'falling')
        total_indicators = len(trends)
        
        if total_indicators == 0:
            return "Unknown", 0
        
        rising_pct = rising_count / total_indicators
        falling_pct = falling_count / total_indicators
        
        # Determine phase
        if rising_pct >= 0.6:
            phase = "Expansion"
            confidence = min(5, int(rising_pct * 6))
        elif falling_pct >= 0.6:
            phase = "Contraction"
            confidence = min(5, int(falling_pct * 6))
        elif rising_pct > falling_pct:
            phase = "Early Recovery"
            confidence = 3
        elif falling_pct > rising_pct:
            phase = "Late Cycle/Peak"
            confidence = 3
        else:
            phase = "Transitional"
            confidence = 2
        
        return phase, confidence
    
    def _get_overall_assessment(self, indicators_data: Dict) -> str:
        """Get overall economic assessment"""
        assessment = ""
        
        # Count positive and negative indicators
        positive_signals = 0
        negative_signals = 0
        total_signals = 0
        
        for series_id, data_info in indicators_data.items():
            data = data_info.get('data', pd.Series())
            if not data.empty:
                total_signals += 1
                # Simplified assessment logic
                if series_id == 'UNRATE':
                    if data.iloc[-1] <= 5.0:
                        positive_signals += 1
                    else:
                        negative_signals += 1
                elif series_id in ['GDPC1', 'UMCSENT']:
                    if len(data) > 1 and data.iloc[-1] > data.iloc[-2]:
                        positive_signals += 1
                    else:
                        negative_signals += 1
        
        if total_signals > 0:
            positive_pct = positive_signals / total_signals
            
            if positive_pct >= 0.7:
                assessment = "🟢 **Strong Economic Conditions** - Most indicators suggest healthy economic momentum."
            elif positive_pct >= 0.5:
                assessment = "🟡 **Mixed Economic Conditions** - Indicators show balanced but uncertain outlook."
            else:
                assessment = "🟠 **Concerning Economic Conditions** - Several indicators suggest economic weakness."
        else:
            assessment = "❓ **Insufficient Data** - Unable to assess economic conditions."
        
        return assessment
    
    def _get_health_interpretation(self, score: float, components: Dict) -> str:
        """Get health score interpretation"""
        interpretation = ""
        
        if score >= 80:
            interpretation += "The economy shows strong fundamentals across most key indicators. "
            interpretation += "This suggests a healthy, growing economy with low recession risk."
        elif score >= 70:
            interpretation += "The economy displays generally positive conditions with some areas of concern. "
            interpretation += "Overall trajectory remains favorable but monitoring is advised."
        elif score >= 60:
            interpretation += "The economy shows mixed signals with balanced strengths and weaknesses. "
            interpretation += "Increased caution and monitoring recommended."
        else:
            interpretation += "The economy exhibits significant weaknesses across multiple indicators. "
            interpretation += "Heightened attention to economic developments is warranted."
        
        # Find strongest and weakest components
        if components:
            strongest = max(components.items(), key=lambda x: x[1]['score'])
            weakest = min(components.items(), key=lambda x: x[1]['score'])
            
            interpretation += f"\n\nStrongest area: {strongest[1]['name']} ({strongest[1]['score']*100:.0f}/100)"
            interpretation += f"\nWeakest area: {weakest[1]['name']} ({weakest[1]['score']*100:.0f}/100)"
        
        return interpretation
    
    def _get_cycle_implications(self, phase: str, trends: Dict) -> str:
        """Get economic cycle implications"""
        implications = ""
        
        if phase == "Expansion":
            implications += "🟢 **Expansion Phase**: Economy is growing robustly. "
            implications += "Expect continued job growth, rising corporate profits, and potential inflationary pressures. "
            implications += "Fed may consider tightening monetary policy."
        
        elif phase == "Late Cycle/Peak":
            implications += "🟡 **Late Cycle**: Economic growth may be peaking. "
            implications += "Monitor for signs of overheating, inflation pressures, and potential policy tightening. "
            implications += "Consider defensive positioning in portfolios."
        
        elif phase == "Contraction":
            implications += "🔴 **Contraction Phase**: Economic activity is declining. "
            implications += "Expect rising unemployment, falling corporate profits, and potential recession risks. "
            implications += "Fed likely to ease monetary policy."
        
        elif phase == "Early Recovery":
            implications += "🟢 **Early Recovery**: Economy is stabilizing and beginning to recover. "
            implications += "Early signs of improvement suggest potential for sustained growth. "
            implications += "Accommodative policy likely to continue."
        
        else:
            implications += "🟡 **Transitional Phase**: Economic conditions are mixed and changing. "
            implications += "Direction unclear - monitor key indicators closely for clearer signals."
        
        return implications
    
    def _check_for_alerts(self, series_id: str, data: pd.Series, info: Dict) -> Optional[str]:
        """Check for significant changes that warrant alerts"""
        if len(data) < 2:
            return None
        
        latest = data.iloc[-1]
        previous = data.iloc[-2]
        change = latest - previous
        
        # Define alert thresholds
        if series_id == 'UNRATE' and abs(change) >= 0.5:
            direction = "increased" if change > 0 else "decreased"
            severity = "🔴" if abs(change) >= 1.0 else "🟠"
            return f"{severity} **Unemployment Alert**: Rate {direction} by {abs(change):.1f}pp to {latest:.1f}%"
        
        elif series_id == 'FEDFUNDS' and abs(change) >= 0.25:
            direction = "raised" if change > 0 else "cut"
            severity = "🟡"
            return f"{severity} **Fed Funds Alert**: Rate {direction} by {abs(change):.2f}pp to {latest:.2f}%"
        
        elif series_id == 'UMCSENT' and abs(change) >= 10:
            direction = "surged" if change > 0 else "plunged"
            severity = "🟡"
            return f"{severity} **Consumer Sentiment Alert**: Index {direction} by {abs(change):.1f} to {latest:.1f}"
        
        # Check for yield curve inversion
        elif series_id == 'DGS10' and len(data) > 1:
            # Would need to compare with shorter rates for proper inversion detection
            # This is a simplified check
            pass
        
        return None


def create_fred_economic_dashboard_tool() -> FREDEconomicDashboardTool:
    """Factory function to create FRED economic dashboard tool"""
    return FREDEconomicDashboardTool()