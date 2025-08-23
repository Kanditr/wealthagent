"""
YFinance Investment Quality Score Tool for LangChain Integration
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
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not installed. Investment quality score features will be disabled.")


class YFinanceQualityScoreInput(BaseModel):
    """Input schema for YFinance investment quality score tool"""
    symbol: str = Field(description="Stock symbol (e.g., 'AAPL', 'TSLA', 'GOOGL')")
    scoring_method: Optional[str] = Field(
        default="comprehensive",
        description="Scoring method: 'comprehensive', 'value', 'growth', or 'quality'"
    )


class YFinanceQualityScoreTool(BaseTool):
    """Tool for generating investment quality scores from YFinance data"""
    
    name: str = "yfinance_quality_score"
    description: str = (
        "Generate comprehensive investment quality scores based on Yahoo Finance data. "
        "Analyzes financial health, valuation, growth prospects, and market metrics "
        "to provide an overall investment quality assessment with detailed scoring breakdown. "
        "Essential for quantitative investment screening and decision-making."
    )
    args_schema: Type[BaseModel] = YFinanceQualityScoreInput
    
    def _run(self, symbol: str, scoring_method: str = "comprehensive") -> str:
        """Execute the quality score tool"""
        if not YFINANCE_AVAILABLE:
            return "❌ yfinance not available. Please install yfinance to use quality score features."
        
        try:
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            
            # Check if we got valid data
            if not info or len(info) < 5:
                return f"❌ No data found for symbol '{symbol.upper()}'. Please verify the ticker symbol."
            
            if scoring_method.lower() == "comprehensive":
                return self._calculate_comprehensive_score(symbol.upper(), info)
            elif scoring_method.lower() == "value":
                return self._calculate_value_score(symbol.upper(), info)
            elif scoring_method.lower() == "growth":
                return self._calculate_growth_score(symbol.upper(), info)
            elif scoring_method.lower() == "quality":
                return self._calculate_quality_score(symbol.upper(), info)
            else:
                return "❌ Invalid scoring method. Use 'comprehensive', 'value', 'growth', or 'quality'."
        
        except Exception as e:
            logger.error(f"YFinance error for {symbol}: {e}")
            return f"❌ Error calculating quality score for {symbol.upper()}: {str(e)}"
    
    def _calculate_comprehensive_score(self, symbol: str, info: dict) -> str:
        """Calculate comprehensive investment quality score"""
        company_name = info.get('longName', symbol)
        
        result = f"🎯 **{company_name} ({symbol}) - Investment Quality Score**\n\n"
        
        # Initialize scoring components
        scores = {
            'financial_health': 0,
            'valuation': 0,
            'growth': 0,
            'profitability': 0,
            'market_position': 0
        }
        max_scores = {
            'financial_health': 20,
            'valuation': 20,
            'growth': 20,
            'profitability': 20,
            'market_position': 20
        }
        
        details = {}
        
        # 1. Financial Health Score (20 points)
        health_score, health_details = self._score_financial_health(info)
        scores['financial_health'] = health_score
        details['financial_health'] = health_details
        
        # 2. Valuation Score (20 points)
        valuation_score, valuation_details = self._score_valuation(info)
        scores['valuation'] = valuation_score
        details['valuation'] = valuation_details
        
        # 3. Growth Score (20 points)
        growth_score, growth_details = self._score_growth(info)
        scores['growth'] = growth_score
        details['growth'] = growth_details
        
        # 4. Profitability Score (20 points)
        profitability_score, profitability_details = self._score_profitability(info)
        scores['profitability'] = profitability_score
        details['profitability'] = profitability_details
        
        # 5. Market Position Score (20 points)
        market_score, market_details = self._score_market_position(info)
        scores['market_position'] = market_score
        details['market_position'] = market_details
        
        # Calculate total score
        total_score = sum(scores.values())
        total_possible = sum(max_scores.values())
        overall_percentage = (total_score / total_possible) * 100
        
        # Overall assessment
        result += f"📊 **OVERALL QUALITY SCORE: {total_score}/{total_possible} ({overall_percentage:.1f}%)**\n\n"
        
        # Grade and recommendation
        grade, recommendation, risk_level = self._get_overall_assessment(overall_percentage)
        result += f"🏆 **Investment Grade**: {grade}\n"
        result += f"💡 **Recommendation**: {recommendation}\n"
        result += f"⚠️ **Risk Level**: {risk_level}\n\n"
        
        # Detailed breakdown
        result += "📋 **DETAILED SCORING BREAKDOWN:**\n\n"
        
        # Financial Health
        health_pct = (scores['financial_health'] / max_scores['financial_health']) * 100
        result += f"🏥 **Financial Health**: {scores['financial_health']}/{max_scores['financial_health']} ({health_pct:.1f}%)\n"
        for detail in details['financial_health']:
            result += f"  {detail}\n"
        result += "\n"
        
        # Valuation
        val_pct = (scores['valuation'] / max_scores['valuation']) * 100
        result += f"💰 **Valuation**: {scores['valuation']}/{max_scores['valuation']} ({val_pct:.1f}%)\n"
        for detail in details['valuation']:
            result += f"  {detail}\n"
        result += "\n"
        
        # Growth
        growth_pct = (scores['growth'] / max_scores['growth']) * 100
        result += f"📈 **Growth Prospects**: {scores['growth']}/{max_scores['growth']} ({growth_pct:.1f}%)\n"
        for detail in details['growth']:
            result += f"  {detail}\n"
        result += "\n"
        
        # Profitability
        profit_pct = (scores['profitability'] / max_scores['profitability']) * 100
        result += f"💹 **Profitability**: {scores['profitability']}/{max_scores['profitability']} ({profit_pct:.1f}%)\n"
        for detail in details['profitability']:
            result += f"  {detail}\n"
        result += "\n"
        
        # Market Position
        market_pct = (scores['market_position'] / max_scores['market_position']) * 100
        result += f"🏛️ **Market Position**: {scores['market_position']}/{max_scores['market_position']} ({market_pct:.1f}%)\n"
        for detail in details['market_position']:
            result += f"  {detail}\n"
        result += "\n"
        
        # Key strengths and weaknesses
        result += self._identify_strengths_weaknesses(scores, max_scores)
        
        return result.strip()
    
    def _score_financial_health(self, info: dict) -> tuple:
        """Score financial health metrics"""
        score = 0
        details = []
        
        # Current Ratio (4 points)
        current_ratio = info.get('currentRatio')
        if current_ratio:
            if current_ratio >= 2.0:
                score += 4
                details.append("✅ Current Ratio: Excellent (>= 2.0)")
            elif current_ratio >= 1.5:
                score += 3
                details.append("✅ Current Ratio: Good (>= 1.5)")
            elif current_ratio >= 1.0:
                score += 2
                details.append("⚠️ Current Ratio: Adequate (>= 1.0)")
            else:
                score += 0
                details.append("❌ Current Ratio: Poor (< 1.0)")
        else:
            details.append("❓ Current Ratio: Not available")
        
        # Debt-to-Equity (4 points)
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity:
            debt_ratio = debt_to_equity / 100
            if debt_ratio <= 0.3:
                score += 4
                details.append("✅ Debt-to-Equity: Conservative (<= 0.3)")
            elif debt_ratio <= 0.5:
                score += 3
                details.append("✅ Debt-to-Equity: Moderate (<= 0.5)")
            elif debt_ratio <= 0.8:
                score += 2
                details.append("⚠️ Debt-to-Equity: High (<= 0.8)")
            else:
                score += 0
                details.append("❌ Debt-to-Equity: Very High (> 0.8)")
        else:
            details.append("❓ Debt-to-Equity: Not available")
        
        # Quick Ratio (3 points)
        quick_ratio = info.get('quickRatio')
        if quick_ratio:
            if quick_ratio >= 1.5:
                score += 3
                details.append("✅ Quick Ratio: Strong (>= 1.5)")
            elif quick_ratio >= 1.0:
                score += 2
                details.append("✅ Quick Ratio: Good (>= 1.0)")
            elif quick_ratio >= 0.8:
                score += 1
                details.append("⚠️ Quick Ratio: Adequate (>= 0.8)")
            else:
                score += 0
                details.append("❌ Quick Ratio: Weak (< 0.8)")
        else:
            details.append("❓ Quick Ratio: Not available")
        
        # Cash Position (4 points)
        total_cash = info.get('totalCash')
        total_debt = info.get('totalDebt')
        if total_cash and total_debt:
            cash_to_debt = total_cash / total_debt if total_debt > 0 else float('inf')
            if cash_to_debt >= 1.0:
                score += 4
                details.append("✅ Cash Position: Excellent (Cash > Debt)")
            elif cash_to_debt >= 0.5:
                score += 3
                details.append("✅ Cash Position: Good (Cash >= 50% of Debt)")
            elif cash_to_debt >= 0.25:
                score += 2
                details.append("⚠️ Cash Position: Moderate (Cash >= 25% of Debt)")
            else:
                score += 1
                details.append("⚠️ Cash Position: Limited (Cash < 25% of Debt)")
        else:
            details.append("❓ Cash Position: Insufficient data")
        
        # Revenue Growth Stability (5 points)
        revenue_growth = info.get('revenueGrowth')
        quarterly_revenue_growth = info.get('revenueQuarterlyGrowth')
        if revenue_growth and quarterly_revenue_growth:
            if revenue_growth > 0 and quarterly_revenue_growth > 0:
                if abs(revenue_growth - quarterly_revenue_growth) < 0.1:  # Consistent growth
                    score += 5
                    details.append("✅ Revenue Growth: Consistent and positive")
                else:
                    score += 3
                    details.append("✅ Revenue Growth: Positive but variable")
            elif revenue_growth > 0 or quarterly_revenue_growth > 0:
                score += 2
                details.append("⚠️ Revenue Growth: Mixed signals")
            else:
                score += 0
                details.append("❌ Revenue Growth: Declining")
        else:
            details.append("❓ Revenue Growth: Insufficient data")
        
        return score, details
    
    def _score_valuation(self, info: dict) -> tuple:
        """Score valuation metrics"""
        score = 0
        details = []
        
        # P/E Ratio (5 points)
        trailing_pe = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        
        pe_to_use = forward_pe if forward_pe else trailing_pe
        if pe_to_use and pe_to_use > 0:
            if pe_to_use <= 15:
                score += 5
                details.append("✅ P/E Ratio: Attractive (<= 15)")
            elif pe_to_use <= 20:
                score += 4
                details.append("✅ P/E Ratio: Reasonable (<= 20)")
            elif pe_to_use <= 25:
                score += 3
                details.append("⚠️ P/E Ratio: Fair (<= 25)")
            elif pe_to_use <= 35:
                score += 2
                details.append("⚠️ P/E Ratio: High (<= 35)")
            else:
                score += 0
                details.append("❌ P/E Ratio: Very High (> 35)")
        else:
            details.append("❓ P/E Ratio: Not available or negative")
        
        # PEG Ratio (5 points)
        peg_ratio = info.get('pegRatio')
        if peg_ratio and peg_ratio > 0:
            if peg_ratio <= 0.5:
                score += 5
                details.append("✅ PEG Ratio: Undervalued (<= 0.5)")
            elif peg_ratio <= 1.0:
                score += 4
                details.append("✅ PEG Ratio: Fair Value (<= 1.0)")
            elif peg_ratio <= 1.5:
                score += 2
                details.append("⚠️ PEG Ratio: Somewhat Expensive (<= 1.5)")
            else:
                score += 0
                details.append("❌ PEG Ratio: Overvalued (> 1.5)")
        else:
            details.append("❓ PEG Ratio: Not available")
        
        # Price-to-Book (3 points)
        price_to_book = info.get('priceToBook')
        if price_to_book and price_to_book > 0:
            if price_to_book <= 1.0:
                score += 3
                details.append("✅ P/B Ratio: Undervalued (<= 1.0)")
            elif price_to_book <= 2.0:
                score += 2
                details.append("✅ P/B Ratio: Reasonable (<= 2.0)")
            elif price_to_book <= 3.0:
                score += 1
                details.append("⚠️ P/B Ratio: High (<= 3.0)")
            else:
                score += 0
                details.append("❌ P/B Ratio: Very High (> 3.0)")
        else:
            details.append("❓ P/B Ratio: Not available")
        
        # Price-to-Sales (3 points)
        price_to_sales = info.get('priceToSalesTrailing12Months')
        if price_to_sales and price_to_sales > 0:
            if price_to_sales <= 2.0:
                score += 3
                details.append("✅ P/S Ratio: Attractive (<= 2.0)")
            elif price_to_sales <= 4.0:
                score += 2
                details.append("✅ P/S Ratio: Reasonable (<= 4.0)")
            elif price_to_sales <= 6.0:
                score += 1
                details.append("⚠️ P/S Ratio: High (<= 6.0)")
            else:
                score += 0
                details.append("❌ P/S Ratio: Very High (> 6.0)")
        else:
            details.append("❓ P/S Ratio: Not available")
        
        # Enterprise Value Ratios (4 points)
        ev_to_revenue = info.get('enterpriseToRevenue')
        ev_to_ebitda = info.get('enterpriseToEbitda')
        
        if ev_to_revenue and ev_to_revenue > 0:
            if ev_to_revenue <= 3.0:
                score += 2
                details.append("✅ EV/Revenue: Good (<= 3.0)")
            elif ev_to_revenue <= 6.0:
                score += 1
                details.append("⚠️ EV/Revenue: Moderate (<= 6.0)")
            else:
                score += 0
                details.append("❌ EV/Revenue: High (> 6.0)")
        else:
            details.append("❓ EV/Revenue: Not available")
        
        if ev_to_ebitda and ev_to_ebitda > 0:
            if ev_to_ebitda <= 12:
                score += 2
                details.append("✅ EV/EBITDA: Attractive (<= 12)")
            elif ev_to_ebitda <= 18:
                score += 1
                details.append("⚠️ EV/EBITDA: Fair (<= 18)")
            else:
                score += 0
                details.append("❌ EV/EBITDA: Expensive (> 18)")
        else:
            details.append("❓ EV/EBITDA: Not available")
        
        return score, details
    
    def _score_growth(self, info: dict) -> tuple:
        """Score growth metrics"""
        score = 0
        details = []
        
        # Revenue Growth (6 points)
        revenue_growth = info.get('revenueGrowth')
        if revenue_growth:
            if revenue_growth >= 0.20:
                score += 6
                details.append("✅ Revenue Growth: Excellent (>= 20%)")
            elif revenue_growth >= 0.10:
                score += 4
                details.append("✅ Revenue Growth: Good (>= 10%)")
            elif revenue_growth >= 0.05:
                score += 2
                details.append("⚠️ Revenue Growth: Moderate (>= 5%)")
            elif revenue_growth >= 0:
                score += 1
                details.append("⚠️ Revenue Growth: Slow (>= 0%)")
            else:
                score += 0
                details.append("❌ Revenue Growth: Declining (< 0%)")
        else:
            details.append("❓ Revenue Growth: Not available")
        
        # Earnings Growth (6 points)
        earnings_growth = info.get('earningsGrowth')
        if earnings_growth:
            if earnings_growth >= 0.25:
                score += 6
                details.append("✅ Earnings Growth: Excellent (>= 25%)")
            elif earnings_growth >= 0.15:
                score += 4
                details.append("✅ Earnings Growth: Good (>= 15%)")
            elif earnings_growth >= 0.05:
                score += 2
                details.append("⚠️ Earnings Growth: Moderate (>= 5%)")
            elif earnings_growth >= 0:
                score += 1
                details.append("⚠️ Earnings Growth: Slow (>= 0%)")
            else:
                score += 0
                details.append("❌ Earnings Growth: Declining (< 0%)")
        else:
            details.append("❓ Earnings Growth: Not available")
        
        # Quarterly Growth Consistency (4 points)
        quarterly_revenue_growth = info.get('revenueQuarterlyGrowth')
        quarterly_earnings_growth = info.get('earningsQuarterlyGrowth')
        
        consistency_score = 0
        if quarterly_revenue_growth and quarterly_revenue_growth > 0:
            consistency_score += 2
            details.append("✅ Quarterly Revenue: Growing")
        elif quarterly_revenue_growth and quarterly_revenue_growth <= 0:
            details.append("❌ Quarterly Revenue: Declining")
        
        if quarterly_earnings_growth and quarterly_earnings_growth > 0:
            consistency_score += 2
            details.append("✅ Quarterly Earnings: Growing")
        elif quarterly_earnings_growth and quarterly_earnings_growth <= 0:
            details.append("❌ Quarterly Earnings: Declining")
        
        score += consistency_score
        
        # Market Share Growth Proxy (4 points) - using relative metrics
        market_cap = info.get('marketCap')
        enterprise_value = info.get('enterpriseValue')
        
        if market_cap and enterprise_value:
            # Companies with strong market position typically have market cap close to EV
            market_strength = min(market_cap / enterprise_value, enterprise_value / market_cap) if enterprise_value > 0 else 0
            
            if market_strength >= 0.8:
                score += 4
                details.append("✅ Market Position: Strong market presence")
            elif market_strength >= 0.6:
                score += 3
                details.append("✅ Market Position: Good market presence")
            elif market_strength >= 0.4:
                score += 2
                details.append("⚠️ Market Position: Moderate market presence")
            else:
                score += 1
                details.append("⚠️ Market Position: Weak market presence")
        else:
            details.append("❓ Market Position: Insufficient data")
        
        return score, details
    
    def _score_profitability(self, info: dict) -> tuple:
        """Score profitability metrics"""
        score = 0
        details = []
        
        # Profit Margins (5 points)
        profit_margins = info.get('profitMargins')
        if profit_margins:
            if profit_margins >= 0.20:
                score += 5
                details.append("✅ Net Margin: Excellent (>= 20%)")
            elif profit_margins >= 0.15:
                score += 4
                details.append("✅ Net Margin: Very Good (>= 15%)")
            elif profit_margins >= 0.10:
                score += 3
                details.append("✅ Net Margin: Good (>= 10%)")
            elif profit_margins >= 0.05:
                score += 2
                details.append("⚠️ Net Margin: Moderate (>= 5%)")
            else:
                score += 0
                details.append("❌ Net Margin: Poor (< 5%)")
        else:
            details.append("❓ Net Margin: Not available")
        
        # Operating Margins (4 points)
        operating_margins = info.get('operatingMargins')
        if operating_margins:
            if operating_margins >= 0.25:
                score += 4
                details.append("✅ Operating Margin: Excellent (>= 25%)")
            elif operating_margins >= 0.15:
                score += 3
                details.append("✅ Operating Margin: Good (>= 15%)")
            elif operating_margins >= 0.08:
                score += 2
                details.append("⚠️ Operating Margin: Moderate (>= 8%)")
            else:
                score += 0
                details.append("❌ Operating Margin: Poor (< 8%)")
        else:
            details.append("❓ Operating Margin: Not available")
        
        # Return on Equity (4 points)
        return_on_equity = info.get('returnOnEquity')
        if return_on_equity:
            if return_on_equity >= 0.20:
                score += 4
                details.append("✅ ROE: Excellent (>= 20%)")
            elif return_on_equity >= 0.15:
                score += 3
                details.append("✅ ROE: Good (>= 15%)")
            elif return_on_equity >= 0.10:
                score += 2
                details.append("⚠️ ROE: Moderate (>= 10%)")
            else:
                score += 0
                details.append("❌ ROE: Poor (< 10%)")
        else:
            details.append("❓ ROE: Not available")
        
        # Return on Assets (3 points)
        return_on_assets = info.get('returnOnAssets')
        if return_on_assets:
            if return_on_assets >= 0.15:
                score += 3
                details.append("✅ ROA: Excellent (>= 15%)")
            elif return_on_assets >= 0.08:
                score += 2
                details.append("✅ ROA: Good (>= 8%)")
            elif return_on_assets >= 0.04:
                score += 1
                details.append("⚠️ ROA: Moderate (>= 4%)")
            else:
                score += 0
                details.append("❌ ROA: Poor (< 4%)")
        else:
            details.append("❓ ROA: Not available")
        
        # Gross Margins (4 points)
        gross_margins = info.get('grossMargins')
        if gross_margins:
            if gross_margins >= 0.60:
                score += 4
                details.append("✅ Gross Margin: Excellent (>= 60%)")
            elif gross_margins >= 0.40:
                score += 3
                details.append("✅ Gross Margin: Good (>= 40%)")
            elif gross_margins >= 0.25:
                score += 2
                details.append("⚠️ Gross Margin: Moderate (>= 25%)")
            else:
                score += 1
                details.append("⚠️ Gross Margin: Low (< 25%)")
        else:
            details.append("❓ Gross Margin: Not available")
        
        return score, details
    
    def _score_market_position(self, info: dict) -> tuple:
        """Score market position and competitive metrics"""
        score = 0
        details = []
        
        # Market Cap Size (4 points)
        market_cap = info.get('marketCap')
        if market_cap:
            if market_cap >= 200e9:  # $200B+
                score += 4
                details.append("✅ Market Cap: Large Cap (>$200B)")
            elif market_cap >= 10e9:  # $10B+
                score += 3
                details.append("✅ Market Cap: Mid-Large Cap (>$10B)")
            elif market_cap >= 2e9:  # $2B+
                score += 2
                details.append("⚠️ Market Cap: Small-Mid Cap (>$2B)")
            else:
                score += 1
                details.append("⚠️ Market Cap: Small Cap (<$2B)")
        else:
            details.append("❓ Market Cap: Not available")
        
        # Beta/Volatility (3 points)
        beta = info.get('beta')
        if beta:
            if 0.5 <= beta <= 1.2:
                score += 3
                details.append("✅ Beta: Stable (0.5-1.2)")
            elif 0.3 <= beta <= 1.5:
                score += 2
                details.append("⚠️ Beta: Moderate volatility (0.3-1.5)")
            else:
                score += 1
                details.append("⚠️ Beta: High volatility (outside 0.3-1.5)")
        else:
            details.append("❓ Beta: Not available")
        
        # Analyst Coverage (3 points)
        num_analyst_opinions = info.get('numberOfAnalystOpinions')
        if num_analyst_opinions:
            if num_analyst_opinions >= 20:
                score += 3
                details.append("✅ Analyst Coverage: Extensive (20+ analysts)")
            elif num_analyst_opinions >= 10:
                score += 2
                details.append("✅ Analyst Coverage: Good (10+ analysts)")
            elif num_analyst_opinions >= 5:
                score += 1
                details.append("⚠️ Analyst Coverage: Limited (5+ analysts)")
            else:
                score += 0
                details.append("❌ Analyst Coverage: Minimal (<5 analysts)")
        else:
            details.append("❓ Analyst Coverage: Not available")
        
        # Trading Volume/Liquidity (3 points)
        avg_volume = info.get('averageVolume')
        if avg_volume:
            if avg_volume >= 5000000:  # 5M+ shares
                score += 3
                details.append("✅ Liquidity: High (5M+ avg volume)")
            elif avg_volume >= 1000000:  # 1M+ shares
                score += 2
                details.append("✅ Liquidity: Good (1M+ avg volume)")
            elif avg_volume >= 100000:  # 100K+ shares
                score += 1
                details.append("⚠️ Liquidity: Moderate (100K+ avg volume)")
            else:
                score += 0
                details.append("❌ Liquidity: Low (<100K avg volume)")
        else:
            details.append("❓ Liquidity: Not available")
        
        # Dividend Consistency (3 points)
        dividend_rate = info.get('dividendRate')
        dividend_yield = info.get('dividendYield')
        payout_ratio = info.get('payoutRatio')
        
        dividend_score = 0
        if dividend_rate and dividend_rate > 0:
            dividend_score += 1
            if dividend_yield and 0.02 <= dividend_yield <= 0.06:  # 2-6% yield
                dividend_score += 1
                if payout_ratio and 0.3 <= payout_ratio <= 0.7:  # 30-70% payout
                    dividend_score += 1
                    details.append("✅ Dividend: Consistent & sustainable")
                else:
                    details.append("⚠️ Dividend: Good yield, check sustainability")
            else:
                details.append("⚠️ Dividend: Paying but yield may be extreme")
        else:
            details.append("ℹ️ Dividend: No dividend (growth focus)")
        
        score += dividend_score
        
        # Share Buyback Activity (4 points) - proxy using shares outstanding trend
        shares_outstanding = info.get('sharesOutstanding')
        if shares_outstanding:
            # This is a simplified approach - in practice, you'd compare historical data
            score += 2  # Neutral score as we can't determine trend from single point
            details.append("ℹ️ Share Count: Data available (trend analysis needed)")
        else:
            details.append("❓ Share Count: Not available")
        
        return score, details
    
    def _get_overall_assessment(self, percentage: float) -> tuple:
        """Get overall investment assessment"""
        if percentage >= 90:
            return "A+", "Strong Buy - Exceptional Quality", "Very Low"
        elif percentage >= 80:
            return "A", "Buy - High Quality", "Low"
        elif percentage >= 70:
            return "B+", "Buy - Above Average Quality", "Low-Medium"
        elif percentage >= 60:
            return "B", "Hold/Buy - Average Quality", "Medium"
        elif percentage >= 50:
            return "C+", "Hold - Below Average Quality", "Medium-High"
        elif percentage >= 40:
            return "C", "Hold/Avoid - Poor Quality", "High"
        else:
            return "D", "Avoid - Very Poor Quality", "Very High"
    
    def _identify_strengths_weaknesses(self, scores: dict, max_scores: dict) -> str:
        """Identify key strengths and weaknesses"""
        result = "🎯 **KEY INSIGHTS:**\n\n"
        
        # Calculate percentages for each category
        percentages = {}
        for category in scores:
            percentages[category] = (scores[category] / max_scores[category]) * 100
        
        # Identify top strengths (>= 75%)
        strengths = [(cat, pct) for cat, pct in percentages.items() if pct >= 75]
        strengths.sort(key=lambda x: x[1], reverse=True)
        
        # Identify weaknesses (< 50%)
        weaknesses = [(cat, pct) for cat, pct in percentages.items() if pct < 50]
        weaknesses.sort(key=lambda x: x[1])
        
        # Display strengths
        if strengths:
            result += "💪 **Key Strengths:**\n"
            for category, percentage in strengths[:3]:  # Top 3
                category_name = category.replace('_', ' ').title()
                result += f"  ✅ {category_name}: {percentage:.0f}%\n"
            result += "\n"
        
        # Display weaknesses
        if weaknesses:
            result += "⚠️ **Areas of Concern:**\n"
            for category, percentage in weaknesses[:3]:  # Top 3
                category_name = category.replace('_', ' ').title()
                result += f"  ❌ {category_name}: {percentage:.0f}%\n"
            result += "\n"
        
        # Investment recommendation based on pattern
        if len(strengths) >= 3 and len(weaknesses) <= 1:
            result += "🎯 **Pattern**: Well-rounded investment with multiple strengths\n"
        elif percentages.get('growth', 0) >= 75 and percentages.get('profitability', 0) >= 75:
            result += "🎯 **Pattern**: Growth stock with strong profitability\n"
        elif percentages.get('valuation', 0) >= 75 and percentages.get('financial_health', 0) >= 75:
            result += "🎯 **Pattern**: Value stock with solid fundamentals\n"
        elif len(weaknesses) >= 3:
            result += "🎯 **Pattern**: Multiple concerns require careful analysis\n"
        else:
            result += "🎯 **Pattern**: Mixed signals - detailed analysis recommended\n"
        
        return result
    
    def _calculate_value_score(self, symbol: str, info: dict) -> str:
        """Calculate value-focused investment score"""
        result = f"💰 **{symbol} - Value Investment Score**\n\n"
        # Implementation would focus on valuation metrics
        # This is a simplified version
        return result + "Value scoring feature - detailed implementation pending"
    
    def _calculate_growth_score(self, symbol: str, info: dict) -> str:
        """Calculate growth-focused investment score"""
        result = f"📈 **{symbol} - Growth Investment Score**\n\n"
        # Implementation would focus on growth metrics
        return result + "Growth scoring feature - detailed implementation pending"
    
    def _calculate_quality_score(self, symbol: str, info: dict) -> str:
        """Calculate quality-focused investment score"""
        result = f"⭐ **{symbol} - Quality Investment Score**\n\n"
        # Implementation would focus on quality metrics
        return result + "Quality scoring feature - detailed implementation pending"


def create_yfinance_quality_score_tool() -> YFinanceQualityScoreTool:
    """Factory function to create YFinance quality score tool"""
    return YFinanceQualityScoreTool()