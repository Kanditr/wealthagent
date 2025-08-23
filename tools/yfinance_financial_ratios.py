"""
YFinance Financial Ratios & Valuation Tool for LangChain Integration
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
    logger.warning("yfinance not installed. Financial ratios features will be disabled.")


class YFinanceFinancialRatiosInput(BaseModel):
    """Input schema for YFinance financial ratios tool"""
    symbol: str = Field(description="Stock symbol (e.g., 'AAPL', 'TSLA', 'GOOGL')")


class YFinanceFinancialRatiosTool(BaseTool):
    """Tool for retrieving financial ratios and valuation metrics from YFinance"""
    
    name: str = "yfinance_financial_ratios"
    description: str = (
        "Get comprehensive financial ratios and valuation metrics from Yahoo Finance. "
        "Provides P/E ratios, profitability metrics (ROE, ROA, margins), liquidity ratios, "
        "growth metrics, and market valuation data. Essential for fundamental analysis and "
        "investment decision making."
    )
    args_schema: Type[BaseModel] = YFinanceFinancialRatiosInput
    
    def _run(self, symbol: str) -> str:
        """Execute the financial ratios tool"""
        if not YFINANCE_AVAILABLE:
            return "❌ yfinance not available. Please install yfinance to use financial ratios features."
        
        try:
            # Get ticker object and info
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            
            # Check if we got valid data
            if not info or len(info) < 5:
                return f"❌ No financial data found for symbol '{symbol.upper()}'. Please verify the ticker symbol."
            
            return self._format_financial_ratios(symbol.upper(), info)
        
        except Exception as e:
            logger.error(f"YFinance error for {symbol}: {e}")
            return f"❌ Error retrieving financial ratios for {symbol.upper()}: {str(e)}"
    
    def _format_financial_ratios(self, symbol: str, info: dict) -> str:
        """Format financial ratios and valuation metrics for display"""
        
        company_name = info.get('longName', info.get('shortName', symbol))
        current_price = info.get('currentPrice', info.get('regularMarketPrice'))
        
        result = f"📊 **{company_name} ({symbol})** Financial Ratios & Valuation\n\n"
        
        # Current Price
        if current_price:
            result += f"💵 **Current Price**: ${current_price:.2f}\n\n"
        
        # Valuation Ratios
        result += "🏷️ **Valuation Ratios:**\n"
        
        # P/E Ratios
        trailing_pe = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        if trailing_pe:
            result += f"📈 Trailing P/E: {trailing_pe:.2f}\n"
        if forward_pe:
            result += f"🔮 Forward P/E: {forward_pe:.2f}\n"
        
        # PEG Ratio
        peg_ratio = info.get('pegRatio')
        if peg_ratio:
            result += f"⚖️ PEG Ratio: {peg_ratio:.2f}\n"
        
        # Price-to-Book
        price_to_book = info.get('priceToBook')
        if price_to_book:
            result += f"📚 Price-to-Book: {price_to_book:.2f}\n"
        
        # Price-to-Sales
        price_to_sales = info.get('priceToSalesTrailing12Months')
        if price_to_sales:
            result += f"💰 Price-to-Sales (TTM): {price_to_sales:.2f}\n"
        
        # Enterprise Value ratios
        ev_to_revenue = info.get('enterpriseToRevenue')
        ev_to_ebitda = info.get('enterpriseToEbitda')
        if ev_to_revenue:
            result += f"🏢 EV/Revenue: {ev_to_revenue:.2f}\n"
        if ev_to_ebitda:
            result += f"🏭 EV/EBITDA: {ev_to_ebitda:.2f}\n"
        
        result += "\n"
        
        # Profitability Metrics
        result += "💹 **Profitability Metrics:**\n"
        
        # Return ratios
        return_on_equity = info.get('returnOnEquity')
        return_on_assets = info.get('returnOnAssets')
        if return_on_equity:
            result += f"🔄 Return on Equity (ROE): {return_on_equity:.2%}\n"
        if return_on_assets:
            result += f"📊 Return on Assets (ROA): {return_on_assets:.2%}\n"
        
        # Profit margins
        gross_margins = info.get('grossMargins')
        operating_margins = info.get('operatingMargins')
        profit_margins = info.get('profitMargins')
        ebitda_margins = info.get('ebitdaMargins')
        
        if gross_margins:
            result += f"📈 Gross Margin: {gross_margins:.2%}\n"
        if operating_margins:
            result += f"⚙️ Operating Margin: {operating_margins:.2%}\n"
        if ebitda_margins:
            result += f"🏭 EBITDA Margin: {ebitda_margins:.2%}\n"
        if profit_margins:
            result += f"💰 Net Profit Margin: {profit_margins:.2%}\n"
        
        result += "\n"
        
        # Liquidity & Financial Health
        result += "🏥 **Liquidity & Financial Health:**\n"
        
        # Liquidity ratios
        current_ratio = info.get('currentRatio')
        quick_ratio = info.get('quickRatio')
        if current_ratio:
            result += f"💧 Current Ratio: {current_ratio:.2f}\n"
        if quick_ratio:
            result += f"⚡ Quick Ratio: {quick_ratio:.2f}\n"
        
        # Debt ratios
        debt_to_equity = info.get('debtToEquity')
        if debt_to_equity:
            debt_to_equity_ratio = debt_to_equity / 100  # Convert from percentage
            result += f"🏦 Debt-to-Equity: {debt_to_equity_ratio:.2f}\n"
        
        # Cash and debt
        total_cash = info.get('totalCash')
        total_debt = info.get('totalDebt')
        if total_cash:
            cash_formatted = self._format_large_number(total_cash)
            result += f"💵 Total Cash: {cash_formatted}\n"
        if total_debt:
            debt_formatted = self._format_large_number(total_debt)
            result += f"💳 Total Debt: {debt_formatted}\n"
        
        result += "\n"
        
        # Growth Metrics
        result += "📈 **Growth Metrics:**\n"
        
        revenue_growth = info.get('revenueGrowth')
        earnings_growth = info.get('earningsGrowth')
        if revenue_growth:
            result += f"💰 Revenue Growth (YoY): {revenue_growth:.2%}\n"
        if earnings_growth:
            result += f"📊 Earnings Growth (YoY): {earnings_growth:.2%}\n"
        
        # Quarterly growth
        quarterly_revenue_growth = info.get('revenueQuarterlyGrowth')
        quarterly_earnings_growth = info.get('earningsQuarterlyGrowth')
        if quarterly_revenue_growth:
            result += f"📅 Revenue Growth (QoQ): {quarterly_revenue_growth:.2%}\n"
        if quarterly_earnings_growth:
            result += f"📈 Earnings Growth (QoQ): {quarterly_earnings_growth:.2%}\n"
        
        result += "\n"
        
        # Market & Trading Metrics
        result += "📊 **Market & Trading Metrics:**\n"
        
        # Market cap
        market_cap = info.get('marketCap')
        if market_cap:
            market_cap_formatted = self._format_large_number(market_cap)
            result += f"🏛️ Market Cap: {market_cap_formatted}\n"
        
        # Beta
        beta = info.get('beta')
        if beta:
            risk_level = self._get_beta_risk_level(beta)
            result += f"📉 Beta: {beta:.2f} ({risk_level})\n"
        
        # 52-week performance
        fifty_two_week_high = info.get('fiftyTwoWeekHigh')
        fifty_two_week_low = info.get('fiftyTwoWeekLow')
        if current_price and fifty_two_week_high and fifty_two_week_low:
            high_distance = ((current_price - fifty_two_week_high) / fifty_two_week_high) * 100
            low_distance = ((current_price - fifty_two_week_low) / fifty_two_week_low) * 100
            result += f"📈 52W High: ${fifty_two_week_high:.2f} ({high_distance:+.1f}%)\n"
            result += f"📉 52W Low: ${fifty_two_week_low:.2f} ({low_distance:+.1f}%)\n"
        
        # Dividend metrics
        dividend_rate = info.get('dividendRate')
        dividend_yield = info.get('dividendYield')
        payout_ratio = info.get('payoutRatio')
        
        if dividend_rate or dividend_yield:
            result += "\n💸 **Dividend Information:**\n"
            if dividend_rate:
                result += f"💰 Annual Dividend: ${dividend_rate:.2f}\n"
            if dividend_yield:
                result += f"🎯 Dividend Yield: {dividend_yield:.2%}\n"
            if payout_ratio:
                result += f"📊 Payout Ratio: {payout_ratio:.2%}\n"
        
        # Analyst targets
        target_high = info.get('targetHighPrice')
        target_low = info.get('targetLowPrice')
        target_mean = info.get('targetMeanPrice')
        
        if target_mean:
            result += "\n🎯 **Analyst Targets:**\n"
            if target_mean and current_price:
                upside = ((target_mean - current_price) / current_price) * 100
                result += f"🎯 Mean Target: ${target_mean:.2f} ({upside:+.1f}% upside)\n"
            if target_high:
                result += f"📈 High Target: ${target_high:.2f}\n"
            if target_low:
                result += f"📉 Low Target: ${target_low:.2f}\n"
        
        return result.strip()
    
    def _format_large_number(self, number: float) -> str:
        """Format large numbers with appropriate units"""
        if number >= 1e12:
            return f"${number/1e12:.2f}T"
        elif number >= 1e9:
            return f"${number/1e9:.2f}B"
        elif number >= 1e6:
            return f"${number/1e6:.2f}M"
        elif number >= 1e3:
            return f"${number/1e3:.2f}K"
        else:
            return f"${number:.2f}"
    
    def _get_beta_risk_level(self, beta: float) -> str:
        """Get risk level description based on beta"""
        if beta < 0.5:
            return "Very Low Risk"
        elif beta < 0.8:
            return "Low Risk"
        elif beta < 1.2:
            return "Market Risk"
        elif beta < 1.5:
            return "High Risk"
        else:
            return "Very High Risk"


def create_yfinance_financial_ratios_tool() -> YFinanceFinancialRatiosTool:
    """Factory function to create YFinance financial ratios tool"""
    return YFinanceFinancialRatiosTool()