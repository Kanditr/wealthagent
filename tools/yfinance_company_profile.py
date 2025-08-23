"""
YFinance Company Overview & Profile Tool for LangChain Integration
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
    logger.warning("yfinance not installed. Company profile features will be disabled.")


class YFinanceCompanyProfileInput(BaseModel):
    """Input schema for YFinance company profile tool"""
    symbol: str = Field(description="Stock symbol (e.g., 'AAPL', 'TSLA', 'GOOGL')")


class YFinanceCompanyProfileTool(BaseTool):
    """Tool for retrieving comprehensive company information from YFinance"""
    
    name: str = "yfinance_company_profile"
    description: str = (
        "Get comprehensive company overview and profile information from Yahoo Finance. "
        "Provides company description, sector, industry, market cap, employee count, "
        "headquarters location, key executives, and basic trading information. "
        "Use this for initial company research and due diligence."
    )
    args_schema: Type[BaseModel] = YFinanceCompanyProfileInput
    
    def _run(self, symbol: str) -> str:
        """Execute the company profile tool"""
        if not YFINANCE_AVAILABLE:
            return "❌ yfinance not available. Please install yfinance to use company profile features."
        
        try:
            # Get ticker object and info
            ticker = yf.Ticker(symbol.upper())
            info = ticker.info
            
            # Check if we got valid data
            if not info or len(info) < 5:
                return f"❌ No company information found for symbol '{symbol.upper()}'. Please verify the ticker symbol."
            
            return self._format_company_profile(symbol.upper(), info)
        
        except Exception as e:
            logger.error(f"YFinance error for {symbol}: {e}")
            return f"❌ Error retrieving company information for {symbol.upper()}: {str(e)}"
    
    def _format_company_profile(self, symbol: str, info: dict) -> str:
        """Format company profile information for display"""
        
        # Company Basic Information
        company_name = info.get('longName', info.get('shortName', symbol))
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        exchange = info.get('exchange', 'N/A')
        
        # Market Data
        market_cap = info.get('marketCap')
        enterprise_value = info.get('enterpriseValue')
        shares_outstanding = info.get('sharesOutstanding')
        current_price = info.get('currentPrice', info.get('regularMarketPrice'))
        
        # Company Details
        employees = info.get('fullTimeEmployees')
        business_summary = info.get('longBusinessSummary', 'N/A')
        website = info.get('website', 'N/A')
        
        # Location
        city = info.get('city', 'N/A')
        state = info.get('state', 'N/A')
        country = info.get('country', 'N/A')
        
        # Format the response
        result = f"🏢 **{company_name} ({symbol})** Company Profile\n\n"
        
        # Basic Information
        result += "📋 **Company Information:**\n"
        result += f"🏷️ Sector: {sector}\n"
        result += f"🏭 Industry: {industry}\n"
        result += f"📈 Exchange: {exchange}\n"
        if employees:
            result += f"👥 Employees: {employees:,}\n"
        
        # Location
        location_parts = [part for part in [city, state, country] if part != 'N/A']
        if location_parts:
            result += f"🌍 Headquarters: {', '.join(location_parts)}\n"
        
        if website != 'N/A':
            result += f"🌐 Website: {website}\n"
        
        result += "\n"
        
        # Market Data
        result += "💰 **Market Information:**\n"
        if current_price:
            result += f"💵 Current Price: ${current_price:,.2f}\n"
        
        if market_cap:
            market_cap_formatted = self._format_large_number(market_cap)
            result += f"🏛️ Market Cap: {market_cap_formatted}\n"
        
        if enterprise_value:
            ev_formatted = self._format_large_number(enterprise_value)
            result += f"🏢 Enterprise Value: {ev_formatted}\n"
        
        if shares_outstanding:
            shares_formatted = self._format_large_number(shares_outstanding, is_shares=True)
            result += f"📊 Shares Outstanding: {shares_formatted}\n"
        
        result += "\n"
        
        # Business Summary
        if business_summary != 'N/A' and len(business_summary) > 10:
            result += "📝 **Business Summary:**\n"
            # Truncate summary if too long
            if len(business_summary) > 500:
                summary = business_summary[:500] + "..."
            else:
                summary = business_summary
            result += f"{summary}\n\n"
        
        # Key Executives (if available)
        officers = info.get('companyOfficers', [])
        if officers and len(officers) > 0:
            result += "👔 **Key Executives:**\n"
            for i, officer in enumerate(officers[:3]):  # Show top 3
                name = officer.get('name', 'N/A')
                title = officer.get('title', 'N/A')
                age = officer.get('age')
                
                exec_info = f"• **{name}** - {title}"
                if age:
                    exec_info += f" (Age: {age})"
                result += exec_info + "\n"
            
            if len(officers) > 3:
                result += f"• ... and {len(officers) - 3} more executives\n"
            result += "\n"
        
        # Trading Information
        result += "📊 **Trading Information:**\n"
        
        # 52-week range
        fifty_two_week_high = info.get('fiftyTwoWeekHigh')
        fifty_two_week_low = info.get('fiftyTwoWeekLow')
        if fifty_two_week_high and fifty_two_week_low:
            result += f"📈 52-Week Range: ${fifty_two_week_low:.2f} - ${fifty_two_week_high:.2f}\n"
        
        # Volume
        avg_volume = info.get('averageVolume')
        if avg_volume:
            result += f"📊 Average Volume: {avg_volume:,}\n"
        
        # Beta
        beta = info.get('beta')
        if beta:
            result += f"📉 Beta: {beta:.2f}\n"
        
        # Dividend info
        dividend_rate = info.get('dividendRate')
        dividend_yield = info.get('dividendYield')
        if dividend_rate:
            result += f"💸 Dividend Rate: ${dividend_rate:.2f}\n"
        if dividend_yield:
            result += f"💰 Dividend Yield: {dividend_yield:.2%}\n"
        
        return result.strip()
    
    def _format_large_number(self, number: float, is_shares: bool = False) -> str:
        """Format large numbers with appropriate units"""
        if number >= 1e12:
            return f"${number/1e12:.2f}T" if not is_shares else f"{number/1e12:.2f}T shares"
        elif number >= 1e9:
            return f"${number/1e9:.2f}B" if not is_shares else f"{number/1e9:.2f}B shares"
        elif number >= 1e6:
            return f"${number/1e6:.2f}M" if not is_shares else f"{number/1e6:.2f}M shares"
        elif number >= 1e3:
            return f"${number/1e3:.2f}K" if not is_shares else f"{number/1e3:.2f}K shares"
        else:
            return f"${number:.2f}" if not is_shares else f"{number:,.0f} shares"


def create_yfinance_company_profile_tool() -> YFinanceCompanyProfileTool:
    """Factory function to create YFinance company profile tool"""
    return YFinanceCompanyProfileTool()