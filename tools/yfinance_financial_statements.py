"""
YFinance Financial Statements Tool for LangChain Integration
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
    logger.warning("yfinance not installed. Financial statements features will be disabled.")


class YFinanceFinancialStatementsInput(BaseModel):
    """Input schema for YFinance financial statements tool"""
    symbol: str = Field(description="Stock symbol (e.g., 'AAPL', 'TSLA', 'GOOGL')")
    statement: Optional[str] = Field(
        default="all", 
        description="Statement type: 'income', 'balance', 'cashflow', or 'all' for summary"
    )
    period: Optional[str] = Field(
        default="annual", 
        description="Period: 'annual' or 'quarterly'"
    )


class YFinanceFinancialStatementsTool(BaseTool):
    """Tool for retrieving financial statements from YFinance"""
    
    name: str = "yfinance_financial_statements"
    description: str = (
        "Get comprehensive financial statements from Yahoo Finance. "
        "Provides income statement, balance sheet, and cash flow statement data. "
        "Can retrieve annual or quarterly data. Essential for fundamental analysis "
        "and detailed financial performance evaluation."
    )
    args_schema: Type[BaseModel] = YFinanceFinancialStatementsInput
    
    def _run(
        self, 
        symbol: str, 
        statement: str = "all", 
        period: str = "annual"
    ) -> str:
        """Execute the financial statements tool"""
        if not YFINANCE_AVAILABLE:
            return "❌ yfinance not available. Please install yfinance to use financial statements features."
        
        try:
            ticker = yf.Ticker(symbol.upper())
            
            # Validate period
            if period.lower() not in ["annual", "quarterly"]:
                return "❌ Invalid period. Use 'annual' or 'quarterly'."
            
            # Get statements based on request
            if statement.lower() == "all":
                return self._get_financial_summary(ticker, symbol.upper(), period)
            elif statement.lower() == "income":
                return self._get_income_statement(ticker, symbol.upper(), period)
            elif statement.lower() == "balance":
                return self._get_balance_sheet(ticker, symbol.upper(), period)
            elif statement.lower() == "cashflow":
                return self._get_cash_flow(ticker, symbol.upper(), period)
            else:
                return "❌ Invalid statement type. Use 'income', 'balance', 'cashflow', or 'all'."
        
        except Exception as e:
            logger.error(f"YFinance error for {symbol}: {e}")
            return f"❌ Error retrieving financial statements for {symbol.upper()}: {str(e)}"
    
    def _get_financial_summary(self, ticker, symbol: str, period: str) -> str:
        """Get summary of all financial statements"""
        try:
            # Get all statements
            if period.lower() == "quarterly":
                income_stmt = ticker.quarterly_income_stmt
                balance_sheet = ticker.quarterly_balance_sheet
                cash_flow = ticker.quarterly_cashflow
                period_label = "Quarterly"
            else:
                income_stmt = ticker.income_stmt
                balance_sheet = ticker.balance_sheet
                cash_flow = ticker.cashflow
                period_label = "Annual"
            
            if income_stmt.empty and balance_sheet.empty and cash_flow.empty:
                return f"❌ No financial statement data found for {symbol}."
            
            result = f"📊 **{symbol} {period_label} Financial Statements Summary**\n\n"
            
            # Income Statement Key Metrics
            if not income_stmt.empty:
                result += "💰 **Income Statement Highlights:**\n"
                latest_col = income_stmt.columns[0] if len(income_stmt.columns) > 0 else None
                
                if latest_col is not None:
                    revenue = income_stmt.loc['Total Revenue', latest_col] if 'Total Revenue' in income_stmt.index else None
                    gross_profit = income_stmt.loc['Gross Profit', latest_col] if 'Gross Profit' in income_stmt.index else None
                    operating_income = income_stmt.loc['Operating Income', latest_col] if 'Operating Income' in income_stmt.index else None
                    net_income = income_stmt.loc['Net Income', latest_col] if 'Net Income' in income_stmt.index else None
                    
                    if revenue is not None:
                        result += f"📈 Total Revenue: {self._format_currency(revenue)}\n"
                    if gross_profit is not None:
                        result += f"💵 Gross Profit: {self._format_currency(gross_profit)}\n"
                    if operating_income is not None:
                        result += f"⚙️ Operating Income: {self._format_currency(operating_income)}\n"
                    if net_income is not None:
                        result += f"🎯 Net Income: {self._format_currency(net_income)}\n"
                    
                    result += f"📅 Period: {latest_col.strftime('%Y-%m-%d')}\n"
                
                result += "\n"
            
            # Balance Sheet Key Metrics
            if not balance_sheet.empty:
                result += "🏦 **Balance Sheet Highlights:**\n"
                latest_col = balance_sheet.columns[0] if len(balance_sheet.columns) > 0 else None
                
                if latest_col is not None:
                    total_assets = balance_sheet.loc['Total Assets', latest_col] if 'Total Assets' in balance_sheet.index else None
                    total_debt = balance_sheet.loc['Total Debt', latest_col] if 'Total Debt' in balance_sheet.index else None
                    cash_equiv = balance_sheet.loc['Cash And Cash Equivalents', latest_col] if 'Cash And Cash Equivalents' in balance_sheet.index else None
                    stockholder_equity = balance_sheet.loc['Stockholders Equity', latest_col] if 'Stockholders Equity' in balance_sheet.index else None
                    
                    if total_assets is not None:
                        result += f"🏛️ Total Assets: {self._format_currency(total_assets)}\n"
                    if cash_equiv is not None:
                        result += f"💵 Cash & Equivalents: {self._format_currency(cash_equiv)}\n"
                    if total_debt is not None:
                        result += f"💳 Total Debt: {self._format_currency(total_debt)}\n"
                    if stockholder_equity is not None:
                        result += f"👥 Stockholders' Equity: {self._format_currency(stockholder_equity)}\n"
                    
                    result += f"📅 Period: {latest_col.strftime('%Y-%m-%d')}\n"
                
                result += "\n"
            
            # Cash Flow Key Metrics
            if not cash_flow.empty:
                result += "💸 **Cash Flow Highlights:**\n"
                latest_col = cash_flow.columns[0] if len(cash_flow.columns) > 0 else None
                
                if latest_col is not None:
                    operating_cf = cash_flow.loc['Operating Cash Flow', latest_col] if 'Operating Cash Flow' in cash_flow.index else None
                    investing_cf = cash_flow.loc['Investing Cash Flow', latest_col] if 'Investing Cash Flow' in cash_flow.index else None
                    financing_cf = cash_flow.loc['Financing Cash Flow', latest_col] if 'Financing Cash Flow' in cash_flow.index else None
                    free_cf = cash_flow.loc['Free Cash Flow', latest_col] if 'Free Cash Flow' in cash_flow.index else None
                    
                    if operating_cf is not None:
                        result += f"⚙️ Operating Cash Flow: {self._format_currency(operating_cf)}\n"
                    if investing_cf is not None:
                        result += f"💼 Investing Cash Flow: {self._format_currency(investing_cf)}\n"
                    if financing_cf is not None:
                        result += f"🏦 Financing Cash Flow: {self._format_currency(financing_cf)}\n"
                    if free_cf is not None:
                        result += f"🎯 Free Cash Flow: {self._format_currency(free_cf)}\n"
                    
                    result += f"📅 Period: {latest_col.strftime('%Y-%m-%d')}\n"
            
            return result.strip()
        
        except Exception as e:
            return f"❌ Error generating financial summary: {str(e)}"
    
    def _get_income_statement(self, ticker, symbol: str, period: str) -> str:
        """Get detailed income statement"""
        try:
            if period.lower() == "quarterly":
                income_stmt = ticker.quarterly_income_stmt
                period_label = "Quarterly"
            else:
                income_stmt = ticker.income_stmt
                period_label = "Annual"
            
            if income_stmt.empty:
                return f"❌ No income statement data found for {symbol}."
            
            result = f"💰 **{symbol} {period_label} Income Statement**\n\n"
            
            # Show data for the most recent periods (up to 3)
            num_periods = min(3, len(income_stmt.columns))
            periods = income_stmt.columns[:num_periods]
            
            result += f"📊 **Showing {num_periods} most recent periods**\n\n"
            
            # Key income statement items
            key_items = [
                ('Total Revenue', '📈 Total Revenue'),
                ('Cost Of Revenue', '💸 Cost of Revenue'),
                ('Gross Profit', '💵 Gross Profit'),
                ('Research And Development', '🔬 R&D Expenses'),
                ('Selling General And Administrative', '🏢 SG&A Expenses'),
                ('Operating Income', '⚙️ Operating Income'),
                ('Interest Expense', '💳 Interest Expense'),
                ('Tax Provision', '🏛️ Tax Provision'),
                ('Net Income', '🎯 Net Income'),
                ('Diluted EPS', '📊 Diluted EPS')
            ]
            
            for item_key, item_label in key_items:
                if item_key in income_stmt.index:
                    result += f"**{item_label}:**\n"
                    values = []
                    for period_date in periods:
                        value = income_stmt.loc[item_key, period_date]
                        if item_key == 'Diluted EPS':
                            values.append(f"${value:.2f}" if value is not None else "N/A")
                        else:
                            values.append(self._format_currency(value) if value is not None else "N/A")
                    
                    # Show periods with dates
                    for i, period_date in enumerate(periods):
                        result += f"  {period_date.strftime('%Y-%m-%d')}: {values[i]}\n"
                    result += "\n"
            
            # Calculate margins if possible
            if 'Total Revenue' in income_stmt.index and 'Gross Profit' in income_stmt.index:
                result += "**📊 Key Margins (Most Recent Period):**\n"
                latest_period = periods[0]
                
                revenue = income_stmt.loc['Total Revenue', latest_period]
                gross_profit = income_stmt.loc['Gross Profit', latest_period]
                operating_income = income_stmt.loc['Operating Income', latest_period] if 'Operating Income' in income_stmt.index else None
                net_income = income_stmt.loc['Net Income', latest_period] if 'Net Income' in income_stmt.index else None
                
                if revenue and revenue != 0:
                    if gross_profit:
                        gross_margin = (gross_profit / revenue) * 100
                        result += f"  Gross Margin: {gross_margin:.1f}%\n"
                    if operating_income:
                        operating_margin = (operating_income / revenue) * 100
                        result += f"  Operating Margin: {operating_margin:.1f}%\n"
                    if net_income:
                        net_margin = (net_income / revenue) * 100
                        result += f"  Net Margin: {net_margin:.1f}%\n"
            
            return result.strip()
        
        except Exception as e:
            return f"❌ Error retrieving income statement: {str(e)}"
    
    def _get_balance_sheet(self, ticker, symbol: str, period: str) -> str:
        """Get detailed balance sheet"""
        try:
            if period.lower() == "quarterly":
                balance_sheet = ticker.quarterly_balance_sheet
                period_label = "Quarterly"
            else:
                balance_sheet = ticker.balance_sheet
                period_label = "Annual"
            
            if balance_sheet.empty:
                return f"❌ No balance sheet data found for {symbol}."
            
            result = f"🏦 **{symbol} {period_label} Balance Sheet**\n\n"
            
            # Show data for the most recent periods (up to 3)
            num_periods = min(3, len(balance_sheet.columns))
            periods = balance_sheet.columns[:num_periods]
            
            result += f"📊 **Showing {num_periods} most recent periods**\n\n"
            
            # Assets
            result += "**🏛️ ASSETS:**\n"
            asset_items = [
                ('Cash And Cash Equivalents', '💵 Cash & Cash Equivalents'),
                ('Short Term Investments', '📊 Short Term Investments'),
                ('Accounts Receivable', '📋 Accounts Receivable'),
                ('Inventory', '📦 Inventory'),
                ('Current Assets', '⚡ Total Current Assets'),
                ('Properties Plant Equipment', '🏭 Property, Plant & Equipment'),
                ('Goodwill', '⭐ Goodwill'),
                ('Total Assets', '🏛️ TOTAL ASSETS')
            ]
            
            for item_key, item_label in asset_items:
                if item_key in balance_sheet.index:
                    result += f"  **{item_label}:**\n"
                    for period_date in periods:
                        value = balance_sheet.loc[item_key, period_date]
                        formatted_value = self._format_currency(value) if value is not None else "N/A"
                        result += f"    {period_date.strftime('%Y-%m-%d')}: {formatted_value}\n"
                    result += "\n"
            
            # Liabilities & Equity
            result += "**💳 LIABILITIES & EQUITY:**\n"
            liability_items = [
                ('Accounts Payable', '📝 Accounts Payable'),
                ('Current Debt', '💳 Current Debt'),
                ('Current Liabilities', '⚡ Total Current Liabilities'),
                ('Long Term Debt', '🏦 Long Term Debt'),
                ('Total Liabilities Net Minority Interest', '💳 Total Liabilities'),
                ('Stockholders Equity', '👥 Stockholders\' Equity'),
                ('Retained Earnings', '💰 Retained Earnings')
            ]
            
            for item_key, item_label in liability_items:
                if item_key in balance_sheet.index:
                    result += f"  **{item_label}:**\n"
                    for period_date in periods:
                        value = balance_sheet.loc[item_key, period_date]
                        formatted_value = self._format_currency(value) if value is not None else "N/A"
                        result += f"    {period_date.strftime('%Y-%m-%d')}: {formatted_value}\n"
                    result += "\n"
            
            # Calculate key ratios if possible
            result += "**📊 Key Balance Sheet Ratios (Most Recent Period):**\n"
            latest_period = periods[0]
            
            current_assets = balance_sheet.loc['Current Assets', latest_period] if 'Current Assets' in balance_sheet.index else None
            current_liabilities = balance_sheet.loc['Current Liabilities', latest_period] if 'Current Liabilities' in balance_sheet.index else None
            total_debt = balance_sheet.loc['Total Debt', latest_period] if 'Total Debt' in balance_sheet.index else None
            stockholder_equity = balance_sheet.loc['Stockholders Equity', latest_period] if 'Stockholders Equity' in balance_sheet.index else None
            
            if current_assets and current_liabilities and current_liabilities != 0:
                current_ratio = current_assets / current_liabilities
                result += f"  Current Ratio: {current_ratio:.2f}\n"
            
            if total_debt and stockholder_equity and stockholder_equity != 0:
                debt_to_equity = total_debt / stockholder_equity
                result += f"  Debt-to-Equity: {debt_to_equity:.2f}\n"
            
            return result.strip()
        
        except Exception as e:
            return f"❌ Error retrieving balance sheet: {str(e)}"
    
    def _get_cash_flow(self, ticker, symbol: str, period: str) -> str:
        """Get detailed cash flow statement"""
        try:
            if period.lower() == "quarterly":
                cash_flow = ticker.quarterly_cashflow
                period_label = "Quarterly"
            else:
                cash_flow = ticker.cashflow
                period_label = "Annual"
            
            if cash_flow.empty:
                return f"❌ No cash flow data found for {symbol}."
            
            result = f"💸 **{symbol} {period_label} Cash Flow Statement**\n\n"
            
            # Show data for the most recent periods (up to 3)
            num_periods = min(3, len(cash_flow.columns))
            periods = cash_flow.columns[:num_periods]
            
            result += f"📊 **Showing {num_periods} most recent periods**\n\n"
            
            # Operating Cash Flow
            result += "**⚙️ OPERATING CASH FLOW:**\n"
            operating_items = [
                ('Net Income', '🎯 Net Income'),
                ('Depreciation And Amortization', '📉 Depreciation & Amortization'),
                ('Changes In Working Capital', '🔄 Changes in Working Capital'),
                ('Operating Cash Flow', '⚙️ Total Operating Cash Flow')
            ]
            
            for item_key, item_label in operating_items:
                if item_key in cash_flow.index:
                    result += f"  **{item_label}:**\n"
                    for period_date in periods:
                        value = cash_flow.loc[item_key, period_date]
                        formatted_value = self._format_currency(value) if value is not None else "N/A"
                        result += f"    {period_date.strftime('%Y-%m-%d')}: {formatted_value}\n"
                    result += "\n"
            
            # Investing Cash Flow
            result += "**💼 INVESTING CASH FLOW:**\n"
            investing_items = [
                ('Capital Expenditure', '🏭 Capital Expenditure'),
                ('Investments In Property Plant And Equipment', '🔧 Investments in PP&E'),
                ('Investing Cash Flow', '💼 Total Investing Cash Flow')
            ]
            
            for item_key, item_label in investing_items:
                if item_key in cash_flow.index:
                    result += f"  **{item_label}:**\n"
                    for period_date in periods:
                        value = cash_flow.loc[item_key, period_date]
                        formatted_value = self._format_currency(value) if value is not None else "N/A"
                        result += f"    {period_date.strftime('%Y-%m-%d')}: {formatted_value}\n"
                    result += "\n"
            
            # Financing Cash Flow
            result += "**🏦 FINANCING CASH FLOW:**\n"
            financing_items = [
                ('Cash Dividends Paid', '💸 Dividends Paid'),
                ('Repurchase Of Capital Stock', '📉 Share Repurchases'),
                ('Long Term Debt Issuance', '💳 Long Term Debt Issued'),
                ('Financing Cash Flow', '🏦 Total Financing Cash Flow')
            ]
            
            for item_key, item_label in financing_items:
                if item_key in cash_flow.index:
                    result += f"  **{item_label}:**\n"
                    for period_date in periods:
                        value = cash_flow.loc[item_key, period_date]
                        formatted_value = self._format_currency(value) if value is not None else "N/A"
                        result += f"    {period_date.strftime('%Y-%m-%d')}: {formatted_value}\n"
                    result += "\n"
            
            # Free Cash Flow
            if 'Free Cash Flow' in cash_flow.index:
                result += "**🎯 FREE CASH FLOW:**\n"
                result += f"  **💎 Free Cash Flow:**\n"
                for period_date in periods:
                    value = cash_flow.loc['Free Cash Flow', period_date]
                    formatted_value = self._format_currency(value) if value is not None else "N/A"
                    result += f"    {period_date.strftime('%Y-%m-%d')}: {formatted_value}\n"
            
            return result.strip()
        
        except Exception as e:
            return f"❌ Error retrieving cash flow statement: {str(e)}"
    
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
                return f"${value:,.0f}"
        except (ValueError, TypeError):
            return "N/A"


def create_yfinance_financial_statements_tool() -> YFinanceFinancialStatementsTool:
    """Factory function to create YFinance financial statements tool"""
    return YFinanceFinancialStatementsTool()