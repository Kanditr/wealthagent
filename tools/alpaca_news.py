"""
Alpaca Market News Tool for LangChain Integration
"""

import os
from typing import Optional, Type, List
from datetime import datetime, timezone, timedelta
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from alpaca.data.historical import NewsClient
    from alpaca.data.requests import NewsRequest
    from alpaca.common.exceptions import APIError
    ALPACA_NEWS_AVAILABLE = True
except ImportError:
    ALPACA_NEWS_AVAILABLE = False
    logger.warning("Alpaca-py news modules not installed. News features will be disabled.")


class AlpacaNewsInput(BaseModel):
    """Input schema for Alpaca news tool"""
    action: str = Field(description="Action to perform: 'latest', 'search'")
    symbols: Optional[str] = Field(default=None, description="Symbol(s) to filter news (e.g., 'AAPL' or 'AAPL,TSLA')")
    limit: Optional[int] = Field(default=10, description="Number of news articles to retrieve (max 50)")
    start_date: Optional[str] = Field(default=None, description="Start date for news search (YYYY-MM-DD format)")
    end_date: Optional[str] = Field(default=None, description="End date for news search (YYYY-MM-DD format)")


class AlpacaNewsTool(BaseTool):
    """Tool for retrieving market news from Alpaca"""
    
    name: str = "alpaca_news"
    description: str = (
        "Get market news from Alpaca. "
        "Actions: 'latest' (recent market news), 'search' (news with date filters). "
        "Can filter by symbols (e.g., 'AAPL,TSLA') or get general market news. "
        "Date format: YYYY-MM-DD. Returns news with headlines, summaries, and sentiment where available."
    )
    args_schema: Type[BaseModel] = AlpacaNewsInput
    
    def _get_client(self):
        """Get Alpaca news client"""
        if not ALPACA_NEWS_AVAILABLE:
            return None
        
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        
        if not api_key or not secret_key:
            return None
        
        try:
            return NewsClient(api_key, secret_key)
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca news client: {e}")
            return None
    
    def _parse_symbols(self, symbols: Optional[str]) -> Optional[List[str]]:
        """Parse symbols string into list"""
        if not symbols:
            return None
        return [s.strip().upper() for s in symbols.split(',') if s.strip()]
    
    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime object"""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    
    def _run(
        self,
        action: str,
        symbols: Optional[str] = None,
        limit: int = 10,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> str:
        """Execute the news tool"""
        client = self._get_client()
        if not client:
            return "❌ Alpaca news not available. Please check your API credentials in .env file."
        
        symbol_list = self._parse_symbols(symbols)
        limit = min(limit, 50)  # Cap at 50 for performance
        
        try:
            if action == "latest":
                return self._get_latest_news(client, symbol_list, limit)
            elif action == "search":
                start_dt = self._parse_date(start_date)
                end_dt = self._parse_date(end_date)
                return self._search_news(client, symbol_list, start_dt, end_dt, limit)
            else:
                return f"❌ Unknown action: {action}. Available actions: latest, search"
        
        except APIError as e:
            logger.error(f"Alpaca API error: {e}")
            return f"❌ Alpaca API error: {str(e)}"
        except Exception as e:
            logger.error(f"News error: {e}")
            return f"❌ News error: {str(e)}"
    
    def _get_latest_news(self, client, symbols: Optional[List[str]], limit: int) -> str:
        """Get latest market news"""
        try:
            # Set default timeframe to last 24 hours for latest news
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(hours=24)
            
            request = NewsRequest(
                symbols=symbols[0] if symbols and len(symbols) == 1 else None,
                start=start_time,
                end=end_time,
                sort="desc",
                include_content=True,
                exclude_contentless=True,
                limit=limit
            )
            
            news_data = client.get_news(request)
            
            return self._format_news_response(news_data, symbols, "Latest News (24h)", start_time, end_time)
        
        except Exception as e:
            return f"❌ Failed to get latest news: {str(e)}"
    
    def _search_news(
        self,
        client,
        symbols: Optional[List[str]],
        start_dt: Optional[datetime],
        end_dt: Optional[datetime],
        limit: int
    ) -> str:
        """Search news with date filters"""
        try:
            # Set default date range if not provided (last 7 days)
            if not end_dt:
                end_dt = datetime.now(timezone.utc)
            if not start_dt:
                start_dt = end_dt - timedelta(days=7)
            
            request = NewsRequest(
                symbols=symbols[0] if symbols and len(symbols) == 1 else None,
                start=start_dt,
                end=end_dt,
                sort="desc",
                include_content=True,
                exclude_contentless=True,
                limit=limit
            )
            
            news_data = client.get_news(request)
            
            return self._format_news_response(news_data, symbols, "News Search", start_dt, end_dt)
        
        except Exception as e:
            return f"❌ Failed to search news: {str(e)}"
    
    def _format_news_response(
        self,
        news_data,
        symbols: Optional[List[str]],
        title: str,
        start_dt: datetime,
        end_dt: datetime
    ) -> str:
        """Format news response"""
        # Convert dates to Thailand timezone for display
        thailand_tz = timezone(timedelta(hours=7))
        start_thai = start_dt.astimezone(thailand_tz)
        end_thai = end_dt.astimezone(thailand_tz)
        
        symbol_filter = f" for {', '.join(symbols)}" if symbols else ""
        
        result = (
            f"📰 **{title}**{symbol_filter}\n"
            f"📅 Period: {start_thai.strftime('%Y-%m-%d %H:%M')} to {end_thai.strftime('%Y-%m-%d %H:%M')} (Thailand time)\n\n"
        )
        
        # Extract news articles from NewsSet data
        news_list = []
        if hasattr(news_data, 'data') and isinstance(news_data.data, dict) and 'news' in news_data.data:
            news_list = news_data.data['news']
        
        if not news_list or len(news_list) == 0:
            result += "❌ No news articles found for the specified criteria."
            return result
        
        result += f"📊 **Found {len(news_list)} articles**\n\n"
        
        for i, article in enumerate(news_list, 1):
            # Convert article timestamp to Thailand timezone
            article_time_thai = article.created_at.astimezone(thailand_tz)
            
            # Get sentiment emoji
            sentiment_emoji = self._get_sentiment_emoji(getattr(article, 'sentiment', None))
            
            # Format article symbols if available
            article_symbols = ""
            if hasattr(article, 'symbols') and article.symbols:
                article_symbols = f" | 🏷️ {', '.join(article.symbols[:3])}"  # Show first 3 symbols
                if len(article.symbols) > 3:
                    article_symbols += f" +{len(article.symbols) - 3} more"
            
            # Format headline and summary
            headline = article.headline[:100] + "..." if len(article.headline) > 100 else article.headline
            
            summary = ""
            if hasattr(article, 'summary') and article.summary:
                summary = article.summary[:200] + "..." if len(article.summary) > 200 else article.summary
                summary = f"\n📝 {summary}"
            
            # Format author and source
            source_info = ""
            if hasattr(article, 'author') and article.author:
                source_info += f" | ✍️ {article.author}"
            if hasattr(article, 'source') and article.source:
                source_info += f" | 📡 {article.source}"
            
            result += (
                f"**{i}.** {sentiment_emoji} **{headline}**\n"
                f"🕒 {article_time_thai.strftime('%m/%d %H:%M')}{article_symbols}{source_info}"
                f"{summary}\n"
            )
            
            # Add URL if available
            if hasattr(article, 'url') and article.url:
                result += f"🔗 [Read more]({article.url})\n"
            
            result += "\n"
        
        return result.strip()
    
    def _get_sentiment_emoji(self, sentiment) -> str:
        """Get emoji for news sentiment"""
        if not sentiment:
            return "📰"
        
        sentiment_str = str(sentiment).lower()
        
        if "positive" in sentiment_str or "bullish" in sentiment_str:
            return "📈"
        elif "negative" in sentiment_str or "bearish" in sentiment_str:
            return "📉"
        elif "neutral" in sentiment_str:
            return "📊"
        else:
            return "📰"


def create_alpaca_news_tool() -> AlpacaNewsTool:
    """Factory function to create Alpaca news tool"""
    return AlpacaNewsTool()