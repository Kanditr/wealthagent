"""
Tavily Breaking News Tool for LangChain Integration
"""

import os
from typing import Optional, Type, List
from datetime import datetime, timedelta
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from langchain_tavily import TavilySearch
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logger.warning("langchain-tavily not installed. Tavily news features will be disabled.")


class TavilyNewsInput(BaseModel):
    """Input schema for Tavily news tool"""
    action: str = Field(description="Action to perform: 'breaking_news', 'company_research', 'market_events', 'sentiment_analysis'")
    query: Optional[str] = Field(
        default=None, 
        description="Search query. Required for most actions. Examples: 'Apple earnings', 'Federal Reserve interest rates', 'Tesla stock news'"
    )
    symbol: Optional[str] = Field(
        default=None,
        description="Stock symbol for company-specific research (e.g., 'AAPL', 'TSLA'). Used with 'company_research' action."
    )
    days: Optional[int] = Field(
        default=1,
        description="Number of days back to search (1-30). Default is 1 for breaking news, 7 for research."
    )
    max_results: Optional[int] = Field(
        default=5,
        description="Maximum number of results to return (1-20). Default is 5."
    )


class TavilyNewsTool(BaseTool):
    """Tool for real-time breaking news and market research using Tavily Search API"""
    
    name: str = "tavily_news"
    description: str = (
        "Get real-time breaking news and comprehensive market research using Tavily's multi-source search. "
        "Actions: 'breaking_news' (latest market updates), 'company_research' (deep company analysis), "
        "'market_events' (economic events, Fed announcements), 'sentiment_analysis' (market sentiment from multiple sources). "
        "Provides broader context and analysis beyond traditional financial news sources. "
        "Superior for investment decision-making with real-time updates and multi-source intelligence."
    )
    args_schema: Type[BaseModel] = TavilyNewsInput
    
    def _get_tavily_client(self, max_results: int = 5, days: int = 1):
        """Get Tavily search client with configuration"""
        if not TAVILY_AVAILABLE:
            return None
        
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return None
        
        try:
            return TavilySearch(
                api_key=api_key,
                max_results=max_results,
                topic="finance",  # Focus on financial topics
                days=days,
                search_depth="advanced",  # Higher relevance
                include_answer=True,
                include_raw_content=False  # Cleaner for LLM consumption
            )
        except Exception as e:
            logger.error(f"Failed to initialize Tavily client: {e}")
            return None
    
    def _run(
        self,
        action: str,
        query: Optional[str] = None,
        symbol: Optional[str] = None,
        days: int = 1,
        max_results: int = 5
    ) -> str:
        """Execute the Tavily news tool"""
        if not TAVILY_AVAILABLE:
            return "❌ langchain-tavily not available. Please install langchain-tavily to use Tavily news features."
        
        # Validate max_results range
        max_results = max(1, min(max_results, 20))
        days = max(1, min(days, 30))
        
        try:
            if action == "breaking_news":
                return self._get_breaking_news(query, days, max_results)
            elif action == "company_research":
                if not symbol and not query:
                    return "❌ Either 'symbol' or 'query' required for company research."
                return self._get_company_research(symbol, query, days, max_results)
            elif action == "market_events":
                return self._get_market_events(query, days, max_results)
            elif action == "sentiment_analysis":
                if not query and not symbol:
                    return "❌ Either 'query' or 'symbol' required for sentiment analysis."
                return self._get_sentiment_analysis(symbol, query, days, max_results)
            else:
                return f"❌ Unknown action: {action}. Available actions: breaking_news, company_research, market_events, sentiment_analysis"
        
        except Exception as e:
            logger.error(f"Tavily News error: {e}")
            return f"❌ Tavily News error: {str(e)}"
    
    def _get_breaking_news(self, query: Optional[str], days: int, max_results: int) -> str:
        """Get breaking financial news"""
        client = self._get_tavily_client(max_results, days)
        if not client:
            return "❌ Tavily API not available. Please set TAVILY_API_KEY in your .env file."
        
        try:
            # Default breaking news queries if none provided
            if not query:
                base_queries = [
                    "stock market news today",
                    "financial markets breaking news", 
                    "economic news today",
                    "cryptocurrency news today"
                ]
                # Use the most general query for breaking news
                search_query = base_queries[0]
            else:
                search_query = f"{query} latest news today"
            
            results = client.invoke({"query": search_query})
            
            return self._format_breaking_news(results, search_query, days)
            
        except Exception as e:
            return f"❌ Error getting breaking news: {str(e)}"
    
    def _get_company_research(self, symbol: Optional[str], query: Optional[str], days: int, max_results: int) -> str:
        """Get comprehensive company research"""
        client = self._get_tavily_client(max_results, days)
        if not client:
            return "❌ Tavily API not available. Please set TAVILY_API_KEY in your .env file."
        
        try:
            # Build research query
            if symbol:
                if query:
                    search_query = f"{symbol} {query} stock news analysis"
                else:
                    search_query = f"{symbol} stock news earnings analyst reports recent developments"
            else:
                search_query = f"{query} stock analysis news"
            
            results = client.invoke({"query": search_query})
            
            return self._format_company_research(results, symbol or query, days, max_results)
            
        except Exception as e:
            return f"❌ Error getting company research: {str(e)}"
    
    def _get_market_events(self, query: Optional[str], days: int, max_results: int) -> str:
        """Get market events and economic news"""
        client = self._get_tavily_client(max_results, days)
        if not client:
            return "❌ Tavily API not available. Please set TAVILY_API_KEY in your .env file."
        
        try:
            # Default market events queries if none provided
            if not query:
                search_query = "Federal Reserve interest rates economic data GDP inflation unemployment market moving events"
            else:
                search_query = f"{query} market impact economic news"
            
            results = client.invoke({"query": search_query})
            
            return self._format_market_events(results, search_query, days, max_results)
            
        except Exception as e:
            return f"❌ Error getting market events: {str(e)}"
    
    def _get_sentiment_analysis(self, symbol: Optional[str], query: Optional[str], days: int, max_results: int) -> str:
        """Get sentiment analysis from multiple sources"""
        client = self._get_tavily_client(max_results, days)
        if not client:
            return "❌ Tavily API not available. Please set TAVILY_API_KEY in your .env file."
        
        try:
            # Build sentiment analysis query
            if symbol:
                search_query = f"{symbol} stock sentiment analysis investor opinion market outlook bullish bearish"
            else:
                search_query = f"{query} market sentiment investor opinion analysis"
            
            results = client.invoke({"query": search_query})
            
            return self._format_sentiment_analysis(results, symbol or query, days, max_results)
            
        except Exception as e:
            return f"❌ Error analyzing sentiment: {str(e)}"
    
    def _format_breaking_news(self, results, query: str, days: int) -> str:
        """Format breaking news results"""
        if not results or len(results) == 0:
            return "❌ No breaking news found for the specified query."
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        result = f"🚨 **BREAKING NEWS** ({days} day{'s' if days > 1 else ''})\n"
        result += f"🔍 Query: {query}\n"
        result += f"🕒 Retrieved: {current_time}\n\n"
        
        # Handle different result formats from Tavily
        items = []
        if isinstance(results, list):
            items = results
        elif isinstance(results, dict):
            if 'results' in results:
                items = results['results']
            elif 'answer' in results:
                # If Tavily returns a direct answer
                result += f"📋 **Key Insight**: {results['answer']}\n\n"
                items = results.get('sources', [])
            else:
                items = [results]
        
        if items:
            result += f"📊 **Found {len(items)} breaking news items:**\n\n"
            
            for i, item in enumerate(items, 1):
                title = item.get('title', 'No title')
                url = item.get('url', '')
                content = item.get('content', item.get('snippet', ''))
                
                # Truncate content for readability
                if len(content) > 200:
                    content = content[:200] + "..."
                
                result += f"**{i}. {title}**\n"
                if content:
                    result += f"📝 {content}\n"
                if url:
                    result += f"🔗 [Read more]({url})\n"
                result += "\n"
        
        result += "💡 **Investment Context**: Monitor these developments for potential market impact and portfolio implications."
        
        return result.strip()
    
    def _format_company_research(self, results, company: str, days: int, max_results: int) -> str:
        """Format company research results"""
        if not results or len(results) == 0:
            return f"❌ No research found for {company}."
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        result = f"🏢 **COMPANY RESEARCH: {company.upper()}**\n"
        result += f"📅 Period: Last {days} day{'s' if days > 1 else ''}\n"
        result += f"🕒 Retrieved: {current_time}\n\n"
        
        # Handle different result formats
        items = []
        key_insight = ""
        
        if isinstance(results, list):
            items = results
        elif isinstance(results, dict):
            if 'results' in results:
                items = results['results']
            if 'answer' in results:
                key_insight = results['answer']
            else:
                items = [results] if results else []
        
        if key_insight:
            result += f"🎯 **Key Insight**: {key_insight}\n\n"
        
        if items:
            result += f"📊 **Research Sources ({len(items)} items):**\n\n"
            
            # Categorize news types
            earnings_news = []
            analyst_reports = []
            general_news = []
            
            for item in items:
                title = item.get('title', '').lower()
                if any(word in title for word in ['earnings', 'quarterly', 'revenue', 'eps']):
                    earnings_news.append(item)
                elif any(word in title for word in ['analyst', 'rating', 'price target', 'upgrade', 'downgrade']):
                    analyst_reports.append(item)
                else:
                    general_news.append(item)
            
            # Display categorized results
            if earnings_news:
                result += "📈 **Earnings & Financial Results:**\n"
                for item in earnings_news[:2]:  # Limit to 2 most recent
                    result += self._format_news_item(item)
                result += "\n"
            
            if analyst_reports:
                result += "👔 **Analyst Reports & Ratings:**\n"
                for item in analyst_reports[:2]:
                    result += self._format_news_item(item)
                result += "\n"
            
            if general_news:
                result += "📰 **General News & Developments:**\n"
                for item in general_news[:3]:
                    result += self._format_news_item(item)
        
        result += f"\n💡 **Research Summary**: Based on {len(items)} sources, use this information for comprehensive investment analysis and decision-making."
        
        return result.strip()
    
    def _format_market_events(self, results, query: str, days: int, max_results: int) -> str:
        """Format market events results"""
        if not results or len(results) == 0:
            return f"❌ No market events found for: {query}"
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        result = f"📊 **MARKET EVENTS & ECONOMIC NEWS**\n"
        result += f"🔍 Focus: {query}\n"
        result += f"📅 Period: Last {days} day{'s' if days > 1 else ''}\n"
        result += f"🕒 Retrieved: {current_time}\n\n"
        
        # Handle results
        items = []
        key_insight = ""
        
        if isinstance(results, list):
            items = results
        elif isinstance(results, dict):
            if 'results' in results:
                items = results['results']
            if 'answer' in results:
                key_insight = results['answer']
            else:
                items = [results] if results else []
        
        if key_insight:
            result += f"🎯 **Market Overview**: {key_insight}\n\n"
        
        if items:
            result += f"📋 **Key Events ({len(items)} items):**\n\n"
            
            # Categorize by importance/type
            fed_news = []
            economic_data = []
            market_news = []
            
            for item in items:
                title = item.get('title', '').lower()
                content = item.get('content', item.get('snippet', '')).lower()
                
                if any(word in title or word in content for word in ['federal reserve', 'fed', 'interest rate', 'powell']):
                    fed_news.append(item)
                elif any(word in title or word in content for word in ['gdp', 'unemployment', 'inflation', 'cpi', 'economic data']):
                    economic_data.append(item)
                else:
                    market_news.append(item)
            
            # Display by priority
            if fed_news:
                result += "🏦 **Federal Reserve & Monetary Policy:**\n"
                for item in fed_news[:2]:
                    result += self._format_news_item(item, include_impact=True)
                result += "\n"
            
            if economic_data:
                result += "📊 **Economic Data & Indicators:**\n"
                for item in economic_data[:2]:
                    result += self._format_news_item(item, include_impact=True)
                result += "\n"
            
            if market_news:
                result += "📈 **Market Developments:**\n"
                for item in market_news[:3]:
                    result += self._format_news_item(item, include_impact=True)
        
        result += f"\n🎯 **Investment Implications**: Monitor these events for portfolio positioning and risk management decisions."
        
        return result.strip()
    
    def _format_sentiment_analysis(self, results, target: str, days: int, max_results: int) -> str:
        """Format sentiment analysis results"""
        if not results or len(results) == 0:
            return f"❌ No sentiment data found for {target}."
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        result = f"📊 **SENTIMENT ANALYSIS: {target.upper()}**\n"
        result += f"📅 Period: Last {days} day{'s' if days > 1 else ''}\n"
        result += f"🕒 Retrieved: {current_time}\n\n"
        
        # Handle results
        items = []
        key_insight = ""
        
        if isinstance(results, list):
            items = results
        elif isinstance(results, dict):
            if 'results' in results:
                items = results['results']
            if 'answer' in results:
                key_insight = results['answer']
            else:
                items = [results] if results else []
        
        if key_insight:
            result += f"🎯 **Sentiment Overview**: {key_insight}\n\n"
        
        if items:
            # Analyze sentiment from titles and content
            positive_indicators = ['bullish', 'positive', 'optimistic', 'growth', 'strong', 'outperform', 'buy', 'upgrade', 'beat', 'exceed']
            negative_indicators = ['bearish', 'negative', 'pessimistic', 'decline', 'weak', 'underperform', 'sell', 'downgrade', 'miss', 'below']
            
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            sentiment_items = []
            
            for item in items:
                title = item.get('title', '').lower()
                content = item.get('content', item.get('snippet', '')).lower()
                combined_text = f"{title} {content}"
                
                positive_score = sum(1 for indicator in positive_indicators if indicator in combined_text)
                negative_score = sum(1 for indicator in negative_indicators if indicator in combined_text)
                
                if positive_score > negative_score:
                    sentiment = "📈 Positive"
                    positive_count += 1
                elif negative_score > positive_score:
                    sentiment = "📉 Negative"
                    negative_count += 1
                else:
                    sentiment = "⚖️ Neutral"
                    neutral_count += 1
                
                sentiment_items.append((item, sentiment))
            
            # Overall sentiment calculation
            total_items = len(items)
            if total_items > 0:
                positive_pct = (positive_count / total_items) * 100
                negative_pct = (negative_count / total_items) * 100
                neutral_pct = (neutral_count / total_items) * 100
                
                result += f"📊 **Sentiment Breakdown** (from {total_items} sources):\n"
                result += f"  📈 Positive: {positive_count} items ({positive_pct:.1f}%)\n"
                result += f"  📉 Negative: {negative_count} items ({negative_pct:.1f}%)\n"
                result += f"  ⚖️ Neutral: {neutral_count} items ({neutral_pct:.1f}%)\n\n"
                
                # Overall sentiment
                if positive_pct > 50:
                    overall_sentiment = "📈 **Overall Sentiment: BULLISH**"
                elif negative_pct > 50:
                    overall_sentiment = "📉 **Overall Sentiment: BEARISH**"
                elif positive_pct > negative_pct:
                    overall_sentiment = "📊 **Overall Sentiment: MILDLY POSITIVE**"
                elif negative_pct > positive_pct:
                    overall_sentiment = "📊 **Overall Sentiment: MILDLY NEGATIVE**"
                else:
                    overall_sentiment = "⚖️ **Overall Sentiment: NEUTRAL**"
                
                result += f"{overall_sentiment}\n\n"
            
            # Display sentiment-categorized items
            result += f"📋 **Sentiment Sources:**\n\n"
            
            for item, sentiment in sentiment_items[:max_results]:
                title = item.get('title', 'No title')
                url = item.get('url', '')
                content = item.get('content', item.get('snippet', ''))
                
                if len(content) > 150:
                    content = content[:150] + "..."
                
                result += f"{sentiment} **{title}**\n"
                if content:
                    result += f"📝 {content}\n"
                if url:
                    result += f"🔗 [Source]({url})\n"
                result += "\n"
        
        result += f"💡 **Investment Context**: Use this sentiment analysis alongside fundamental and technical analysis for comprehensive investment decisions."
        
        return result.strip()
    
    def _format_news_item(self, item: dict, include_impact: bool = False) -> str:
        """Format individual news item"""
        title = item.get('title', 'No title')
        url = item.get('url', '')
        content = item.get('content', item.get('snippet', ''))
        
        # Truncate content
        if len(content) > 120:
            content = content[:120] + "..."
        
        formatted = f"  • **{title}**\n"
        if content:
            formatted += f"    {content}\n"
        if url:
            formatted += f"    🔗 [Read more]({url})\n"
        
        if include_impact:
            # Simple impact assessment based on keywords
            impact_keywords = ['major', 'significant', 'breaking', 'unprecedented', 'record', 'surge', 'plunge', 'crisis']
            text_lower = f"{title} {content}".lower()
            
            if any(keyword in text_lower for keyword in impact_keywords):
                formatted += f"    ⚠️ **High Impact**: May significantly affect markets\n"
        
        return formatted


def create_tavily_news_tool() -> TavilyNewsTool:
    """Factory function to create Tavily news tool"""
    return TavilyNewsTool()