#!/usr/bin/env python3
"""
Yahoo Finance & Multi-Source News Collector for Intelligence Pipeline
Replaces broken GDELT with free, reliable sources for sentiment analysis
"""

import requests
import feedparser
import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsCollector:
    def __init__(self):
        self.data_dir = "Intelligence/data/raw/news"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Sentiment keywords for market analysis
        self.positive_words = [
            'bullish', 'rally', 'surge', 'gains', 'strong', 'growth', 'buy', 'calls', 
            'upgrade', 'beat', 'moon', 'rocket', 'soar', 'breakout', 'support', 'bounce',
            'oversold', 'undervalued', 'earnings_beat', 'guidance_raised'
        ]
        
        self.negative_words = [
            'bearish', 'crash', 'fall', 'losses', 'weak', 'decline', 'sell', 'puts',
            'downgrade', 'miss', 'dump', 'fear', 'panic', 'plunge', 'resistance', 'overbought',
            'overvalued', 'earnings_miss', 'guidance_cut', 'recession'
        ]
        
        self.volatility_words = [
            'fomc', 'fed', 'powell', 'cpi', 'inflation', 'earnings', 'merger', 'fda',
            'circuit_breaker', 'halted', 'squeeze', 'gamma', 'options_expiry', 'rebalancing'
        ]
        
    def collect_yahoo_finance_rss(self) -> List[Dict[str, Any]]:
        """Collect news from Yahoo Finance RSS feeds"""
        feeds = [
            'https://finance.yahoo.com/rss/topfinstories',
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://finance.yahoo.com/rss/headline?s=^GSPC',  # S&P 500
            'https://finance.yahoo.com/rss/headline?s=^IXIC',  # NASDAQ
        ]
        
        news_items = []
        
        for feed_url in feeds:
            try:
                logger.info(f"Fetching Yahoo Finance RSS: {feed_url}")
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:20]:  # Limit to 20 articles per feed
                    news_items.append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'source': 'yahoo_finance',
                        'title': entry.get('title', ''),
                        'url': entry.get('link', ''),
                        'summary': entry.get('summary', ''),
                        'published': entry.get('published', ''),
                        'tags': [tag.term for tag in entry.get('tags', [])]
                    })
                
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"Failed to fetch Yahoo RSS {feed_url}: {e}")
                continue
        
        return news_items
    
    def collect_reddit_wallstreetbets(self) -> List[Dict[str, Any]]:
        """Collect sentiment from Reddit r/wallstreetbets"""
        try:
            # Reddit JSON API (no authentication needed for public posts)
            url = "https://www.reddit.com/r/wallstreetbets/hot.json"
            headers = {'User-Agent': 'TradingBotNewsCollector/1.0'}
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            news_items = []
            
            for post in data['data']['children'][:50]:  # Top 50 hot posts
                post_data = post['data']
                
                # Calculate sentiment intensity based on score and comments
                score = post_data.get('score', 0)
                num_comments = post_data.get('num_comments', 0)
                intensity = min(100, (score + num_comments * 2) / 50)  # Scale to 0-100
                
                news_items.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'source': 'reddit_wsb',
                    'title': post_data.get('title', ''),
                    'url': f"https://reddit.com{post_data.get('permalink', '')}",
                    'score': score,
                    'comments': num_comments,
                    'intensity': intensity,
                    'selftext': post_data.get('selftext', '')
                })
            
            return news_items
            
        except Exception as e:
            logger.warning(f"Failed to collect Reddit data: {e}")
            return []
    
    def collect_finviz_news(self) -> List[Dict[str, Any]]:
        """Collect headlines from Finviz (basic scraping)"""
        try:
            # Simple headlines from Finviz main page
            url = "https://finviz.com/"
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; NewsBot/1.0)'}
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Basic regex extraction for news headlines
            # This is a simple approach - in production, use proper HTML parsing
            headlines = re.findall(r'<a[^>]*class="[^"]*news[^"]*"[^>]*>([^<]+)</a>', response.text)
            
            news_items = []
            for headline in headlines[:20]:  # Limit to 20 headlines
                news_items.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'source': 'finviz',
                    'title': headline.strip(),
                    'url': 'https://finviz.com/',
                })
            
            return news_items
            
        except Exception as e:
            logger.warning(f"Failed to collect Finviz news: {e}")
            return []
    
    def analyze_sentiment(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhanced sentiment analysis with market-specific keywords"""
        if not news_items:
            return {
                'intensity': 0,
                'sentiment': 'neutral',
                'confidence': 0,
                'article_count': 0,
                'volatility_score': 0,
                'bullish_score': 0,
                'bearish_score': 0
            }
        
        total_sentiment = 0
        volatility_score = 0
        sentiment_scores = []
        
        for item in news_items:
            text = f"{item.get('title', '')} {item.get('summary', '')} {item.get('selftext', '')}".lower()
            
            # Count positive/negative/volatility words
            pos_count = sum(1 for word in self.positive_words if word in text)
            neg_count = sum(1 for word in self.negative_words if word in text)
            vol_count = sum(1 for word in self.volatility_words if word in text)
            
            # Reddit-specific scoring (higher weight for high-engagement posts)
            if item.get('source') == 'reddit_wsb':
                engagement_weight = min(2.0, 1 + item.get('intensity', 0) / 100)
                pos_count *= engagement_weight
                neg_count *= engagement_weight
                vol_count *= engagement_weight
            
            # Calculate item sentiment (-1 to +1)
            if pos_count + neg_count > 0:
                item_sentiment = (pos_count - neg_count) / (pos_count + neg_count)
            else:
                item_sentiment = 0
            
            sentiment_scores.append(item_sentiment)
            total_sentiment += item_sentiment
            volatility_score += vol_count
        
        # Overall sentiment
        avg_sentiment = total_sentiment / len(news_items) if news_items else 0
        
        # Classify sentiment
        if avg_sentiment > 0.2:
            sentiment = 'bullish'
        elif avg_sentiment < -0.2:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        # Intensity based on absolute sentiment and volatility
        intensity = min(100, (abs(avg_sentiment) * 100 + volatility_score * 5))
        
        # Confidence based on consistency and sample size
        if len(sentiment_scores) > 1:
            sentiment_variance = sum((s - avg_sentiment) ** 2 for s in sentiment_scores) / len(sentiment_scores)
            consistency = max(0, 1 - sentiment_variance)
        else:
            consistency = 0.5
        
        volume_factor = min(len(news_items) / 30, 1)  # More articles = higher confidence
        confidence = (consistency * 0.7 + volume_factor * 0.3) * 100
        
        return {
            'intensity': round(intensity, 1),
            'sentiment': sentiment,
            'confidence': round(confidence, 1),
            'article_count': len(news_items),
            'volatility_score': min(10, volatility_score),  # Scale 0-10
            'bullish_score': round(max(0, avg_sentiment) * 100, 1),
            'bearish_score': round(max(0, -avg_sentiment) * 100, 1),
            'avg_sentiment': round(avg_sentiment, 3)
        }
    
    def detect_events(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect major market events from news"""
        events = {
            'fomc_detected': False,
            'cpi_detected': False,
            'earnings_season': False,
            'merger_activity': False,
            'fed_speak': False
        }
        
        all_text = ' '.join([
            f"{item.get('title', '')} {item.get('summary', '')}" 
            for item in news_items
        ]).lower()
        
        # FOMC detection
        fomc_keywords = ['fomc', 'federal reserve meeting', 'fed meeting', 'rate decision']
        events['fomc_detected'] = any(keyword in all_text for keyword in fomc_keywords)
        
        # CPI detection
        cpi_keywords = ['cpi', 'consumer price index', 'inflation data', 'pce']
        events['cpi_detected'] = any(keyword in all_text for keyword in cpi_keywords)
        
        # Earnings season
        earnings_keywords = ['earnings', 'quarterly results', 'eps', 'guidance']
        earnings_count = sum(all_text.count(keyword) for keyword in earnings_keywords)
        events['earnings_season'] = earnings_count >= 3
        
        # Fed speak
        fed_keywords = ['powell', 'jerome powell', 'fed chair', 'fed governor']
        events['fed_speak'] = any(keyword in all_text for keyword in fed_keywords)
        
        # Merger activity
        merger_keywords = ['merger', 'acquisition', 'buyout', 'takeover']
        events['merger_activity'] = any(keyword in all_text for keyword in merger_keywords)
        
        return events
    
    def save_news_data(self, news_items: List[Dict[str, Any]], analysis: Dict[str, Any]) -> tuple:
        """Save collected news and analysis to files"""
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Save raw news items
        news_file = os.path.join(self.data_dir, f"news_{timestamp_str}.json")
        with open(news_file, 'w') as f:
            json.dump(news_items, f, indent=2)
        
        # Save analysis summary
        analysis_file = os.path.join(self.data_dir, f"analysis_{timestamp_str}.json")
        analysis['timestamp'] = datetime.utcnow().isoformat()
        analysis['news_file'] = news_file
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Update latest analysis for bot consumption
        latest_file = os.path.join(self.data_dir, "latest_analysis.json")
        with open(latest_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Saved news data: {len(news_items)} articles, sentiment={analysis['sentiment']}, intensity={analysis['intensity']:.1f}")
        
        return news_file, analysis_file
    
    def collect_all_news(self) -> Dict[str, Any]:
        """Main function to collect from all sources and analyze"""
        logger.info("Starting news collection from multiple sources...")
        
        all_news = []
        
        # Collect from Yahoo Finance RSS
        yahoo_news = self.collect_yahoo_finance_rss()
        all_news.extend(yahoo_news)
        logger.info(f"Collected {len(yahoo_news)} articles from Yahoo Finance")
        
        # Collect from Reddit WSB
        reddit_news = self.collect_reddit_wallstreetbets()
        all_news.extend(reddit_news)
        logger.info(f"Collected {len(reddit_news)} posts from Reddit WSB")
        
        # Collect from Finviz
        finviz_news = self.collect_finviz_news()
        all_news.extend(finviz_news)
        logger.info(f"Collected {len(finviz_news)} headlines from Finviz")
        
        # Analyze sentiment
        analysis = self.analyze_sentiment(all_news)
        
        # Detect events
        events = self.detect_events(all_news)
        analysis.update(events)
        
        # Save data
        news_file, analysis_file = self.save_news_data(all_news, analysis)
        
        logger.info(f"News collection complete: {analysis['sentiment']} sentiment, "
                   f"{analysis['intensity']:.1f} intensity, {analysis['confidence']:.1f}% confidence")
        
        return analysis

def main():
    """Main execution function"""
    collector = NewsCollector()
    try:
        result = collector.collect_all_news()
        print(f"Collection complete: {result['sentiment']} sentiment with {result['confidence']:.1f}% confidence")
        return result
    except Exception as e:
        logger.error(f"News collection failed: {e}")
        return None

if __name__ == "__main__":
    main()