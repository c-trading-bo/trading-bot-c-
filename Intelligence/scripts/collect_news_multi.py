#!/usr/bin/env python3
"""
Multi-source news collection system for trading bot intelligence.
Replaces GDELT with reliable, free news sources.
"""

import feedparser
import requests
import json
import re
from datetime import datetime, timedelta
import time
import os
import logging
from typing import List, Dict, Any
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TradingBot/2.0 (+https://github.com/kevinsuero072897-collab/trading-bot-c-)'
        })
        
    def collect_all_news(self) -> Dict[str, Any]:
        """Collect news from multiple FREE sources that actually work"""
        
        all_articles = []
        sentiment_scores = []
        volatility_indicators = {
            'fomc_detected': False,
            'cpi_detected': False,
            'nfp_detected': False,
            'earnings_detected': False,
            'panic_detected': False,
            'euphoria_detected': False,
            'major_event_score': 0
        }
        
        logger.info(f"Starting news collection at {datetime.utcnow()}")
        
        # 1. Yahoo Finance RSS - Unlimited and reliable
        yahoo_feeds = [
            "https://finance.yahoo.com/rss/topfinstories",
            "https://feeds.finance.yahoo.com/rss/2.0/headline",
            "https://finance.yahoo.com/news/rssindex"
        ]
        
        for feed_url in yahoo_feeds:
            try:
                logger.info(f"Fetching Yahoo Finance: {feed_url}")
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries[:15]:  # Top 15 from each feed
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    full_text = f"{title} {summary}".lower()
                    
                    # Calculate sentiment
                    sentiment = self.calculate_advanced_sentiment(full_text)
                    sentiment_scores.append(sentiment)
                    
                    # Check for volatility events
                    self.check_volatility_indicators(full_text, volatility_indicators)
                    
                    all_articles.append({
                        'title': title,
                        'summary': summary[:500],
                        'published': entry.get('published', datetime.utcnow().isoformat()),
                        'link': entry.get('link', ''),
                        'source': 'yahoo_finance',
                        'sentiment': sentiment,
                        'has_numbers': bool(re.search(r'\d+\.?\d*%', full_text)),
                        'urgency': self.detect_urgency(title)
                    })
                
                time.sleep(0.5)  # Be respectful
                
            except Exception as e:
                logger.error(f"Yahoo feed error: {e}")
        
        # 2. Reddit WallStreetBets - Sentiment gauge
        try:
            logger.info("Fetching Reddit WSB sentiment...")
            headers = {'User-Agent': 'TradingBot/2.0 (by u/trader)'}
            
            wsb_url = "https://www.reddit.com/r/wallstreetbets/hot.json?limit=30"
            response = self.session.get(wsb_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                total_score = 0
                total_comments = 0
                bearish_count = 0
                bullish_count = 0
                
                for post in data['data']['children'][:20]:
                    post_data = post['data']
                    title = post_data.get('title', '')
                    selftext = post_data.get('selftext', '')
                    full_text = f"{title} {selftext}".lower()
                    score = post_data.get('score', 0)
                    comments = post_data.get('num_comments', 0)
                    
                    total_score += score
                    total_comments += comments
                    
                    # Sentiment analysis
                    if any(word in full_text for word in ['puts', 'bear', 'crash', 'dump', 'sell']):
                        bearish_count += 1
                    if any(word in full_text for word in ['calls', 'bull', 'moon', 'pump', 'buy']):
                        bullish_count += 1
                    
                    # Check for squeeze/panic
                    if 'squeeze' in full_text or 'gamma' in full_text:
                        volatility_indicators['euphoria_detected'] = True
                        volatility_indicators['major_event_score'] += 2
                    
                    if score > 1000:  # High engagement post
                        all_articles.append({
                            'title': f"WSB: {title}",
                            'summary': selftext[:300],
                            'published': datetime.fromtimestamp(post_data['created_utc']).isoformat(),
                            'link': f"https://reddit.com{post_data['permalink']}",
                            'source': 'reddit_wsb',
                            'sentiment': 0.5 if bullish_count > bearish_count else -0.5,
                            'engagement': score + comments,
                            'is_viral': score > 5000
                        })
                
                # Calculate WSB sentiment
                wsb_sentiment = (bullish_count - bearish_count) / max(bullish_count + bearish_count, 1)
                sentiment_scores.append(wsb_sentiment)
                
                # Detect market extremes
                if total_score > 50000:  # Very high engagement
                    volatility_indicators['major_event_score'] += 3
                    logger.info(f"WSB highly active: {total_score} total score")
            
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Reddit error: {e}")
        
        # 3. MarketWatch RSS
        try:
            logger.info("Fetching MarketWatch...")
            mw_feeds = [
                "http://feeds.marketwatch.com/marketwatch/topstories/",
                "http://feeds.marketwatch.com/marketwatch/realtimeheadlines/"
            ]
            
            for feed_url in mw_feeds:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    sentiment = self.calculate_advanced_sentiment(f"{title} {summary}")
                    sentiment_scores.append(sentiment)
                    
                    self.check_volatility_indicators(f"{title} {summary}".lower(), volatility_indicators)
                    
                    all_articles.append({
                        'title': title,
                        'summary': summary[:500],
                        'published': entry.get('published', ''),
                        'source': 'marketwatch',
                        'sentiment': sentiment
                    })
                    
        except Exception as e:
            logger.error(f"MarketWatch error: {e}")
        
        # 4. CNBC RSS
        try:
            logger.info("Fetching CNBC...")
            cnbc_feeds = [
                "https://www.cnbc.com/id/100003114/device/rss/rss.html",  # Top News
                "https://www.cnbc.com/id/15839135/device/rss/rss.html",   # Markets
            ]
            
            for feed_url in cnbc_feeds:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:
                    title = entry.get('title', '')
                    summary = entry.get('summary', '')
                    sentiment = self.calculate_advanced_sentiment(f"{title} {summary}")
                    sentiment_scores.append(sentiment)
                    
                    self.check_volatility_indicators(f"{title} {summary}".lower(), volatility_indicators)
                    
                    all_articles.append({
                        'title': title,
                        'summary': summary[:500],
                        'published': entry.get('published', ''),
                        'source': 'cnbc',
                        'sentiment': sentiment
                    })
                    
        except Exception as e:
            logger.error(f"CNBC error: {e}")
        
        # Calculate aggregate metrics
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        news_intensity = min(100.0, len(all_articles) * 2)  # Normalize to 0-100 scale
        
        # Boost intensity based on volatility indicators
        event_boost = volatility_indicators['major_event_score'] * 10
        news_intensity = min(100.0, news_intensity + event_boost)
        
        # Detect market regime from news
        if volatility_indicators['fomc_detected'] or volatility_indicators['cpi_detected']:
            regime_hint = "Volatile"
        elif avg_sentiment > 0.3:
            regime_hint = "Trending"
        elif avg_sentiment < -0.3:
            regime_hint = "Volatile"
        else:
            regime_hint = "Ranging"
        
        # Save results
        self.save_results(all_articles, avg_sentiment, news_intensity, volatility_indicators, regime_hint)
        
        logger.info(f"Collected {len(all_articles)} articles, sentiment: {avg_sentiment:.2f}, intensity: {news_intensity:.1f}")
        
        return {
            'articles': all_articles,
            'avg_sentiment': avg_sentiment,
            'news_intensity': news_intensity,
            'volatility_indicators': volatility_indicators,
            'regime_hint': regime_hint,
            'article_count': len(all_articles)
        }
    
    def calculate_advanced_sentiment(self, text: str) -> float:
        """Advanced sentiment analysis using keyword scoring"""
        if not text:
            return 0.0
        
        text = text.lower()
        
        # Positive indicators
        positive_words = [
            'bull', 'bullish', 'rally', 'surge', 'gains', 'up', 'higher', 'strong',
            'positive', 'optimistic', 'confident', 'growth', 'expansion', 'boom',
            'breakthrough', 'success', 'profits', 'earnings beat', 'upgrade'
        ]
        
        # Negative indicators
        negative_words = [
            'bear', 'bearish', 'crash', 'plunge', 'fall', 'down', 'lower', 'weak',
            'negative', 'pessimistic', 'concerned', 'recession', 'contraction', 'bust',
            'crisis', 'failure', 'losses', 'earnings miss', 'downgrade'
        ]
        
        # High impact words get extra weight
        high_impact_positive = ['surge', 'rally', 'boom', 'breakthrough']
        high_impact_negative = ['crash', 'plunge', 'crisis', 'bust']
        
        pos_score = 0
        neg_score = 0
        
        for word in positive_words:
            count = text.count(word)
            weight = 2 if word in high_impact_positive else 1
            pos_score += count * weight
        
        for word in negative_words:
            count = text.count(word)
            weight = 2 if word in high_impact_negative else 1
            neg_score += count * weight
        
        # Numbers indicate significance
        if re.search(r'\d+\.?\d*%', text):
            pos_score += 0.5  # Any percentage adds slight positive bias (newsworthy)
        
        total_score = pos_score + neg_score
        if total_score == 0:
            return 0.0
        
        sentiment = (pos_score - neg_score) / total_score
        return max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]
    
    def check_volatility_indicators(self, text: str, indicators: Dict[str, Any]):
        """Check for volatility-inducing events"""
        text = text.lower()
        
        # FOMC/Fed events
        if any(word in text for word in ['fomc', 'federal reserve', 'fed meeting', 'powell', 'fed chairman']):
            indicators['fomc_detected'] = True
            indicators['major_event_score'] += 5
        
        # CPI events
        if any(word in text for word in ['cpi', 'inflation', 'consumer price']):
            indicators['cpi_detected'] = True
            indicators['major_event_score'] += 4
        
        # NFP events
        if any(word in text for word in ['nfp', 'non-farm payroll', 'jobs report', 'employment']):
            indicators['nfp_detected'] = True
            indicators['major_event_score'] += 3
        
        # Earnings events
        if any(word in text for word in ['earnings', 'quarterly results', 'guidance']):
            indicators['earnings_detected'] = True
            indicators['major_event_score'] += 1
        
        # Market panic indicators
        if any(word in text for word in ['panic', 'selloff', 'crash', 'meltdown', 'circuit breaker']):
            indicators['panic_detected'] = True
            indicators['major_event_score'] += 3
        
        # Market euphoria indicators
        if any(word in text for word in ['ath', 'all-time high', 'euphoria', 'bubble', 'mania']):
            indicators['euphoria_detected'] = True
            indicators['major_event_score'] += 2
    
    def detect_urgency(self, title: str) -> int:
        """Detect urgency level from title (1-10 scale)"""
        if not title:
            return 1
        
        title_lower = title.lower()
        urgency = 1
        
        # Breaking news indicators
        if any(word in title_lower for word in ['breaking', 'urgent', 'alert', 'developing']):
            urgency += 5
        
        # Market timing indicators
        if any(word in title_lower for word in ['now', 'today', 'this morning', 'just in']):
            urgency += 3
        
        # Magnitude indicators
        if any(word in title_lower for word in ['massive', 'huge', 'dramatic', 'shocking']):
            urgency += 2
        
        # Percentage indicators
        if re.search(r'\d+\.?\d*%', title):
            urgency += 2
        
        return min(10, urgency)
    
    def save_results(self, articles: List[Dict], sentiment: float, intensity: float, 
                    volatility_indicators: Dict, regime_hint: str):
        """Save news results to file"""
        output_dir = "Intelligence/data/news"
        os.makedirs(f"{output_dir}/raw", exist_ok=True)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        
        # Save raw articles
        raw_file = f"{output_dir}/raw/{timestamp}.json"
        with open(raw_file, 'w') as f:
            json.dump({
                'timestamp': datetime.utcnow().isoformat(),
                'articles': articles,
                'count': len(articles),
                'avg_sentiment': sentiment,
                'volatility_indicators': volatility_indicators,
                'regime_hint': regime_hint
            }, f, indent=2)
        
        # Save summary for bot consumption
        latest_file = f"{output_dir}/latest.json"
        summary = {
            'timestamp': datetime.utcnow().isoformat(),
            'article_count': len(articles),
            'avg_sentiment': sentiment,
            'news_intensity': intensity,
            'volatility_score': volatility_indicators['major_event_score'],
            'regime_hint': regime_hint,
            'events': {
                'fomc': volatility_indicators['fomc_detected'],
                'cpi': volatility_indicators['cpi_detected'],
                'nfp': volatility_indicators['nfp_detected'],
                'earnings': volatility_indicators['earnings_detected'],
                'panic': volatility_indicators['panic_detected'],
                'euphoria': volatility_indicators['euphoria_detected']
            },
            'top_headlines': [
                {
                    'title': article['title'],
                    'sentiment': article['sentiment'],
                    'source': article['source']
                }
                for article in articles[:5]  # Top 5 headlines
            ]
        }
        
        with open(latest_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"News data saved to {latest_file}")

def main():
    """Main function to run news collection"""
    collector = NewsCollector()
    results = collector.collect_all_news()
    
    print(f"✅ News collection complete:")
    print(f"   Articles: {results['article_count']}")
    print(f"   Sentiment: {results['avg_sentiment']:.2f}")
    print(f"   Intensity: {results['news_intensity']:.1f}")
    print(f"   Regime: {results['regime_hint']}")
    print(f"   Major Events: {results['volatility_indicators']['major_event_score']}")

if __name__ == "__main__":
    main()