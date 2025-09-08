#!/usr/bin/env python3
"""
GDELT News Collector for Intelligence Pipeline
Collects and analyzes market-relevant news from GDELT for sentiment analysis
"""

import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NewsCollector:
    def __init__(self):
        self.data_dir = "Intelligence/data/raw/news"
        os.makedirs(self.data_dir, exist_ok=True)
        
    def collect_gdelt_news(self, hours_back=1):
        """Collect GDELT news from the last N hours"""
        try:
            # GDELT GKG (Global Knowledge Graph) API for recent news
            # Focus on financial and economic themes
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=hours_back)
            
            # Format timestamps for GDELT API
            start_str = start_time.strftime("%Y%m%d%H%M%S")
            end_str = end_time.strftime("%Y%m%d%H%M%S")
            
            # GDELT themes related to markets and economy
            market_themes = [
                "ECON_STOCKMARKET",
                "ECON_ECONOMICS", 
                "ECON_INFLATION",
                "ECON_UNEMPLOYMENT",
                "ECON_TRADE",
                "GOV_CENTRALBANK",
                "GOV_MONETARY_POLICY"
            ]
            
            news_items = []
            
            for theme in market_themes:
                url = f"https://api.gdeltproject.org/api/v2/doc/doc"
                params = {
                    'query': f'theme:{theme}',
                    'mode': 'timelinevolinfo',
                    'timespan': f'{start_str}-{end_str}',
                    'format': 'json',
                    'maxrecords': 50
                }
                
                try:
                    response = requests.get(url, params=params, timeout=30)
                    response.raise_for_status()
                    
                    data = response.json()
                    if 'articles' in data:
                        for article in data['articles']:
                            news_items.append({
                                'timestamp': datetime.utcnow().isoformat(),
                                'theme': theme,
                                'title': article.get('title', ''),
                                'url': article.get('url', ''),
                                'source': article.get('source', ''),
                                'tone': article.get('tone', 0),
                                'language': article.get('language', 'en')
                            })
                    
                    # Rate limiting
                    time.sleep(1)
                    
                except requests.RequestException as e:
                    logger.warning(f"Failed to fetch GDELT data for theme {theme}: {e}")
                    continue
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error collecting GDELT news: {e}")
            return []
    
    def analyze_sentiment(self, news_items):
        """Simple sentiment analysis based on GDELT tone scores"""
        if not news_items:
            return {
                'intensity': 0,
                'sentiment': 'neutral',
                'confidence': 0,
                'article_count': 0
            }
        
        # GDELT tone scores range from -100 (most negative) to +100 (most positive)
        tones = [item.get('tone', 0) for item in news_items if item.get('tone') is not None]
        
        if not tones:
            return {
                'intensity': 0,
                'sentiment': 'neutral', 
                'confidence': 0,
                'article_count': len(news_items)
            }
        
        avg_tone = sum(tones) / len(tones)
        intensity = min(abs(avg_tone), 100)  # Cap at 100
        
        if avg_tone > 5:
            sentiment = 'bullish'
        elif avg_tone < -5:
            sentiment = 'bearish'
        else:
            sentiment = 'neutral'
        
        # Confidence based on consistency and volume
        tone_variance = sum((t - avg_tone) ** 2 for t in tones) / len(tones)
        consistency = max(0, 1 - (tone_variance / 1000))  # Normalize variance
        volume_factor = min(len(news_items) / 20, 1)  # More articles = higher confidence
        confidence = (consistency * 0.7 + volume_factor * 0.3) * 100
        
        return {
            'intensity': intensity,
            'sentiment': sentiment,
            'confidence': confidence,
            'article_count': len(news_items),
            'avg_tone': avg_tone
        }
    
    def save_news_data(self, news_items, analysis):
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

def main():
    """Main execution function"""
    logger.info("Starting GDELT news collection...")
    
    collector = NewsCollector()
    
    # Collect news from last hour
    news_items = collector.collect_gdelt_news(hours_back=1)
    
    if not news_items:
        logger.warning("No news items collected")
        # Still save empty analysis for consistency
        analysis = {
            'intensity': 0,
            'sentiment': 'neutral',
            'confidence': 0,
            'article_count': 0,
            'timestamp': datetime.utcnow().isoformat()
        }
    else:
        # Analyze sentiment
        analysis = collector.analyze_sentiment(news_items)
    
    # Save data
    collector.save_news_data(news_items, analysis)
    
    logger.info(f"News collection completed: {analysis['article_count']} articles processed")

if __name__ == "__main__":
    main()