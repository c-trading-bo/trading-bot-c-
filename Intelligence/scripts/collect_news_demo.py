#!/usr/bin/env python3
"""
Demo News Collector for Intelligence Pipeline
Creates simulated market news data for testing intelligence integration
"""

import json
import os
from datetime import datetime, timedelta
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoNewsCollector:
    def __init__(self):
        self.data_dir = "Intelligence/data/raw/news"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Sample news scenarios for different market conditions
        self.bullish_headlines = [
            "S&P 500 rallies to new highs on strong earnings",
            "Fed signals dovish stance, markets surge",
            "Tech stocks lead broad market advance",
            "Economic data beats expectations across the board",
            "Corporate earnings exceed analyst forecasts",
            "Bull market momentum accelerates into year-end"
        ]
        
        self.bearish_headlines = [
            "Markets plunge on inflation concerns",
            "Fed hawkish rhetoric triggers selloff",
            "Tech stocks lead market decline",
            "Economic indicators show slowdown",
            "Earnings disappoint, guidance cut",
            "Risk-off sentiment grips markets"
        ]
        
        self.volatile_headlines = [
            "FOMC meeting sparks wild market swings",
            "CPI data causes massive volatility",
            "Options expiry creates chaotic trading",
            "Fed Chairman speech rocks markets",
            "Emergency rate decision shakes confidence",
            "Geopolitical tensions spike market fear"
        ]
        
        self.neutral_headlines = [
            "Markets trade sideways ahead of data",
            "Mixed earnings results keep indices flat",
            "Investors await Fed guidance",
            "Steady trading continues in major indices",
            "Markets consolidate recent gains",
            "Range-bound trading persists"
        ]
    
    def generate_demo_news(self, market_regime: str = "neutral") -> list:
        """Generate realistic demo news based on market regime"""
        news_items = []
        
        # Select headlines based on regime
        if market_regime == "bullish":
            headlines = self.bullish_headlines
            base_sentiment = 0.6
        elif market_regime == "bearish":
            headlines = self.bearish_headlines
            base_sentiment = -0.6
        elif market_regime == "volatile":
            headlines = self.volatile_headlines
            base_sentiment = 0.0
        else:
            headlines = self.neutral_headlines
            base_sentiment = 0.0
        
        # Generate 15-25 news items
        num_articles = random.randint(15, 25)
        
        for i in range(num_articles):
            # Mix of different sources
            source = random.choice(['yahoo_finance', 'reuters', 'bloomberg', 'marketwatch', 'cnbc'])
            
            # Select headline with some variation
            headline = random.choice(headlines)
            if i > len(headlines):
                # Add variation to prevent duplicates
                prefixes = ["Breaking:", "Update:", "Analysis:", "Report:", ""]
                headline = f"{random.choice(prefixes)} {headline}"
            
            # Generate realistic metrics
            timestamp = datetime.now() - timedelta(minutes=random.randint(0, 360))
            
            news_items.append({
                'timestamp': timestamp.isoformat(),
                'source': source,
                'title': headline,
                'url': f'https://{source}.com/article/{i}',
                'summary': f'Market analysis regarding {headline.lower()}',
                'published': timestamp.isoformat(),
                'sentiment_score': base_sentiment + random.uniform(-0.3, 0.3)
            })
        
        return news_items
    
    def detect_market_events(self, regime: str = "neutral") -> dict:
        """Generate event detection based on current scenario"""
        events = {
            'fomc_detected': False,
            'cpi_detected': False,
            'earnings_season': False,
            'merger_activity': False,
            'fed_speak': False
        }
        
        # Simulate event detection based on regime
        if regime == "volatile":
            events['fomc_detected'] = random.choice([True, False])
            events['cpi_detected'] = random.choice([True, False])
            events['fed_speak'] = True
        elif regime == "bullish":
            events['earnings_season'] = True
        elif regime == "bearish":
            events['fed_speak'] = random.choice([True, False])
        
        # Random chance of other events
        events['merger_activity'] = random.random() < 0.1
        
        return events
    
    def analyze_sentiment(self, news_items: list, regime: str = "neutral") -> dict:
        """Analyze sentiment of demo news items"""
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
        
        # Calculate aggregate sentiment
        sentiments = [item.get('sentiment_score', 0) for item in news_items]
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        # Map regime to expected outcomes
        regime_mapping = {
            "bullish": {
                'sentiment': 'bullish',
                'intensity': random.uniform(60, 85),
                'confidence': random.uniform(70, 90),
                'volatility_score': random.uniform(2, 4),
                'bullish_score': random.uniform(60, 85),
                'bearish_score': random.uniform(5, 20)
            },
            "bearish": {
                'sentiment': 'bearish',
                'intensity': random.uniform(55, 80),
                'confidence': random.uniform(65, 85),
                'volatility_score': random.uniform(3, 6),
                'bullish_score': random.uniform(10, 25),
                'bearish_score': random.uniform(55, 80)
            },
            "volatile": {
                'sentiment': 'neutral',
                'intensity': random.uniform(75, 95),
                'confidence': random.uniform(50, 70),
                'volatility_score': random.uniform(7, 10),
                'bullish_score': random.uniform(30, 50),
                'bearish_score': random.uniform(30, 50)
            },
            "neutral": {
                'sentiment': 'neutral',
                'intensity': random.uniform(20, 40),
                'confidence': random.uniform(60, 80),
                'volatility_score': random.uniform(1, 3),
                'bullish_score': random.uniform(35, 55),
                'bearish_score': random.uniform(35, 55)
            }
        }
        
        result = regime_mapping.get(regime, regime_mapping["neutral"]).copy()
        result['article_count'] = len(news_items)
        result['avg_sentiment'] = round(avg_sentiment, 3)
        
        # Round values for realism
        for key in ['intensity', 'confidence', 'volatility_score', 'bullish_score', 'bearish_score']:
            result[key] = round(result[key], 1)
        
        return result
    
    def save_news_data(self, news_items: list, analysis: dict) -> tuple:
        """Save demo news and analysis to files"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw news items
        news_file = os.path.join(self.data_dir, f"news_{timestamp_str}.json")
        with open(news_file, 'w') as f:
            json.dump(news_items, f, indent=2)
        
        # Save analysis summary
        analysis_file = os.path.join(self.data_dir, f"analysis_{timestamp_str}.json")
        analysis['timestamp'] = datetime.now().isoformat()
        analysis['news_file'] = news_file
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Update latest analysis for bot consumption
        latest_file = os.path.join(self.data_dir, "latest_analysis.json")
        with open(latest_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Saved demo news data: {len(news_items)} articles, sentiment={analysis['sentiment']}, intensity={analysis['intensity']:.1f}")
        
        return news_file, analysis_file
    
    def collect_demo_news(self, regime: str = "neutral") -> dict:
        """Main function to generate demo news and analysis"""
        logger.info(f"Generating demo news for regime: {regime}")
        
        # Generate news items
        news_items = self.generate_demo_news(regime)
        
        # Analyze sentiment
        analysis = self.analyze_sentiment(news_items, regime)
        
        # Detect events
        events = self.detect_market_events(regime)
        analysis.update(events)
        
        # Save data
        news_file, analysis_file = self.save_news_data(news_items, analysis)
        
        logger.info(f"Demo news collection complete: {analysis['sentiment']} sentiment, "
                   f"{analysis['intensity']:.1f} intensity, {analysis['confidence']:.1f}% confidence")
        
        return analysis

def main():
    """Main execution function with different regime scenarios"""
    collector = DemoNewsCollector()
    
    # Determine regime based on time or random selection for demo
    regimes = ["bullish", "bearish", "volatile", "neutral"]
    
    # For demo purposes, cycle through regimes or pick randomly
    import sys
    if len(sys.argv) > 1:
        regime = sys.argv[1].lower()
        if regime not in regimes:
            regime = "neutral"
    else:
        # Random regime for demo
        regime = random.choice(regimes)
    
    try:
        result = collector.collect_demo_news(regime)
        print(f"Demo collection complete: {result['sentiment']} sentiment with {result['confidence']:.1f}% confidence")
        print(f"Regime: {regime}, Intensity: {result['intensity']:.1f}, Volatility: {result['volatility_score']:.1f}")
        print(f"Events: FOMC={result['fomc_detected']}, CPI={result['cpi_detected']}, Earnings={result['earnings_season']}")
        return result
    except Exception as e:
        logger.error(f"Demo news collection failed: {e}")
        return None

if __name__ == "__main__":
    main()