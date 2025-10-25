"""
News Sentiment Analyzer using FinBERT
Provides basic news sentiment from free sources (GDELT, Reddit)
Addresses HEDGE_FUND_GAP_ANALYSIS.md - Section 4: News Sentiment (Alternative Data)
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not installed. Run: pip install transformers torch")


class NewsSentimentAnalyzer:
    """
    Analyzes news sentiment using pre-trained FinBERT model
    Provides hedge fund level alternative data from news sources
    """
    
    def __init__(self, data_path: str = "./data/news_sentiment"):
        """
        Initialize sentiment analyzer
        
        Args:
            data_path: Path to store sentiment data
        """
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.tokenizer = None
        
        if TRANSFORMERS_AVAILABLE:
            self._load_finbert_model()
        else:
            print("FinBERT model not available - transformers library required")
    
    def _load_finbert_model(self):
        """Load pre-trained FinBERT model for financial sentiment analysis"""
        try:
            print("Loading FinBERT model...")
            
            # Use FinBERT-tone for sentiment classification
            model_name = "ProsusAI/finbert"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Set to evaluation mode
            self.model.eval()
            
            print("FinBERT model loaded successfully")
        except Exception as e:
            print(f"Error loading FinBERT model: {e}")
            self.model = None
            self.tokenizer = None
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: News text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            # Return neutral if model not available
            return {
                'score': 0.0,
                'confidence': 0.0,
                'label': 'neutral'
            }
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT outputs: [negative, neutral, positive]
            negative, neutral, positive = predictions[0].tolist()
            
            # Calculate overall sentiment score (-1 to +1)
            sentiment_score = positive - negative
            
            # Determine label
            max_prob = max(negative, neutral, positive)
            if max_prob == positive:
                label = 'bullish'
            elif max_prob == negative:
                label = 'bearish'
            else:
                label = 'neutral'
            
            return {
                'score': sentiment_score,
                'confidence': max_prob,
                'label': label,
                'positive': positive,
                'neutral': neutral,
                'negative': negative
            }
        
        except Exception as e:
            print(f"Error analyzing text: {e}")
            return {
                'score': 0.0,
                'confidence': 0.0,
                'label': 'neutral'
            }
    
    def analyze_news_for_symbol(
        self,
        symbol: str,
        news_items: List[Dict[str, str]]
    ) -> Dict[str, any]:
        """
        Analyze sentiment for a symbol from multiple news items
        
        Args:
            symbol: Trading symbol (e.g., 'ES', 'SPY')
            news_items: List of news items with 'title' and 'text' keys
            
        Returns:
            Aggregated sentiment analysis
        """
        if not news_items:
            return {
                'symbol': symbol,
                'overall_score': 0.0,
                'confidence': 0.0,
                'sentiment_trend': 'neutral',
                'article_count': 0
            }
        
        sentiments = []
        
        for item in news_items:
            # Combine title and text
            text = f"{item.get('title', '')} {item.get('text', '')}"
            
            if text.strip():
                sentiment = self.analyze_text(text)
                sentiments.append(sentiment)
        
        if not sentiments:
            return {
                'symbol': symbol,
                'overall_score': 0.0,
                'confidence': 0.0,
                'sentiment_trend': 'neutral',
                'article_count': 0
            }
        
        # Aggregate sentiments
        avg_score = sum(s['score'] for s in sentiments) / len(sentiments)
        avg_confidence = sum(s['confidence'] for s in sentiments) / len(sentiments)
        
        # Determine trend
        if avg_score > 0.3:
            trend = 'bullish'
        elif avg_score < -0.3:
            trend = 'bearish'
        else:
            trend = 'neutral'
        
        return {
            'symbol': symbol,
            'overall_score': avg_score,
            'confidence': avg_confidence,
            'sentiment_trend': trend,
            'article_count': len(sentiments),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def fetch_gdelt_news(self, symbol: str, days: int = 1) -> List[Dict[str, str]]:
        """
        Fetch news from GDELT (free news source)
        
        Note: This is a placeholder. In production, integrate with GDELT API
        https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/
        
        Args:
            symbol: Trading symbol
            days: Number of days to look back
            
        Returns:
            List of news items
        """
        print(f"Fetching GDELT news for {symbol} (last {days} days)...")
        print("Note: GDELT integration placeholder - implement GDELT API in production")
        
        # Placeholder - return empty list
        # In production, make HTTP request to GDELT API
        return []
    
    def fetch_reddit_sentiment(self, symbol: str, subreddit: str = "stocks") -> List[Dict[str, str]]:
        """
        Fetch sentiment from Reddit
        
        Note: This is a placeholder. In production, integrate with Reddit API
        https://www.reddit.com/dev/api/
        
        Args:
            symbol: Trading symbol
            subreddit: Subreddit to search
            
        Returns:
            List of Reddit posts
        """
        print(f"Fetching Reddit sentiment for {symbol} from r/{subreddit}...")
        print("Note: Reddit integration placeholder - implement Reddit API in production")
        
        # Placeholder - return empty list
        # In production, use PRAW (Python Reddit API Wrapper)
        return []
    
    def generate_demo_news(self, symbol: str) -> List[Dict[str, str]]:
        """
        Generate demo news items for testing
        
        Args:
            symbol: Trading symbol
            
        Returns:
            List of demo news items
        """
        return [
            {
                'title': f'{symbol} Shows Strong Performance',
                'text': 'Market analysts are optimistic about future growth potential with strong earnings expected.',
                'source': 'demo'
            },
            {
                'title': f'Concerns Rise Over {symbol} Volatility',
                'text': 'Some investors express caution as market volatility increases amid uncertain conditions.',
                'source': 'demo'
            },
            {
                'title': f'{symbol} Trading Volume Increases',
                'text': 'Neutral trading day with increased volume but no clear direction from market participants.',
                'source': 'demo'
            }
        ]
    
    def update_sentiment_cache(self, symbols: List[str]):
        """
        Update sentiment cache for multiple symbols
        
        Args:
            symbols: List of trading symbols
        """
        print(f"Updating sentiment cache for {len(symbols)} symbols...")
        
        sentiments = {}
        
        for symbol in symbols:
            # Fetch news (using demo for now)
            news_items = self.generate_demo_news(symbol)
            
            # Analyze sentiment
            sentiment = self.analyze_news_for_symbol(symbol, news_items)
            
            sentiments[symbol] = {
                'score': sentiment['overall_score'],
                'confidence': sentiment['confidence'],
                'timestamp': sentiment['timestamp'],
                'source': 'finbert_demo'
            }
            
            print(f"{symbol}: Score={sentiment['overall_score']:.3f}, "
                  f"Trend={sentiment['sentiment_trend']}")
        
        # Save to latest_news_sentiment.json
        output_file = self.data_path / "latest_news_sentiment.json"
        with open(output_file, 'w') as f:
            json.dump(sentiments, f, indent=2)
        
        print(f"Sentiment cache saved to: {output_file}")
        
        return sentiments


def main():
    """Main entry point for news sentiment analysis"""
    
    # Default symbols for futures trading
    symbols = ['ES', 'NQ', 'SPY', 'QQQ']
    
    if len(sys.argv) > 1:
        # Allow custom symbols from command line
        symbols = sys.argv[1].split(',')
    
    analyzer = NewsSentimentAnalyzer()
    
    if not TRANSFORMERS_AVAILABLE:
        print("\nError: transformers library not installed")
        print("Install with: pip install transformers torch")
        print("\nGenerating placeholder sentiment data...")
        
        # Generate placeholder data even without FinBERT
        placeholder_sentiments = {}
        for symbol in symbols:
            placeholder_sentiments[symbol] = {
                'score': 0.0,
                'confidence': 0.0,
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'placeholder'
            }
        
        output_file = Path("./data/news_sentiment") / "latest_news_sentiment.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(placeholder_sentiments, f, indent=2)
        
        print(f"Placeholder sentiment data saved to: {output_file}")
        return
    
    # Update sentiment cache
    sentiments = analyzer.update_sentiment_cache(symbols)
    
    print("\n=== News Sentiment Analysis Complete ===")
    for symbol, data in sentiments.items():
        print(f"{symbol}: {data['score']:.3f} (confidence: {data['confidence']:.3f})")


if __name__ == '__main__':
    main()
