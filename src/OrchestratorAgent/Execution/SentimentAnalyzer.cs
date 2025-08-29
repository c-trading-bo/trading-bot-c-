using System;
using System.Collections.Generic;

namespace OrchestratorAgent.Execution
{
    // Hot-swappable sentiment analysis for news-driven trading
    public sealed class SentimentAnalyzer
    {
        readonly Queue<double> _sentimentHistory = new();
        readonly int _maxHistory = 100;
        
        public double CurrentSentiment { get; private set; } = 0.5; // Neutral
        
        public SentimentAnalyzer()
        {
            Console.WriteLine("[HOT-RELOAD] SentimentAnalyzer loaded dynamically!");
        }
        
        public void UpdateSentiment(string newsText)
        {
            // Simple sentiment scoring (can be enhanced with ML models)
            var sentiment = AnalyzeSentiment(newsText);
            
            _sentimentHistory.Enqueue(sentiment);
            while (_sentimentHistory.Count > _maxHistory) 
                _sentimentHistory.Dequeue();
                
            CurrentSentiment = _sentimentHistory.Average();
            
            Console.WriteLine($"[Sentiment] Updated: {CurrentSentiment:F3} from news: {newsText.Substring(0, Math.Min(50, newsText.Length))}...");
        }
        
        private double AnalyzeSentiment(string text)
        {
            // Basic sentiment analysis
            var positive = new[] { "bullish", "rally", "surge", "gains", "strong", "growth" };
            var negative = new[] { "bearish", "crash", "fall", "losses", "weak", "decline" };
            
            var lowerText = text.ToLowerInvariant();
            int posCount = 0, negCount = 0;
            
            foreach (var word in positive)
                if (lowerText.Contains(word)) posCount++;
                
            foreach (var word in negative)
                if (lowerText.Contains(word)) negCount++;
                
            // Return 0-1 sentiment score
            if (posCount + negCount == 0) return 0.5; // Neutral
            return (double)posCount / (posCount + negCount);
        }
        
        public double GetSentimentMultiplier()
        {
            // Convert sentiment to position size multiplier
            // 0.5 = neutral (1.0x), 0.8+ = bullish (1.2x), 0.2- = bearish (0.8x)
            if (CurrentSentiment > 0.7) return 1.2;
            if (CurrentSentiment < 0.3) return 0.8;
            return 1.0;
        }
    }
}
