using System;
using System.Collections.Generic;

namespace TradingBot.UnifiedOrchestrator.Models
{
    /// <summary>
    /// Comprehensive market data model for the unified orchestrator
    /// Enhanced to support ALL sophisticated service integrations
    /// </summary>
    public class MarketData
    {
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        // Core Prices & Volume
        public decimal ESPrice { get; set; }
        public decimal NQPrice { get; set; }
        public long ESVolume { get; set; }
        public long NQVolume { get; set; }
        
        // Technical Indicators (enhanced)
        public Dictionary<string, decimal> Indicators { get; set; } = new();
        
        // Market Internals (comprehensive)
        public MarketInternals Internals { get; set; } = new();
        
        // Correlation & instrument
        public decimal Correlation { get; set; } = 0.8m;
        public string PrimaryInstrument { get; set; } = "ES";
        
        // Enhanced metadata from ALL sophisticated services
        public Dictionary<string, object> Metadata { get; set; } = new();
    }
    
    /// <summary>
    /// Comprehensive market internals supporting ALL sophisticated service data
    /// </summary>
    public class MarketInternals
    {
        // Traditional Market Internals
        public decimal VIX { get; set; }
        public int TICK { get; set; }
        public int ADD { get; set; }
        public int VOLD { get; set; }
        
        // Intelligence & Sentiment Data
        public decimal MarketSentiment { get; set; }
        public string NewsContext { get; set; } = "NEUTRAL";
        public string EconomicContext { get; set; } = "NORMAL";
        
        // Portfolio & Risk Management
        public decimal PortfolioHeat { get; set; }
        public decimal DailyPnL { get; set; }
        
        // Time-based Context
        public string CurrentSession { get; set; } = "UNKNOWN";
        public bool IsHighVolatilityEvent { get; set; }
        
        // ML & Strategy Context
        public string PreferredStrategy { get; set; } = "AUTO";
        public decimal PositionSizeMultiplier { get; set; } = 1.0m;
        public bool ShouldTrade { get; set; } = true;
    }
}
