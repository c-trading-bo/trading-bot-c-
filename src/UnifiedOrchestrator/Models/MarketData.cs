using System;
using System.Collections.Generic;

namespace TradingBot.UnifiedOrchestrator.Models
{
    /// <summary>
    /// Market data model for the unified orchestrator
    /// </summary>
    public class MarketData
    {
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        // Prices & Volume
        public decimal ESPrice { get; set; }
        public decimal NQPrice { get; set; }
        public long ESVolume { get; set; }
        public long NQVolume { get; set; }
        
        // Calculated indicators
        public Dictionary<string, double> Indicators { get; set; } = new();
        
        // Correlation & instrument
        public decimal Correlation { get; set; } = 0.8m;
        public string PrimaryInstrument { get; set; } = "ES";
    }
}
