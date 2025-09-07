using System;
using System.Collections.Generic;

namespace TradingBot.Core
{
    /// <summary>
    /// UNIFIED MarketData for Core layer - used by all components
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
        
        // Market internals
        public MarketInternals Internals { get; set; }
        
        // Correlation & instrument
        public decimal Correlation { get; set; } = 0.8m;
        public string PrimaryInstrument { get; set; } = "ES";
    }
    
    public class MarketInternals
    {
        public decimal VIX { get; set; } = 15m;
        public int TICK { get; set; }
        public int ADD { get; set; }
        public long VOLD { get; set; }
    }
}
