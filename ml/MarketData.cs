using System;

namespace TradingBot.ML
{
    /// <summary>
    /// ML-specific MarketData format used by machine learning components
    /// </summary>
    public class MarketData
    {
        // Core prices and volume
        public decimal ESPrice { get; set; }
        public decimal NQPrice { get; set; }
        public long ESVolume { get; set; }
        public long NQVolume { get; set; }
        
        // Technical indicators
        public decimal ES_ATR { get; set; }
        public decimal NQ_ATR { get; set; }
        public decimal RSI_ES { get; set; }
        public decimal RSI_NQ { get; set; }
        
        // Market internals
        public decimal VIX { get; set; }
        public int TICK { get; set; }
        public int ADD { get; set; }
        
        // Correlation and instrument
        public decimal Correlation { get; set; }
        public string PrimaryInstrument { get; set; }
        
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    }
}
