using System.Collections.Generic;

namespace TradingBot.ML
{
    public static class MarketDataExtensions
    {
        /// <summary>
        /// Convert Core.MarketData to ML.MarketData format
        /// </summary>
        public static MarketData ToMLMarketData(this TradingBot.Core.MarketData m)
        {
            // Helper to safely get indicator values
            double GetIndicator(string key, double defaultValue = 0) => 
                (m.Indicators != null && m.Indicators.TryGetValue(key, out var v)) ? v : defaultValue;
            
            return new MarketData
            {
                ESPrice = m.ESPrice,
                NQPrice = m.NQPrice,
                ESVolume = m.ESVolume,
                NQVolume = m.NQVolume,
                ES_ATR = (decimal)GetIndicator("ATR_ES", 10),
                NQ_ATR = (decimal)GetIndicator("ATR_NQ", 25),
                VIX = m.Internals?.VIX ?? 15m,
                TICK = m.Internals?.TICK ?? 0,
                ADD = m.Internals?.ADD ?? 0,
                Correlation = m.Correlation,  // FIXED - no casting needed
                RSI_ES = (decimal)GetIndicator("RSI_ES", 50),
                RSI_NQ = (decimal)GetIndicator("RSI_NQ", 50),
                PrimaryInstrument = m.PrimaryInstrument ?? "ES"
            };
        }
    }
}
