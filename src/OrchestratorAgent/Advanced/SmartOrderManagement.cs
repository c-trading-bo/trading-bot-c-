using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCore.Models;

namespace OrchestratorAgent.Advanced
{
    /// <summary>
    /// Advanced order management with dynamic sizing, correlation analysis, and adaptive execution
    /// </summary>
    public sealed class SmartOrderManagement
    {
        private readonly ILogger<SmartOrderManagement> _logger;
        private readonly Dictionary<string, PositionCorrelation> _correlations = new();
        private readonly Dictionary<string, MarketRegime> _regimes = new();

        public SmartOrderManagement(ILogger<SmartOrderManagement> logger)
        {
            _logger = logger;
        }

        /// <summary>
        /// Advanced position sizing based on market regime, correlation, and volatility
        /// </summary>
        public int CalculateOptimalSize(Signal signal, MarketSnapshot snapshot, Dictionary<string, decimal> currentPositions)
        {
            var baseSize = signal.Size;
            var regime = GetMarketRegime(signal.Symbol);
            var correlation = CalculatePortfolioCorrelation(signal.Symbol, currentPositions);
            var volatility = snapshot.SignalBarAtrMult; // Normalized volatility (using ATR multiplier)

            // Regime-based sizing
            var regimeMultiplier = regime switch
            {
                MarketRegime.HighVol => 0.5m,      // Reduce size in high volatility
                MarketRegime.LowVol => 1.2m,       // Increase size in low volatility
                MarketRegime.Trending => 1.0m,     // Normal size when trending
                MarketRegime.Ranging => 0.8m,      // Slightly reduce in range-bound
                _ => 1.0m
            };

            // Correlation adjustment (reduce size if highly correlated positions exist)
            var correlationMultiplier = correlation > 0.7m ? 0.6m : 
                                       correlation > 0.5m ? 0.8m : 1.0m;

            // Volatility adjustment
            var volMultiplier = volatility > 0.03m ? 0.7m :     // High vol = smaller size
                               volatility < 0.015m ? 1.2m : 1.0m; // Low vol = larger size

            var adjustedSize = (int)(baseSize * regimeMultiplier * correlationMultiplier * volMultiplier);
            
            // Ensure minimum size and lot compliance
            var minSize = BotCore.Models.InstrumentMeta.LotStep(signal.Symbol);
            adjustedSize = Math.Max(minSize, adjustedSize);
            adjustedSize -= adjustedSize % minSize; // Round to lot size

            _logger.LogInformation("[SMART_SIZE] {Symbol}: Base={Base} -> Adjusted={Adj} (Regime={Regime:F2}, Corr={Corr:F2}, Vol={Vol:F2})",
                signal.Symbol, baseSize, adjustedSize, regimeMultiplier, correlationMultiplier, volMultiplier);

            return adjustedSize;
        }

        /// <summary>
        /// Dynamic stop loss adjustment based on market conditions
        /// </summary>
        public decimal CalculateDynamicStop(Signal signal, MarketSnapshot snapshot, TimeSpan timeInPosition)
        {
            var baseStop = signal.Stop;
            var currentPrice = snapshot.LastPrice;
            var isLong = signal.Side.Equals("BUY", StringComparison.OrdinalIgnoreCase);
            
            // Time-based stop tightening (trail stop closer as time passes)
            var timeMultiplier = timeInPosition.TotalMinutes switch
            {
                > 240 => 0.7m,  // After 4 hours, tighten to 70% of original
                > 120 => 0.8m,  // After 2 hours, tighten to 80%
                > 60 => 0.9m,   // After 1 hour, tighten to 90%
                _ => 1.0m       // First hour, keep original
            };

            // Volatility-based adjustment (using available ATR multiplier data)
            var atr = snapshot.SignalBarAtrMult * snapshot.LastPrice; // Approximate ATR from multiplier
            var normalizedAtr = atr / currentPrice;
            var volMultiplier = normalizedAtr > 0.02m ? 1.3m :  // High vol = wider stops
                               normalizedAtr < 0.01m ? 0.8m : 1.0m; // Low vol = tighter stops

            // Calculate adjusted stop
            var riskAmount = Math.Abs(signal.Entry - baseStop);
            var adjustedRisk = riskAmount * timeMultiplier * volMultiplier;
            var newStop = isLong ? signal.Entry - adjustedRisk : signal.Entry + adjustedRisk;

            // Ensure we don't move stop against us
            if (isLong && newStop < baseStop) newStop = baseStop;
            if (!isLong && newStop > baseStop) newStop = baseStop;

            return BotCore.Models.InstrumentMeta.RoundToTick(signal.Symbol, newStop);
        }

        /// <summary>
        /// Intelligent partial profit taking with market structure awareness
        /// </summary>
        public List<PartialTarget> CalculatePartialTargets(Signal signal, MarketSnapshot snapshot)
        {
            var entry = signal.Entry;
            var stop = signal.Stop;
            var finalTarget = signal.Target;
            var isLong = signal.Side.Equals("BUY", StringComparison.OrdinalIgnoreCase);
            
            var riskAmount = Math.Abs(entry - stop);
            var targets = new List<PartialTarget>();

            // Support/Resistance based targets
            var levels = GetKeyLevels(signal.Symbol, snapshot);
            
            // T1: Quick scalp at 0.5R or nearest support/resistance
            var t1Distance = riskAmount * 0.5m;
            var t1Price = isLong ? entry + t1Distance : entry - t1Distance;
            
            // Adjust to nearest level if within 25% of calculated target
            var nearestLevel = FindNearestLevel(t1Price, levels, isLong);
            if (nearestLevel.HasValue && Math.Abs(nearestLevel.Value - t1Price) / riskAmount < 0.25m)
                t1Price = nearestLevel.Value;

            targets.Add(new PartialTarget
            {
                Price = BotCore.Models.InstrumentMeta.RoundToTick(signal.Symbol, t1Price),
                Quantity = (int)(signal.Size * 0.3m), // 30% at T1
                RMultiple = 0.5m,
                Reason = "Quick scalp"
            });

            // T2: 1R target
            var t2Distance = riskAmount * 1.0m;
            var t2Price = isLong ? entry + t2Distance : entry - t2Distance;
            targets.Add(new PartialTarget
            {
                Price = BotCore.Models.InstrumentMeta.RoundToTick(signal.Symbol, t2Price),
                Quantity = (int)(signal.Size * 0.4m), // 40% at T2
                RMultiple = 1.0m,
                Reason = "1R profit"
            });

            // T3: Extended target based on ATR and market regime
            var regime = GetMarketRegime(signal.Symbol);
            var extendedR = regime == MarketRegime.Trending ? 3.0m : 2.0m;
            var t3Distance = riskAmount * extendedR;
            var t3Price = isLong ? entry + t3Distance : entry - t3Distance;
            
            targets.Add(new PartialTarget
            {
                Price = BotCore.Models.InstrumentMeta.RoundToTick(signal.Symbol, t3Price),
                Quantity = signal.Size - targets.Sum(t => t.Quantity), // Remaining quantity
                RMultiple = extendedR,
                Reason = $"Extended target ({regime})"
            });

            return targets;
        }

        private MarketRegime GetMarketRegime(string symbol)
        {
            // Simplified regime detection - in practice, use your ML models
            return _regimes.GetValueOrDefault(symbol, MarketRegime.Trending);
        }

        private decimal CalculatePortfolioCorrelation(string symbol, Dictionary<string, decimal> positions)
        {
            // Simplified correlation calculation
            // In practice, use historical price correlation matrix
            if (!positions.Any()) return 0m;

            return symbol.ToUpper() switch
            {
                "ES" when positions.ContainsKey("NQ") => 0.8m,  // High correlation ES/NQ
                "NQ" when positions.ContainsKey("ES") => 0.8m,
                "GC" when positions.ContainsKey("SI") => 0.6m,  // Moderate correlation metals
                _ => 0.2m // Low correlation default
            };
        }

        private List<decimal> GetKeyLevels(string symbol, MarketSnapshot snapshot)
        {
            // Simplified - in practice, calculate from historical data
            var price = snapshot.LastPrice;
            var atr = snapshot.SignalBarAtrMult * snapshot.LastPrice; // Approximate ATR from multiplier
            
            return new List<decimal>
            {
                price + atr,     // Resistance
                price - atr,     // Support
                price + atr * 2, // Strong resistance
                price - atr * 2  // Strong support
            };
        }

        private decimal? FindNearestLevel(decimal targetPrice, List<decimal> levels, bool isLong)
        {
            if (!levels.Any()) return null;

            return isLong 
                ? levels.Where(l => l >= targetPrice).OrderBy(l => l).FirstOrDefault()
                : levels.Where(l => l <= targetPrice).OrderByDescending(l => l).FirstOrDefault();
        }
    }

    public enum MarketRegime
    {
        HighVol,
        LowVol,
        Trending,
        Ranging
    }

    public class PositionCorrelation
    {
        public string Symbol1 { get; set; } = "";
        public string Symbol2 { get; set; } = "";
        public decimal Correlation { get; set; }
        public DateTime LastUpdated { get; set; }
    }

    public class PartialTarget
    {
        public decimal Price { get; set; }
        public int Quantity { get; set; }
        public decimal RMultiple { get; set; }
        public string Reason { get; set; } = "";
    }
}
