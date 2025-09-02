// ES/NQ Correlation Matrix Manager
// File: Services/ES_NQ_CorrelationManager.cs
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace BotCore.Services
{
    public class CorrelationData
    {
        public double Correlation5Min { get; set; }
        public double Correlation20Min { get; set; }
        public double Correlation60Min { get; set; }
        public double CorrelationDaily { get; set; }
        public double LeadLagRatio { get; set; } // Who's leading
        public string Leader { get; set; } = "NEUTRAL"; // ES or NQ
        public double Divergence { get; set; }
        public DateTime LastUpdate { get; set; }
    }
    
    public class SignalFilter
    {
        public bool Allow { get; set; } = true;
        public double ConfidenceMultiplier { get; set; } = 1.0;
        public double PositionSizeMultiplier { get; set; } = 1.0;
        public string Reason { get; set; } = "";
    }
    
    public class SignalResult
    {
        public string Action { get; set; } = "NEUTRAL";
        public double Confidence { get; set; }
        public string Symbol { get; set; } = "";
    }

    public interface ICorrelationManager
    {
        Task<SignalFilter> GetCorrelationFilterAsync(string instrument, SignalResult signal);
        Task<CorrelationData> GetCorrelationDataAsync();
    }

    public class ES_NQ_CorrelationManager : ICorrelationManager
    {
        private readonly IMarketDataService _marketData;
        private readonly ILogger<ES_NQ_CorrelationManager> _logger;
        
        // Dynamic correlation windows
        private readonly int[] _correlationWindows = { 5, 20, 60, 252 }; // minutes
        private Dictionary<string, CorrelationData> _correlationMatrix = new();
        
        public ES_NQ_CorrelationManager(IMarketDataService marketData, ILogger<ES_NQ_CorrelationManager> logger)
        {
            _marketData = marketData ?? throw new ArgumentNullException(nameof(marketData));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }
        
        public async Task<SignalFilter> GetCorrelationFilterAsync(string instrument, SignalResult signal)
        {
            try
            {
                var correlation = await CalculateES_NQ_CorrelationAsync();
                var filter = new SignalFilter { Allow = true };
                
                // CRITICAL: ES/NQ divergence detection
                if (correlation.Divergence > 2.0) // Significant divergence
                {
                    _logger.LogWarning("ES/NQ divergence detected: {Divergence:F2}σ", correlation.Divergence);
                    
                    // Trade the laggard
                    if (instrument == correlation.Leader)
                    {
                        filter.Allow = false;
                        filter.Reason = $"{instrument} leading, wait for {GetOther(instrument)}";
                    }
                    else
                    {
                        filter.ConfidenceMultiplier = 1.3; // Higher confidence on laggard
                        filter.Reason = $"{instrument} lagging, catch-up trade";
                    }
                }
                
                // Correlation regime filtering
                if (correlation.Correlation5Min < 0.3) // Decorrelated
                {
                    _logger.LogInformation("ES/NQ decorrelated - reduce position size");
                    filter.PositionSizeMultiplier = 0.5;
                }
                else if (correlation.Correlation5Min > 0.9) // Highly correlated
                {
                    // Don't take opposing positions
                    var otherPosition = await GetCurrentPositionAsync(GetOther(instrument));
                    if (otherPosition != null && OppositeDirection(signal, otherPosition))
                    {
                        filter.Allow = false;
                        filter.Reason = "Would create opposing ES/NQ positions";
                    }
                }
                
                return filter;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error calculating correlation filter for {Instrument}", instrument);
                return new SignalFilter { Allow = true };
            }
        }
        
        public async Task<CorrelationData> GetCorrelationDataAsync()
        {
            return await CalculateES_NQ_CorrelationAsync();
        }
        
        private async Task<CorrelationData> CalculateES_NQ_CorrelationAsync()
        {
            try
            {
                var esData = await GetRecentBarsAsync("ES", 252);
                var nqData = await GetRecentBarsAsync("NQ", 252);
                
                var correlation = new CorrelationData
                {
                    LastUpdate = DateTime.UtcNow
                };
                
                // Calculate correlations at different timeframes
                correlation.Correlation5Min = CalculateCorrelation(esData.Take(5).ToList(), nqData.Take(5).ToList());
                correlation.Correlation20Min = CalculateCorrelation(esData.Take(20).ToList(), nqData.Take(20).ToList());
                correlation.Correlation60Min = CalculateCorrelation(esData.Take(60).ToList(), nqData.Take(60).ToList());
                correlation.CorrelationDaily = CalculateCorrelation(esData, nqData);
                
                // Detect lead/lag
                var leadLag = DetectLeadLag(esData, nqData);
                correlation.Leader = leadLag.Leader;
                correlation.LeadLagRatio = leadLag.Ratio;
                
                // Calculate divergence
                correlation.Divergence = CalculateDivergence(esData, nqData);
                
                _correlationMatrix["ES_NQ"] = correlation;
                
                return correlation;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error calculating ES/NQ correlation");
                return new CorrelationData
                {
                    Correlation5Min = 0.5,
                    Correlation20Min = 0.5,
                    Correlation60Min = 0.5,
                    CorrelationDaily = 0.5,
                    Leader = "NEUTRAL",
                    LeadLagRatio = 0,
                    Divergence = 0,
                    LastUpdate = DateTime.UtcNow
                };
            }
        }
        
        private async Task<List<decimal>> GetRecentBarsAsync(string symbol, int count)
        {
            try
            {
                // This would normally interface with your market data service
                // For now, return mock data
                var random = new Random();
                var bars = new List<decimal>();
                var basePrice = symbol == "ES" ? 4500m : 15000m;
                
                for (int i = 0; i < count; i++)
                {
                    var change = (decimal)(random.NextDouble() - 0.5) * 0.02m; // ±1% change
                    basePrice = basePrice * (1 + change);
                    bars.Add(basePrice);
                }
                
                return bars;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error getting recent bars for {Symbol}", symbol);
                return new List<decimal>();
            }
        }
        
        private double CalculateCorrelation(List<decimal> series1, List<decimal> series2)
        {
            try
            {
                if (series1.Count != series2.Count || series1.Count < 2)
                    return 0.0;
                
                // Calculate returns
                var returns1 = new List<double>();
                var returns2 = new List<double>();
                
                for (int i = 1; i < series1.Count; i++)
                {
                    if (series1[i - 1] != 0 && series2[i - 1] != 0)
                    {
                        returns1.Add((double)((series1[i] - series1[i - 1]) / series1[i - 1]));
                        returns2.Add((double)((series2[i] - series2[i - 1]) / series2[i - 1]));
                    }
                }
                
                if (returns1.Count < 2)
                    return 0.0;
                
                // Calculate correlation coefficient
                var mean1 = returns1.Average();
                var mean2 = returns2.Average();
                
                var numerator = returns1.Zip(returns2, (x, y) => (x - mean1) * (y - mean2)).Sum();
                var denominator1 = Math.Sqrt(returns1.Sum(x => Math.Pow(x - mean1, 2)));
                var denominator2 = Math.Sqrt(returns2.Sum(y => Math.Pow(y - mean2, 2)));
                
                if (denominator1 == 0 || denominator2 == 0)
                    return 0.0;
                
                return numerator / (denominator1 * denominator2);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error calculating correlation");
                return 0.0;
            }
        }
        
        private (string Leader, double Ratio) DetectLeadLag(List<decimal> esData, List<decimal> nqData)
        {
            try
            {
                // Simplified lead/lag detection using price momentum
                if (esData.Count < 5 || nqData.Count < 5)
                    return ("NEUTRAL", 0.0);
                
                var esReturn = (double)((esData[0] - esData[4]) / esData[4]);
                var nqReturn = (double)((nqData[0] - nqData[4]) / nqData[4]);
                
                var ratio = Math.Abs(esReturn) - Math.Abs(nqReturn);
                
                if (Math.Abs(ratio) < 0.001) // Too close to call
                    return ("NEUTRAL", 0.0);
                
                return ratio > 0 ? ("ES", ratio) : ("NQ", Math.Abs(ratio));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error detecting lead/lag");
                return ("NEUTRAL", 0.0);
            }
        }
        
        private double CalculateDivergence(List<decimal> esData, List<decimal> nqData)
        {
            try
            {
                if (esData.Count < 20 || nqData.Count < 20)
                    return 0.0;
                
                // Calculate z-score of the spread between normalized returns
                var esReturns = new List<double>();
                var nqReturns = new List<double>();
                
                for (int i = 1; i < Math.Min(esData.Count, 20); i++)
                {
                    if (esData[i - 1] != 0 && nqData[i - 1] != 0)
                    {
                        esReturns.Add((double)((esData[i] - esData[i - 1]) / esData[i - 1]));
                        nqReturns.Add((double)((nqData[i] - nqData[i - 1]) / nqData[i - 1]));
                    }
                }
                
                if (esReturns.Count < 10)
                    return 0.0;
                
                var spreads = esReturns.Zip(nqReturns, (es, nq) => es - nq).ToList();
                var mean = spreads.Average();
                var stdDev = Math.Sqrt(spreads.Sum(x => Math.Pow(x - mean, 2)) / spreads.Count);
                
                if (stdDev == 0)
                    return 0.0;
                
                return Math.Abs(spreads.Last() - mean) / stdDev;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error calculating divergence");
                return 0.0;
            }
        }
        
        private string GetOther(string instrument)
        {
            return instrument == "ES" ? "NQ" : "ES";
        }
        
        private async Task<Position?> GetCurrentPositionAsync(string symbol)
        {
            // This would interface with your position service
            // For now, return null (no position)
            await Task.Delay(1);
            return null;
        }
        
        private bool OppositeDirection(SignalResult signal, Position position)
        {
            // Check if signal direction is opposite to current position
            return (signal.Action == "BUY" && position.Size < 0) ||
                   (signal.Action == "SELL" && position.Size > 0);
        }
    }
    
    // Supporting classes
    public class Position
    {
        public string Symbol { get; set; } = "";
        public decimal Size { get; set; }
        public decimal CurrentPrice { get; set; }
    }
    
    public interface IMarketDataService
    {
        // Interface definition for market data service
    }
}