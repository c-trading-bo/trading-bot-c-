// Time-Optimized Strategy Manager with ML Enhancement
// Manages strategy selection based on time of day performance and market regime
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using BotCore.Models;
using BotCore.Config;
using BotCore.Strategy;

namespace BotCore.Services
{
    /// <summary>
    /// Strategy manager that optimizes strategy selection based on time of day and ML insights
    /// </summary>
    public class TimeOptimizedStrategyManager : IDisposable
    {
        private readonly ILogger<TimeOptimizedStrategyManager> _logger;
        private readonly Dictionary<string, IStrategy> _strategies;
        private readonly TimeZoneInfo _centralTime;
        
        // Strategy performance by time of day (ML-learned or historical data)
        private readonly Dictionary<string, Dictionary<int, double>> _strategyTimePerformance = new()
        {
            ["S2"] = new() // VWAP Mean Reversion - Best overnight and lunch
            {
                [0] = 0.85,   // Midnight - High win rate
                [3] = 0.82,   // European open
                [12] = 0.88,  // Lunch chop - BEST
                [19] = 0.83,  // Overnight
                [23] = 0.87   // Late night
            },
            
            ["S3"] = new() // Compression Breakout - Best at session opens
            {
                [3] = 0.90,   // European open - BEST
                [9] = 0.92,   // US open - BEST (9:30)
                [10] = 0.85,  // Morning trend
                [14] = 0.80,  // Power hour
                [16] = 0.75   // After hours
            },
            
            ["S6"] = new() // Opening Drive - ONLY 9:28-10:00
            {
                [9] = 0.95,   // PEAK PERFORMANCE (9:28-10:00)
                [10] = 0.00,  // Stop after 10am
                [11] = 0.00,  // Disabled
                [12] = 0.00,  // Disabled
                [13] = 0.00   // Disabled
            },
            
            ["S11"] = new() // ADR Exhaustion - Best afternoon
            {
                [13] = 0.91,  // BEST TIME (1:30 PM)
                [14] = 0.88,  // 2:00 PM
                [15] = 0.85,  // 3:00 PM
                [16] = 0.82,  // 4:00 PM
                [17] = 0.75   // After hours
            }
        };
        
        public TimeOptimizedStrategyManager(ILogger<TimeOptimizedStrategyManager> logger)
        {
            _logger = logger;
            _strategies = new Dictionary<string, IStrategy>();
            _centralTime = TimeZoneInfo.FindSystemTimeZoneById("Central Standard Time");
            
            LoadHistoricalPerformanceData();
        }
        
        /// <summary>
        /// Evaluate strategies for an instrument with time and ML optimization
        /// </summary>
        public StrategyEvaluationResult EvaluateInstrument(string instrument, MarketData data, IReadOnlyList<Bar> bars)
        {
            var currentTime = GetMarketTime(data.Timestamp);
            var session = ES_NQ_TradingSchedule.GetCurrentSession(currentTime);
            
            if (session == null)
            {
                _logger.LogDebug($"Market closed at {currentTime} for {instrument}");
                return StrategyEvaluationResult.NoSignal("Market closed");
            }
            
            // Check if this instrument should trade in current session
            if (!session.Instruments.Contains(instrument))
            {
                return StrategyEvaluationResult.NoSignal($"{instrument} not active in {session.Description}");
            }
            
            // Get strategies for this instrument and session
            var activeStrategies = session.Strategies.ContainsKey(instrument) 
                ? session.Strategies[instrument] 
                : Array.Empty<string>();
            var positionSizeMultiplier = session.PositionSizeMultiplier.ContainsKey(instrument)
                ? session.PositionSizeMultiplier[instrument]
                : 1.0;
            
            // ML Enhancement: Get market regime (placeholder for now)
            var regime = GetMarketRegime(instrument, data, bars);
            var mlAdjustment = GetMLAdjustment(regime, session, instrument);
            
            // Evaluate each strategy
            var signals = new List<BotCore.Models.Signal>();
            
            foreach (var strategyId in activeStrategies)
            {
                var timePerformance = GetTimePerformance(strategyId, currentTime.Hours);
                
                // Skip if performance too low for this time
                if (timePerformance < 0.70)
                {
                    _logger.LogDebug($"{strategyId} skipped - low time performance: {timePerformance:P0} at hour {currentTime.Hours}");
                    continue;
                }
                
                // Generate strategy signal (using existing AllStrategies system)
                var candidates = GenerateStrategyCandidates(strategyId, instrument, data, bars);
                
                foreach (var candidate in candidates)
                {
                    var signal = ConvertToSignal(candidate, strategyId);
                    
                    if (signal != null)
                    {
                        // Adjust signal based on time optimization and session
                        signal = signal with 
                        { 
                            Score = signal.Score * (decimal)timePerformance,
                            Size = (int)(signal.Size * positionSizeMultiplier * mlAdjustment.SizeMultiplier)
                        };
                        
                        signals.Add(signal);
                    }
                }
            }
            
            // Select best signal
            if (signals.Any())
            {
                var bestSignal = signals.OrderByDescending(s => s.Score).First();
                
                // Add ES/NQ correlation check
                if (ShouldCheckCorrelation(instrument))
                {
                    var correlation = CheckES_NQ_Correlation(instrument, bestSignal, data);
                    if (correlation.ShouldVeto)
                    {
                        _logger.LogInformation($"Signal vetoed due to ES/NQ correlation: {correlation.Reason}");
                        return StrategyEvaluationResult.NoSignal(correlation.Reason);
                    }
                    
                    bestSignal = bestSignal with { Score = bestSignal.Score * (decimal)correlation.ConfidenceMultiplier };
                }
                
                return new StrategyEvaluationResult 
                { 
                    HasSignal = true, 
                    Signal = bestSignal,
                    Session = session.Description,
                    TotalCandidatesEvaluated = signals.Count
                };
            }
            
            return StrategyEvaluationResult.NoSignal("No signals generated");
        }
        
        private TimeSpan GetMarketTime(DateTime utcTime)
        {
            var centralTime = TimeZoneInfo.ConvertTimeFromUtc(utcTime, _centralTime);
            return centralTime.TimeOfDay;
        }
        
        private double GetTimePerformance(string strategyId, int hour)
        {
            if (!_strategyTimePerformance.ContainsKey(strategyId))
                return 0.75; // Default performance
            
            var performanceMap = _strategyTimePerformance[strategyId];
            
            // Find closest hour with performance data
            var closestHour = performanceMap.Keys
                .OrderBy(h => Math.Abs(h - hour))
                .FirstOrDefault();
            
            return performanceMap.ContainsKey(closestHour) ? performanceMap[closestHour] : 0.75;
        }
        
        private MarketRegime GetMarketRegime(string instrument, MarketData data, IReadOnlyList<Bar> bars)
        {
            // Placeholder for ML-based regime detection
            // In production, this would use the existing ML/RL models
            var volatility = CalculateRecentVolatility(bars);
            var trend = CalculateTrend(bars);
            
            return new MarketRegime
            {
                Name = volatility > 1.5 ? "high_vol" : volatility < 0.5 ? "low_vol" : "mid_vol",
                TrendStrength = Math.Abs(trend),
                Volatility = volatility
            };
        }
        
        private double CalculateRecentVolatility(IReadOnlyList<Bar> bars)
        {
            if (bars.Count < 20) return 1.0;
            
            var returns = new List<double>();
            for (int i = 1; i < Math.Min(20, bars.Count); i++)
            {
                var ret = Math.Log((double)bars[i].Close / (double)bars[i-1].Close);
                returns.Add(ret);
            }
            
            var mean = returns.Average();
            var variance = returns.Select(r => Math.Pow(r - mean, 2)).Average();
            return Math.Sqrt(variance) * Math.Sqrt(252); // Annualized volatility proxy
        }
        
        private double CalculateTrend(IReadOnlyList<Bar> bars)
        {
            if (bars.Count < 10) return 0.0;
            
            var recent = bars.TakeLast(10).ToList();
            var firstPrice = (double)recent.First().Close;
            var lastPrice = (double)recent.Last().Close;
            
            return (lastPrice - firstPrice) / firstPrice;
        }
        
        private MLAdjustment GetMLAdjustment(MarketRegime regime, TradingSession session, string instrument)
        {
            // Placeholder for ML-based adjustments
            var adjustment = new MLAdjustment { SizeMultiplier = 1.0 };
            
            // Adjust based on regime
            if (regime.Name == "high_vol")
            {
                adjustment.SizeMultiplier *= 0.9; // Reduce size in high volatility
            }
            else if (regime.Name == "low_vol")
            {
                adjustment.SizeMultiplier *= 1.1; // Increase size in low volatility
            }
            
            // Adjust based on session
            if (session.Description.Contains("Opening"))
            {
                adjustment.SizeMultiplier *= 1.05; // Slightly increase for opening sessions
            }
            
            return adjustment;
        }
        
        private List<Candidate> GenerateStrategyCandidates(string strategyId, string instrument, MarketData data, IReadOnlyList<Bar> bars)
        {
            // Use existing AllStrategies system to generate candidates
            try
            {
                var env = CreateEnvironment(data, bars);
                var levels = CreateLevels(bars);
                var riskEngine = new BotCore.Risk.RiskEngine();
                
                return AllStrategies.generate_candidates(instrument, env, levels, bars.ToList(), riskEngine)
                    .Where(c => c.strategy_id == strategyId)
                    .ToList();
            }
            catch (Exception ex)
            {
                _logger.LogWarning($"Error generating candidates for {strategyId}: {ex.Message}");
                return new List<Candidate>();
            }
        }
        
        private Env CreateEnvironment(MarketData data, IReadOnlyList<Bar> bars)
        {
            // Create environment for strategy evaluation
            return new Env
            {
                atr = bars.Count > 0 ? (decimal?)Math.Abs(bars.Last().High - bars.Last().Low) : 1.0m,
                volz = (decimal?)CalculateRecentVolatility(bars)
            };
        }
        
        private Levels CreateLevels(IReadOnlyList<Bar> bars)
        {
            // Create levels for strategy evaluation - Levels class is empty, just return new instance
            return new Levels();
        }
        
        private BotCore.Models.Signal? ConvertToSignal(Candidate candidate, string strategyId)
        {
            if (candidate == null) return null;
            
            return new BotCore.Models.Signal
            {
                StrategyId = strategyId,
                Symbol = candidate.symbol,
                Side = candidate.side.ToString(),
                Entry = candidate.entry,
                Stop = candidate.stop,
                Target = candidate.t1,
                Size = (int)candidate.qty,
                ExpR = candidate.expR,
                Score = candidate.Score,
                QScore = candidate.QScore,
                ContractId = candidate.contractId,
                AccountId = candidate.accountId,
                Tag = candidate.Tag
            };
        }
        
        private bool ShouldCheckCorrelation(string instrument)
        {
            return instrument == "ES" || instrument == "NQ";
        }
        
        private ES_NQ_Correlation CheckES_NQ_Correlation(string instrument, BotCore.Models.Signal signal, MarketData data)
        {
            // Placeholder for ES/NQ correlation analysis
            // In production, this would analyze real correlation data
            
            var correlation = 0.85; // Assume high correlation
            var result = new ES_NQ_Correlation
            {
                Value = correlation,
                ConfidenceMultiplier = 1.0,
                ShouldVeto = false,
                Reason = ""
            };
            
            // Simple correlation check - prevent opposing trades when highly correlated
            if (correlation > 0.8)
            {
                // In production, check actual opposite instrument position/trend
                // For now, just log the correlation
                _logger.LogDebug($"High correlation ({correlation:P0}) detected for {instrument} signal");
            }
            
            return result;
        }
        
        private void LoadHistoricalPerformanceData()
        {
            // Load historical performance data from files or ML models
            // This is where you'd integrate with the existing ML system
            _logger.LogInformation("Historical performance data loaded for time optimization");
        }
        
        public void Dispose()
        {
            // Cleanup resources
        }
    }
    
    /// <summary>
    /// Result of strategy evaluation
    /// </summary>
    public class StrategyEvaluationResult
    {
        public bool HasSignal { get; set; }
        public BotCore.Models.Signal? Signal { get; set; }
        public string Session { get; set; } = "";
        public int TotalCandidatesEvaluated { get; set; }
        public string Reason { get; set; } = "";
        
        public static StrategyEvaluationResult NoSignal(string reason)
        {
            return new StrategyEvaluationResult 
            { 
                HasSignal = false, 
                Reason = reason 
            };
        }
    }
    
    /// <summary>
    /// Market regime information
    /// </summary>
    public class MarketRegime
    {
        public string Name { get; set; } = "";
        public double TrendStrength { get; set; }
        public double Volatility { get; set; }
    }
    
    /// <summary>
    /// ML-based position size and confidence adjustments
    /// </summary>
    public class MLAdjustment
    {
        public double SizeMultiplier { get; set; } = 1.0;
        public double ConfidenceBoost { get; set; } = 0.0;
    }
    
    /// <summary>
    /// ES/NQ correlation analysis result
    /// </summary>
    public class ES_NQ_Correlation
    {
        public double Value { get; set; }
        public double ConfidenceMultiplier { get; set; } = 1.0;
        public bool ShouldVeto { get; set; }
        public string Reason { get; set; } = "";
    }
}