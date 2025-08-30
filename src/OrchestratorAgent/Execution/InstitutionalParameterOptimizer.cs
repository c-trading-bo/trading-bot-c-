using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;

namespace OrchestratorAgent.Execution
{
    /// <summary>
    /// Optimized parameter configuration for institutional-level performance.
    /// Implements recommended institutional trading parameter settings.
    /// </summary>
    public static class InstitutionalParameterOptimizer
    {
        /// <summary>
        /// Gets optimized parameters based on institutional best practices
        /// </summary>
        public static OptimizedParameters GetOptimizedParameters()
        {
            return new OptimizedParameters
            {
                // Orchestrator parameters - reduce flip-flopping
                OrchUpdateCooldownBars = 35, // Increased from default ~20
                OrchStickiness = 0.025, // Increased from default ~0.01
                
                // Canary parameters - tighter promotion criteria
                CanaryRatio = 0.09, // Sweet spot for ES/NQ
                CanaryPValueMax = 0.08, // Tighter than 0.10 default, looser than 0.05
                
                // Risk parameters - institutional CVaR settings
                CvarLevel = 0.95, // 95th percentile risk
                CvarTargetR = 0.65, // Lower target = safer sizing, often higher WR
                
                // Bandit parameters - balanced exploration/exploitation
                WeightFloor = 0.10, // All strategies keep learning
                ExplorationRate = 0.05, // Conservative exploration
                
                // Position sizing limits
                MaxPositionMultiplier = 2.5, // Conservative maximum
                MinPositionMultiplier = 0.1, // Minimum viable size
                
                // Session-specific adjustments
                RthVolatilityMultiplier = 1.0, // Normal sizing during RTH
                EthVolatilityMultiplier = 0.75, // Reduced sizing during ETH
                
                // News trading parameters
                NewsConfidenceThreshold = 0.70, // Higher threshold for news trades
                NewsPositionMultiplier = 1.5, // Increased size for high-confidence news
                
                // Drift detection sensitivity
                DriftDetectionThreshold = 3.0, // Standard Page-Hinkley threshold
                SafeModeHours = 4.0, // Hours to stay in safe mode after drift
                
                // Meta-labeling parameters
                MetaLabelingThreshold = 0.65, // Only trade when >65% win probability
                MetaLabelingConfidence = 0.75, // Minimum confidence for meta-labeling
                
                // Execution parameters
                LimitOrderFillThreshold = 0.75, // Use limit if >75% fill probability
                MaxSlippageAllowed = 0.5, // Maximum acceptable slippage
                
                // Calibration parameters
                CalibrationMinSamples = 20, // Minimum samples for calibration
                CalibrationUpdateRate = 0.1, // Conservative learning rate
                
                // Walk-forward validation
                ValidationTrainingDays = 30, // 30-day training window
                ValidationTestDays = 1, // 1-day test window
                ValidationMinTrades = 50, // Minimum trades for valid training
                
                LastUpdated = DateTime.Now
            };
        }
        
        /// <summary>
        /// Applies optimized parameters to environment variables
        /// </summary>
        public static void ApplyOptimizedParameters(string configPath = "state/setup/optimized-params.json")
        {
            try
            {
                var parameters = GetOptimizedParameters();
                
                // Apply to environment variables
                Environment.SetEnvironmentVariable("ORCH_UPDATE_COOLDOWN_BARS", parameters.OrchUpdateCooldownBars.ToString());
                Environment.SetEnvironmentVariable("ORCH_STICKINESS", parameters.OrchStickiness.ToString("F3"));
                Environment.SetEnvironmentVariable("CANARY_RATIO", parameters.CanaryRatio.ToString("F3"));
                Environment.SetEnvironmentVariable("CANARY_PVALUE_MAX", parameters.CanaryPValueMax.ToString("F3"));
                Environment.SetEnvironmentVariable("CVAR_LEVEL", parameters.CvarLevel.ToString("F3"));
                Environment.SetEnvironmentVariable("CVAR_TARGET_R", parameters.CvarTargetR.ToString("F3"));
                Environment.SetEnvironmentVariable("BANDIT_WEIGHT_FLOOR", parameters.WeightFloor.ToString("F3"));
                Environment.SetEnvironmentVariable("BANDIT_EXPLORATION_RATE", parameters.ExplorationRate.ToString("F3"));
                Environment.SetEnvironmentVariable("MAX_POSITION_MULTIPLIER", parameters.MaxPositionMultiplier.ToString("F1"));
                Environment.SetEnvironmentVariable("NEWS_CONFIDENCE_THRESHOLD", parameters.NewsConfidenceThreshold.ToString("F2"));
                Environment.SetEnvironmentVariable("META_LABELING_THRESHOLD", parameters.MetaLabelingThreshold.ToString("F2"));
                Environment.SetEnvironmentVariable("DRIFT_THRESHOLD", parameters.DriftDetectionThreshold.ToString("F1"));
                
                // Save to config file
                SaveParametersToFile(parameters, configPath);
                
                Console.WriteLine("[PARAM-OPT] ✅ Applied institutional parameter optimization");
                Console.WriteLine($"[PARAM-OPT] Key changes: cooldown={parameters.OrchUpdateCooldownBars}, stickiness={parameters.OrchStickiness:F3}, canary={parameters.CanaryRatio:F3}");
                Console.WriteLine($"[PARAM-OPT] Risk: CVaR={parameters.CvarLevel:F2}, targetR={parameters.CvarTargetR:F2}, exploration={parameters.ExplorationRate:F3}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[PARAM-OPT] Error applying parameters: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Gets symbol-specific parameter adjustments
        /// </summary>
        public static SymbolParameters GetSymbolParameters(string symbol)
        {
            return symbol.ToUpper() switch
            {
                "ES" => new SymbolParameters
                {
                    Symbol = "ES",
                    TickSize = 0.25m,
                    TypicalSpread = 0.25m,
                    VolatilityMultiplier = 1.0,
                    LiquidityMultiplier = 1.0,
                    MaxPositionSize = 3,
                    OptimalRMultiple = 1.5m,
                    PreferredStrategies = new[] { "S2a", "S2b", "S3a" }
                },
                "NQ" => new SymbolParameters
                {
                    Symbol = "NQ",
                    TickSize = 0.25m,
                    TypicalSpread = 0.50m,
                    VolatilityMultiplier = 0.8, // Reduce due to higher volatility
                    LiquidityMultiplier = 0.9,
                    MaxPositionSize = 2,
                    OptimalRMultiple = 1.8m,
                    PreferredStrategies = new[] { "S2a", "S3a" }
                },
                "MES" => new SymbolParameters
                {
                    Symbol = "MES",
                    TickSize = 0.25m,
                    TypicalSpread = 0.25m,
                    VolatilityMultiplier = 1.2, // Can be more aggressive with micro
                    LiquidityMultiplier = 0.8,
                    MaxPositionSize = 10,
                    OptimalRMultiple = 1.3m,
                    PreferredStrategies = new[] { "S2a", "S2b", "S3a", "S3b" }
                },
                "MNQ" => new SymbolParameters
                {
                    Symbol = "MNQ",
                    TickSize = 0.25m,
                    TypicalSpread = 0.50m,
                    VolatilityMultiplier = 1.0,
                    LiquidityMultiplier = 0.7,
                    MaxPositionSize = 8,
                    OptimalRMultiple = 1.6m,
                    PreferredStrategies = new[] { "S2a", "S3a" }
                },
                _ => new SymbolParameters { Symbol = symbol }
            };
        }
        
        /// <summary>
        /// Gets session-specific parameter adjustments
        /// </summary>
        public static SessionParameters GetSessionParameters(SessionType session)
        {
            return session switch
            {
                SessionType.RTH => new SessionParameters
                {
                    Session = SessionType.RTH,
                    VolatilityMultiplier = 1.0,
                    SpreadMultiplier = 1.0,
                    ConfidenceBonus = 0.05, // Higher confidence during RTH
                    MaxTradesPerHour = 8,
                    PreferredOrderType = "LIMIT",
                    TimeoutMinutes = 15
                },
                SessionType.ETH => new SessionParameters
                {
                    Session = SessionType.ETH,
                    VolatilityMultiplier = 0.75, // More conservative
                    SpreadMultiplier = 1.3, // Wider spreads
                    ConfidenceBonus = -0.05, // Lower confidence
                    MaxTradesPerHour = 4,
                    PreferredOrderType = "MARKET", // Fill certainty more important
                    TimeoutMinutes = 30
                },
                _ => new SessionParameters()
            };
        }
        
        /// <summary>
        /// Creates comprehensive parameter profile for specific market conditions
        /// </summary>
        public static ParameterProfile CreateParameterProfile(string symbol, SessionType session, string regime)
        {
            var baseParams = GetOptimizedParameters();
            var symbolParams = GetSymbolParameters(symbol);
            var sessionParams = GetSessionParameters(session);
            
            return new ParameterProfile
            {
                Symbol = symbol,
                Session = session,
                Regime = regime,
                
                // Combined position sizing
                BasePositionSize = Math.Min(symbolParams.MaxPositionSize, 
                    baseParams.MaxPositionMultiplier * symbolParams.VolatilityMultiplier * sessionParams.VolatilityMultiplier),
                
                // Combined confidence adjustments
                ConfidenceThreshold = baseParams.MetaLabelingThreshold + sessionParams.ConfidenceBonus,
                
                // Risk parameters
                TargetRMultiple = (double)symbolParams.OptimalRMultiple,
                MaxRisk = baseParams.CvarTargetR,
                
                // Execution preferences
                PreferredOrderType = sessionParams.PreferredOrderType,
                TimeoutMinutes = sessionParams.TimeoutMinutes,
                
                // Strategy filtering
                AllowedStrategies = symbolParams.PreferredStrategies,
                
                CreatedAt = DateTime.Now
            };
        }
        
        private static void SaveParametersToFile(OptimizedParameters parameters, string filePath)
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(filePath)!);
                var json = JsonSerializer.Serialize(parameters, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(filePath, json);
                Console.WriteLine($"[PARAM-OPT] Saved parameters to {filePath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[PARAM-OPT] Failed to save parameters: {ex.Message}");
            }
        }
    }
    
    public class OptimizedParameters
    {
        // Orchestrator parameters
        public int OrchUpdateCooldownBars { get; set; }
        public double OrchStickiness { get; set; }
        
        // Canary parameters
        public double CanaryRatio { get; set; }
        public double CanaryPValueMax { get; set; }
        
        // Risk parameters
        public double CvarLevel { get; set; }
        public double CvarTargetR { get; set; }
        
        // Bandit parameters
        public double WeightFloor { get; set; }
        public double ExplorationRate { get; set; }
        
        // Position sizing
        public double MaxPositionMultiplier { get; set; }
        public double MinPositionMultiplier { get; set; }
        
        // Session adjustments
        public double RthVolatilityMultiplier { get; set; }
        public double EthVolatilityMultiplier { get; set; }
        
        // News trading
        public double NewsConfidenceThreshold { get; set; }
        public double NewsPositionMultiplier { get; set; }
        
        // Drift detection
        public double DriftDetectionThreshold { get; set; }
        public double SafeModeHours { get; set; }
        
        // Meta-labeling
        public double MetaLabelingThreshold { get; set; }
        public double MetaLabelingConfidence { get; set; }
        
        // Execution
        public double LimitOrderFillThreshold { get; set; }
        public double MaxSlippageAllowed { get; set; }
        
        // Calibration
        public int CalibrationMinSamples { get; set; }
        public double CalibrationUpdateRate { get; set; }
        
        // Validation
        public int ValidationTrainingDays { get; set; }
        public int ValidationTestDays { get; set; }
        public int ValidationMinTrades { get; set; }
        
        public DateTime LastUpdated { get; set; }
    }
    
    public class SymbolParameters
    {
        public string Symbol { get; set; } = "";
        public decimal TickSize { get; set; } = 0.25m;
        public decimal TypicalSpread { get; set; } = 0.25m;
        public double VolatilityMultiplier { get; set; } = 1.0;
        public double LiquidityMultiplier { get; set; } = 1.0;
        public int MaxPositionSize { get; set; } = 3;
        public decimal OptimalRMultiple { get; set; } = 1.5m;
        public string[] PreferredStrategies { get; set; } = Array.Empty<string>();
    }
    
    public class SessionParameters
    {
        public SessionType Session { get; set; }
        public double VolatilityMultiplier { get; set; } = 1.0;
        public double SpreadMultiplier { get; set; } = 1.0;
        public double ConfidenceBonus { get; set; } = 0.0;
        public int MaxTradesPerHour { get; set; } = 6;
        public string PreferredOrderType { get; set; } = "LIMIT";
        public int TimeoutMinutes { get; set; } = 15;
    }
    
    public class ParameterProfile
    {
        public string Symbol { get; set; } = "";
        public SessionType Session { get; set; }
        public string Regime { get; set; } = "";
        
        public double BasePositionSize { get; set; }
        public double ConfidenceThreshold { get; set; }
        public double TargetRMultiple { get; set; }
        public double MaxRisk { get; set; }
        
        public string PreferredOrderType { get; set; } = "";
        public int TimeoutMinutes { get; set; }
        public string[] AllowedStrategies { get; set; } = Array.Empty<string>();
        
        public DateTime CreatedAt { get; set; }
    }
}
