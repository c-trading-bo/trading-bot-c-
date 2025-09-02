using System;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Collections.Generic;
using System.Linq;

namespace TopstepX.Bot.Intelligence
{
    /// <summary>
    /// Cloud Data Integration for Local Trading Bot
    /// Consumes data collected by GitHub Actions cloud workflows
    /// Separates cloud (free data) from local (TopstepX API) responsibilities
    /// </summary>
    public class CloudDataIntegration
    {
        private readonly ILogger<CloudDataIntegration> _logger;
        private readonly string _intelligenceDir;
        
        public CloudDataIntegration(ILogger<CloudDataIntegration> logger, string intelligenceDir = "Intelligence")
        {
            _logger = logger;
            _intelligenceDir = intelligenceDir;
        }
        
        /// <summary>
        /// Get latest market signals from cloud-collected data
        /// </summary>
        public async Task<MarketSignals?> GetLatestMarketSignalsAsync()
        {
            try
            {
                var signalsFile = Path.Combine(_intelligenceDir, "data", "combined_trading_signals.json");
                
                if (!File.Exists(signalsFile))
                {
                    _logger.LogWarning("No combined trading signals file found. Cloud data may not be available.");
                    return null;
                }
                
                // Check if data is fresh (less than 1 hour old)
                var fileInfo = new FileInfo(signalsFile);
                var age = DateTime.Now - fileInfo.LastWriteTime;
                
                if (age.TotalHours > 1)
                {
                    _logger.LogWarning($"Trading signals data is {age.TotalMinutes:F1} minutes old. Consider refreshing cloud data.");
                }
                
                var jsonContent = await File.ReadAllTextAsync(signalsFile);
                var signals = JsonSerializer.Deserialize<MarketSignals>(jsonContent, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });
                
                _logger.LogInformation($"✅ Cloud market signals loaded - Regime: {signals?.RiskRegime}, Bias: {signals?.MarketBias}");
                return signals;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error loading cloud market signals");
                return null;
            }
        }
        
        /// <summary>
        /// Get enhanced options flow data from cloud collection
        /// </summary>
        public async Task<OptionsFlowData?> GetLatestOptionsDataAsync()
        {
            try
            {
                var optionsFile = Path.Combine(_intelligenceDir, "data", "options", "flow", "latest_enhanced_options.json");
                
                if (!File.Exists(optionsFile))
                {
                    _logger.LogWarning("No enhanced options data found");
                    return null;
                }
                
                var jsonContent = await File.ReadAllTextAsync(optionsFile);
                var optionsData = JsonSerializer.Deserialize<OptionsFlowData>(jsonContent, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });
                
                _logger.LogInformation($"✅ Options flow data loaded - Market P/C Ratio: {optionsData?.MarketSummary?.MarketPutCallRatio:F3}");
                return optionsData;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error loading options flow data");
                return null;
            }
        }
        
        /// <summary>
        /// Get macro economic data from cloud collection
        /// </summary>
        public async Task<MacroData?> GetLatestMacroDataAsync()
        {
            try
            {
                var macroFile = Path.Combine(_intelligenceDir, "data", "macro", "latest_macro_data.json");
                
                if (!File.Exists(macroFile))
                {
                    _logger.LogWarning("No macro data found");
                    return null;
                }
                
                var jsonContent = await File.ReadAllTextAsync(macroFile);
                var macroData = JsonSerializer.Deserialize<MacroData>(jsonContent, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });
                
                _logger.LogInformation($"✅ Macro data loaded - VIX: {macroData?.VolatilityIndices?.Vix?.Current:F2}");
                return macroData;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error loading macro data");
                return null;
            }
        }
        
        /// <summary>
        /// Check if cloud data collection is working properly
        /// </summary>
        public async Task<CloudDataStatus> GetCloudDataStatusAsync()
        {
            var status = new CloudDataStatus
            {
                Timestamp = DateTime.Now,
                IsHealthy = true,
                Issues = new List<string>()
            };
            
            try
            {
                // Check for required data files
                var requiredFiles = new Dictionary<string, string>
                {
                    ["Options"] = Path.Combine(_intelligenceDir, "data", "options", "flow", "latest_enhanced_options.json"),
                    ["Macro"] = Path.Combine(_intelligenceDir, "data", "macro", "latest_macro_data.json"),
                    ["News"] = Path.Combine(_intelligenceDir, "data", "news", "latest_news_sentiment.json"),
                    ["Signals"] = Path.Combine(_intelligenceDir, "data", "combined_trading_signals.json")
                };
                
                foreach (var (dataType, filePath) in requiredFiles)
                {
                    if (!File.Exists(filePath))
                    {
                        status.Issues.Add($"Missing {dataType} data file");
                        status.IsHealthy = false;
                        continue;
                    }
                    
                    // Check data freshness (should be updated every 30 minutes max)
                    var fileInfo = new FileInfo(filePath);
                    var age = DateTime.Now - fileInfo.LastWriteTime;
                    
                    if (age.TotalMinutes > 60) // Allow 1 hour for macro data
                    {
                        status.Issues.Add($"{dataType} data is stale ({age.TotalMinutes:F0} minutes old)");
                        if (age.TotalHours > 4) // Critical if over 4 hours
                        {
                            status.IsHealthy = false;
                        }
                    }
                }
                
                // Additional health checks
                var signals = await GetLatestMarketSignalsAsync();
                if (signals?.DataAvailability?.Options == false && 
                    signals?.DataAvailability?.Macro == false && 
                    signals?.DataAvailability?.News == false)
                {
                    status.Issues.Add("No data sources available in combined signals");
                    status.IsHealthy = false;
                }
                
                status.AvailableDataSources = new Dictionary<string, bool>
                {
                    ["Options"] = signals?.DataAvailability?.Options ?? false,
                    ["Macro"] = signals?.DataAvailability?.Macro ?? false,
                    ["News"] = signals?.DataAvailability?.News ?? false
                };
                
                _logger.LogInformation($"Cloud data status: {(status.IsHealthy ? "✅ Healthy" : "⚠️ Issues detected")}");
                
                if (status.Issues.Any())
                {
                    _logger.LogWarning($"Cloud data issues: {string.Join(", ", status.Issues)}");
                }
                
                return status;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error checking cloud data status");
                status.IsHealthy = false;
                status.Issues.Add($"Status check failed: {ex.Message}");
                return status;
            }
        }
        
        /// <summary>
        /// Get position sizing recommendation based on cloud intelligence
        /// </summary>
        public async Task<PositionSizingRecommendation> GetPositionSizingRecommendationAsync(decimal basePositionSize)
        {
            var recommendation = new PositionSizingRecommendation
            {
                BaseSize = basePositionSize,
                RecommendedSize = basePositionSize,
                SizingFactor = 1.0m,
                Reasoning = "No cloud data available - using base size"
            };
            
            try
            {
                var signals = await GetLatestMarketSignalsAsync();
                
                if (signals == null)
                {
                    return recommendation;
                }
                
                // Apply position sizing factor from cloud intelligence
                var sizingFactor = (decimal)(signals.PositionSizingFactor ?? 1.0);
                recommendation.SizingFactor = sizingFactor;
                recommendation.RecommendedSize = basePositionSize * sizingFactor;
                
                // Build reasoning
                var reasons = new List<string>();
                
                if (signals.RiskRegime == "HIGH_RISK")
                {
                    reasons.Add("High risk regime detected");
                }
                else if (signals.RiskRegime == "LOW_RISK")
                {
                    reasons.Add("Low risk regime - increased sizing");
                }
                
                if (signals.EntryConfidence < 0.5)
                {
                    reasons.Add("Low entry confidence");
                }
                else if (signals.EntryConfidence > 0.8)
                {
                    reasons.Add("High entry confidence");
                }
                
                if (signals.RiskFactors?.Any() == true)
                {
                    reasons.Add($"Risk factors: {string.Join(", ", signals.RiskFactors)}");
                }
                
                recommendation.Reasoning = reasons.Any() ? string.Join("; ", reasons) : "Standard market conditions";
                
                _logger.LogInformation($"Position sizing: {basePositionSize:F2} → {recommendation.RecommendedSize:F2} " +
                                     $"(factor: {sizingFactor:F2}) - {recommendation.Reasoning}");
                
                return recommendation;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error calculating position sizing recommendation");
                return recommendation;
            }
        }
    }
    
    // Data models for cloud intelligence integration
    
    public class MarketSignals
    {
        public string? Timestamp { get; set; }
        public DataAvailability? DataAvailability { get; set; }
        public string? RiskRegime { get; set; }
        public string? MarketBias { get; set; }
        public string? VolatilityRegime { get; set; }
        public double? PositionSizingFactor { get; set; }
        public double? EntryConfidence { get; set; }
        public List<string>? RiskFactors { get; set; }
    }
    
    public class DataAvailability
    {
        public bool Options { get; set; }
        public bool Macro { get; set; }
        public bool News { get; set; }
    }
    
    public class OptionsFlowData
    {
        public string? Timestamp { get; set; }
        public Dictionary<string, object>? Symbols { get; set; }
        public MarketSummary? MarketSummary { get; set; }
    }
    
    public class MarketSummary
    {
        public int TotalCallVolume { get; set; }
        public int TotalPutVolume { get; set; }
        public int TotalCallOI { get; set; }
        public int TotalPutOI { get; set; }
        public double MarketPutCallRatio { get; set; }
        public string? CollectionTime { get; set; }
    }
    
    public class MacroData
    {
        public string? Timestamp { get; set; }
        public Dictionary<string, TreasuryYield>? TreasuryYields { get; set; }
        public Dictionary<string, EconomicIndicator>? EconomicIndicators { get; set; }
        public Dictionary<string, Currency>? Currencies { get; set; }
        public Dictionary<string, Commodity>? Commodities { get; set; }
        public VolatilityIndices? VolatilityIndices { get; set; }
        public Dictionary<string, SentimentIndicator>? SentimentIndicators { get; set; }
    }
    
    public class TreasuryYield
    {
        public double Current { get; set; }
        public double Previous { get; set; }
        public double Change { get; set; }
        public double ChangeBps { get; set; }
    }
    
    public class EconomicIndicator
    {
        public double? Current { get; set; }
        public double? Previous { get; set; }
        public string? LastUpdated { get; set; }
        public string? SeriesId { get; set; }
    }
    
    public class Currency
    {
        public double Current { get; set; }
        public double Previous { get; set; }
        public double ChangePct { get; set; }
    }
    
    public class Commodity
    {
        public double Current { get; set; }
        public double Previous { get; set; }
        public double ChangePct { get; set; }
    }
    
    public class VolatilityIndex
    {
        public double Current { get; set; }
        public double Previous { get; set; }
        public double Change { get; set; }
        public double ChangePct { get; set; }
    }
    
    public class VolatilityIndices
    {
        public VolatilityIndex? Vix { get; set; }
        public VolatilityIndex? Vxn { get; set; }
        public VolatilityIndex? Rvx { get; set; }
        public VolatilityIndex? Vxd { get; set; }
    }
    
    public class SentimentIndicator
    {
        public double Score { get; set; }
        public string? Label { get; set; }
        public string? CalculationBasis { get; set; }
    }
    
    public class CloudDataStatus
    {
        public DateTime Timestamp { get; set; }
        public bool IsHealthy { get; set; }
        public List<string> Issues { get; set; } = new();
        public Dictionary<string, bool>? AvailableDataSources { get; set; }
    }
    
    public class PositionSizingRecommendation
    {
        public decimal BaseSize { get; set; }
        public decimal RecommendedSize { get; set; }
        public decimal SizingFactor { get; set; }
        public string? Reasoning { get; set; }
    }
}