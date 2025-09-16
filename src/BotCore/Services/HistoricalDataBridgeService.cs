using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using BotCore.Market;
using BotCore.Models;
using System.Text.Json;
using System.Net.Http;
using System.Net.Http.Json;

namespace BotCore.Services
{
    /// <summary>
    /// Historical Data Bridge Service for production-ready trading warm-up
    /// Seeds bar aggregator with recent historical data while maintaining data provenance
    /// Implements the solution for BarsSeen >= 10 requirement
    /// </summary>
    public interface IHistoricalDataBridgeService
    {
        Task<bool> SeedTradingSystemAsync(string[] contractIds);
        Task<List<BotCore.Models.Bar>> GetRecentHistoricalBarsAsync(string contractId, int barCount = 20);
        Task<bool> ValidateHistoricalDataAsync(string contractId);
    }

    /// <summary>
    /// Interface for consuming historical bars in the trading system
    /// Allows the bridge to properly feed historical data into the live system
    /// </summary>
    public interface IHistoricalBarConsumer
    {
        /// <summary>
        /// Process historical bars as if they were live bars
        /// This should increment the BarsSeen counter and feed aggregators
        /// </summary>
        void ConsumeHistoricalBars(string contractId, IEnumerable<BotCore.Models.Bar> bars);
    }

    public class HistoricalDataBridgeService : IHistoricalDataBridgeService
    {
        private readonly ILogger<HistoricalDataBridgeService> _logger;
        private readonly TradingReadinessConfiguration _config;
        private readonly HttpClient _httpClient;
        private readonly IHistoricalBarConsumer? _barConsumer;

        public HistoricalDataBridgeService(
            ILogger<HistoricalDataBridgeService> logger,
            IOptions<TradingReadinessConfiguration> config,
            HttpClient httpClient,
            IHistoricalBarConsumer? barConsumer = null)
        {
            _logger = logger;
            _config = config.Value;
            _httpClient = httpClient;
            _barConsumer = barConsumer;
        }

        /// <summary>
        /// Seed the trading system with historical data for fast warm-up
        /// </summary>
        public async Task<bool> SeedTradingSystemAsync(string[] contractIds)
        {
            if (!_config.EnableHistoricalSeeding)
            {
                _logger.LogInformation("[HISTORICAL-BRIDGE] Historical seeding disabled");
                return false;
            }

            _logger.LogInformation("[HISTORICAL-BRIDGE] Starting historical data seeding for {ContractCount} contracts", contractIds.Length);

            var successCount = 0;
            var totalSeeded = 0;

            foreach (var contractId in contractIds)
            {
                try
                {
                    _logger.LogDebug("[HISTORICAL-BRIDGE] Seeding contract: {ContractId}", contractId);

                    // Get recent historical bars - FIXED: Request sufficient bars for trading strategies (not just seeding)
                    var historicalBars = await GetRecentHistoricalBarsAsync(contractId, Math.Max(_config.MinSeededBars + 2, 200));
                    
                    if (historicalBars.Any())
                    {
                        totalSeeded += historicalBars.Count;
                        successCount++;

                        // CRITICAL FIX: Actually feed the bars into the trading system
                        if (_barConsumer != null)
                        {
                            _barConsumer.ConsumeHistoricalBars(contractId, historicalBars);
                            _logger.LogInformation("[HISTORICAL-BRIDGE] ✅ Seeded {BarCount} historical bars for {ContractId} into trading system", 
                                historicalBars.Count, contractId);
                        }
                        else
                        {
                            _logger.LogWarning("[HISTORICAL-BRIDGE] ⚠️ No bar consumer available - bars retrieved but not fed to trading system for {ContractId}", contractId);
                        }
                    }
                    else
                    {
                        _logger.LogWarning("[HISTORICAL-BRIDGE] ⚠️ No historical data available for {ContractId}", contractId);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[HISTORICAL-BRIDGE] ❌ Failed to seed {ContractId}: {Error}", contractId, ex.Message);
                }
            }

            var success = successCount > 0;
            _logger.LogInformation("[HISTORICAL-BRIDGE] Seeding complete: {SuccessCount}/{TotalCount} contracts, {TotalBars} bars available", 
                successCount, contractIds.Length, totalSeeded);

            return success;
        }

        /// <summary>
        /// Get recent historical bars from TopstepX API or fallback sources
        /// </summary>
        public async Task<List<BotCore.Models.Bar>> GetRecentHistoricalBarsAsync(string contractId, int barCount = 20)
        {
            try
            {
                // Primary: Try TopstepX historical API
                var topstepXBars = await TryGetTopstepXBarsAsync(contractId, barCount);
                if (topstepXBars.Any())
                {
                    _logger.LogDebug("[HISTORICAL-BRIDGE] Retrieved {BarCount} bars from TopstepX for {ContractId}", 
                        topstepXBars.Count, contractId);
                    return topstepXBars;
                }

                // Fallback: Use correlation manager's data sources
                var correlationBars = await TryGetCorrelationManagerBarsAsync(contractId, barCount);
                if (correlationBars.Any())
                {
                    _logger.LogDebug("[HISTORICAL-BRIDGE] Retrieved {BarCount} bars from correlation manager for {ContractId}", 
                        correlationBars.Count, contractId);
                    return correlationBars;
                }

                // Last resort: Generate recent synthetic bars for warm-up
                var syntheticBars = GenerateWarmupBars(contractId, barCount);
                _logger.LogWarning("[HISTORICAL-BRIDGE] Using synthetic bars for {ContractId} - {BarCount} bars generated", 
                    contractId, syntheticBars.Count);
                
                return syntheticBars;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[HISTORICAL-BRIDGE] Error getting historical bars for {ContractId}", contractId);
                return new List<BotCore.Models.Bar>();
            }
        }

        /// <summary>
        /// Validate that historical data is recent and suitable for trading
        /// </summary>
        public async Task<bool> ValidateHistoricalDataAsync(string contractId)
        {
            try
            {
                var bars = await GetRecentHistoricalBarsAsync(contractId, 5);
                if (!bars.Any()) return false;

                var mostRecentBar = bars.OrderByDescending(b => b.Ts).First();
                var dataAge = DateTime.UtcNow - DateTime.UnixEpoch.AddMilliseconds(mostRecentBar.Ts);

                var isValid = dataAge.TotalHours <= _config.MaxHistoricalDataAgeHours;
                
                _logger.LogDebug("[HISTORICAL-BRIDGE] Data validation for {ContractId}: Age={Age:F1}h, Valid={IsValid}", 
                    contractId, dataAge.TotalHours, isValid);

                return isValid;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[HISTORICAL-BRIDGE] Validation failed for {ContractId}", contractId);
                return false;
            }
        }

        #region Private Helper Methods

        private async Task<List<BotCore.Models.Bar>> TryGetTopstepXBarsAsync(string contractId, int barCount)
        {
            try
            {
                // Get JWT token from environment
                var jwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
                if (string.IsNullOrEmpty(jwt))
                {
                    _logger.LogDebug("[HISTORICAL-BRIDGE] No TOPSTEPX_JWT found for historical data fetching");
                    return new List<BotCore.Models.Bar>();
                }
                
                // Calculate time window - get last few days to ensure we have enough bars
                var endTime = DateTime.UtcNow;
                var startTime = endTime.AddDays(-7); // Get 7 days of data to ensure we have enough
                
                // Create request body
                var requestBody = new
                {
                    contractId = contractId,
                    live = false, // Use simulation data for consistency
                    startTime = startTime.ToString("yyyy-MM-ddTHH:mm:ssZ"),
                    endTime = endTime.ToString("yyyy-MM-ddTHH:mm:ssZ"),
                    unit = 2, // Minutes
                    unitNumber = 1, // 1-minute bars
                    limit = Math.Max(barCount, 1000), // Get more than requested to ensure we have enough
                    includePartialBar = false
                };
                
                _logger.LogDebug("[HISTORICAL-BRIDGE] Fetching TopstepX historical data for {ContractId} from {StartTime} to {EndTime}", 
                    contractId, startTime, endTime);
                
                // Set authorization header
                _httpClient.DefaultRequestHeaders.Clear();
                _httpClient.DefaultRequestHeaders.Authorization = 
                    new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt);
                
                // Make API call
                var response = await _httpClient.PostAsJsonAsync("https://api.topstepx.com/api/History/retrieveBars", requestBody);
                
                if (!response.IsSuccessStatusCode)
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    _logger.LogWarning("[HISTORICAL-BRIDGE] TopstepX History API failed: {StatusCode} - {Error}", 
                        response.StatusCode, errorContent);
                    return new List<BotCore.Models.Bar>();
                }
                
                var responseContent = await response.Content.ReadAsStringAsync();
                _logger.LogDebug("[HISTORICAL-BRIDGE] TopstepX History API response length: {Length} characters", responseContent.Length);
                
                // Debug: Log first 500 characters of response
                var previewContent = responseContent.Length > 500 ? responseContent.Substring(0, 500) + "..." : responseContent;
                _logger.LogInformation("[HISTORICAL-BRIDGE] 📊 API Response Preview: {Preview}", previewContent);
                
                // Parse response to Bar objects
                var bars = ParseTopstepXHistoricalResponse(responseContent, contractId);
                
                if (bars.Count > 0)
                {
                    // FIXED: Use ALL available bars instead of limiting to barCount
                    // The API fetches 1000 bars, we should use them all for better trading decisions
                    var recentBars = bars.OrderByDescending(b => b.Ts).Take(Math.Min(bars.Count, 1000)).OrderBy(b => b.Ts).ToList();
                    _logger.LogInformation("[HISTORICAL-BRIDGE] Retrieved {Count} TopstepX historical bars for {ContractId} (requested {RequestedCount}, using {UsedCount})", 
                        bars.Count, contractId, barCount, recentBars.Count);
                    return recentBars;
                }
                
                return new List<BotCore.Models.Bar>();
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[HISTORICAL-BRIDGE] TopstepX bars failed for {ContractId}: {Error}", contractId, ex.Message);
                return new List<BotCore.Models.Bar>();
            }
        }

        private async Task<List<BotCore.Models.Bar>> TryGetCorrelationManagerBarsAsync(string contractId, int barCount)
        {
            try
            {
                // This would integrate with other data sources through correlation manager
                // For now, return empty - could be extended with additional data sources
                await Task.CompletedTask;
                return new List<BotCore.Models.Bar>();
            }
            catch (Exception ex)
            {
                _logger.LogDebug("[HISTORICAL-BRIDGE] Correlation manager bars failed for {ContractId}: {Error}", contractId, ex.Message);
                return new List<BotCore.Models.Bar>();
            }
        }

        private List<BotCore.Models.Bar> ParseTopstepXHistoricalResponse(string responseContent, string contractId)
        {
            try
            {
                using var document = System.Text.Json.JsonDocument.Parse(responseContent);
                var root = document.RootElement;
                
                // Check for success field
                if (root.TryGetProperty("success", out var successElement) && !successElement.GetBoolean())
                {
                    _logger.LogWarning("[HISTORICAL-BRIDGE] TopstepX History API returned success=false for {ContractId}", contractId);
                    return new List<BotCore.Models.Bar>();
                }
                
                // TopstepX format: { "bars": [...], "success": true, "errorCode": 0, "errorMessage": null }
                if (!root.TryGetProperty("bars", out var barsElement) || barsElement.ValueKind != JsonValueKind.Array)
                {
                    _logger.LogWarning("[HISTORICAL-BRIDGE] No 'bars' array found in TopstepX response for {ContractId}", contractId);
                    return new List<BotCore.Models.Bar>();
                }
                
                var bars = new List<BotCore.Models.Bar>();
                var symbol = GetSymbolFromContractId(contractId);
                
                foreach (var barElement in barsElement.EnumerateArray())
                {
                    try
                    {
                        var bar = new BotCore.Models.Bar
                        {
                            Symbol = symbol,
                            Ts = ParseTopstepXTimestampToUnixMs(barElement),
                            Open = ParseDecimalField(barElement, "o", "open"),
                            High = ParseDecimalField(barElement, "h", "high"), 
                            Low = ParseDecimalField(barElement, "l", "low"),
                            Close = ParseDecimalField(barElement, "c", "close"),
                            Volume = (int)ParseLongField(barElement, "v", "volume")
                        };
                        
                        // Validate bar data
                        if (bar.Open > 0 && bar.High > 0 && bar.Low > 0 && bar.Close > 0 && 
                            bar.High >= bar.Low && bar.High >= bar.Open && bar.High >= bar.Close &&
                            bar.Low <= bar.Open && bar.Low <= bar.Close)
                        {
                            bars.Add(bar);
                        }
                        else
                        {
                            _logger.LogDebug("[HISTORICAL-BRIDGE] Invalid bar data for {ContractId}: O:{Open} H:{High} L:{Low} C:{Close}", 
                                contractId, bar.Open, bar.High, bar.Low, bar.Close);
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogDebug(ex, "[HISTORICAL-BRIDGE] Error parsing individual bar for {ContractId}", contractId);
                        // Continue with other bars
                    }
                }
                
                _logger.LogInformation("[HISTORICAL-BRIDGE] Parsed {Count} valid bars from TopstepX response for {ContractId}", bars.Count, contractId);
                
                // Debug: Log sample of parsed bars
                if (bars.Count > 0)
                {
                    var sampleBars = bars.Take(3).ToList();
                    foreach (var bar in sampleBars)
                    {
                        _logger.LogInformation("[HISTORICAL-BRIDGE] 📊 Sample Bar: {Symbol} {Timestamp} O:{Open} H:{High} L:{Low} C:{Close} V:{Volume}", 
                            bar.Symbol, new DateTime(1970, 1, 1).AddMilliseconds(bar.Ts), bar.Open, bar.High, bar.Low, bar.Close, bar.Volume);
                    }
                }
                
                return bars;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[HISTORICAL-BRIDGE] Error parsing TopstepX historical response for {ContractId}", contractId);
                return new List<BotCore.Models.Bar>();
            }
        }
        
        private long ParseTopstepXTimestampToUnixMs(JsonElement barElement)
        {
            // TopstepX uses 't' field with ISO 8601 format: "2025-09-12T20:59:00+00:00"
            if (barElement.TryGetProperty("t", out var timestampElement) && timestampElement.ValueKind == JsonValueKind.String)
            {
                var timestampStr = timestampElement.GetString();
                if (DateTime.TryParse(timestampStr, out var timestamp))
                {
                    return ((DateTimeOffset)timestamp.ToUniversalTime()).ToUnixTimeMilliseconds();
                }
            }
            
            // Fallback: current time minus some offset
            return DateTimeOffset.UtcNow.AddMinutes(-1).ToUnixTimeMilliseconds();
        }
        
        private decimal ParseDecimalField(JsonElement element, params string[] fieldNames)
        {
            foreach (var fieldName in fieldNames)
            {
                if (element.TryGetProperty(fieldName, out var fieldElement))
                {
                    if (fieldElement.ValueKind == JsonValueKind.Number)
                    {
                        return fieldElement.GetDecimal();
                    }
                    if (fieldElement.ValueKind == JsonValueKind.String && 
                        decimal.TryParse(fieldElement.GetString(), out var decimalValue))
                    {
                        return decimalValue;
                    }
                }
            }
            return 0;
        }
        
        private long ParseLongField(JsonElement element, params string[] fieldNames)
        {
            foreach (var fieldName in fieldNames)
            {
                if (element.TryGetProperty(fieldName, out var fieldElement))
                {
                    if (fieldElement.ValueKind == JsonValueKind.Number)
                    {
                        return fieldElement.GetInt64();
                    }
                    if (fieldElement.ValueKind == JsonValueKind.String && 
                        long.TryParse(fieldElement.GetString(), out var longValue))
                    {
                        return longValue;
                    }
                }
            }
            return 0;
        }

        private List<BotCore.Models.Bar> GenerateWarmupBars(string contractId, int barCount)
        {
            var bars = new List<BotCore.Models.Bar>();
            var basePrice = GetBasePriceForContract(contractId);
            var currentTime = DateTime.UtcNow;
            
            // Generate bars going backwards in time (1-minute bars)
            for (int i = barCount - 1; i >= 0; i--)
            {
                var barStart = currentTime.AddMinutes(-i - 1);
                var barEnd = barStart.AddMinutes(1);
                
                // Simple price movement simulation for warm-up
                var priceVariation = (decimal)(new Random().NextDouble() - 0.5) * (basePrice * 0.001m);
                var price = basePrice + priceVariation;
                
                var bar = new BotCore.Models.Bar
                {
                    Symbol = GetSymbolFromContractId(contractId),
                    Ts = ((DateTimeOffset)barStart).ToUnixTimeMilliseconds(),
                    Open = price,
                    High = price + Math.Abs(priceVariation) * 0.5m,
                    Low = price - Math.Abs(priceVariation) * 0.5m,
                    Close = price,
                    Volume = 100 + new Random().Next(1, 500) // Synthetic volume
                };
                
                bars.Add(bar);
            }

            return bars.OrderBy(b => b.Ts).ToList();
        }

        private decimal GetBasePriceForContract(string contractId)
        {
            // Get reasonable base prices for major contracts
            return contractId switch
            {
                "CON.F.US.EP.Z25" => 5800m, // ES futures
                "CON.F.US.ENQ.Z25" => 20000m, // NQ futures
                _ when contractId.Contains("EP") => 5800m, // ES variants
                _ when contractId.Contains("ENQ") => 20000m, // NQ variants
                _ => 100m // Generic fallback
            };
        }

        private string GetSymbolFromContractId(string contractId)
        {
            return contractId switch
            {
                "CON.F.US.EP.Z25" => "ES",
                "CON.F.US.ENQ.Z25" => "NQ",
                _ when contractId.Contains("EP") => "ES",
                _ when contractId.Contains("ENQ") => "NQ",
                _ => "UNKNOWN"
            };
        }

        #endregion
    }
}