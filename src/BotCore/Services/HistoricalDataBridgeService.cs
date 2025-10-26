using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using BotCore.Market;
using BotCore.Models;
using System.Text.Json;
using System.Net.Http;
using System.Net.Http.Json;
using System.IO;
using System.Diagnostics;
using System.Globalization;
using Microsoft.Extensions.Hosting;
using TradingBot.Abstractions;

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
        bool IsSeeded { get; }
        int TotalBarsLoaded { get; }
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

    public class HistoricalDataBridgeService : BackgroundService, IHistoricalDataBridgeService
    {
        // Base price constants for major futures contracts
        private const decimal EsFuturesBasePrice = 5800m;
        private const decimal NqFuturesBasePrice = 20000m;
        private const decimal GenericContractFallbackPrice = 100m;

        // TopstepX History API endpoint
        private const string TopstepXHistoryApiUrl = "https://api.topstepx.com/api/History/retrieveBars";

        private readonly ILogger<HistoricalDataBridgeService> _logger;
        private readonly TradingReadinessConfiguration _config;
        private readonly HttpClient _httpClient;
        private readonly IHistoricalBarConsumer? _barConsumer;
        private readonly ITopstepXAdapterService? _topstepXAdapter;

        private bool _isSeeded = false;
        private int _totalBarsLoaded = 0;
        private readonly object _seedLock = new object();
        private string[]? _contractsToSeed;

        public bool IsSeeded
        {
            get { lock (_seedLock) { return _isSeeded; } }
            private set { lock (_seedLock) { _isSeeded = value; } }
        }

        public int TotalBarsLoaded
        {
            get { lock (_seedLock) { return _totalBarsLoaded; } }
            private set { lock (_seedLock) { _totalBarsLoaded = value; } }
        }

        public HistoricalDataBridgeService(
            ILogger<HistoricalDataBridgeService> logger,
            IOptions<TradingReadinessConfiguration> config,
            HttpClient httpClient,
            ITopstepXAdapterService? topstepXAdapter = null,
            IHistoricalBarConsumer? barConsumer = null)
        {
            _logger = logger;
            ArgumentNullException.ThrowIfNull(config);
            _config = config.Value;
            _httpClient = httpClient;
            _topstepXAdapter = topstepXAdapter;
            _barConsumer = barConsumer;
        }

        protected override async Task ExecuteAsync(CancellationToken stoppingToken)
        {
            // CRITICAL: Lab Mode uses pre-loaded JSON files, NOT live API connections
            // Skip background seeding entirely when in Lab Mode to maintain complete segregation
            var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
            var isLabMode = labMode == "1" || labMode?.ToLowerInvariant() == "true";

            if (isLabMode)
            {
                _logger.LogInformation("[HISTORICAL-BRIDGE] Lab Mode detected - skipping API-based historical data seeding");
                _logger.LogInformation("[HISTORICAL-BRIDGE] Lab Mode uses pre-loaded JSON files for complete API segregation");
                // FIXED: Wait indefinitely instead of returning to prevent shutdown signal
                await Task.Delay(Timeout.Infinite, stoppingToken).ConfigureAwait(false);
                return;
            }

            // Continuous background retry for historical data loading (Terminal Mode only)
            _logger.LogInformation("[HISTORICAL-BRIDGE] Background retry service started");

            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    if (!IsSeeded && _contractsToSeed != null)
                    {
                        _logger.LogInformation("[HISTORICAL-BRIDGE] ⏳ Retrying historical data load (not yet seeded)...");
                        var success = await SeedTradingSystemAsync(_contractsToSeed).ConfigureAwait(false);

                        if (success)
                        {
                            _logger.LogInformation("[HISTORICAL-BRIDGE] ✅ Historical data successfully loaded on retry!");
                        }
                        else
                        {
                            _logger.LogWarning("[HISTORICAL-BRIDGE] ⚠️ Retry failed, will try again in 30 seconds");
                        }
                    }

                    await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    // Normal shutdown
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[HISTORICAL-BRIDGE] Error in background retry service");
                    await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken).ConfigureAwait(false);
                }
            }

            _logger.LogInformation("[HISTORICAL-BRIDGE] Background retry service stopped");
        }

        /// <summary>
        /// Seed the trading system with historical data for fast warm-up
        /// </summary>
        public async Task<bool> SeedTradingSystemAsync(string[] contractIds)
        {
            ArgumentNullException.ThrowIfNull(contractIds);

            // CRITICAL: Lab Mode uses pre-loaded JSON files, NOT live API connections
            var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
            var isLabMode = labMode == "1" || labMode?.ToLowerInvariant() == "true";

            if (isLabMode)
            {
                _logger.LogInformation("[HISTORICAL-BRIDGE] Lab Mode - skipping API seeding (uses pre-loaded JSON data)");
                return false;
            }

            // Store contracts for background retry if initial attempt fails
            if (_contractsToSeed == null)
            {
                _contractsToSeed = contractIds;
            }

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
                    var historicalBars = await GetRecentHistoricalBarsAsync(contractId, Math.Max(_config.MinSeededBars + 2, 200)).ConfigureAwait(false);

                    if (historicalBars.Count > 0)
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
                catch (InvalidOperationException ex)
                {
                    _logger.LogError(ex, "[HISTORICAL-BRIDGE] ❌ Invalid operation seeding {ContractId}", contractId);
                }
                catch (TimeoutException ex)
                {
                    _logger.LogError(ex, "[HISTORICAL-BRIDGE] ❌ Timeout seeding {ContractId}", contractId);
                }
                catch (HttpRequestException ex)
                {
                    _logger.LogError(ex, "[HISTORICAL-BRIDGE] ❌ HTTP error seeding {ContractId}", contractId);
                }
            }

            var success = successCount > 0;

            // Update seeding status
            if (success)
            {
                IsSeeded = true;
                TotalBarsLoaded = totalSeeded;
                _logger.LogInformation("[HISTORICAL-BRIDGE] ✅ SEEDED! {SuccessCount}/{TotalCount} contracts, {TotalBars} bars loaded",
                    successCount, contractIds.Length, totalSeeded);
            }
            else
            {
                _logger.LogWarning("[HISTORICAL-BRIDGE] ⚠️ Seeding FAILED: 0/{TotalCount} contracts loaded - will retry in background",
                    contractIds.Length);
            }

            return success;
        }

        /// <summary>
        /// Get recent historical bars from SDK adapter or fallback sources
        /// </summary>
        public async Task<List<BotCore.Models.Bar>> GetRecentHistoricalBarsAsync(string contractId, int barCount = 20)
        {
            // CRITICAL: Lab Mode uses pre-loaded JSON files, NOT live API connections
            var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
            var isLabMode = labMode == "1" || labMode?.ToLowerInvariant() == "true";

            if (isLabMode)
            {
                _logger.LogDebug("[HISTORICAL-BRIDGE] Lab Mode - returning empty bars (uses pre-loaded JSON data, not API)");
                return new List<BotCore.Models.Bar>();
            }

            try
            {
                // PRIMARY: Try TopstepX adapter service for historical data (uses stdin/stdout commands)
                var sdkAdapterBars = await TryGetSdkAdapterBarsAsync(contractId, barCount, CancellationToken.None).ConfigureAwait(false);
                if (sdkAdapterBars.Count > 0)
                {
                    _logger.LogDebug("[HISTORICAL-BRIDGE] ✅ Retrieved {BarCount} bars from TopstepX adapter for {ContractId}",
                        sdkAdapterBars.Count, contractId);
                    return sdkAdapterBars;
                }

                // FALLBACK 1: Try TopstepX historical API
                var topstepXBars = await TryGetTopstepXBarsAsync(contractId, barCount).ConfigureAwait(false);
                if (topstepXBars.Count > 0)
                {
                    _logger.LogDebug("[HISTORICAL-BRIDGE] Retrieved {BarCount} bars from TopstepX API for {ContractId}",
                        topstepXBars.Count, contractId);
                    return topstepXBars;
                }

                // FALLBACK 2: Use correlation manager's data sources
                var correlationBars = await TryGetCorrelationManagerBarsAsync(contractId, barCount).ConfigureAwait(false);
                if (correlationBars.Count > 0)
                {
                    _logger.LogDebug("[HISTORICAL-BRIDGE] Retrieved {BarCount} bars from correlation manager for {ContractId}",
                        correlationBars.Count, contractId);
                    return correlationBars;
                }

                // NO SYNTHETIC DATA - Log warning but return empty list (don't throw exception)
                _logger.LogWarning("[HISTORICAL-BRIDGE] NO real historical data available for {ContractId}. Bot will wait for live data.", contractId);
                return new List<BotCore.Models.Bar>();
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "[HISTORICAL-BRIDGE] Invalid operation getting historical bars for {ContractId}", contractId);
                return new List<BotCore.Models.Bar>();
            }
            catch (TimeoutException ex)
            {
                _logger.LogError(ex, "[HISTORICAL-BRIDGE] Timeout getting historical bars for {ContractId}", contractId);
                return new List<BotCore.Models.Bar>();
            }
            catch (HttpRequestException ex)
            {
                _logger.LogError(ex, "[HISTORICAL-BRIDGE] HTTP error getting historical bars for {ContractId}", contractId);
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
                var bars = await GetRecentHistoricalBarsAsync(contractId, 5).ConfigureAwait(false);
                if (bars.Count == 0) return false;

                var mostRecentBar = bars.OrderByDescending(b => b.Ts).First();
                var dataAge = DateTime.UtcNow - DateTime.UnixEpoch.AddMilliseconds(mostRecentBar.Ts);

                var isValid = dataAge.TotalHours <= _config.MaxHistoricalDataAgeHours;

                _logger.LogDebug("[HISTORICAL-BRIDGE] Data validation for {ContractId}: Age={Age:F1}h, Valid={IsValid}",
                    contractId, dataAge.TotalHours, isValid);

                return isValid;
            }
            catch (InvalidOperationException ex)
            {
                _logger.LogError(ex, "[HISTORICAL-BRIDGE] Invalid operation during validation for {ContractId}", contractId);
                return false;
            }
            catch (ArgumentException ex)
            {
                _logger.LogError(ex, "[HISTORICAL-BRIDGE] Invalid argument during validation for {ContractId}", contractId);
                return false;
            }
        }

        #region Private Helper Methods

        /// <summary>
        /// Try to get historical bars using the TopstepX adapter service (preferred method)
        /// Uses stdin/stdout command protocol, not HTTP
        /// </summary>
        private async Task<List<BotCore.Models.Bar>> TryGetSdkAdapterBarsAsync(string contractId, int barCount, CancellationToken cancellationToken = default)
        {
            try
            {
                // Check if TopstepX adapter is available
                if (_topstepXAdapter == null)
                {
                    _logger.LogWarning("[HISTORICAL-BRIDGE] TopstepX adapter not available, cannot fetch historical data");
                    return new List<BotCore.Models.Bar>();
                }

                _logger.LogInformation("[HISTORICAL-BRIDGE] ✅ Using TopstepX adapter service for historical data");

                // Map contract ID to symbol (CON.F.US.EP.Z25 -> ES, CON.F.US.ENQ.Z25 -> NQ)
                var symbol = contractId.Contains("EP") ? "ES" : contractId.Contains("ENQ") ? "NQ" : contractId;

                // Calculate how many days of data we need to get the requested bar count
                // Assuming 5-minute bars, ~78 bars per trading session (6.5 hours)
                // Request more days to ensure we have enough bars
                var daysNeeded = Math.Max(1, (barCount / 78) + 2); // Add buffer

                _logger.LogInformation("[HISTORICAL-BRIDGE] Fetching {Days} days of historical data for {Symbol} via TopstepX adapter (target: {BarCount} bars)",
                    daysNeeded, symbol, barCount);

                // Call TopstepX adapter service using stdin/stdout commands
                var historicalBars = await _topstepXAdapter.GetHistoricalBarsAsync(
                    symbol,
                    days: daysNeeded,
                    intervalMinutes: 5,
                    cancellationToken
                ).ConfigureAwait(false);

                if (historicalBars == null || historicalBars.Count == 0)
                {
                    _logger.LogWarning("[HISTORICAL-BRIDGE] TopstepX adapter returned no historical bars for {Symbol}", symbol);
                    return new List<BotCore.Models.Bar>();
                }

                _logger.LogInformation("[HISTORICAL-BRIDGE] TopstepX adapter returned {Count} bars, converting to BotCore.Models.Bar format", historicalBars.Count);

                // Convert HistoricalBar to BotCore.Models.Bar
                var bars = new List<BotCore.Models.Bar>();

                foreach (var histBar in historicalBars.TakeLast(barCount)) // Take only the requested number of most recent bars
                {
                    try
                    {
                        var botBar = new BotCore.Models.Bar
                        {
                            Symbol = contractId, // Use original contract ID, not simplified symbol
                            Open = histBar.Open,
                            High = histBar.High,
                            Low = histBar.Low,
                            Close = histBar.Close,
                            Volume = (int)histBar.Volume, // Convert long to int
                            Ts = ((DateTimeOffset)histBar.Timestamp).ToUnixTimeMilliseconds(),
                            Start = histBar.Timestamp
                        };
                        bars.Add(botBar);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "[HISTORICAL-BRIDGE] Failed to convert historical bar at {Timestamp}", histBar.Timestamp);
                    }
                }

                _logger.LogInformation("[HISTORICAL-BRIDGE] ✅ Retrieved {Count} bars via TopstepX adapter for {ContractId}", bars.Count, contractId);
                return bars;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[HISTORICAL-BRIDGE] TopstepX adapter fetch failed for {ContractId}: {Error}", contractId, ex.Message);
                return new List<BotCore.Models.Bar>();
            }
        }

        /// <summary>
        /// Try to get historical bars from TopstepX API (fallback method)
        /// </summary>

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
                    startTime = startTime.ToString("yyyy-MM-ddTHH:mm:ssZ", CultureInfo.InvariantCulture),
                    endTime = endTime.ToString("yyyy-MM-ddTHH:mm:ssZ", CultureInfo.InvariantCulture),
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
                var response = await _httpClient.PostAsJsonAsync(TopstepXHistoryApiUrl, requestBody).ConfigureAwait(false);

                if (!response.IsSuccessStatusCode)
                {
                    var errorContent = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
                    _logger.LogWarning("[HISTORICAL-BRIDGE] TopstepX History API failed: {StatusCode} - {Error}",
                        response.StatusCode, errorContent);
                    return new List<BotCore.Models.Bar>();
                }

                var responseContent = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
                _logger.LogDebug("[HISTORICAL-BRIDGE] TopstepX History API response length: {Length} characters", responseContent.Length);

                // Debug: Log first 500 characters of response
                var previewContent = responseContent.Length > 500 ? string.Concat(responseContent.AsSpan(0, 500), "...") : responseContent;
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
                await Task.CompletedTask.ConfigureAwait(false);
                return new List<BotCore.Models.Bar>();
            }
            catch (Exception ex)
            {
                _logger.LogDebug(ex, "[HISTORICAL-BRIDGE] Correlation manager bars failed for {ContractId}", contractId);
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
                            bar.Symbol, DateTime.UnixEpoch.AddMilliseconds(bar.Ts), bar.Open, bar.High, bar.Low, bar.Close, bar.Volume);
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

        private static long ParseTopstepXTimestampToUnixMs(JsonElement barElement)
        {
            // TopstepX uses 't' field with ISO 8601 format: "2025-09-12T20:59:00+00:00"
            if (barElement.TryGetProperty("t", out var timestampElement) && timestampElement.ValueKind == JsonValueKind.String)
            {
                var timestampStr = timestampElement.GetString();
                if (DateTime.TryParse(timestampStr, System.Globalization.CultureInfo.InvariantCulture, System.Globalization.DateTimeStyles.None, out var timestamp))
                {
                    return ((DateTimeOffset)timestamp.ToUniversalTime()).ToUnixTimeMilliseconds();
                }
            }

            // Fallback: current time minus some offset
            return DateTimeOffset.UtcNow.AddMinutes(-1).ToUnixTimeMilliseconds();
        }

        private static decimal ParseDecimalField(JsonElement element, params string[] fieldNames)
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

        private static long ParseLongField(JsonElement element, params string[] fieldNames)
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

        private static decimal GetBasePriceForContract(string contractId)
        {
            // Get reasonable base prices for major contracts
            return contractId switch
            {
                "CON.F.US.EP.Z25" => EsFuturesBasePrice, // ES futures
                "CON.F.US.ENQ.Z25" => NqFuturesBasePrice, // NQ futures
                _ when contractId.Contains("EP", StringComparison.Ordinal) => EsFuturesBasePrice, // ES variants
                _ when contractId.Contains("ENQ", StringComparison.Ordinal) => NqFuturesBasePrice, // NQ variants
                _ => GenericContractFallbackPrice // Generic fallback
            };
        }

        private static string GetSymbolFromContractId(string contractId)
        {
            return contractId switch
            {
                "CON.F.US.EP.Z25" => "ES",
                "CON.F.US.ENQ.Z25" => "NQ",
                _ when contractId.Contains("EP", StringComparison.Ordinal) => "ES",
                _ when contractId.Contains("ENQ", StringComparison.Ordinal) => "NQ",
                _ => "UNKNOWN"
            };
        }

        #endregion
    }
}
