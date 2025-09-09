using Microsoft.Extensions.Logging;
using System.Collections.Concurrent;
using System.Text.Json;

namespace TradingBot.RLAgent;

/// <summary>
/// Feature Engineering system with configurable lookbacks, microstructure features, and null/NaN policy
/// Implements requirement 1.3: Replace hardcoded lookbacks, add microstructure features, null policy, feature importance
/// </summary>
public class FeatureEngineering : IDisposable
{
    private readonly ILogger<FeatureEngineering> _logger;
    private readonly FeatureConfig _config;
    private readonly ConcurrentDictionary<string, FeatureState> _featureStates = new();
    private readonly ConcurrentDictionary<string, CircularBuffer<MarketData>> _marketDataBuffers = new();
    private readonly ConcurrentDictionary<string, FeatureImportanceTracker> _importanceTrackers = new();
    private readonly Timer _dailyReportTimer;
    private bool _disposed = false;

    public FeatureEngineering(
        ILogger<FeatureEngineering> logger,
        FeatureConfig config)
    {
        _logger = logger;
        _config = config;

        // Daily feature importance reporting timer
        _dailyReportTimer = new Timer(GenerateDailyFeatureReport, null, 
            TimeSpan.FromDays(1), TimeSpan.FromDays(1));

        _logger.LogInformation("[FEATURE_ENG] Initialized with {ProfileCount} regime profiles", 
            _config.RegimeProfiles.Count);
    }

    /// <summary>
    /// Generate feature vector for ML inference
    /// </summary>
    public async Task<FeatureVector> GenerateFeaturesAsync(
        string symbol,
        string strategy,
        RegimeType regime,
        MarketData currentData,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var featureKey = GetFeatureKey(symbol, strategy, regime);
            var profile = GetRegimeProfile(regime);
            
            // Update market data buffer
            UpdateMarketDataBuffer(featureKey, currentData);
            
            // Get or create feature state
            var state = _featureStates.GetOrAdd(featureKey, _ => new FeatureState());
            
            // Generate feature vector
            var features = new List<double>();
            var featureNames = new List<string>();
            
            // Price features
            await AddPriceFeatures(features, featureNames, featureKey, currentData, profile);
            
            // Volume features
            await AddVolumeFeatures(features, featureNames, featureKey, currentData, profile);
            
            // Technical indicator features
            await AddTechnicalFeatures(features, featureNames, featureKey, currentData, profile);
            
            // Microstructure features (requirement: bid-ask spread, z-score, order-flow, tick-run)
            await AddMicrostructureFeatures(features, featureNames, featureKey, currentData, profile);
            
            // Regime features
            await AddRegimeFeatures(features, featureNames, regime, currentData);
            
            // Time-based features
            await AddTimeFeatures(features, featureNames, currentData);
            
            // Apply null/NaN policy
            var cleanedFeatures = ApplyNullNaNPolicy(features, featureNames, state);
            
            var featureVector = new FeatureVector
            {
                Symbol = symbol,
                Strategy = strategy,
                Regime = regime,
                Timestamp = currentData.Timestamp,
                Features = cleanedFeatures.ToArray(),
                FeatureNames = featureNames.ToArray(),
                FeatureCount = cleanedFeatures.Count,
                HasMissingValues = features.Count != cleanedFeatures.Count
            };

            // Update feature state
            state.LastUpdate = DateTime.UtcNow;
            state.FeatureCount = cleanedFeatures.Count;
            
            _logger.LogDebug("[FEATURE_ENG] Generated {FeatureCount} features for {Symbol} {Strategy} {Regime}", 
                cleanedFeatures.Count, symbol, strategy, regime);

            return featureVector;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURE_ENG] Error generating features for {Symbol} {Strategy} {Regime}", 
                symbol, strategy, regime);
            
            // Return empty feature vector as fallback
            return new FeatureVector
            {
                Symbol = symbol,
                Strategy = strategy,
                Regime = regime,
                Timestamp = currentData.Timestamp,
                Features = Array.Empty<double>(),
                FeatureNames = Array.Empty<string>(),
                FeatureCount = 0,
                HasMissingValues = true
            };
        }
    }

    /// <summary>
    /// Update feature importance tracking
    /// </summary>
    public void UpdateFeatureImportance(
        string symbol,
        string strategy,
        RegimeType regime,
        string[] featureNames,
        double[] importanceScores)
    {
        try
        {
            var featureKey = GetFeatureKey(symbol, strategy, regime);
            var tracker = _importanceTrackers.GetOrAdd(featureKey, _ => new FeatureImportanceTracker());
            
            tracker.UpdateImportance(featureNames, importanceScores);
            
            _logger.LogDebug("[FEATURE_ENG] Updated importance for {FeatureCount} features: {FeatureKey}", 
                featureNames.Length, featureKey);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[FEATURE_ENG] Error updating feature importance for {Symbol} {Strategy} {Regime}", 
                symbol, strategy, regime);
        }
    }

    /// <summary>
    /// Add price-based features with configurable lookbacks
    /// </summary>
    private async Task AddPriceFeatures(
        List<double> features,
        List<string> featureNames,
        string featureKey,
        MarketData currentData,
        RegimeProfile profile)
    {
        var buffer = GetMarketDataBuffer(featureKey);
        
        if (buffer.Count < 2)
        {
            // Not enough data, use sentinel values
            features.AddRange(new double[] { 0.0, 0.0, 0.0, 0.0, 0.0 });
            featureNames.AddRange(new[] { "price_return_1", "price_return_5", "price_return_20", "price_volatility", "price_trend" });
            return;
        }

        // Price returns with configurable lookbacks
        var returns1 = CalculateReturn(currentData.Close, buffer.GetFromEnd(1)?.Close ?? currentData.Close);
        var returns5 = buffer.Count >= 5 ? CalculateReturn(currentData.Close, buffer.GetFromEnd(5)?.Close ?? currentData.Close) : 0.0;
        var returns20 = buffer.Count >= 20 ? CalculateReturn(currentData.Close, buffer.GetFromEnd(20)?.Close ?? currentData.Close) : 0.0;

        // Price volatility (configurable window)
        var volatilityWindow = Math.Min(profile.VolatilityLookback, buffer.Count);
        var recentPrices = buffer.GetLast(volatilityWindow).Select(d => d.Close).ToArray();
        var volatility = CalculateVolatility(recentPrices);

        // Price trend (using SMA difference)
        var trendWindow = Math.Min(profile.TrendLookback, buffer.Count);
        var trend = CalculateTrend(buffer.GetLast(trendWindow).Select(d => d.Close).ToArray());

        features.AddRange(new[] { returns1, returns5, returns20, volatility, trend });
        featureNames.AddRange(new[] { "price_return_1", "price_return_5", "price_return_20", "price_volatility", "price_trend" });
    }

    /// <summary>
    /// Add volume-based features
    /// </summary>
    private async Task AddVolumeFeatures(
        List<double> features,
        List<string> featureNames,
        string featureKey,
        MarketData currentData,
        RegimeProfile profile)
    {
        var buffer = GetMarketDataBuffer(featureKey);
        
        if (buffer.Count < 2)
        {
            features.AddRange(new double[] { 0.0, 0.0, 0.0 });
            featureNames.AddRange(new[] { "volume_ratio", "volume_trend", "volume_volatility" });
            return;
        }

        // Volume ratio (current vs average)
        var volumeWindow = Math.Min(profile.VolumeLookback, buffer.Count);
        var recentVolumes = buffer.GetLast(volumeWindow).Select(d => d.Volume).ToArray();
        var avgVolume = recentVolumes.Length > 0 ? recentVolumes.Average() : 1.0;
        var volumeRatio = avgVolume > 0 ? currentData.Volume / avgVolume : 1.0;

        // Volume trend
        var volumeTrend = CalculateTrend(recentVolumes);

        // Volume volatility
        var volumeVolatility = CalculateVolatility(recentVolumes);

        features.AddRange(new[] { volumeRatio, volumeTrend, volumeVolatility });
        featureNames.AddRange(new[] { "volume_ratio", "volume_trend", "volume_volatility" });
    }

    /// <summary>
    /// Add technical indicator features
    /// </summary>
    private async Task AddTechnicalFeatures(
        List<double> features,
        List<string> featureNames,
        string featureKey,
        MarketData currentData,
        RegimeProfile profile)
    {
        var buffer = GetMarketDataBuffer(featureKey);

        // RSI (Relative Strength Index)
        var rsiWindow = Math.Min(profile.RsiLookback, buffer.Count);
        var rsi = buffer.Count >= rsiWindow ? CalculateRSI(buffer.GetLast(rsiWindow), currentData) : 50.0;

        // Bollinger Bands position
        var bbWindow = Math.Min(profile.BollingerLookback, buffer.Count);
        var bollingerPosition = buffer.Count >= bbWindow ? CalculateBollingerPosition(buffer.GetLast(bbWindow), currentData) : 0.5;

        // ATR (Average True Range)
        var atrWindow = Math.Min(profile.AtrLookback, buffer.Count);
        var atr = buffer.Count >= atrWindow ? CalculateATR(buffer.GetLast(atrWindow), currentData) : 0.0;

        // MACD
        var macdResult = buffer.Count >= 26 ? CalculateMACD(buffer.GetLast(26), currentData) : (0.0, 0.0);

        features.AddRange(new[] { rsi / 100.0, bollingerPosition, atr, macdResult.macd, macdResult.signal });
        featureNames.AddRange(new[] { "rsi_normalized", "bollinger_position", "atr", "macd", "macd_signal" });
    }

    /// <summary>
    /// Add microstructure features (requirement: bid-ask spread, z-score, order-flow, tick-run)
    /// </summary>
    private async Task AddMicrostructureFeatures(
        List<double> features,
        List<string> featureNames,
        string featureKey,
        MarketData currentData,
        RegimeProfile profile)
    {
        var buffer = GetMarketDataBuffer(featureKey);

        // Bid-ask spread
        var spread = currentData.Ask - currentData.Bid;
        var midPrice = (currentData.Bid + currentData.Ask) / 2.0;
        var spreadBps = midPrice > 0 ? (spread / midPrice) * 10000.0 : 0.0;

        // Spread z-score (spread relative to recent history)
        var spreadWindow = Math.Min(profile.MicrostructureLookback, buffer.Count);
        var recentSpreads = buffer.GetLast(spreadWindow)
            .Select(d => d.Ask - d.Bid)
            .Where(s => s > 0)
            .ToArray();
        
        var spreadZScore = 0.0;
        if (recentSpreads.Length > 5)
        {
            var avgSpread = recentSpreads.Average();
            var stdSpread = Math.Sqrt(recentSpreads.Select(s => Math.Pow(s - avgSpread, 2)).Average());
            spreadZScore = stdSpread > 0 ? (spread - avgSpread) / stdSpread : 0.0;
        }

        // Order flow imbalance (approximated using tick direction)
        var tickDirection = GetTickDirection(buffer, currentData);
        var imbalanceWindow = Math.Min(profile.OrderFlowLookback, buffer.Count);
        var orderFlowImbalance = CalculateOrderFlowImbalance(buffer.GetLast(imbalanceWindow), tickDirection);

        // Tick run (consecutive ticks in same direction)
        var tickRun = CalculateTickRun(buffer, currentData);

        // Last trade direction EMA
        var tradeDirectionEma = CalculateTradeDirectionEMA(buffer, currentData, profile.TradeDirectionDecay);

        features.AddRange(new[] { spreadBps, spreadZScore, orderFlowImbalance, tickRun, tradeDirectionEma });
        featureNames.AddRange(new[] { "spread_bps", "spread_zscore", "order_flow_imbalance", "tick_run", "trade_direction_ema" });
    }

    /// <summary>
    /// Add regime-based features
    /// </summary>
    private async Task AddRegimeFeatures(
        List<double> features,
        List<string> featureNames,
        RegimeType regime,
        MarketData currentData)
    {
        // One-hot encoding for regime
        features.AddRange(new[]
        {
            regime == RegimeType.Range ? 1.0 : 0.0,
            regime == RegimeType.Trend ? 1.0 : 0.0,
            regime == RegimeType.Volatility ? 1.0 : 0.0,
            regime == RegimeType.LowVol ? 1.0 : 0.0,
            regime == RegimeType.HighVol ? 1.0 : 0.0
        });

        featureNames.AddRange(new[] { "regime_range", "regime_trend", "regime_volatility", "regime_lowvol", "regime_highvol" });
    }

    /// <summary>
    /// Add time-based features
    /// </summary>
    private async Task AddTimeFeatures(
        List<double> features,
        List<string> featureNames,
        MarketData currentData)
    {
        var timestamp = currentData.Timestamp;
        
        // Time of day (normalized)
        var timeOfDay = timestamp.TimeOfDay.TotalHours / 24.0;
        
        // Day of week (one-hot)
        var dayOfWeek = (int)timestamp.DayOfWeek;
        var isMonday = dayOfWeek == 1 ? 1.0 : 0.0;
        var isFriday = dayOfWeek == 5 ? 1.0 : 0.0;
        
        // Market session (US market hours approximation)
        var hour = timestamp.Hour;
        var isMarketHours = (hour >= 9 && hour < 16) ? 1.0 : 0.0;
        var isOpeningHour = (hour == 9) ? 1.0 : 0.0;
        var isClosingHour = (hour == 15) ? 1.0 : 0.0;

        features.AddRange(new[] { timeOfDay, isMonday, isFriday, isMarketHours, isOpeningHour, isClosingHour });
        featureNames.AddRange(new[] { "time_of_day", "is_monday", "is_friday", "is_market_hours", "is_opening_hour", "is_closing_hour" });
    }

    /// <summary>
    /// Apply null/NaN policy: forward-fill bounded, default sentinels, skip-logic for missing book
    /// </summary>
    private List<double> ApplyNullNaNPolicy(
        List<double> features,
        List<string> featureNames,
        FeatureState state)
    {
        var cleanedFeatures = new List<double>();
        
        for (int i = 0; i < features.Count; i++)
        {
            var feature = features[i];
            var featureName = i < featureNames.Count ? featureNames[i] : $"feature_{i}";
            
            if (double.IsNaN(feature) || double.IsInfinity(feature))
            {
                // Forward-fill from previous value if available
                if (state.LastValidValues.TryGetValue(featureName, out var lastValid))
                {
                    feature = lastValid;
                    _logger.LogDebug("[FEATURE_ENG] Forward-filled NaN for feature: {FeatureName}", featureName);
                }
                else
                {
                    // Use default sentinel value
                    feature = GetDefaultSentinelValue(featureName);
                    _logger.LogDebug("[FEATURE_ENG] Applied sentinel value for feature: {FeatureName}", featureName);
                }
            }
            
            // Bound extreme values
            feature = BoundFeatureValue(feature, featureName);
            
            // Store as last valid value
            state.LastValidValues[featureName] = feature;
            
            cleanedFeatures.Add(feature);
        }
        
        return cleanedFeatures;
    }

    /// <summary>
    /// Get default sentinel value for missing features
    /// </summary>
    private double GetDefaultSentinelValue(string featureName)
    {
        return featureName.ToLower() switch
        {
            var name when name.Contains("return") => 0.0,
            var name when name.Contains("ratio") => 1.0,
            var name when name.Contains("volatility") => 0.15,
            var name when name.Contains("rsi") => 0.5,
            var name when name.Contains("bollinger") => 0.5,
            var name when name.Contains("regime") => 0.0,
            var name when name.Contains("spread") => 1.0,
            _ => 0.0
        };
    }

    /// <summary>
    /// Bound feature values to prevent extreme outliers
    /// </summary>
    private double BoundFeatureValue(double value, string featureName)
    {
        var bounds = featureName.ToLower() switch
        {
            var name when name.Contains("return") => (-0.1, 0.1),    // ±10% max return
            var name when name.Contains("ratio") => (0.01, 100.0),   // Volume ratio bounds
            var name when name.Contains("volatility") => (0.0, 2.0), // 0-200% volatility
            var name when name.Contains("zscore") => (-5.0, 5.0),    // Z-score bounds
            var name when name.Contains("spread_bps") => (0.0, 100.0), // 0-100 bps spread
            _ => (double.MinValue, double.MaxValue)
        };
        
        return Math.Max(bounds.Item1, Math.Min(bounds.Item2, value));
    }

    // Helper calculation methods
    private double CalculateReturn(double current, double previous)
    {
        return previous > 0 ? (current - previous) / previous : 0.0;
    }

    private double CalculateVolatility(double[] prices)
    {
        if (prices.Length < 2) return 0.0;
        
        var returns = new List<double>();
        for (int i = 1; i < prices.Length; i++)
        {
            returns.Add(CalculateReturn(prices[i], prices[i - 1]));
        }
        
        var mean = returns.Average();
        var variance = returns.Select(r => Math.Pow(r - mean, 2)).Average();
        return Math.Sqrt(variance);
    }

    private double CalculateTrend(double[] values)
    {
        if (values.Length < 2) return 0.0;
        
        // Simple linear trend calculation
        var n = values.Length;
        var x = Enumerable.Range(0, n).Select(i => (double)i).ToArray();
        var y = values;
        
        var sumX = x.Sum();
        var sumY = y.Sum();
        var sumXY = x.Zip(y, (xi, yi) => xi * yi).Sum();
        var sumX2 = x.Select(xi => xi * xi).Sum();
        
        var slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        return slope;
    }

    private double CalculateRSI(MarketData[] buffer, MarketData current)
    {
        var allData = buffer.Append(current).ToArray();
        if (allData.Length < 15) return 50.0; // Default neutral RSI
        
        var gains = new List<double>();
        var losses = new List<double>();
        
        for (int i = 1; i < allData.Length; i++)
        {
            var change = allData[i].Close - allData[i - 1].Close;
            if (change > 0)
            {
                gains.Add(change);
                losses.Add(0);
            }
            else
            {
                gains.Add(0);
                losses.Add(-change);
            }
        }
        
        var avgGain = gains.TakeLast(14).Average();
        var avgLoss = losses.TakeLast(14).Average();
        
        if (avgLoss == 0) return 100.0;
        
        var rs = avgGain / avgLoss;
        return 100.0 - (100.0 / (1.0 + rs));
    }

    private double CalculateBollingerPosition(MarketData[] buffer, MarketData current)
    {
        var prices = buffer.Select(d => d.Close).Append(current.Close).ToArray();
        if (prices.Length < 20) return 0.5; // Default middle position
        
        var sma = prices.TakeLast(20).Average();
        var variance = prices.TakeLast(20).Select(p => Math.Pow(p - sma, 2)).Average();
        var stdDev = Math.Sqrt(variance);
        
        var upperBand = sma + (2.0 * stdDev);
        var lowerBand = sma - (2.0 * stdDev);
        
        if (upperBand <= lowerBand) return 0.5;
        
        return (current.Close - lowerBand) / (upperBand - lowerBand);
    }

    private double CalculateATR(MarketData[] buffer, MarketData current)
    {
        var allData = buffer.Append(current).ToArray();
        if (allData.Length < 2) return 0.0;
        
        var trueRanges = new List<double>();
        for (int i = 1; i < allData.Length; i++)
        {
            var high = allData[i].High;
            var low = allData[i].Low;
            var prevClose = allData[i - 1].Close;
            
            var tr = Math.Max(high - low, Math.Max(Math.Abs(high - prevClose), Math.Abs(low - prevClose)));
            trueRanges.Add(tr);
        }
        
        return trueRanges.TakeLast(14).Average();
    }

    private (double macd, double signal) CalculateMACD(MarketData[] buffer, MarketData current)
    {
        var prices = buffer.Select(d => d.Close).Append(current.Close).ToArray();
        if (prices.Length < 26) return (0.0, 0.0);
        
        // Simplified MACD calculation
        var ema12 = CalculateEMA(prices, 12);
        var ema26 = CalculateEMA(prices, 26);
        var macd = ema12 - ema26;
        
        // Signal line (9-period EMA of MACD) - simplified
        var signal = macd * 0.1; // Simplified signal calculation
        
        return (macd, signal);
    }

    private double CalculateEMA(double[] prices, int period)
    {
        if (prices.Length < period) return prices.LastOrDefault();
        
        var multiplier = 2.0 / (period + 1);
        var ema = prices.Take(period).Average(); // Start with SMA
        
        for (int i = period; i < prices.Length; i++)
        {
            ema = (prices[i] * multiplier) + (ema * (1 - multiplier));
        }
        
        return ema;
    }

    private int GetTickDirection(CircularBuffer<MarketData> buffer, MarketData current)
    {
        if (buffer.Count == 0) return 0;
        
        var last = buffer.GetFromEnd(0);
        if (last == null) return 0;
        
        return current.Close > last.Close ? 1 : (current.Close < last.Close ? -1 : 0);
    }

    private double CalculateOrderFlowImbalance(MarketData[] buffer, int currentTickDirection)
    {
        if (buffer.Length < 2) return 0.0;
        
        var upTicks = 0;
        var downTicks = 0;
        
        for (int i = 1; i < buffer.Length; i++)
        {
            var direction = buffer[i].Close > buffer[i - 1].Close ? 1 : (buffer[i].Close < buffer[i - 1].Close ? -1 : 0);
            if (direction > 0) upTicks++;
            else if (direction < 0) downTicks++;
        }
        
        // Add current tick direction
        if (currentTickDirection > 0) upTicks++;
        else if (currentTickDirection < 0) downTicks++;
        
        var totalTicks = upTicks + downTicks;
        return totalTicks > 0 ? (upTicks - downTicks) / (double)totalTicks : 0.0;
    }

    private double CalculateTickRun(CircularBuffer<MarketData> buffer, MarketData current)
    {
        if (buffer.Count < 2) return 0.0;
        
        var currentDirection = GetTickDirection(buffer, current);
        if (currentDirection == 0) return 0.0;
        
        var run = 1;
        for (int i = 1; i < Math.Min(buffer.Count, 10); i++) // Check last 10 ticks
        {
            var prevData = buffer.GetFromEnd(i);
            var prevPrevData = buffer.GetFromEnd(i + 1);
            
            if (prevData == null || prevPrevData == null) break;
            
            var prevDirection = prevData.Close > prevPrevData.Close ? 1 : (prevData.Close < prevPrevData.Close ? -1 : 0);
            
            if (prevDirection == currentDirection)
                run++;
            else
                break;
        }
        
        return Math.Min(run, 10) / 10.0; // Normalize to 0-1
    }

    private double CalculateTradeDirectionEMA(CircularBuffer<MarketData> buffer, MarketData current, double decay)
    {
        var currentDirection = GetTickDirection(buffer, current);
        
        // Get previous EMA value from feature state (simplified)
        var state = _featureStates.Values.FirstOrDefault();
        var prevEma = state?.LastValidValues.GetValueOrDefault("trade_direction_ema", 0.0) ?? 0.0;
        
        // EMA calculation: EMA = (Current * Alpha) + (Previous * (1 - Alpha))
        var alpha = 1.0 - decay;
        return (currentDirection * alpha) + (prevEma * decay);
    }

    private string GetFeatureKey(string symbol, string strategy, RegimeType regime)
    {
        return $"{symbol}_{strategy}_{regime}";
    }

    private RegimeProfile GetRegimeProfile(RegimeType regime)
    {
        return _config.RegimeProfiles.GetValueOrDefault(regime, _config.DefaultProfile);
    }

    private CircularBuffer<MarketData> GetMarketDataBuffer(string featureKey)
    {
        return _marketDataBuffers.GetOrAdd(featureKey, _ => new CircularBuffer<MarketData>(_config.MaxBufferSize));
    }

    private void UpdateMarketDataBuffer(string featureKey, MarketData data)
    {
        var buffer = GetMarketDataBuffer(featureKey);
        buffer.Add(data);
    }

    /// <summary>
    /// Generate daily feature importance report
    /// </summary>
    private async void GenerateDailyFeatureReport(object? state)
    {
        try
        {
            _logger.LogInformation("[FEATURE_ENG] Generating daily feature importance report...");
            
            var report = new FeatureImportanceReport
            {
                GeneratedAt = DateTime.UtcNow,
                SymbolReports = new Dictionary<string, Dictionary<string, double>>()
            };
            
            foreach (var (featureKey, tracker) in _importanceTrackers)
            {
                var topFeatures = tracker.GetTopKFeatures(_config.TopKFeatures);
                report.SymbolReports[featureKey] = topFeatures;
                
                _logger.LogInformation("[FEATURE_ENG] Top features for {FeatureKey}: {TopFeatures}", 
                    featureKey, string.Join(", ", topFeatures.Take(5).Select(kv => $"{kv.Key}: {kv.Value:F3}")));
            }
            
            // Save report to file
            var reportJson = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
            var reportPath = Path.Combine("reports", $"feature_importance_{DateTime.UtcNow:yyyyMMdd}.json");
            Directory.CreateDirectory(Path.GetDirectoryName(reportPath)!);
            await File.WriteAllTextAsync(reportPath, reportJson);
            
            _logger.LogInformation("[FEATURE_ENG] Daily feature importance report saved: {ReportPath}", reportPath);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[FEATURE_ENG] Error generating daily feature importance report");
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _dailyReportTimer?.Dispose();
            _disposed = true;
            _logger.LogInformation("[FEATURE_ENG] Disposed successfully");
        }
    }
}

#region Supporting Classes

/// <summary>
/// Configuration for feature engineering with regime-specific profiles
/// </summary>
public class FeatureConfig
{
    public Dictionary<RegimeType, RegimeProfile> RegimeProfiles { get; set; } = new();
    public RegimeProfile DefaultProfile { get; set; } = new();
    public int MaxBufferSize { get; set; } = 1000;
    public int TopKFeatures { get; set; } = 10;
}

/// <summary>
/// Regime-specific feature configuration profile
/// </summary>
public class RegimeProfile
{
    public int VolatilityLookback { get; set; } = 20;
    public int TrendLookback { get; set; } = 50;
    public int VolumeLookback { get; set; } = 20;
    public int RsiLookback { get; set; } = 14;
    public int BollingerLookback { get; set; } = 20;
    public int AtrLookback { get; set; } = 14;
    public int MicrostructureLookback { get; set; } = 100;
    public int OrderFlowLookback { get; set; } = 50;
    public double TradeDirectionDecay { get; set; } = 0.9;
}

/// <summary>
/// Feature vector output
/// </summary>
public class FeatureVector
{
    public string Symbol { get; set; } = string.Empty;
    public string Strategy { get; set; } = string.Empty;
    public RegimeType Regime { get; set; }
    public DateTime Timestamp { get; set; }
    public double[] Features { get; set; } = Array.Empty<double>();
    public string[] FeatureNames { get; set; } = Array.Empty<string>();
    public int FeatureCount { get; set; }
    public bool HasMissingValues { get; set; }
}

/// <summary>
/// Feature state for forward-filling and null handling
/// </summary>
public class FeatureState
{
    public DateTime LastUpdate { get; set; }
    public int FeatureCount { get; set; }
    public Dictionary<string, double> LastValidValues { get; set; } = new();
}

/// <summary>
/// Feature importance tracking
/// </summary>
public class FeatureImportanceTracker
{
    private readonly Dictionary<string, List<double>> _importanceHistory = new();
    private readonly object _lock = new();

    public void UpdateImportance(string[] featureNames, double[] importanceScores)
    {
        lock (_lock)
        {
            for (int i = 0; i < Math.Min(featureNames.Length, importanceScores.Length); i++)
            {
                var featureName = featureNames[i];
                var importance = importanceScores[i];

                if (!_importanceHistory.ContainsKey(featureName))
                {
                    _importanceHistory[featureName] = new List<double>();
                }

                _importanceHistory[featureName].Add(importance);

                // Keep only recent history
                if (_importanceHistory[featureName].Count > 100)
                {
                    _importanceHistory[featureName].RemoveAt(0);
                }
            }
        }
    }

    public Dictionary<string, double> GetTopKFeatures(int k)
    {
        lock (_lock)
        {
            return _importanceHistory
                .Where(kv => kv.Value.Count > 0)
                .ToDictionary(kv => kv.Key, kv => kv.Value.Average())
                .OrderByDescending(kv => kv.Value)
                .Take(k)
                .ToDictionary(kv => kv.Key, kv => kv.Value);
        }
    }
}

/// <summary>
/// Daily feature importance report
/// </summary>
public class FeatureImportanceReport
{
    public DateTime GeneratedAt { get; set; }
    public Dictionary<string, Dictionary<string, double>> SymbolReports { get; set; } = new();
}

/// <summary>
/// Circular buffer for efficient market data storage
/// </summary>
public class CircularBuffer<T>
{
    private readonly T[] _buffer;
    private readonly int _size;
    private int _head = 0;
    private int _count = 0;

    public CircularBuffer(int size)
    {
        _size = size;
        _buffer = new T[size];
    }

    public int Count => _count;

    public void Add(T item)
    {
        _buffer[_head] = item;
        _head = (_head + 1) % _size;
        if (_count < _size)
            _count++;
    }

    public T? GetFromEnd(int index)
    {
        if (index >= _count) return default(T);
        var actualIndex = (_head - 1 - index + _size) % _size;
        return _buffer[actualIndex];
    }

    public T[] GetLast(int count)
    {
        count = Math.Min(count, _count);
        var result = new T[count];
        
        for (int i = 0; i < count; i++)
        {
            var item = GetFromEnd(i);
            if (item != null)
                result[count - 1 - i] = item;
        }
        
        return result;
    }
}

/// <summary>
/// Market data structure
/// </summary>
public class MarketData
{
    public DateTime Timestamp { get; set; }
    public double Open { get; set; }
    public double High { get; set; }
    public double Low { get; set; }
    public double Close { get; set; }
    public double Volume { get; set; }
    public double Bid { get; set; }
    public double Ask { get; set; }
}

/// <summary>
/// Regime types for feature configuration
/// </summary>
public enum RegimeType
{
    Range = 0,
    Trend = 1,
    Volatility = 2,
    LowVol = 3,
    HighVol = 4
}

#endregion