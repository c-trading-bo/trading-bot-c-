using Microsoft.Extensions.Logging;
using System.Text.Json;
using System.Collections.Concurrent;
using TradingBot.BotCore.Services;

namespace TradingBot.UnifiedOrchestrator.Runtime;

/// <summary>
/// Symbol-aware execution guards with per-symbol configuration and deterministic behavior
/// Production-grade microstructure filtering for institutional multi-asset trading
/// </summary>
public class SymbolAwareExecutionGuards
{
    private readonly ILogger<SymbolAwareExecutionGuards> _logger;
    private readonly ConcurrentDictionary<string, SymbolMicrostructureConfig> _symbolConfigs;
    private readonly ConcurrentDictionary<string, MicrostructureData> _symbolMicrostructure;
    private readonly object _dataLock = new();

    public SymbolAwareExecutionGuards(ILogger<SymbolAwareExecutionGuards> logger)
    {
        _logger = logger;
        _symbolConfigs = new ConcurrentDictionary<string, SymbolMicrostructureConfig>();
        _symbolMicrostructure = new ConcurrentDictionary<string, MicrostructureData>();
        
        LoadSymbolConfigurations();
        LoadMicrostructureState();
        
        _logger.LogInformation("🛡️ [SYMBOL-AWARE-GUARDS] Initialized with {ConfigCount} symbol configurations", _symbolConfigs.Count);
    }

    /// <summary>
    /// Symbol-aware execution check with specific limits per asset class
    /// </summary>
    public bool Allow(string symbol, decimal bid, decimal ask, decimal volume, int latencyMs)
    {
        var config = GetSymbolConfig(symbol);
        var checks = new List<(bool passed, string reason)>();

        // Check 0: Kill switch (Level 2: Order placement blocking)
        var killSwitchCheck = !ProductionKillSwitchService.IsKillSwitchActive();
        checks.Add((killSwitchCheck, $"Kill switch: {(killSwitchCheck ? "INACTIVE" : "ACTIVE - BLOCKING ALL ORDERS")}"));
        
        if (!killSwitchCheck)
        {
            _logger.LogCritical("🔴 [SYMBOL-AWARE-GUARDS] Kill switch active - blocking all order execution for {Symbol}", symbol);
            return false; // Immediate return for kill switch
        }

        // Check 1: Symbol-specific spread width
        var spreadTicks = CalculateSpreadInTicks(symbol, bid, ask);
        var spreadCheck = spreadTicks <= config.MaxSpreadTicks;
        checks.Add((spreadCheck, $"Spread: {spreadTicks:F2} <= {config.MaxSpreadTicks} ticks"));

        // Check 2: Symbol-specific latency
        var latencyCheck = latencyMs <= config.MaxLatencyMs;
        checks.Add((latencyCheck, $"Latency: {latencyMs}ms <= {config.MaxLatencyMs}ms"));

        // Check 3: Symbol-specific volume
        var volumeCheck = volume >= config.MinVolumeThreshold;
        checks.Add((volumeCheck, $"Volume: {volume:F0} >= {config.MinVolumeThreshold:F0}"));

        // Check 4: Symbol-specific order book imbalance
        var imbalanceCheck = CheckOrderBookImbalance(symbol, config);
        checks.Add((imbalanceCheck.passed, imbalanceCheck.reason));

        // Check 5: Market session validation (EST-based for futures)
        var sessionCheck = IsValidMarketSession(symbol);
        checks.Add((sessionCheck, $"Market session: {(sessionCheck ? "OPEN" : "CLOSED")}"));

        var allPassed = checks.All(c => c.passed);

        if (allPassed)
        {
            _logger.LogDebug("✅ [SYMBOL-AWARE-GUARDS] {Symbol} execution ALLOWED - all symbol-specific checks passed", symbol);
        }
        else
        {
            var failedChecks = checks.Where(c => !c.passed).Select(c => c.reason);
            _logger.LogWarning("🛡️ [SYMBOL-AWARE-GUARDS] {Symbol} execution BLOCKED - failed checks: {FailedChecks}",
                symbol, string.Join(", ", failedChecks));
        }

        // Update microstructure data
        UpdateMicrostructureData(symbol, bid, ask, volume, latencyMs);

        return allPassed;
    }

    /// <summary>
    /// Get symbol-specific configuration with defaults for unknown symbols
    /// </summary>
    private SymbolMicrostructureConfig GetSymbolConfig(string symbol)
    {
        return _symbolConfigs.GetOrAdd(symbol, sym => CreateDefaultConfig(sym));
    }

    /// <summary>
    /// Create symbol-specific default configuration
    /// </summary>
    private static SymbolMicrostructureConfig CreateDefaultConfig(string symbol)
    {
        return symbol.ToUpperInvariant() switch
        {
            // E-mini S&P 500 (ES) - Most liquid
            "ES" or "ESZ25" or "ESH26" or "ESM26" or "ESU26" => new SymbolMicrostructureConfig
            {
                Symbol = symbol,
                MaxSpreadTicks = 2.0m, // 2 ticks = 0.50 points
                MaxLatencyMs = 50,     // Strict latency for ES
                MinVolumeThreshold = 2000m,
                MaxOrderBookImbalance = 0.75m,
                TickSize = 0.25m,
                AssetClass = "IndexFutures"
            },
            
            // Micro E-mini S&P 500 (MES) - Lower volume tolerance
            "MES" or "MESZ25" or "MESH26" or "MESM26" or "MESU26" => new SymbolMicrostructureConfig
            {
                Symbol = symbol,
                MaxSpreadTicks = 3.0m, // 3 ticks = 0.75 points
                MaxLatencyMs = 75,     // More lenient for micro
                MinVolumeThreshold = 500m,
                MaxOrderBookImbalance = 0.80m,
                TickSize = 0.25m,
                AssetClass = "MicroIndexFutures"
            },
            
            // E-mini NASDAQ-100 (NQ) - High volatility tolerance
            "NQ" or "NQZ25" or "NQH26" or "NQM26" or "NQU26" => new SymbolMicrostructureConfig
            {
                Symbol = symbol,
                MaxSpreadTicks = 4.0m, // 4 ticks = 1.00 points
                MaxLatencyMs = 60,     // Moderate latency tolerance
                MinVolumeThreshold = 1500m,
                MaxOrderBookImbalance = 0.70m,
                TickSize = 0.25m,
                AssetClass = "TechIndexFutures"
            },
            
            // Micro E-mini NASDAQ-100 (MNQ) - Most lenient
            "MNQ" or "MNQZ25" or "MNQH26" or "MNQM26" or "MNQU26" => new SymbolMicrostructureConfig
            {
                Symbol = symbol,
                MaxSpreadTicks = 5.0m, // 5 ticks = 1.25 points
                MaxLatencyMs = 100,    // Most lenient latency
                MinVolumeThreshold = 300m,
                MaxOrderBookImbalance = 0.85m,
                TickSize = 0.25m,
                AssetClass = "MicroTechIndexFutures"
            },
            
            // Default configuration for unknown symbols
            _ => new SymbolMicrostructureConfig
            {
                Symbol = symbol,
                MaxSpreadTicks = 3.0m,
                MaxLatencyMs = 100,
                MinVolumeThreshold = 1000m,
                MaxOrderBookImbalance = 0.80m,
                TickSize = 0.01m, // Default to penny tick
                AssetClass = "Unknown"
            }
        };
    }

    private static decimal CalculateSpreadInTicks(string symbol, decimal bid, decimal ask)
    {
        var spread = ask - bid;
        var tickSize = GetTickSizeForSymbol(symbol);
        return spread / tickSize;
    }

    private static decimal GetTickSizeForSymbol(string symbol)
    {
        return symbol.ToUpperInvariant() switch
        {
            "ES" or "ESZ25" or "ESH26" or "ESM26" or "ESU26" => 0.25m,
            "MES" or "MESZ25" or "MESH26" or "MESM26" or "MESU26" => 0.25m,
            "NQ" or "NQZ25" or "NQH26" or "NQM26" or "NQU26" => 0.25m,
            "MNQ" or "MNQZ25" or "MNQH26" or "MNQM26" or "MNQU26" => 0.25m,
            _ => 0.01m // Default penny tick
        };
    }

    private (bool passed, string reason) CheckOrderBookImbalance(string symbol, SymbolMicrostructureConfig config)
    {
        lock (_dataLock)
        {
            if (_symbolMicrostructure.TryGetValue(symbol, out var microData))
            {
                var imbalancePassed = Math.Abs(microData.OrderBookImbalance) <= config.MaxOrderBookImbalance;
                return (imbalancePassed, $"Order book imbalance: {Math.Abs(microData.OrderBookImbalance):F2} <= {config.MaxOrderBookImbalance:F2}");
            }
            
            return (true, "Order book imbalance: NO DATA (defaulting to ALLOW)");
        }
    }

    /// <summary>
    /// Check if market session is valid for the symbol (EST-based for futures)
    /// </summary>
    private static bool IsValidMarketSession(string symbol)
    {
        var estTime = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, 
            TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));
        var dayOfWeek = estTime.DayOfWeek;
        var timeOfDay = estTime.TimeOfDay;

        // Futures trading hours: Sunday 6:00 PM EST to Friday 5:00 PM EST
        return symbol.ToUpperInvariant() switch
        {
            var s when s.StartsWith("ES") || s.StartsWith("MES") || s.StartsWith("NQ") || s.StartsWith("MNQ") => 
                IsFuturesSessionOpen(dayOfWeek, timeOfDay),
            _ => true // Default to always open for non-futures
        };
    }

    private static bool IsFuturesSessionOpen(DayOfWeek dayOfWeek, TimeSpan timeOfDay)
    {
        return dayOfWeek switch
        {
            DayOfWeek.Sunday => timeOfDay >= TimeSpan.FromHours(18), // 6:00 PM EST Sunday
            DayOfWeek.Monday or DayOfWeek.Tuesday or DayOfWeek.Wednesday or DayOfWeek.Thursday => true, // All day
            DayOfWeek.Friday => timeOfDay < TimeSpan.FromHours(17), // Until 5:00 PM EST Friday
            DayOfWeek.Saturday => false, // Closed Saturday
            _ => false
        };
    }

    /// <summary>
    /// Update microstructure data with symbol-specific tracking
    /// </summary>
    public void UpdateMicrostructureData(string symbol, decimal bid, decimal ask, decimal volume, int latencyMs)
    {
        lock (_dataLock)
        {
            var data = _symbolMicrostructure.GetOrAdd(symbol, _ => new MicrostructureData { Symbol = symbol });

            data.LastBid = bid;
            data.LastAsk = ask;
            data.LastVolume = volume;
            data.LastLatencyMs = latencyMs;
            data.LastUpdate = DateTime.UtcNow;
            data.SpreadTicks = CalculateSpreadInTicks(symbol, bid, ask);

            // Calculate order book imbalance (simplified - in production would use real bid/ask sizes)
            var config = GetSymbolConfig(symbol);
            data.OrderBookImbalance = Math.Min(data.SpreadTicks / config.MaxSpreadTicks, 1.0m);

            // Persist state periodically
            SaveMicrostructureState(symbol, data);
        }
    }

    /// <summary>
    /// Load symbol-specific configurations from files
    /// </summary>
    private void LoadSymbolConfigurations()
    {
        try
        {
            var configDir = Path.Combine(Directory.GetCurrentDirectory(), "config", "symbols");
            if (!Directory.Exists(configDir))
            {
                Directory.CreateDirectory(configDir);
                CreateDefaultSymbolConfigs(configDir);
            }

            var configFiles = Directory.GetFiles(configDir, "*.json");
            foreach (var file in configFiles)
            {
                var json = File.ReadAllText(file);
                var config = JsonSerializer.Deserialize<SymbolMicrostructureConfig>(json);
                if (config != null)
                {
                    _symbolConfigs.TryAdd(config.Symbol, config);
                }
            }

            _logger.LogInformation("📂 [SYMBOL-AWARE-GUARDS] Loaded {Count} symbol configurations", _symbolConfigs.Count);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "⚠️ [SYMBOL-AWARE-GUARDS] Failed to load symbol configurations, using defaults");
        }
    }

    /// <summary>
    /// Create default symbol configuration files
    /// </summary>
    private void CreateDefaultSymbolConfigs(string configDir)
    {
        var symbols = new[] { "ES", "MES", "NQ", "MNQ" };
        
        foreach (var symbol in symbols)
        {
            var config = CreateDefaultConfig(symbol);
            var json = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
            var filePath = Path.Combine(configDir, $"{symbol}.json");
            File.WriteAllText(filePath, json);
        }
    }

    private void LoadMicrostructureState()
    {
        try
        {
            var stateDir = Path.Combine(Directory.GetCurrentDirectory(), "state");
            if (!Directory.Exists(stateDir)) return;

            var files = Directory.GetFiles(stateDir, "microstructure.*.json");
            foreach (var file in files)
            {
                var json = File.ReadAllText(file);
                var data = JsonSerializer.Deserialize<MicrostructureData>(json);
                if (data != null)
                {
                    _symbolMicrostructure.TryAdd(data.Symbol, data);
                }
            }

            _logger.LogInformation("📂 [SYMBOL-AWARE-GUARDS] Loaded microstructure state for {Count} symbols", _symbolMicrostructure.Count);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "⚠️ [SYMBOL-AWARE-GUARDS] Failed to load microstructure state");
        }
    }

    private void SaveMicrostructureState(string symbol, MicrostructureData data)
    {
        try
        {
            var stateDir = Path.Combine(Directory.GetCurrentDirectory(), "state");
            Directory.CreateDirectory(stateDir);

            var filePath = Path.Combine(stateDir, $"microstructure.{symbol}.json");
            var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(filePath, json);
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to save microstructure state for {Symbol}", symbol);
        }
    }

    /// <summary>
    /// Get current microstructure state for all symbols
    /// </summary>
    public Dictionary<string, MicrostructureData> GetMicrostructureState()
    {
        lock (_dataLock)
        {
            return _symbolMicrostructure.ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        }
    }

    /// <summary>
    /// Get symbol configurations for monitoring
    /// </summary>
    public Dictionary<string, SymbolMicrostructureConfig> GetSymbolConfigurations()
    {
        return _symbolConfigs.ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
    }
}

/// <summary>
/// Symbol-specific microstructure configuration
/// </summary>
public class SymbolMicrostructureConfig
{
    public string Symbol { get; set; } = string.Empty;
    public decimal MaxSpreadTicks { get; set; }
    public int MaxLatencyMs { get; set; }
    public decimal MinVolumeThreshold { get; set; }
    public decimal MaxOrderBookImbalance { get; set; }
    public decimal TickSize { get; set; }
    public string AssetClass { get; set; } = string.Empty;
    public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
}