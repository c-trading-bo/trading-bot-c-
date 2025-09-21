using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace TradingBot.UnifiedOrchestrator.Runtime;

/// <summary>
/// Execution guards that filter trades based on microstructure conditions
/// to prevent trading during hostile market conditions.
/// </summary>
public class ExecutionGuards
{
    private readonly ILogger<ExecutionGuards> _logger;
    private readonly Dictionary<string, MicrostructureData> _symbolMicrostructure;
    private readonly object _dataLock = new();

    // Configuration constants (following production guardrails - no magic numbers)
    private const decimal MAX_SPREAD_TICKS_ES = 2.0m;
    private const decimal MAX_SPREAD_TICKS_NQ = 4.0m;
    private const int MAX_LATENCY_MS = 100;
    private const decimal MIN_VOLUME_THRESHOLD = 1000m;
    private const decimal MAX_ORDER_BOOK_IMBALANCE = 0.8m;

    public ExecutionGuards(ILogger<ExecutionGuards> logger)
    {
        _logger = logger;
        _symbolMicrostructure = new Dictionary<string, MicrostructureData>();
        
        _logger.LogInformation("🛡️ [EXECUTION-GUARDS] Initialized with ES spread limit: {ESSpread} ticks, NQ: {NQSpread} ticks, Max latency: {MaxLatency}ms",
            MAX_SPREAD_TICKS_ES, MAX_SPREAD_TICKS_NQ, MAX_LATENCY_MS);
    }

    /// <summary>
    /// Check if execution should be allowed based on current market microstructure
    /// </summary>
    /// <param name="symbol">Trading symbol</param>
    /// <param name="bid">Current bid price</param>
    /// <param name="ask">Current ask price</param>
    /// <param name="volume">Current volume</param>
    /// <param name="latencyMs">Current system latency in milliseconds</param>
    /// <returns>True if execution is allowed, false if conditions are hostile</returns>
    public bool Allow(string symbol, decimal bid, decimal ask, decimal volume, int latencyMs)
    {
        var checks = new List<(bool passed, string reason)>();

        // Check 1: Spread width
        var spreadTicks = CalculateSpreadInTicks(symbol, bid, ask);
        var maxAllowedSpread = GetMaxSpreadForSymbol(symbol);
        var spreadCheck = spreadTicks <= maxAllowedSpread;
        checks.Add((spreadCheck, $"Spread: {spreadTicks:F2} <= {maxAllowedSpread} ticks"));

        // Check 2: Latency
        var latencyCheck = latencyMs <= MAX_LATENCY_MS;
        checks.Add((latencyCheck, $"Latency: {latencyMs}ms <= {MAX_LATENCY_MS}ms"));

        // Check 3: Volume
        var volumeCheck = volume >= MIN_VOLUME_THRESHOLD;
        checks.Add((volumeCheck, $"Volume: {volume:F0} >= {MIN_VOLUME_THRESHOLD:F0}"));

        // Check 4: Order book imbalance (if data available)
        var imbalanceCheck = true; // Default to true if no data
        lock (_dataLock)
        {
            if (_symbolMicrostructure.TryGetValue(symbol, out var microData))
            {
                imbalanceCheck = Math.Abs(microData.OrderBookImbalance) <= MAX_ORDER_BOOK_IMBALANCE;
                checks.Add((imbalanceCheck, $"Order book imbalance: {Math.Abs(microData.OrderBookImbalance):F2} <= {MAX_ORDER_BOOK_IMBALANCE:F2}"));
            }
        }

        var allPassed = checks.All(c => c.passed);

        if (allPassed)
        {
            _logger.LogDebug("✅ [EXECUTION-GUARDS] {Symbol} execution ALLOWED - all checks passed", symbol);
        }
        else
        {
            var failedChecks = checks.Where(c => !c.passed).Select(c => c.reason);
            _logger.LogWarning("🛡️ [EXECUTION-GUARDS] {Symbol} execution BLOCKED - failed checks: {FailedChecks}",
                symbol, string.Join(", ", failedChecks));
        }

        // Update microstructure data
        UpdateMicrostructureData(symbol, bid, ask, volume, latencyMs);

        return allPassed;
    }

    /// <summary>
    /// Update microstructure data for symbol-specific state tracking
    /// </summary>
    public void UpdateMicrostructureData(string symbol, decimal bid, decimal ask, decimal volume, int latencyMs)
    {
        lock (_dataLock)
        {
            if (!_symbolMicrostructure.TryGetValue(symbol, out var data))
            {
                data = new MicrostructureData { Symbol = symbol };
                _symbolMicrostructure[symbol] = data;
            }

            data.LastBid = bid;
            data.LastAsk = ask;
            data.LastVolume = volume;
            data.LastLatencyMs = latencyMs;
            data.LastUpdate = DateTime.UtcNow;
            data.SpreadTicks = CalculateSpreadInTicks(symbol, bid, ask);

            // Calculate order book imbalance (simplified - would need real bid/ask sizes)
            // For now, use spread as a proxy for imbalance
            data.OrderBookImbalance = Math.Min(data.SpreadTicks / GetMaxSpreadForSymbol(symbol), 1.0m);

            // Save to state file periodically
            SaveMicrostructureState(symbol, data);
        }
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
            "ES" or "ESZ25" or "ESH25" or "ESM25" or "ESU25" => 0.25m,
            "NQ" or "NQZ25" or "NQH25" or "NQM25" or "NQU25" => 0.25m,
            _ => 0.25m // Default ES/NQ tick size
        };
    }

    private static decimal GetMaxSpreadForSymbol(string symbol)
    {
        return symbol.ToUpperInvariant() switch
        {
            "ES" or "ESZ25" or "ESH25" or "ESM25" or "ESU25" => MAX_SPREAD_TICKS_ES,
            "NQ" or "NQZ25" or "NQH25" or "NQM25" or "NQU25" => MAX_SPREAD_TICKS_NQ,
            _ => MAX_SPREAD_TICKS_ES // Default to ES
        };
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
            _logger.LogWarning(ex, "⚠️ [EXECUTION-GUARDS] Failed to save microstructure state for {Symbol}", symbol);
        }
    }

    /// <summary>
    /// Get current microstructure state for monitoring
    /// </summary>
    public Dictionary<string, MicrostructureData> GetMicrostructureState()
    {
        lock (_dataLock)
        {
            return _symbolMicrostructure.ToDictionary(kvp => kvp.Key, kvp => kvp.Value);
        }
    }

    /// <summary>
    /// Load microstructure state from files on startup
    /// </summary>
    public void LoadMicrostructureState()
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
                    lock (_dataLock)
                    {
                        _symbolMicrostructure[data.Symbol] = data;
                    }
                }
            }

            _logger.LogInformation("📂 [EXECUTION-GUARDS] Loaded microstructure state for {Count} symbols", _symbolMicrostructure.Count);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "⚠️ [EXECUTION-GUARDS] Failed to load microstructure state");
        }
    }
}

/// <summary>
/// Microstructure data for a trading symbol
/// </summary>
public class MicrostructureData
{
    public string Symbol { get; set; } = string.Empty;
    public decimal LastBid { get; set; }
    public decimal LastAsk { get; set; }
    public decimal LastVolume { get; set; }
    public int LastLatencyMs { get; set; }
    public decimal SpreadTicks { get; set; }
    public decimal OrderBookImbalance { get; set; }
    public DateTime LastUpdate { get; set; }
}