using System;
using System.Collections.Concurrent;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCore.Config;

namespace BotCore.Services;

/// <summary>
/// Market data staleness detection service
/// Pauses routing if last tick > N seconds (configurable)
/// Resumes on fresh tick reception
/// </summary>
public interface IMarketDataStalenessService
{
    void UpdateLastTick(string symbol, DateTime tickTime);
    bool IsDataStale(string symbol);
    bool IsRoutingPaused { get; }
    void StartMonitoring();
    void StopMonitoring();
    event Action<string>? OnDataStale;
    event Action<string>? OnDataFresh;
    event Action? OnRoutingPaused;
    event Action? OnRoutingResumed;
}

public class MarketDataStalenessService : IMarketDataStalenessService, IDisposable
{
    private readonly ILogger<MarketDataStalenessService> _logger;
    private readonly ConcurrentDictionary<string, DateTime> _lastTickTimes = new();
    private readonly Timer _stalenessCheckTimer;
    private readonly object _pauseLock = new();
    
    private volatile bool _isRoutingPaused;
    private volatile bool _isMonitoring;
    private readonly int _stalenessThresholdSeconds;
    private readonly int _checkIntervalMs;

    public bool IsRoutingPaused => _isRoutingPaused;

    public event Action<string>? OnDataStale;
    public event Action<string>? OnDataFresh;
    public event Action? OnRoutingPaused;
    public event Action? OnRoutingResumed;

    public MarketDataStalenessService(ILogger<MarketDataStalenessService> logger)
    {
        _logger = logger;
        
        // Get configuration from environment
        _stalenessThresholdSeconds = EnvConfig.GetInt("MARKET_DATA_STALENESS_THRESHOLD_SEC", 30);
        _checkIntervalMs = EnvConfig.GetInt("MARKET_DATA_CHECK_INTERVAL_MS", 5000);

        // Initialize timer but don't start it yet
        _stalenessCheckTimer = new Timer(CheckStaleness, null, Timeout.Infinite, Timeout.Infinite);

        _logger.LogInformation("MarketDataStalenessService initialized. Threshold: {ThresholdSec}s, Check interval: {IntervalMs}ms",
            _stalenessThresholdSeconds, _checkIntervalMs);
    }

    public void UpdateLastTick(string symbol, DateTime tickTime)
    {
        if (string.IsNullOrEmpty(symbol))
            return;

        var previousTime = _lastTickTimes.GetValueOrDefault(symbol, DateTime.MinValue);
        _lastTickTimes[symbol] = tickTime;

        // Check if this symbol was stale and is now fresh
        if (previousTime != DateTime.MinValue)
        {
            var previousAge = DateTime.UtcNow - previousTime;
            var currentAge = DateTime.UtcNow - tickTime;

            if (previousAge.TotalSeconds > _stalenessThresholdSeconds && 
                currentAge.TotalSeconds <= _stalenessThresholdSeconds)
            {
                OnDataFresh?.Invoke(symbol);
                
                var freshData = new
                {
                    timestamp = DateTime.UtcNow,
                    component = "market_data_staleness_service",
                    operation = "data_fresh",
                    symbol = symbol,
                    tick_time = tickTime,
                    age_seconds = currentAge.TotalSeconds
                };

                _logger.LogInformation("DATA_FRESH: {FreshData}", JsonSerializer.Serialize(freshData));

                // Check if we should resume routing
                CheckResumeRouting();
            }
        }

        _logger.LogTrace("Updated last tick for {Symbol}: {TickTime}", symbol, tickTime);
    }

    public bool IsDataStale(string symbol)
    {
        if (!_lastTickTimes.TryGetValue(symbol, out var lastTick))
            return true; // No data is considered stale

        var age = DateTime.UtcNow - lastTick;
        return age.TotalSeconds > _stalenessThresholdSeconds;
    }

    public void StartMonitoring()
    {
        if (_isMonitoring)
        {
            _logger.LogWarning("Market data staleness monitoring is already running");
            return;
        }

        _isMonitoring = true;
        _stalenessCheckTimer.Change(_checkIntervalMs, _checkIntervalMs);
        
        var startData = new
        {
            timestamp = DateTime.UtcNow,
            component = "market_data_staleness_service",
            operation = "monitoring_started",
            threshold_seconds = _stalenessThresholdSeconds,
            check_interval_ms = _checkIntervalMs
        };

        _logger.LogInformation("MONITORING_STARTED: {StartData}", JsonSerializer.Serialize(startData));
    }

    public void StopMonitoring()
    {
        if (!_isMonitoring)
            return;

        _isMonitoring = false; // Stop monitoring
        _stalenessCheckTimer.Change(Timeout.Infinite, Timeout.Infinite);

        var stopData = new
        {
            timestamp = DateTime.UtcNow,
            component = "market_data_staleness_service",
            operation = "monitoring_stopped"
        };

        _logger.LogInformation("MONITORING_STOPPED: {StopData}", JsonSerializer.Serialize(stopData));
    }

    private void CheckStaleness(object? state)
    {
        if (!_isMonitoring)
            return;

        try
        {
            var now = DateTime.UtcNow;
            var staleSymbols = new List<string>();
            var freshSymbols = new List<string>();

            foreach (var kvp in _lastTickTimes)
            {
                var symbol = kvp.Key;
                var lastTick = kvp.Value;
                var age = now - lastTick;

                if (age.TotalSeconds > _stalenessThresholdSeconds)
                {
                    staleSymbols.Add(symbol);
                }
                else
                {
                    freshSymbols.Add(symbol);
                }
            }

            // Log staleness status
            if (staleSymbols.Count > 0)
            {
                foreach (var symbol in staleSymbols)
                {
                    var lastTick = _lastTickTimes[symbol];
                    var age = now - lastTick;

                    var staleData = new
                    {
                        timestamp = now,
                        component = "market_data_staleness_service",
                        operation = "data_stale",
                        symbol = symbol,
                        last_tick = lastTick,
                        age_seconds = age.TotalSeconds,
                        threshold_seconds = _stalenessThresholdSeconds
                    };

                    _logger.LogWarning("DATA_STALE: {StaleData}", JsonSerializer.Serialize(staleData));
                    OnDataStale?.Invoke(symbol);
                }

                // Pause routing if not already paused
                PauseRoutingIfNeeded(staleSymbols);
            }

            // Resume routing if all data is fresh and routing was paused
            if (staleSymbols.Count == 0 && _isRoutingPaused)
            {
                ResumeRouting();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during staleness check");
        }
    }

    private void PauseRoutingIfNeeded(List<string> staleSymbols)
    {
        lock (_pauseLock)
        {
            if (_isRoutingPaused)
                return;

            _isRoutingPaused = true;

            var pauseData = new
            {
                timestamp = DateTime.UtcNow,
                component = "market_data_staleness_service",
                operation = "routing_paused",
                reason = "stale_market_data",
                stale_symbols = staleSymbols.ToArray(),
                threshold_seconds = _stalenessThresholdSeconds
            };

            _logger.LogWarning("ROUTING_PAUSED: {PauseData}", JsonSerializer.Serialize(pauseData));
            OnRoutingPaused?.Invoke();
        }
    }

    private void CheckResumeRouting()
    {
        if (!_isRoutingPaused)
            return;

        // Check if all tracked symbols are now fresh
        var now = DateTime.UtcNow;
        var hasStaleData = false;

        foreach (var kvp in _lastTickTimes)
        {
            var age = now - kvp.Value;
            if (age.TotalSeconds > _stalenessThresholdSeconds)
            {
                hasStaleData = true;
                break;
            }
        }

        if (!hasStaleData)
        {
            ResumeRouting();
        }
    }

    private void ResumeRouting()
    {
        lock (_pauseLock)
        {
            if (!_isRoutingPaused)
                return;

            _isRoutingPaused = false;

            var resumeData = new
            {
                timestamp = DateTime.UtcNow,
                component = "market_data_staleness_service",
                operation = "routing_resumed",
                reason = "fresh_market_data"
            };

            _logger.LogInformation("ROUTING_RESUMED: {ResumeData}", JsonSerializer.Serialize(resumeData));
            OnRoutingResumed?.Invoke();
        }
    }

    public void Dispose()
    {
        StopMonitoring();
        _stalenessCheckTimer?.Dispose();
        _logger.LogInformation("MarketDataStalenessService disposed");
    }
}