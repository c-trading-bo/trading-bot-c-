using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Collections.Concurrent;
using TradingBot.Abstractions;
using Trading.Safety.Models;

namespace Trading.Safety.CircuitBreakers;

/// <summary>
/// Production-grade circuit breaker system for trading operations
/// Monitors volatility spikes, gaps, error thresholds and automatically suspends trading
/// </summary>
public interface ICircuitBreakerManager
{
    bool IsTradingSuspended { get; }
    Task<bool> CheckCircuitBreakersAsync(MarketContext context);
    Task ResetCircuitBreakersAsync(string reason);
    Task ForceCircuitBreakerAsync(string reason, TimeSpan duration);
    event Action<CircuitBreakerEvent> OnCircuitBreakerTriggered;
    event Action<CircuitBreakerEvent> OnCircuitBreakerReset;
}

public class CircuitBreakerManager : ICircuitBreakerManager
{
    private readonly ILogger<CircuitBreakerManager> _logger;
    private readonly CircuitBreakerConfig _config;
    private readonly ConcurrentDictionary<string, CircuitBreakerState> _circuitBreakers = new();
    private readonly object _lock = new object();

    public bool IsTradingSuspended => _circuitBreakers.Values.Any(cb => cb.IsTriggered);

    public event Action<CircuitBreakerEvent> OnCircuitBreakerTriggered = delegate { };
    public event Action<CircuitBreakerEvent> OnCircuitBreakerReset = delegate { };

    public CircuitBreakerManager(
        ILogger<CircuitBreakerManager> logger,
        IOptions<CircuitBreakerConfig> config)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _config = config?.Value ?? throw new ArgumentNullException(nameof(config));
        InitializeCircuitBreakers();
    }

    public async Task<bool> CheckCircuitBreakersAsync(MarketContext context)
    {
        try
        {
            var correlationId = Guid.NewGuid().ToString("N")[..8];
            
            // Check volatility circuit breaker
            await CheckVolatilityCircuitBreakerAsync(context, correlationId).ConfigureAwait(false);
            
            // Check gap circuit breaker
            await CheckGapCircuitBreakerAsync(context, correlationId).ConfigureAwait(false);
            
            // Check error threshold circuit breaker
            await CheckErrorThresholdCircuitBreakerAsync(correlationId).ConfigureAwait(false);
            
            // Check time-based circuit breakers
            await CheckTimeBasedCircuitBreakersAsync(correlationId).ConfigureAwait(false);

            return !IsTradingSuspended;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[CIRCUIT_BREAKER] Error checking circuit breakers");
            // Default to safe state - suspend trading on errors
            await ForceCircuitBreakerAsync("ERROR_CHECKING_CIRCUIT_BREAKERS", TimeSpan.FromMinutes(5)).ConfigureAwait(false);
            return false;
        }
    }

    public async Task ResetCircuitBreakersAsync(string reason)
    {
        lock (_lock)
        {
            foreach (var cb in _circuitBreakers.Values)
            {
                if (cb.IsTriggered)
                {
                    cb.Reset();
                    var resetEvent = new CircuitBreakerEvent
                    {
                        Type = cb.Type,
                        Action = "RESET",
                        Reason = reason,
                        Timestamp = DateTime.UtcNow,
                        Metadata = new Dictionary<string, object>
                        {
                            ["manual_reset"] = true,
                            ["reset_reason"] = reason
                        }
                    };
                    
                    _logger.LogWarning("[CIRCUIT_BREAKER] {Type} reset: {Reason}", cb.Type, reason);
                    OnCircuitBreakerReset.Invoke(resetEvent);
                }
            }
        }

        await Task.CompletedTask.ConfigureAwait(false);
    }

    public async Task ForceCircuitBreakerAsync(string reason, TimeSpan duration)
    {
        var forceBreaker = new CircuitBreakerState
        {
            Type = "FORCED",
            IsTriggered = true,
            TriggerTime = DateTime.UtcNow,
            ResetTime = DateTime.UtcNow.Add(duration),
            TriggerCount = 1,
            LastTriggerReason = reason
        };

        _circuitBreakers.AddOrUpdate("FORCED", forceBreaker, (k, v) => forceBreaker);

        var triggerEvent = new CircuitBreakerEvent
        {
            Type = "FORCED",
            Action = "TRIGGERED",
            Reason = reason,
            Timestamp = DateTime.UtcNow,
            Metadata = new Dictionary<string, object>
            {
                ["duration_minutes"] = duration.TotalMinutes,
                ["forced"] = true
            }
        };

        _logger.LogCritical("[CIRCUIT_BREAKER] FORCED circuit breaker triggered: {Reason} for {Duration}", 
            reason, duration);
        
        OnCircuitBreakerTriggered.Invoke(triggerEvent);
        await Task.CompletedTask.ConfigureAwait(false);
    }

    private void InitializeCircuitBreakers()
    {
        _circuitBreakers["VOLATILITY"] = new CircuitBreakerState { Type = "VOLATILITY" };
        _circuitBreakers["GAP"] = new CircuitBreakerState { Type = "GAP" };
        _circuitBreakers["ERROR_THRESHOLD"] = new CircuitBreakerState { Type = "ERROR_THRESHOLD" };
        _circuitBreakers["NEWS_EVENT"] = new CircuitBreakerState { Type = "NEWS_EVENT" };
        _circuitBreakers["MARKET_OPEN"] = new CircuitBreakerState { Type = "MARKET_OPEN" };
    }

    private async Task CheckVolatilityCircuitBreakerAsync(MarketContext context, string correlationId)
    {
        var volatilityBreaker = _circuitBreakers["VOLATILITY"];
        
        // Calculate recent volatility (simplified - use actual volatility calculation in production)
        var recentVolatility = Math.Abs((context.Ask - context.Bid) / context.Price);
        
        if (recentVolatility > _config.VolatilityThreshold)
        {
            if (!volatilityBreaker.IsTriggered)
            {
                await TriggerCircuitBreakerAsync(volatilityBreaker, 
                    $"Volatility spike detected: {recentVolatility:P2} > {_config.VolatilityThreshold:P2}",
                    correlationId,
                    new Dictionary<string, object>
                    {
                        ["volatility"] = recentVolatility,
                        ["threshold"] = _config.VolatilityThreshold,
                        ["symbol"] = context.Symbol,
                        ["price"] = context.Price
                    }).ConfigureAwait(false);
            }
        }
        else if (volatilityBreaker.IsTriggered && 
                 DateTime.UtcNow > volatilityBreaker.ResetTime)
        {
            await ResetCircuitBreakerAsync(volatilityBreaker, "Volatility normalized", correlationId).ConfigureAwait(false);
        }
    }

    private async Task CheckGapCircuitBreakerAsync(MarketContext context, string correlationId)
    {
        var gapBreaker = _circuitBreakers["GAP"];
        
        // Check for abnormal price gaps (simplified implementation)
        var spread = Math.Abs(context.Ask - context.Bid);
        var normalSpread = context.Price * _config.NormalSpreadPercentage;
        
        if (spread > normalSpread * _config.GapMultiplier)
        {
            if (!gapBreaker.IsTriggered)
            {
                await TriggerCircuitBreakerAsync(gapBreaker, 
                    $"Abnormal spread detected: {spread:F2} vs normal {normalSpread:F2}",
                    correlationId,
                    new Dictionary<string, object>
                    {
                        ["spread"] = spread,
                        ["normal_spread"] = normalSpread,
                        ["gap_multiplier"] = _config.GapMultiplier,
                        ["symbol"] = context.Symbol
                    }).ConfigureAwait(false);
            }
        }
        else if (gapBreaker.IsTriggered && 
                 DateTime.UtcNow > gapBreaker.ResetTime)
        {
            await ResetCircuitBreakerAsync(gapBreaker, "Spread normalized", correlationId).ConfigureAwait(false);
        }
    }

    private async Task CheckErrorThresholdCircuitBreakerAsync(string correlationId)
    {
        var errorBreaker = _circuitBreakers["ERROR_THRESHOLD"];
        
        // This would integrate with error tracking system
        // For now, use a simple time-based reset
        if (errorBreaker.IsTriggered && 
            DateTime.UtcNow > errorBreaker.ResetTime)
        {
            await ResetCircuitBreakerAsync(errorBreaker, "Error threshold timeout", correlationId).ConfigureAwait(false);
        }
    }

    private async Task CheckTimeBasedCircuitBreakersAsync(string correlationId)
    {
        var now = DateTime.UtcNow;
        var easternTime = TimeZoneInfo.ConvertTimeFromUtc(now, 
            TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));
        
        var marketOpenBreaker = _circuitBreakers["MARKET_OPEN"];
        
        // Suspend trading 5 minutes before and after market open
        var marketOpen = easternTime.Date.AddHours(9).AddMinutes(30); // 9:30 AM ET
        var suspendStart = marketOpen.AddMinutes(-5);
        var suspendEnd = marketOpen.AddMinutes(5);
        
        if (easternTime >= suspendStart && easternTime <= suspendEnd)
        {
            if (!marketOpenBreaker.IsTriggered)
            {
                await TriggerCircuitBreakerAsync(marketOpenBreaker, 
                    "Market open suspension period",
                    correlationId,
                    new Dictionary<string, object>
                    {
                        ["market_open"] = marketOpen,
                        ["eastern_time"] = easternTime,
                        ["suspend_period"] = "5_minutes_around_open"
                    }).ConfigureAwait(false);
            }
        }
        else if (marketOpenBreaker.IsTriggered)
        {
            await ResetCircuitBreakerAsync(marketOpenBreaker, "Market open period ended", correlationId).ConfigureAwait(false);
        }
    }

    private async Task TriggerCircuitBreakerAsync(
        CircuitBreakerState breaker, 
        string reason, 
        string correlationId,
        Dictionary<string, object> metadata)
    {
        lock (_lock)
        {
            breaker.IsTriggered = true;
            breaker.TriggerTime = DateTime.UtcNow;
            breaker.ResetTime = DateTime.UtcNow.Add(_config.DefaultResetDuration);
            breaker.TriggerCount++;
            breaker.LastTriggerReason = reason;
        }

        var triggerEvent = new CircuitBreakerEvent
        {
            Type = breaker.Type,
            Action = "TRIGGERED",
            Reason = reason,
            Timestamp = DateTime.UtcNow,
            CorrelationId = correlationId,
            Metadata = metadata
        };

        _logger.LogCritical("[CIRCUIT_BREAKER] {Type} triggered: {Reason} [CorrelationId: {CorrelationId}]", 
            breaker.Type, reason, correlationId);
        
        OnCircuitBreakerTriggered.Invoke(triggerEvent);
        await Task.CompletedTask.ConfigureAwait(false);
    }

    private async Task ResetCircuitBreakerAsync(
        CircuitBreakerState breaker, 
        string reason, 
        string correlationId)
    {
        lock (_lock)
        {
            breaker.Reset();
        }

        var resetEvent = new CircuitBreakerEvent
        {
            Type = breaker.Type,
            Action = "RESET",
            Reason = reason,
            Timestamp = DateTime.UtcNow,
            CorrelationId = correlationId
        };

        _logger.LogWarning("[CIRCUIT_BREAKER] {Type} reset: {Reason} [CorrelationId: {CorrelationId}]", 
            breaker.Type, reason, correlationId);
        
        OnCircuitBreakerReset.Invoke(resetEvent);
        await Task.CompletedTask.ConfigureAwait(false);
    }
}

public class CircuitBreakerState
{
    public string Type { get; set; } = string.Empty;
    public bool IsTriggered { get; set; }
    public DateTime TriggerTime { get; set; }
    public DateTime ResetTime { get; set; }
    public int TriggerCount { get; set; }
    public string LastTriggerReason { get; set; } = string.Empty;

    public void Reset()
    {
        IsTriggered = false;
        ResetTime = DateTime.MinValue;
        LastTriggerReason = string.Empty;
    }
}

public class CircuitBreakerEvent
{
    public string Type { get; set; } = string.Empty;
    public string Action { get; set; } = string.Empty;
    public string Reason { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string CorrelationId { get; set; } = string.Empty;
    public Dictionary<string, object> Metadata { get; } = new();
}

public class CircuitBreakerConfig
{
    public double VolatilityThreshold { get; set; } = 0.02; // 2%
    public double NormalSpreadPercentage { get; set; } = 0.0001; // 0.01%
    public double GapMultiplier { get; set; } = 5.0;
    public TimeSpan DefaultResetDuration { get; set; } = TimeSpan.FromMinutes(15);
    public int ErrorThreshold { get; set; } = 5;
    public TimeSpan ErrorWindowDuration { get; set; } = TimeSpan.FromMinutes(5);
}