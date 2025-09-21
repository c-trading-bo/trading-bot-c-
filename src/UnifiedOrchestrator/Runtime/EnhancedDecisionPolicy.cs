using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;
using System.Collections.Concurrent;

namespace TradingBot.UnifiedOrchestrator.Runtime;

/// <summary>
/// Enhanced Decision Policy with UTC timing, mathematical clamping, and futures EST time awareness
/// Production-grade autonomous decision making for institutional trading
/// </summary>
public class EnhancedDecisionPolicy
{
    private readonly ILogger<EnhancedDecisionPolicy> _logger;
    private readonly EnhancedDecisionPolicyOptions _options;
    private readonly ConcurrentDictionary<string, SymbolDecisionState> _symbolStates;
    private readonly object _rateLimitLock = new();
    
    // UTC-based timing for global trading hours
    private DateTime _lastDecisionTimeUtc = DateTime.MinValue;
    private int _decisionsThisMinute = 0;
    private DateTime _currentMinuteStartUtc = DateTime.MinValue;
    
    // EST-based futures trading hours
    private static readonly TimeZoneInfo EstTimeZone = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");

    public EnhancedDecisionPolicy(ILogger<EnhancedDecisionPolicy> logger, IOptions<EnhancedDecisionPolicyOptions> options)
    {
        _logger = logger;
        _options = options.Value;
        ValidateAndClampThresholds();
        _symbolStates = new ConcurrentDictionary<string, SymbolDecisionState>();
        
        _logger.LogInformation("🎯 [ENHANCED-DECISION-POLICY] Initialized with UTC timing and mathematical clamping");
        _logger.LogInformation("📊 [ENHANCED-DECISION-POLICY] Bull:{BullThreshold:F3} Bear:{BearThreshold:F3} Hysteresis:{Hysteresis:F3}",
            _options.BullThreshold, _options.BearThreshold, _options.HysteresisBuffer);
    }

    /// <summary>
    /// Make intelligent decision with UTC timing and EST futures market hours awareness
    /// </summary>
    public TradingAction Decide(decimal confidence, int currentPosition, string symbol, DateTime timestampUtc)
    {
        // Ensure UTC timestamp
        var utcTimestamp = timestampUtc.Kind == DateTimeKind.Utc ? timestampUtc : timestampUtc.ToUniversalTime();
        
        // Check EST futures trading hours
        if (!IsWithinFuturesTradingHours(utcTimestamp, symbol))
        {
            _logger.LogDebug("📅 [ENHANCED-DECISION-POLICY] Outside futures trading hours for {Symbol} (EST)", symbol);
            return TradingAction.Hold;
        }

        // Mathematical clamping of confidence
        var clampedConfidence = MathematicalClamp(confidence, 0.0m, 1.0m);
        
        // Check rate limiting with UTC timing
        if (!IsRateLimitAllowed(utcTimestamp))
        {
            _logger.LogDebug("⏱️ [ENHANCED-DECISION-POLICY] Rate limit exceeded (UTC-based), returning HOLD for {Symbol}", symbol);
            return TradingAction.Hold;
        }

        // Get or create symbol state
        var symbolState = _symbolStates.GetOrAdd(symbol, _ => new SymbolDecisionState());

        // Apply position bias with mathematical clamping
        var adjustedThresholds = CalculateAdjustedThresholds(currentPosition);

        // Apply hysteresis with mathematical clamping
        var (bullThreshold, bearThreshold) = ApplyHysteresisWithClamping(symbolState.LastAction, adjustedThresholds);

        // Make decision based on clamped thresholds
        var action = DetermineActionWithClamping(clampedConfidence, bullThreshold, bearThreshold);

        // Update symbol state with UTC timing
        symbolState.LastAction = action;
        symbolState.LastDecisionTimeUtc = utcTimestamp;
        symbolState.LastConfidence = clampedConfidence;

        // Update rate limiting if action taken
        if (action != TradingAction.Hold)
        {
            UpdateRateLimit(utcTimestamp);
        }

        _logger.LogInformation("🎯 [ENHANCED-DECISION-POLICY] {Symbol} UTC:{UtcTime:HH:mm:ss} EST:{EstTime:HH:mm:ss} Confidence:{Confidence:F3} Position:{Position} → {Action} (Thresholds: Bull:{Bull:F3} Bear:{Bear:F3})",
            symbol, utcTimestamp, TimeZoneInfo.ConvertTimeFromUtc(utcTimestamp, EstTimeZone), clampedConfidence, currentPosition, action, bullThreshold, bearThreshold);

        return action;
    }

    /// <summary>
    /// Check if current time is within futures trading hours (EST-based)
    /// </summary>
    private static bool IsWithinFuturesTradingHours(DateTime utcTimestamp, string symbol)
    {
        var estTime = TimeZoneInfo.ConvertTimeFromUtc(utcTimestamp, EstTimeZone);
        var dayOfWeek = estTime.DayOfWeek;
        var timeOfDay = estTime.TimeOfDay;

        // Futures trading hours: Sunday 6:00 PM EST to Friday 5:00 PM EST
        return symbol.ToUpperInvariant() switch
        {
            "ES" or "ESZ25" or "ESH26" or "ESM26" or "ESU26" => IsFuturesHours(dayOfWeek, timeOfDay),
            "MES" or "MESZ25" or "MESH26" or "MESM26" or "MESU26" => IsFuturesHours(dayOfWeek, timeOfDay),
            "NQ" or "NQZ25" or "NQH26" or "NQM26" or "NQU26" => IsFuturesHours(dayOfWeek, timeOfDay),
            "MNQ" or "MNQZ25" or "MNQH26" or "MNQM26" or "MNQU26" => IsFuturesHours(dayOfWeek, timeOfDay),
            _ => true // Default to always allow for non-futures symbols
        };
    }

    private static bool IsFuturesHours(DayOfWeek dayOfWeek, TimeSpan timeOfDay)
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
    /// Mathematical clamping to ensure values stay within bounds
    /// </summary>
    private static decimal MathematicalClamp(decimal value, decimal min, decimal max)
    {
        if (min > max) throw new ArgumentException($"Min ({min}) cannot be greater than max ({max})");
        return Math.Max(min, Math.Min(max, value));
    }

    /// <summary>
    /// Validate and clamp configuration thresholds to prevent invalid states
    /// </summary>
    private void ValidateAndClampThresholds()
    {
        // Clamp bull threshold between 0.51 and 0.95
        _options.BullThreshold = MathematicalClamp(_options.BullThreshold, 0.51m, 0.95m);
        
        // Clamp bear threshold between 0.05 and 0.49
        _options.BearThreshold = MathematicalClamp(_options.BearThreshold, 0.05m, 0.49m);
        
        // Ensure bull > bear with minimum gap
        if (_options.BullThreshold <= _options.BearThreshold)
        {
            _options.BullThreshold = _options.BearThreshold + 0.10m; // Minimum 10% gap
        }
        
        // Clamp hysteresis buffer
        _options.HysteresisBuffer = MathematicalClamp(_options.HysteresisBuffer, 0.005m, 0.05m);
        
        // Clamp rate limiting parameters
        _options.MaxDecisionsPerMinute = Math.Max(1, Math.Min(60, _options.MaxDecisionsPerMinute));
        _options.MinTimeBetweenDecisionsSeconds = Math.Max(1, Math.Min(300, _options.MinTimeBetweenDecisionsSeconds));
    }

    private bool IsRateLimitAllowed(DateTime utcTimestamp)
    {
        lock (_rateLimitLock)
        {
            // Check if we're in a new minute (UTC-based)
            var currentMinute = new DateTime(utcTimestamp.Year, utcTimestamp.Month, utcTimestamp.Day, 
                utcTimestamp.Hour, utcTimestamp.Minute, 0, DateTimeKind.Utc);
            
            if (currentMinute != _currentMinuteStartUtc)
            {
                _currentMinuteStartUtc = currentMinute;
                _decisionsThisMinute = 0;
            }

            // Check decisions per minute limit
            if (_decisionsThisMinute >= _options.MaxDecisionsPerMinute)
            {
                return false;
            }

            // Check minimum time between decisions
            var timeSinceLastDecision = utcTimestamp - _lastDecisionTimeUtc;
            if (timeSinceLastDecision.TotalSeconds < _options.MinTimeBetweenDecisionsSeconds)
            {
                return false;
            }

            return true;
        }
    }

    private void UpdateRateLimit(DateTime utcTimestamp)
    {
        lock (_rateLimitLock)
        {
            _lastDecisionTimeUtc = utcTimestamp;
            _decisionsThisMinute++;
        }
    }

    private (decimal bullThreshold, decimal bearThreshold) CalculateAdjustedThresholds(int currentPosition)
    {
        if (!_options.EnablePositionBias || currentPosition == 0)
        {
            return (_options.BullThreshold, _options.BearThreshold);
        }

        // Apply mathematical clamping to position bias
        var positionBias = MathematicalClamp(Math.Abs(currentPosition), 0, _options.MaxPositionForBias) * 0.01m;

        if (currentPosition > 0) // Long position - make it harder to add more longs
        {
            var adjustedBull = MathematicalClamp(_options.BullThreshold + positionBias, 0.51m, 0.95m);
            return (adjustedBull, _options.BearThreshold);
        }
        else // Short position - make it harder to add more shorts
        {
            var adjustedBear = MathematicalClamp(_options.BearThreshold - positionBias, 0.05m, 0.49m);
            return (_options.BullThreshold, adjustedBear);
        }
    }

    private (decimal bullThreshold, decimal bearThreshold) ApplyHysteresisWithClamping(TradingAction lastAction, (decimal bull, decimal bear) thresholds)
    {
        var (bullThreshold, bearThreshold) = thresholds;

        switch (lastAction)
        {
            case TradingAction.Buy:
                // If last action was BUY, make it harder to switch to SELL
                bearThreshold = MathematicalClamp(bearThreshold - _options.HysteresisBuffer, 0.05m, 0.49m);
                break;
            case TradingAction.Sell:
                // If last action was SELL, make it harder to switch to BUY
                bullThreshold = MathematicalClamp(bullThreshold + _options.HysteresisBuffer, 0.51m, 0.95m);
                break;
            case TradingAction.Hold:
            default:
                // No hysteresis adjustment for HOLD
                break;
        }

        return (bullThreshold, bearThreshold);
    }

    private static TradingAction DetermineActionWithClamping(decimal confidence, decimal bullThreshold, decimal bearThreshold)
    {
        // Ensure thresholds are properly ordered
        var clampedBullThreshold = Math.Max(bearThreshold + 0.01m, bullThreshold);
        var clampedBearThreshold = Math.Min(bullThreshold - 0.01m, bearThreshold);

        if (confidence >= clampedBullThreshold)
        {
            return TradingAction.Buy;
        }
        else if (confidence <= clampedBearThreshold)
        {
            return TradingAction.Sell;
        }
        else
        {
            return TradingAction.Hold; // In the neutral band
        }
    }

    /// <summary>
    /// Get current state for monitoring with UTC timestamps
    /// </summary>
    public EnhancedDecisionPolicyState GetState()
    {
        lock (_rateLimitLock)
        {
            return new EnhancedDecisionPolicyState
            {
                DecisionsThisMinute = _decisionsThisMinute,
                LastDecisionTimeUtc = _lastDecisionTimeUtc,
                CurrentMinuteStartUtc = _currentMinuteStartUtc,
                SymbolStates = _symbolStates.ToDictionary(kvp => kvp.Key, kvp => kvp.Value),
                ClampedBullThreshold = _options.BullThreshold,
                ClampedBearThreshold = _options.BearThreshold,
                ClampedHysteresisBuffer = _options.HysteresisBuffer
            };
        }
    }
}

/// <summary>
/// Enhanced configuration options with mathematical bounds checking
/// </summary>
public class EnhancedDecisionPolicyOptions
{
    public decimal BullThreshold { get; set; } = 0.55m;
    public decimal BearThreshold { get; set; } = 0.45m;
    public decimal HysteresisBuffer { get; set; } = 0.01m;
    public int MaxDecisionsPerMinute { get; set; } = 5;
    public int MinTimeBetweenDecisionsSeconds { get; set; } = 30;
    public bool EnablePositionBias { get; set; } = true;
    public int MaxPositionForBias { get; set; } = 5;
    public bool EnableFuturesHoursFiltering { get; set; } = true;
    public bool EnableUtcTimingOnly { get; set; } = true;
}

/// <summary>
/// Enhanced decision state with UTC timing
/// </summary>
public class EnhancedDecisionPolicyState
{
    public int DecisionsThisMinute { get; set; }
    public DateTime LastDecisionTimeUtc { get; set; }
    public DateTime CurrentMinuteStartUtc { get; set; }
    public Dictionary<string, SymbolDecisionState> SymbolStates { get; set; } = new();
    public decimal ClampedBullThreshold { get; set; }
    public decimal ClampedBearThreshold { get; set; }
    public decimal ClampedHysteresisBuffer { get; set; }
}