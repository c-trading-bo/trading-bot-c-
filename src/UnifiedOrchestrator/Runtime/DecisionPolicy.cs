using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;
using System.Collections.Concurrent;

namespace TradingBot.UnifiedOrchestrator.Runtime;

/// <summary>
/// Smart decision policy that implements neutral band, hysteresis, and rate limiting
/// to prevent over-trading and provide intelligent HOLD decisions.
/// </summary>
public class DecisionPolicy
{
    private readonly ILogger<DecisionPolicy> _logger;
    private readonly DecisionPolicyOptions _options;
    private readonly ConcurrentDictionary<string, SymbolDecisionState> _symbolStates;
    private readonly object _rateLimitLock = new();
    private DateTime _lastDecisionTime = DateTime.MinValue;
    private int _decisionsThisMinute = 0;
    private DateTime _currentMinuteStart = DateTime.MinValue;

    public DecisionPolicy(ILogger<DecisionPolicy> logger, IOptions<DecisionPolicyOptions> options)
    {
        _logger = logger;
        _options = options.Value;
        _options.Validate();
        _symbolStates = new ConcurrentDictionary<string, SymbolDecisionState>();
        
        _logger.LogInformation("🎯 [DECISION-POLICY] Initialized with Bull:{BullThreshold} Bear:{BearThreshold} Hysteresis:{Hysteresis}",
            _options.BullThreshold, _options.BearThreshold, _options.HysteresisBuffer);
    }

    /// <summary>
    /// Make intelligent decision considering neutral band, hysteresis, and rate limiting
    /// </summary>
    /// <param name="confidence">Final confidence score from trading system</param>
    /// <param name="currentPosition">Current position quantity (positive=long, negative=short, 0=flat)</param>
    /// <param name="symbol">Trading symbol</param>
    /// <param name="timestamp">Decision timestamp</param>
    /// <returns>Trading action: BUY, SELL, or HOLD</returns>
    public TradingAction Decide(decimal confidence, int currentPosition, string symbol, DateTime timestamp)
    {
        // Check rate limiting first
        if (!IsRateLimitAllowed(timestamp))
        {
            _logger.LogDebug("📊 [DECISION-POLICY] Rate limit exceeded, returning HOLD for {Symbol}", symbol);
            return TradingAction.Hold;
        }

        // Get or create symbol state
        var symbolState = _symbolStates.GetOrAdd(symbol, _ => new SymbolDecisionState());

        // Apply position bias if enabled
        var adjustedThresholds = CalculateAdjustedThresholds(currentPosition);

        // Apply hysteresis based on last decision
        var (bullThreshold, bearThreshold) = ApplyHysteresis(symbolState.LastAction, adjustedThresholds);

        // Make decision based on thresholds
        var action = DetermineAction(confidence, bullThreshold, bearThreshold);

        // Update symbol state
        symbolState.LastAction = action;
        symbolState.LastDecisionTime = timestamp;
        symbolState.LastConfidence = confidence;

        // Update rate limiting
        if (action != TradingAction.Hold)
        {
            UpdateRateLimit(timestamp);
        }

        _logger.LogInformation("🎯 [DECISION-POLICY] {Symbol} Confidence:{Confidence:F3} Position:{Position} → {Action} (Thresholds: Bull:{Bull:F3} Bear:{Bear:F3})",
            symbol, confidence, currentPosition, action, bullThreshold, bearThreshold);

        return action;
    }

    private bool IsRateLimitAllowed(DateTime timestamp)
    {
        lock (_rateLimitLock)
        {
            // Check if we're in a new minute
            var currentMinute = new DateTime(timestamp.Year, timestamp.Month, timestamp.Day, timestamp.Hour, timestamp.Minute, 0);
            if (currentMinute != _currentMinuteStart)
            {
                _currentMinuteStart = currentMinute;
                _decisionsThisMinute = 0;
            }

            // Check decisions per minute limit
            if (_decisionsThisMinute >= _options.MaxDecisionsPerMinute)
            {
                return false;
            }

            // Check minimum time between decisions
            var timeSinceLastDecision = timestamp - _lastDecisionTime;
            if (timeSinceLastDecision.TotalSeconds < _options.MinTimeBetweenDecisionsSeconds)
            {
                return false;
            }

            return true;
        }
    }

    private void UpdateRateLimit(DateTime timestamp)
    {
        lock (_rateLimitLock)
        {
            _lastDecisionTime = timestamp;
            _decisionsThisMinute++;
        }
    }

    private (decimal bullThreshold, decimal bearThreshold) CalculateAdjustedThresholds(int currentPosition)
    {
        if (!_options.EnablePositionBias || currentPosition == 0)
        {
            return (_options.BullThreshold, _options.BearThreshold);
        }

        // Adjust thresholds based on position size to avoid over-concentration
        var positionBias = Math.Min(Math.Abs(currentPosition), _options.MaxPositionForBias) * 0.01m;

        if (currentPosition > 0) // Long position - make it harder to add more longs
        {
            return (_options.BullThreshold + positionBias, _options.BearThreshold);
        }
        else // Short position - make it harder to add more shorts
        {
            return (_options.BullThreshold, _options.BearThreshold - positionBias);
        }
    }

    private (decimal bullThreshold, decimal bearThreshold) ApplyHysteresis(TradingAction lastAction, (decimal bull, decimal bear) thresholds)
    {
        var (bullThreshold, bearThreshold) = thresholds;

        switch (lastAction)
        {
            case TradingAction.Buy:
                // If last action was BUY, make it harder to switch to SELL
                bearThreshold -= _options.HysteresisBuffer;
                break;
            case TradingAction.Sell:
                // If last action was SELL, make it harder to switch to BUY
                bullThreshold += _options.HysteresisBuffer;
                break;
            case TradingAction.Hold:
            default:
                // No hysteresis adjustment for HOLD
                break;
        }

        return (bullThreshold, bearThreshold);
    }

    private static TradingAction DetermineAction(decimal confidence, decimal bullThreshold, decimal bearThreshold)
    {
        if (confidence >= bullThreshold)
        {
            return TradingAction.Buy;
        }
        else if (confidence <= bearThreshold)
        {
            return TradingAction.Sell;
        }
        else
        {
            return TradingAction.Hold; // In the neutral band
        }
    }

    /// <summary>
    /// Get current state for monitoring/debugging
    /// </summary>
    public DecisionPolicyState GetState()
    {
        lock (_rateLimitLock)
        {
            return new DecisionPolicyState
            {
                DecisionsThisMinute = _decisionsThisMinute,
                LastDecisionTime = _lastDecisionTime,
                CurrentMinuteStart = _currentMinuteStart,
                SymbolStates = _symbolStates.ToDictionary(kvp => kvp.Key, kvp => kvp.Value)
            };
        }
    }
}

/// <summary>
/// Tracks decision state per symbol for hysteresis calculations
/// </summary>
public class SymbolDecisionState
{
    public TradingAction LastAction { get; set; } = TradingAction.Hold;
    public DateTime LastDecisionTime { get; set; } = DateTime.MinValue;
    public decimal LastConfidence { get; set; } = 0.5m;
}

/// <summary>
/// Current state of the decision policy for monitoring
/// </summary>
public class DecisionPolicyState
{
    public int DecisionsThisMinute { get; set; }
    public DateTime LastDecisionTime { get; set; }
    public DateTime CurrentMinuteStart { get; set; }
    public Dictionary<string, SymbolDecisionState> SymbolStates { get; set; } = new();
}