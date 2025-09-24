using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using BotCore.Services;

namespace BotCore.References;

/// <summary>
/// Integration reference showing how to use the new neutral band decision policy
/// and enhanced session-aware runtime gates in trading applications
/// </summary>
public class EnhancedTradingIntegrationReference
{
    private readonly SafeHoldDecisionPolicy _safeHoldPolicy;
    private readonly SessionAwareRuntimeGates _sessionGates;
    private readonly ILogger<EnhancedTradingIntegrationReference> _logger;

    public EnhancedTradingIntegrationReference(
        SafeHoldDecisionPolicy safeHoldPolicy,
        SessionAwareRuntimeGates sessionGates,
        ILogger<EnhancedTradingIntegrationReference> logger)
    {
        _safeHoldPolicy = safeHoldPolicy;
        _sessionGates = sessionGates;
        _logger = logger;
    }

    /// <summary>
    /// Reference of complete trading decision workflow with neutral band and session awareness
    /// </summary>
    public async Task<bool> EvaluateCompleteTradingDecisionAsync(
        string symbol,
        string strategyId,
        double mlConfidence,
        CancellationToken cancellationToken = default)
    {
        try
        {
            // STEP 1: Check session-aware runtime gates first
            var tradingAllowed = await _sessionGates.IsTradingAllowedAsync(symbol, cancellationToken).ConfigureAwait(false);
            if (!tradingAllowed)
            {
                var sessionStatus = _sessionGates.GetSessionStatus();
                _logger.LogInformation("[TRADING_GATE] Trading blocked by session gates: {Session} " +
                                     "RTH={IsRth} ETH={IsEth} Maintenance={IsMaintenanceBreak} Weekend={IsWeekendClosed} " +
                                     "SundayReopen={IsSundayReopenCurb} EthFirstMins={IsEthFirstMinsCurb}",
                    sessionStatus.CurrentSession, sessionStatus.IsRth, sessionStatus.IsEth,
                    sessionStatus.IsMaintenanceBreak, sessionStatus.IsWeekendClosed,
                    sessionStatus.IsSundayReopenCurb, sessionStatus.IsEthFirstMinsCurb);

                // Log reopen curb details if applicable
                if (sessionStatus.IsWithinReopenCurbWindow && sessionStatus.ReopenCurbTimeRemaining.HasValue)
                {
                    _logger.LogInformation("[REOPEN_CURB] Trading blocked by reopen curb window. Time remaining: {TimeRemaining}",
                        sessionStatus.ReopenCurbTimeRemaining.Value);
                }

                return false;
            }

            // STEP 2: Apply neutral band decision policy
            var decision = await _safeHoldPolicy.EvaluateDecisionAsync(mlConfidence, symbol, strategyId, cancellationToken).ConfigureAwait(false);

            // STEP 3: Log decision with complete context
            _logger.LogInformation("[COMPLETE_DECISION] {Symbol} {Strategy}: confidence={Confidence:F3} → {Action} ({Reason})",
                symbol, strategyId, mlConfidence, decision.Action, decision.Reason);

            // STEP 4: Return trading permission based on decision
            switch (decision.Action)
            {
                case TradingAction.BUY:
                case TradingAction.SELL:
                    _logger.LogInformation("[TRADING_APPROVED] {Symbol} {Strategy}: {Action} signal approved with confidence {Confidence:F3}",
                        symbol, strategyId, decision.Action, mlConfidence);
                    return true;

                case TradingAction.HOLD:
                    _logger.LogDebug("[TRADING_HOLD] {Symbol} {Strategy}: HOLD signal - confidence in neutral band",
                        symbol, strategyId);
                    return false;

                default:
                    _logger.LogWarning("[TRADING_UNKNOWN] {Symbol} {Strategy}: Unknown decision action {Action}",
                        symbol, strategyId, decision.Action);
                    return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[TRADING_ERROR] Error evaluating trading decision for {Symbol} {Strategy}",
                symbol, strategyId);
            return false;
        }
    }

    /// <summary>
    /// Reference showing session-specific risk parameter adjustment
    /// </summary>
    public async Task<decimal> GetSessionAdjustedPositionSizeAsync(
        string symbol,
        decimal basePositionSize,
        CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);

        var sessionStatus = _sessionGates.GetSessionStatus();
        var adjustedSize = basePositionSize;

        // Apply session-specific position sizing adjustments
        if (sessionStatus.IsRth)
        {
            // Regular Trading Hours - use full position size
            adjustedSize = basePositionSize;
            _logger.LogDebug("[POSITION_SIZING] {Symbol} RTH: Using full position size {Size}",
                symbol, adjustedSize);
        }
        else if (sessionStatus.IsEth && sessionStatus.TradingAllowed)
        {
            // Extended Trading Hours - reduce position size by 25%
            adjustedSize = basePositionSize * 0.75m;
            _logger.LogDebug("[POSITION_SIZING] {Symbol} ETH: Reduced position size {OriginalSize} → {AdjustedSize}",
                symbol, basePositionSize, adjustedSize);
        }
        else
        {
            // Market closed or restricted - no trading
            adjustedSize = 0m;
            _logger.LogDebug("[POSITION_SIZING] {Symbol} CLOSED: No trading allowed - position size = 0",
                symbol);
        }

        return adjustedSize;
    }

    /// <summary>
    /// Reference showing neutral band statistics monitoring
    /// </summary>
    public void LogNeutralBandStatistics()
    {
        var stats = _safeHoldPolicy.GetNeutralBandStats();
        
        _logger.LogInformation("[NEUTRAL_BAND_STATS] " +
                             "Bearish threshold: {BearishThreshold:F3} " +
                             "Bullish threshold: {BullishThreshold:F3} " +
                             "Band width: {BandWidth:F3} " +
                             "Hysteresis: {Hysteresis} " +
                             "Buffer: {Buffer:F3}",
            stats.BearishThreshold, stats.BullishThreshold, stats.NeutralBandWidth,
            stats.EnableHysteresis, stats.HysteresisBuffer);
    }

    /// <summary>
    /// Reference showing enhanced reopen curbing logic
    /// </summary>
    public async Task MonitorReopenCurbStatusAsync(CancellationToken cancellationToken = default)
    {
        while (!cancellationToken.IsCancellationRequested)
        {
            var sessionStatus = _sessionGates.GetSessionStatus();
            
            if (sessionStatus.IsWithinReopenCurbWindow)
            {
                var timeRemaining = sessionStatus.ReopenCurbTimeRemaining;
                _logger.LogInformation("[REOPEN_CURB_MONITOR] Active curb window. Time remaining: {TimeRemaining}. " +
                                     "ETH curb: {EthCurb}, Sunday curb: {SundayReopen}",
                    timeRemaining, sessionStatus.IsEthFirstMinsCurb, sessionStatus.IsSundayReopenCurb);
            }

            await Task.Delay(TimeSpan.FromMinutes(1), cancellationToken).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Reference showing integration with existing systems
    /// Demonstrates how CloudRlTrainerV2 and BotSupervisor can use the new components
    /// </summary>
    public async Task<string> GetComprehensiveSystemStatusAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);

        var sessionStatus = _sessionGates.GetSessionStatus();
        var neutralBandStats = _safeHoldPolicy.GetNeutralBandStats();

        return $@"
ENHANCED TRADING SYSTEM STATUS
════════════════════════════════════════════════════════════════

📅 SESSION STATUS
Current Session: {sessionStatus.CurrentSession}
Trading Allowed: {sessionStatus.TradingAllowed}
Eastern Time: {sessionStatus.EasternTime:yyyy-MM-dd HH:mm:ss}
Next Change: {sessionStatus.NextSessionChange?.ToString("HH:mm:ss") ?? "N/A"}

🕐 SESSION DETAILS
RTH (9:30-16:00): {sessionStatus.IsRth}
ETH: {sessionStatus.IsEth}
Maintenance Break: {sessionStatus.IsMaintenanceBreak}
Weekend Closed: {sessionStatus.IsWeekendClosed}

🚫 REOPEN CURBS
Within Curb Window: {sessionStatus.IsWithinReopenCurbWindow}
Sunday Reopen Curb: {sessionStatus.IsSundayReopenCurb}
ETH First Minutes Curb: {sessionStatus.IsEthFirstMinsCurb}
Time Remaining: {sessionStatus.ReopenCurbTimeRemaining?.ToString(@"hh\:mm\:ss") ?? "N/A"}

📊 NEUTRAL BAND CONFIGURATION
Bearish Threshold: {neutralBandStats.BearishThreshold:F3} (45%)
Bullish Threshold: {neutralBandStats.BullishThreshold:F3} (55%)
Neutral Band Width: {neutralBandStats.NeutralBandWidth:F3}
Hysteresis Enabled: {neutralBandStats.EnableHysteresis}
Hysteresis Buffer: {neutralBandStats.HysteresisBuffer:F3}

✅ SYSTEM INTEGRATION STATUS
Session-Aware Runtime Gates: ACTIVE
Neutral Band Decision Policy: ACTIVE
Enhanced Reopen Curbing: ACTIVE
CloudRlTrainerV2 Integration: READY
Neural UCB Bracket Selection: ENABLED

════════════════════════════════════════════════════════════════
";
    }
}

/// <summary>
/// Extension methods for easy service registration and integration
/// </summary>
public static class EnhancedTradingIntegrationExtensions
{
    /// <summary>
    /// Register all enhanced trading integration services
    /// </summary>
    public static IServiceCollection AddEnhancedTradingIntegration(this IServiceCollection services)
    {
        services.AddSingleton<SessionAwareRuntimeGates>();
        services.AddSingleton<SafeHoldDecisionPolicy>();
        services.AddSingleton<EnhancedTradingIntegrationReference>();
        
        return services;
    }

    /// <summary>
    /// Quick check if confidence is in neutral band
    /// </summary>
    public static bool IsNeutral(this double confidence, double bearishThreshold = 0.45, double bullishThreshold = 0.55)
    {
        return confidence > bearishThreshold && confidence < bullishThreshold;
    }

    /// <summary>
    /// Get trading action from confidence with neutral band logic
    /// </summary>
    public static TradingAction GetTradingAction(this double confidence, double bearishThreshold = 0.45, double bullishThreshold = 0.55)
    {
        if (confidence <= bearishThreshold)
            return TradingAction.SELL;
        if (confidence >= bullishThreshold)
            return TradingAction.BUY;
        return TradingAction.HOLD;
    }
}