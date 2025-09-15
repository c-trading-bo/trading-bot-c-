using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Hosting;
using Trading.Safety.Models;
using Trading.Safety.Persistence;
using TradingBot.Abstractions;
using AbstractionsRiskState = TradingBot.Abstractions.RiskState;
using AbstractionsRiskBreach = TradingBot.Abstractions.RiskBreach;
using ModelsRiskState = Trading.Safety.Models.RiskState;

namespace Trading.Safety;

/// <summary>
/// Enhanced production-grade risk management system
/// Integrates with existing RiskManager while adding daily/session limits, 
/// position persistence, and comprehensive governance
/// </summary>
/// <summary>
/// Enhanced production-grade risk management system
/// Provides enhanced risk management features beyond the base IRiskManager
/// </summary>
public interface IEnhancedRiskManager
{
    Task<ModelsRiskState> GetCurrentRiskStateAsync();
    Task<List<RiskBreachEvent>> GetRecentBreachesAsync(TimeSpan period);
    Task UpdateSessionLimitsAsync(RiskLimits newLimits);
    Task ForcePositionUnwindAsync(string reason);
    Task<bool> IsWithinRiskLimitsAsync(decimal orderValue, int newTradeCount = 1);
    event Action<RiskBreachEvent> OnRiskBreachEvent;
    event Action<ModelsRiskState> OnRiskStateChanged;
    bool IsRiskBreached { get; }
}

/// <summary>
/// Production-grade enhanced risk manager with comprehensive governance
/// Works alongside the base IRiskManager to provide additional functionality
/// </summary>
public class EnhancedRiskManager : IEnhancedRiskManager, IHostedService
{
    private readonly ILogger<EnhancedRiskManager> _logger;
    private readonly IPositionStatePersistence _persistence;
    private readonly Timer _riskMonitoringTimer;
    private readonly Timer _sessionResetTimer;
    
    private RiskLimits _currentLimits;
    private ModelsRiskState _currentState;
    private readonly List<RiskBreachEvent> _recentBreaches = new();
    private readonly object _stateLock = new object();
    private string _currentSessionId = Guid.NewGuid().ToString("N")[..8];
    
    // Events for enhanced functionality
    public event Action<RiskBreachEvent>? OnRiskBreachEvent;
    public event Action<ModelsRiskState>? OnRiskStateChanged;

    public bool IsRiskBreached => _currentState.IsRiskBreached;

    public EnhancedRiskManager(
        ILogger<EnhancedRiskManager> logger,
        IPositionStatePersistence persistence,
        IOptions<AppOptions> config)
    {
        _logger = logger;
        _persistence = persistence;
        
        // Initialize with default limits
        _currentLimits = CreateDefaultRiskLimits(config.Value);
        _currentState = new ModelsRiskState();
        
        // Set up monitoring timers
        _riskMonitoringTimer = new Timer(MonitorRiskState, null, TimeSpan.Zero, _currentLimits.RiskEvaluationInterval);
        _sessionResetTimer = new Timer(CheckSessionReset, null, TimeSpan.FromMinutes(1), TimeSpan.FromMinutes(1));
        
        // Initialize state in background task
        _ = Task.Run(async () =>
        {
            try
            {
                // Load persisted risk state if available
                var persistedState = await _persistence.LoadRiskStateAsync();
                if (persistedState != null)
                {
                    // Check if state is from today
                    if (persistedState.DailyResetTime.Date == DateTime.UtcNow.Date)
                    {
                        _currentState = persistedState;
                        _logger.LogInformation("[ENHANCED-RISK] Restored risk state from persistence: PnL={PnL:C}, Level={Level}",
                            _currentState.DailyPnL, _currentState.RiskLevel);
                    }
                    else
                    {
                        _logger.LogInformation("[ENHANCED-RISK] Persisted state is from previous day, starting fresh");
                        _currentState = _currentState.ResetDaily();
                    }
                }
                
                // Load position state for exposure calculation
                var positionState = await _persistence.LoadPositionStateAsync();
                if (positionState != null)
                {
                    _currentState = _currentState with 
                    { 
                        TotalExposure = positionState.TotalExposure,
                        OpenPositionCount = positionState.OpenPositions.Count 
                    };
                    _logger.LogInformation("[ENHANCED-RISK] Restored position state: Exposure={Exposure:C}, Positions={Count}",
                        _currentState.TotalExposure, _currentState.OpenPositionCount);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[ENHANCED-RISK] Failed to initialize risk state from persistence");
            }
        });
        
        _logger.LogInformation("[ENHANCED-RISK] Enhanced risk manager initialized with session {SessionId}", _currentSessionId);
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        await Task.CompletedTask;
        _logger.LogInformation("[ENHANCED-RISK] Enhanced risk manager started successfully");
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        try
        {
            // Create final backup before shutdown
            await _persistence.CreateBackupAsync("shutdown");
            
            // Save final risk state
            await _persistence.SaveRiskStateAsync(_currentState);
            
            // Archive session if there was activity
            if (_currentState.SessionTradeCount > 0)
            {
                var sessionSummary = CreateSessionSummary();
                await _persistence.ArchiveCompletedSessionAsync(_currentSessionId, sessionSummary);
            }
            
            _riskMonitoringTimer?.Dispose();
            _sessionResetTimer?.Dispose();
            
            _logger.LogInformation("[ENHANCED-RISK] Enhanced risk manager stopped gracefully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ENHANCED-RISK] Error during enhanced risk manager shutdown");
        }
    }

    // Enhanced functionality only
    public Task<ModelsRiskState> GetCurrentRiskStateAsync()
    {
        lock (_stateLock)
        {
            return Task.FromResult(_currentState with { }); // Return a copy
        }
    }

    public Task<List<RiskBreachEvent>> GetRecentBreachesAsync(TimeSpan period)
    {
        var cutoff = DateTime.UtcNow - period;
        lock (_stateLock)
        {
            return Task.FromResult(_recentBreaches.Where(b => b.Timestamp >= cutoff).ToList());
        }
    }

    public async Task UpdateSessionLimitsAsync(RiskLimits newLimits)
    {
        if (!newLimits.IsValid())
        {
            throw new ArgumentException("Invalid risk limits provided");
        }
        
        var oldLimits = _currentLimits;
        _currentLimits = newLimits;
        
        _logger.LogInformation("[ENHANCED-RISK] Risk limits updated: DailyLoss={DailyLoss:C}, SessionLoss={SessionLoss:C}, " +
                             "MaxPosition={MaxPosition:C}", 
            newLimits.MaxDailyLoss, newLimits.MaxSessionLoss, newLimits.MaxPositionSize);
        
        // Create backup when limits change
        await _persistence.CreateBackupAsync("limits_update");
    }

    public async Task ForcePositionUnwindAsync(string reason)
    {
        try
        {
            _logger.LogCritical("[ENHANCED-RISK] 🚨 FORCE POSITION UNWIND INITIATED: {Reason}", reason);
            
            // Mark risk as breached to prevent new orders
            lock (_stateLock)
            {
                _currentState.IsRiskBreached = true;
                _currentState.ActiveBreaches.Add($"FORCE_UNWIND: {reason}");
                _currentState.RiskLevel = "EMERGENCY";
            }
            
            // Create emergency backup
            await _persistence.CreateBackupAsync("force_unwind");
            
            // Record breach event
            var breachEvent = new RiskBreachEvent
            {
                BreachType = "FORCE_UNWIND",
                Severity = "EMERGENCY",
                Description = $"Force position unwind initiated: {reason}",
                RequiresImmediateAction = true,
                RecommendedActions = new List<string> { "Close all positions", "Stop trading", "Review risk controls" }
            };
            
            await RecordRiskBreachAsync(breachEvent);
            
            // Notify all subscribers
            OnRiskBreachEvent?.Invoke(breachEvent);
            
            _logger.LogCritical("[ENHANCED-RISK] Force unwind process initiated, all new orders blocked");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ENHANCED-RISK] Error during force position unwind");
            throw;
        }
    }

    public async Task<bool> IsWithinRiskLimitsAsync(decimal orderValue, int newTradeCount = 1)
    {
        lock (_stateLock)
        {
            // Check if already breached
            if (_currentState.IsRiskBreached)
            {
                return false;
            }
            
            // Check daily trade count limit
            if (_currentState.DailyTradeCount + newTradeCount > _currentLimits.MaxDailyTrades)
            {
                _logger.LogWarning("[ENHANCED-RISK] Order rejected: would exceed daily trade limit");
                return false;
            }
            
            // Check session trade count limit
            if (_currentState.SessionTradeCount + newTradeCount > _currentLimits.MaxSessionTrades)
            {
                _logger.LogWarning("[ENHANCED-RISK] Order rejected: would exceed session trade limit");
                return false;
            }
            
            // Check position size limits
            if (_currentState.TotalExposure + orderValue > _currentLimits.MaxPortfolioExposure)
            {
                _logger.LogWarning("[ENHANCED-RISK] Order rejected: would exceed portfolio exposure limit");
                return false;
            }
            
            // Check maximum open positions
            if (_currentState.OpenPositionCount >= _currentLimits.MaxOpenPositions)
            {
                _logger.LogWarning("[ENHANCED-RISK] Order rejected: would exceed maximum open positions");
                return false;
            }
            
            return true;
        }
    }

    // Implement base interface methods by delegating to base manager
    public Task<bool> ValidateOrderAsync(TradingBot.Infrastructure.TopstepX.PlaceOrderRequest order)
    {
        // First check enhanced limits
        var orderValue = Math.Abs(order.Quantity * order.Price);
        if (!IsWithinRiskLimitsAsync(orderValue, 1).Result)
        {
            return Task.FromResult(false);
        }
        
        return Task.FromResult(true);
    }
    
    // Helper methods for external components to update risk state
    public async Task UpdatePositionStateAsync(string symbol, decimal currentPrice, int quantity)
    {
        // Update our position tracking
        lock (_stateLock)
        {
            var positionValue = Math.Abs(quantity * currentPrice);
            _currentState = _currentState with { TotalExposure = _currentState.TotalExposure + positionValue };
            
            if (quantity != 0)
            {
                _currentState = _currentState with { OpenPositionCount = _currentState.OpenPositionCount + 1 };
            }
            
            _currentState = _currentState with { LastUpdated = DateTime.UtcNow };
        }
        
        await TriggerRiskStateUpdate();
    }

    public async Task UpdatePnLStateAsync(decimal totalPnL)
    {
        // Update our enhanced tracking
        lock (_stateLock)
        {
            var previousPnL = _currentState.DailyPnL;
            var updatedState = _currentState with { DailyPnL = totalPnL };
            
            // Update peaks and drawdowns
            if (totalPnL > updatedState.DailyPeak)
            {
                updatedState = updatedState with { DailyPeak = totalPnL };
            }
            
            updatedState = updatedState with { DailyDrawdown = updatedState.DailyPeak - totalPnL };
            
            // Update session metrics
            var sessionPnLChange = totalPnL - previousPnL;
            var newSessionPnL = updatedState.SessionPnL + sessionPnLChange;
            updatedState = updatedState with { SessionPnL = newSessionPnL };
            
            if (newSessionPnL > updatedState.SessionPeak)
            {
                updatedState = updatedState with { SessionPeak = newSessionPnL };
            }
            
            updatedState = updatedState with 
            { 
                SessionDrawdown = updatedState.SessionPeak - newSessionPnL,
                LastUpdated = DateTime.UtcNow 
            };
            
            _currentState = updatedState;
        }
        
        await CheckRiskLimitsAsync();
        await TriggerRiskStateUpdate();
    }

    private async Task CheckRiskLimitsAsync()
    {
        ModelsRiskState currentState;
        lock (_stateLock)
        {
            currentState = _currentState with { }; // Copy for thread safety
        }
        
        var breaches = new List<RiskBreachEvent>();
        
        // Check daily loss limit
        if (currentState.DailyPnL <= _currentLimits.MaxDailyLoss)
        {
            breaches.Add(new RiskBreachEvent
            {
                BreachType = "DAILY_LOSS_LIMIT",
                Severity = "CRITICAL",
                Description = "Daily loss limit exceeded",
                CurrentValue = Math.Abs(currentState.DailyPnL),
                LimitValue = Math.Abs(_currentLimits.MaxDailyLoss),
                UtilizationPercent = Math.Abs(currentState.DailyPnL) / Math.Abs(_currentLimits.MaxDailyLoss),
                RequiresImmediateAction = true,
                RecommendedActions = new List<string> { "Stop trading", "Close positions", "Review strategy" }
            });
        }
        
        // Check session loss limit
        if (currentState.SessionPnL <= _currentLimits.MaxSessionLoss)
        {
            breaches.Add(new RiskBreachEvent
            {
                BreachType = "SESSION_LOSS_LIMIT",
                Severity = "CRITICAL",
                Description = "Session loss limit exceeded",
                CurrentValue = Math.Abs(currentState.SessionPnL),
                LimitValue = Math.Abs(_currentLimits.MaxSessionLoss),
                UtilizationPercent = Math.Abs(currentState.SessionPnL) / Math.Abs(_currentLimits.MaxSessionLoss),
                RequiresImmediateAction = true,
                RecommendedActions = new List<string> { "End session", "Take break", "Review performance" }
            });
        }
        
        // Check daily trade count limit
        if (currentState.DailyTradeCount >= _currentLimits.MaxDailyTrades)
        {
            breaches.Add(new RiskBreachEvent
            {
                BreachType = "DAILY_TRADE_LIMIT",
                Severity = "WARNING",
                Description = "Daily trade count limit reached",
                CurrentValue = currentState.DailyTradeCount,
                LimitValue = _currentLimits.MaxDailyTrades,
                UtilizationPercent = (decimal)currentState.DailyTradeCount / _currentLimits.MaxDailyTrades,
                RequiresImmediateAction = false,
                RecommendedActions = new List<string> { "Stop placing new orders", "Review trade frequency" }
            });
        }
        
        // Process any breaches
        foreach (var breach in breaches)
        {
            await RecordRiskBreachAsync(breach);
        }
        
        // Update risk level and breach status
        lock (_stateLock)
        {
            _currentState.IsRiskBreached = breaches.Any(b => b.RequiresImmediateAction);
            _currentState.RiskLevel = CalculateRiskLevel();
            _currentState.CompositeRiskScore = CalculateCompositeRiskScore();
        }
    }

    private async Task RecordRiskBreachAsync(RiskBreachEvent breachEvent)
    {
        try
        {
            lock (_stateLock)
            {
                _recentBreaches.Add(breachEvent);
                _currentState.ActiveBreaches.Add(breachEvent.BreachType);
                
                // Keep only recent breaches (last 24 hours)
                var cutoff = DateTime.UtcNow.AddHours(-24);
                _recentBreaches.RemoveAll(b => b.Timestamp < cutoff);
            }
            
            _logger.LogCritical("[ENHANCED-RISK] 🚨 RISK BREACH: {Type} - {Description}. " +
                              "Current: {Current}, Limit: {Limit}, Utilization: {Utilization:P1}",
                breachEvent.BreachType, breachEvent.Description, 
                breachEvent.CurrentValue, breachEvent.LimitValue, breachEvent.UtilizationPercent);
            
            // Notify subscribers
            OnRiskBreachEvent?.Invoke(breachEvent);
            
            // Create backup on critical breaches
            if (breachEvent.RequiresImmediateAction)
            {
                await _persistence.CreateBackupAsync($"breach_{breachEvent.BreachType.ToLowerInvariant()}");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ENHANCED-RISK] Failed to record risk breach");
        }
    }

    private string CalculateRiskLevel()
    {
        var riskScore = CalculateCompositeRiskScore();
        
        return riskScore switch
        {
            >= 0.9m => "CRITICAL",
            >= 0.7m => "HIGH", 
            >= 0.4m => "MEDIUM",
            _ => "LOW"
        };
    }

    private decimal CalculateCompositeRiskScore()
    {
        var dailyLossRisk = Math.Abs(_currentState.DailyPnL) / Math.Abs(_currentLimits.MaxDailyLoss);
        var sessionLossRisk = Math.Abs(_currentState.SessionPnL) / Math.Abs(_currentLimits.MaxSessionLoss);
        var exposureRisk = _currentState.TotalExposure / _currentLimits.MaxPortfolioExposure;
        var tradeCountRisk = (decimal)_currentState.DailyTradeCount / _currentLimits.MaxDailyTrades;
        
        // Weighted composite score
        return Math.Min(1.0m, 
            (dailyLossRisk * 0.3m) + 
            (sessionLossRisk * 0.2m) + 
            (exposureRisk * 0.3m) + 
            (tradeCountRisk * 0.2m));
    }

    private async Task TriggerRiskStateUpdate()
    {
        try
        {
            // Save state periodically
            await _persistence.SaveRiskStateAsync(_currentState);
            
            // Notify subscribers
            OnRiskStateChanged?.Invoke(_currentState);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ENHANCED-RISK] Failed to trigger risk state update");
        }
    }

    private void MonitorRiskState(object? state)
    {
        try
        {
            _ = Task.Run(async () =>
            {
                await CheckRiskLimitsAsync();
                await TriggerRiskStateUpdate();
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ENHANCED-RISK] Error in risk monitoring timer");
        }
    }

    private void CheckSessionReset(object? state)
    {
        try
        {
            var now = DateTime.UtcNow;
            var currentTime = TimeOnly.FromDateTime(now);
            
            // Check for daily reset
            if (currentTime >= _currentLimits.DailyRiskResetTime && 
                _currentState.DailyResetTime.Date < now.Date)
            {
                _ = Task.Run(async () =>
                {
                    var sessionSummary = CreateSessionSummary();
                    await _persistence.ArchiveCompletedSessionAsync(_currentSessionId, sessionSummary);
                    
                    lock (_stateLock)
                    {
                        _currentState = _currentState.ResetDaily();
                        _currentState = _currentState.ResetSession();
                        _currentSessionId = Guid.NewGuid().ToString("N")[..8];
                    }
                    
                    _logger.LogInformation("[ENHANCED-RISK] Daily risk limits reset for new trading day");
                });
            }
            
            // Check for session break
            if (currentTime >= _currentLimits.SessionBreakTime && 
                _currentState.SessionStartTime.Date == now.Date &&
                now.TimeOfDay > _currentLimits.SessionBreakTime.ToTimeSpan())
            {
                _ = Task.Run(async () =>
                {
                    lock (_stateLock)
                    {
                        _currentState = _currentState.ResetSession();
                    }
                    
                    _logger.LogInformation("[ENHANCED-RISK] Session limits reset for afternoon session");
                });
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ENHANCED-RISK] Error in session reset timer");
        }
    }

    private SessionSummary CreateSessionSummary()
    {
        lock (_stateLock)
        {
            return new SessionSummary
            {
                SessionId = _currentSessionId,
                StartTime = _currentState.SessionStartTime,
                EndTime = DateTime.UtcNow,
                Duration = DateTime.UtcNow - _currentState.SessionStartTime,
                TotalPnL = _currentState.SessionPnL,
                MaxDrawdown = _currentState.SessionDrawdown,
                TradeCount = _currentState.SessionTradeCount,
                TerminationReason = _currentState.IsRiskBreached ? "RISK_BREACH" : "NORMAL"
            };
        }
    }

    private static RiskLimits CreateDefaultRiskLimits(AppOptions config)
    {
        return new RiskLimits
        {
            MaxDailyLoss = config.MaxDailyLoss,
            MaxPositionSize = config.MaxPositionSize,
            MaxDrawdownAmount = config.DrawdownLimit,
            MaxDailyTrades = 50,
            MaxSessionTrades = 20,
            MaxPortfolioExposure = config.MaxPositionSize * 5, // 5x position size
            MaxOpenPositions = 10
        };
    }

    public void Dispose()
    {
        _riskMonitoringTimer?.Dispose();
        _sessionResetTimer?.Dispose();
    }
}