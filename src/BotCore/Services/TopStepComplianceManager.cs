using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace BotCore.Services;

/// <summary>
/// 🛡️ TOPSTEP COMPLIANCE MANAGER 🛡️
/// 
/// Enforces TopStep evaluation account rules to ensure compliance:
/// - Never exceed $2,400 daily loss limit (using $1,000 safety buffer)
/// - Never breach $2,500 total drawdown (using $2,000 safety buffer)
/// - Only trade approved contracts (ES and NQ full-size)
/// - Respect minimum trading days requirements
/// - Track profit targets for evaluation goals
/// - Automatically stop trading if approaching limits
/// 
/// This is critical for maintaining funded account status and passing evaluations.
/// </summary>
public class TopStepComplianceManager
{
    private readonly ILogger _logger;
    private readonly AutonomousConfig _config;
    
    // TopStep compliance limits (with safety buffers)
    private const decimal TopStepDailyLossLimit = -2400m;
    private const decimal SafeDailyLossLimit = -1000m;    // Conservative safety buffer
    private const decimal TopStepDrawdownLimit = -2500m;
    private const decimal SafeDrawdownLimit = -2000m;     // Conservative safety buffer
    
    // Approved contracts for TopStep evaluation
    private readonly string[] ApprovedContracts = { "ES", "NQ" };
    
    // Account state tracking
    private decimal _currentDrawdown = 0m;
    private decimal _todayPnL = 0m;
    private decimal _accountBalance = 50000m; // Starting balance
    private DateTime _lastResetDate = DateTime.Today;
    private int _tradingDaysCompleted = 0;
    private readonly object _stateLock = new();
    
    public TopStepComplianceManager(ILogger logger, IOptions<AutonomousConfig> config)
    {
        _logger = logger;
        _config = config.Value;
        
        _logger.LogInformation("🛡️ [TOPSTEP-COMPLIANCE] Initialized with safety limits: Daily=${DailyLimit}, Drawdown=${DrawdownLimit}",
            SafeDailyLossLimit, SafeDrawdownLimit);
    }
    
    /// <summary>
    /// Check if trading is allowed based on current compliance status
    /// </summary>
    public async Task<bool> CanTradeAsync(decimal currentPnL, decimal accountBalance, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        lock (_stateLock)
        {
            // Update account state
            UpdateAccountState(currentPnL, accountBalance);
            
            // Check daily loss limit
            if (_todayPnL <= SafeDailyLossLimit)
            {
                _logger.LogWarning("🚫 [TOPSTEP-COMPLIANCE] Daily loss limit reached: ${PnL} <= ${Limit}",
                    _todayPnL, SafeDailyLossLimit);
                return false;
            }
            
            // Check total drawdown limit
            if (_currentDrawdown <= SafeDrawdownLimit)
            {
                _logger.LogWarning("🚫 [TOPSTEP-COMPLIANCE] Drawdown limit reached: ${Drawdown} <= ${Limit}",
                    _currentDrawdown, SafeDrawdownLimit);
                return false;
            }
            
            // Check if approaching limits (80% threshold)
            if (_todayPnL <= SafeDailyLossLimit * 0.8m)
            {
                _logger.LogWarning("⚠️ [TOPSTEP-COMPLIANCE] Approaching daily loss limit: ${PnL} (80% of ${Limit})",
                    _todayPnL, SafeDailyLossLimit);
                // Still allow trading but with caution
            }
            
            if (_currentDrawdown <= SafeDrawdownLimit * 0.8m)
            {
                _logger.LogWarning("⚠️ [TOPSTEP-COMPLIANCE] Approaching drawdown limit: ${Drawdown} (80% of ${Limit})",
                    _currentDrawdown, SafeDrawdownLimit);
                // Still allow trading but with caution
            }
            
            return true;
        }
    }
    
    /// <summary>
    /// Check if a specific contract is approved for TopStep evaluation
    /// </summary>
    public bool IsContractApproved(string symbol)
    {
        var approved = Array.Exists(ApprovedContracts, contract => 
            symbol.StartsWith(contract, StringComparison.OrdinalIgnoreCase));
        
        if (!approved)
        {
            _logger.LogWarning("🚫 [TOPSTEP-COMPLIANCE] Contract not approved: {Symbol}. Approved: {Contracts}",
                symbol, string.Join(", ", ApprovedContracts));
        }
        
        return approved;
    }
    
    /// <summary>
    /// Calculate maximum allowed position size based on remaining risk budget
    /// </summary>
    public decimal CalculateMaxAllowedPositionSize(decimal proposedSize, decimal stopDistance)
    {
        lock (_stateLock)
        {
            // Calculate remaining daily risk budget
            var remainingDailyRisk = Math.Abs(SafeDailyLossLimit - _todayPnL);
            
            // Calculate remaining drawdown risk budget
            var remainingDrawdownRisk = Math.Abs(SafeDrawdownLimit - _currentDrawdown);
            
            // Use the smaller of the two limits
            var remainingRisk = Math.Min(remainingDailyRisk, remainingDrawdownRisk);
            
            // Calculate maximum position size based on stop distance
            decimal maxPositionSize = 0m;
            if (stopDistance > 0)
            {
                maxPositionSize = remainingRisk / stopDistance;
            }
            
            // Use the smaller of proposed size or max allowed size
            var allowedSize = Math.Min(proposedSize, maxPositionSize);
            
            _logger.LogDebug("💰 [TOPSTEP-COMPLIANCE] Position sizing: Proposed=${Proposed}, Max=${Max}, Allowed=${Allowed} (Risk budget: ${Budget})",
                proposedSize, maxPositionSize, allowedSize, remainingRisk);
            
            return Math.Max(0, allowedSize);
        }
    }
    
    /// <summary>
    /// Record a trade result and update compliance tracking
    /// </summary>
    public async Task RecordTradeResultAsync(string symbol, decimal pnl, DateTime tradeTime, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        lock (_stateLock)
        {
            // Check if contract was approved
            if (!IsContractApproved(symbol))
            {
                _logger.LogError("❌ [TOPSTEP-COMPLIANCE] Trade recorded for non-approved contract: {Symbol}", symbol);
                // Could trigger compliance violation alert
            }
            
            // Update daily P&L
            if (tradeTime.Date != _lastResetDate)
            {
                // New trading day - reset daily P&L
                _todayPnL = 0m;
                _lastResetDate = tradeTime.Date;
                _tradingDaysCompleted++;
                
                _logger.LogInformation("📅 [TOPSTEP-COMPLIANCE] New trading day: {Date}, Days completed: {Days}",
                    tradeTime.Date.ToString("yyyy-MM-dd"), _tradingDaysCompleted);
            }
            
            _todayPnL += pnl;
            
            // Update drawdown if this is a loss
            if (pnl < 0)
            {
                _currentDrawdown = Math.Min(_currentDrawdown, _todayPnL);
            }
            
            _logger.LogDebug("📊 [TOPSTEP-COMPLIANCE] Trade recorded: {Symbol} ${PnL:F2}, Daily: ${Daily:F2}, Drawdown: ${Drawdown:F2}",
                symbol, pnl, _todayPnL, _currentDrawdown);
        }
        
        // Check for compliance violations outside the lock
        await CheckComplianceViolationsAsync(cancellationToken).ConfigureAwait(false);
    }
    
    /// <summary>
    /// Get current compliance status
    /// </summary>
    public ComplianceStatus GetComplianceStatus()
    {
        lock (_stateLock)
        {
            return new ComplianceStatus
            {
                TodayPnL = _todayPnL,
                CurrentDrawdown = _currentDrawdown,
                AccountBalance = _accountBalance,
                TradingDaysCompleted = _tradingDaysCompleted,
                DailyLossRemaining = Math.Abs(SafeDailyLossLimit - _todayPnL),
                DrawdownRemaining = Math.Abs(SafeDrawdownLimit - _currentDrawdown),
                IsCompliant = _todayPnL > SafeDailyLossLimit && _currentDrawdown > SafeDrawdownLimit,
                DaysUntilMinimum = Math.Max(0, GetMinimumTradingDays() - _tradingDaysCompleted)
            };
        }
    }
    
    /// <summary>
    /// Get profit target for TopStep evaluation
    /// </summary>
    public decimal GetProfitTarget()
    {
        // TopStep evaluation profit targets
        // Evaluation: $3,000 profit target for $50K account
        // Funded: No specific target, but consistent profitability expected
        return 3000m;
    }
    
    /// <summary>
    /// Check if minimum trading days requirement is met
    /// </summary>
    public bool IsMinimumTradingDaysMet()
    {
        var minimumDays = GetMinimumTradingDays();
        return _tradingDaysCompleted >= minimumDays;
    }
    
    /// <summary>
    /// Get time until daily reset (5 PM ET)
    /// </summary>
    public TimeSpan GetTimeUntilDailyReset()
    {
        var easternTime = GetEasternTime();
        var resetTime = easternTime.Date.AddHours(17); // 5 PM ET
        
        if (easternTime.TimeOfDay >= TimeSpan.FromHours(17))
        {
            resetTime = resetTime.AddDays(1); // Next day's reset
        }
        
        return resetTime - easternTime;
    }
    
    /// <summary>
    /// Generate compliance report
    /// </summary>
    public async Task<ComplianceReport> GenerateComplianceReportAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        var status = GetComplianceStatus();
        var profitTarget = GetProfitTarget();
        var progressToTarget = (status.AccountBalance - 50000m) / profitTarget * 100m;
        
        return new ComplianceReport
        {
            Date = DateTime.Today,
            Status = status,
            ProfitTarget = profitTarget,
            ProgressToTarget = Math.Max(0, progressToTarget),
            IsOnTrack = status.IsCompliant && progressToTarget > 0,
            MinimumDaysRemaining = Math.Max(0, GetMinimumTradingDays() - _tradingDaysCompleted),
            Recommendations = GenerateRecommendations(status)
        };
    }
    
    private void UpdateAccountState(decimal currentPnL, decimal accountBalance)
    {
        // Update account balance
        _accountBalance = accountBalance;
        
        // Reset daily P&L if new day
        var today = DateTime.Today;
        if (_lastResetDate != today)
        {
            _todayPnL = 0m;
            _lastResetDate = today;
            _tradingDaysCompleted++;
            
            _logger.LogInformation("📅 [TOPSTEP-COMPLIANCE] Daily reset: {Date}, Trading days: {Days}",
                today.ToString("yyyy-MM-dd"), _tradingDaysCompleted);
        }
        
        _todayPnL = currentPnL;
        
        // Update drawdown tracking
        var accountHighWaterMark = Math.Max(50000m, _accountBalance); // Assuming we track high water mark
        _currentDrawdown = _accountBalance - accountHighWaterMark;
    }
    
    private async Task CheckComplianceViolationsAsync(CancellationToken cancellationToken)
    {
        // Check for hard violations
        if (_todayPnL <= TopStepDailyLossLimit)
        {
            _logger.LogCritical("🚨 [TOPSTEP-COMPLIANCE] VIOLATION: Daily loss limit exceeded: ${PnL} <= ${Limit}",
                _todayPnL, TopStepDailyLossLimit);
            // Could trigger emergency stop
        }
        
        if (_currentDrawdown <= TopStepDrawdownLimit)
        {
            _logger.LogCritical("🚨 [TOPSTEP-COMPLIANCE] VIOLATION: Drawdown limit exceeded: ${Drawdown} <= ${Limit}",
                _currentDrawdown, TopStepDrawdownLimit);
            // Could trigger emergency stop
        }
        
        // Check for approaching violations (90% threshold)
        if (_todayPnL <= TopStepDailyLossLimit * 0.9m)
        {
            _logger.LogError("🔴 [TOPSTEP-COMPLIANCE] CRITICAL: Approaching daily loss limit: ${PnL}",
                _todayPnL);
        }
        
        if (_currentDrawdown <= TopStepDrawdownLimit * 0.9m)
        {
            _logger.LogError("🔴 [TOPSTEP-COMPLIANCE] CRITICAL: Approaching drawdown limit: ${Drawdown}",
                _currentDrawdown);
        }
        
        await Task.CompletedTask.ConfigureAwait(false);
    }
    
    private int GetMinimumTradingDays()
    {
        // TopStep evaluation requirements
        // Evaluation: Minimum 5 trading days
        // Express Funded: Minimum 5 trading days
        return 5;
    }
    
    private DateTime GetEasternTime()
    {
        try
        {
            var easternZone = TimeZoneInfo.FindSystemTimeZoneById("America/New_York");
            return TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, easternZone);
        }
        catch
        {
            // Fallback to UTC-5 (EST) if timezone not found
            return DateTime.UtcNow.AddHours(-5);
        }
    }
    
    private List<string> GenerateRecommendations(ComplianceStatus status)
    {
        var recommendations = new List<string>();
        
        if (!status.IsCompliant)
        {
            recommendations.Add("STOP TRADING: Compliance violation detected");
            return recommendations;
        }
        
        if (status.DailyLossRemaining < 200m)
        {
            recommendations.Add("Reduce position size - approaching daily loss limit");
        }
        
        if (status.DrawdownRemaining < 300m)
        {
            recommendations.Add("Consider defensive trading - approaching drawdown limit");
        }
        
        if (status.DaysUntilMinimum > 0)
        {
            recommendations.Add($"Continue trading for {status.DaysUntilMinimum} more days to meet minimum requirement");
        }
        
        var profitTarget = GetProfitTarget();
        var remainingTarget = profitTarget - (status.AccountBalance - 50000m);
        if (remainingTarget > 0)
        {
            recommendations.Add($"${remainingTarget:F0} remaining to reach profit target");
        }
        
        return recommendations;
    }
}

/// <summary>
/// Current compliance status
/// </summary>
public class ComplianceStatus
{
    public decimal TodayPnL { get; set; }
    public decimal CurrentDrawdown { get; set; }
    public decimal AccountBalance { get; set; }
    public int TradingDaysCompleted { get; set; }
    public decimal DailyLossRemaining { get; set; }
    public decimal DrawdownRemaining { get; set; }
    public bool IsCompliant { get; set; }
    public int DaysUntilMinimum { get; set; }
}

/// <summary>
/// Compliance report for monitoring and analysis
/// </summary>
public class ComplianceReport
{
    public DateTime Date { get; set; }
    public ComplianceStatus Status { get; set; } = new();
    public decimal ProfitTarget { get; set; }
    public decimal ProgressToTarget { get; set; }
    public bool IsOnTrack { get; set; }
    public int MinimumDaysRemaining { get; set; }
    public List<string> Recommendations { get; } = new();
}