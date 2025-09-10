using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;
using TradingBot.Infrastructure.TopstepX;

namespace Trading.Safety;

/// <summary>
/// Enforces risk limits and automatically unwinds positions when breached
/// Implements MaxDailyLoss, MaxPositionSize, and DrawdownLimit controls
/// </summary>
public interface IRiskManager
{
    event Action<RiskBreach> OnRiskBreach;
    Task<bool> ValidateOrderAsync(PlaceOrderRequest order);
    Task UpdatePositionAsync(string symbol, decimal currentPrice, int quantity);
    Task UpdateDailyPnLAsync(decimal totalPnL);
    bool IsRiskBreached { get; }
    RiskMetrics GetCurrentMetrics();
}

public record RiskBreach(RiskBreachType Type, string Message, decimal CurrentValue, decimal Limit);
public record RiskMetrics(decimal DailyPnL, decimal MaxDrawdown, decimal LargestPosition, bool IsBreached);

public enum RiskBreachType
{
    MaxDailyLoss,
    MaxPositionSize, 
    DrawdownLimit
}

public class RiskManager : TradingBot.Abstractions.IRiskManager
{
    private readonly ILogger<RiskManager> _logger;
    private readonly AppOptions _config;
    private decimal _dailyPnL = 0;
    private decimal _peakPnL = 0;
    private decimal _maxDrawdown = 0;
    private decimal _largestPosition = 0;
    private bool _isBreached = false;

    public event Action<TradingBot.Abstractions.RiskBreach>? OnRiskBreach;
    public event Action<TradingBot.Abstractions.RiskBreach>? RiskBreachDetected;
    public bool IsRiskBreached => _isBreached;

    public RiskManager(ILogger<RiskManager> logger, IOptions<AppOptions> config)
    {
        _logger = logger;
        _config = config.Value;
    }

    public async Task<TradingBot.Abstractions.RiskAssessment> AssessRiskAsync(TradingBot.Abstractions.TradingDecision decision)
    {
        var riskAssessment = new TradingBot.Abstractions.RiskAssessment
        {
            RiskScore = CalculateRiskScore(decision),
            MaxPositionSize = _config.MaxPositionSize,
            CurrentExposure = Math.Abs(_largestPosition),
            VaR = Math.Abs(_maxDrawdown),
            RiskLevel = _isBreached ? "HIGH" : (_dailyPnL < _config.MaxDailyLoss / 2 ? "MEDIUM" : "LOW"),
            Warnings = new List<string>(),
            Timestamp = DateTime.UtcNow
        };

        // Add warnings based on current risk state
        if (_isBreached)
        {
            riskAssessment.Warnings.Add("Risk breach currently active");
        }
        
        if (_dailyPnL < _config.MaxDailyLoss * 0.8m)
        {
            riskAssessment.Warnings.Add("Approaching maximum daily loss limit");
        }

        return await Task.FromResult(riskAssessment);
    }

    private decimal CalculateRiskScore(TradingBot.Abstractions.TradingDecision decision)
    {
        // Simple risk scoring algorithm
        decimal riskScore = 0.1m; // Base risk

        // Increase risk based on position size
        if (decision.MaxPositionSize > _config.MaxPositionSize * 0.5m)
        {
            riskScore += 0.3m;
        }

        // Increase risk if approaching daily loss limit
        if (_dailyPnL < _config.MaxDailyLoss * 0.5m)
        {
            riskScore += 0.4m;
        }

        // Decrease risk for high confidence signals
        if (decision.Confidence > 0.8m)
        {
            riskScore -= 0.2m;
        }

        return Math.Max(0m, Math.Min(1m, riskScore));
    }

    public async Task<bool> ValidateOrderAsync(PlaceOrderRequest order)
    {
        try
        {
            // Check if already breached
            if (_isBreached)
            {
                _logger.LogWarning("[RISK] Order rejected - risk breach active");
                return false;
            }

            // Validate position size limits
            var newPositionSize = Math.Abs(order.Quantity * order.Price);
            if (newPositionSize > _config.MaxPositionSize)
            {
                var breach = new RiskBreach(
                    RiskBreachType.MaxPositionSize,
                    "Order would exceed maximum position size",
                    newPositionSize,
                    _config.MaxPositionSize
                );
                
                await HandleRiskBreachAsync(breach);
                return false;
            }

            _logger.LogDebug("[RISK] Order validation passed - size: {Size}, limit: {Limit}", 
                newPositionSize, _config.MaxPositionSize);
            
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RISK] Error validating order");
            return false; // Fail safe - reject order on error
        }
    }

    public async Task UpdatePositionAsync(string symbol, decimal currentPrice, int quantity)
    {
        try
        {
            var positionValue = Math.Abs(quantity * currentPrice);
            
            if (positionValue > _largestPosition)
            {
                _largestPosition = positionValue;
                _logger.LogDebug("[RISK] Updated largest position: {Position}", _largestPosition);
            }

            // Check position size breach
            if (positionValue > _config.MaxPositionSize && !_isBreached)
            {
                var breach = new RiskBreach(
                    RiskBreachType.MaxPositionSize,
                    $"Position size breach detected for {symbol}",
                    positionValue,
                    _config.MaxPositionSize
                );
                
                await HandleRiskBreachAsync(breach);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RISK] Error updating position");
        }
    }

    public async Task UpdateDailyPnLAsync(decimal totalPnL)
    {
        try
        {
            _dailyPnL = totalPnL;

            // Track peak for drawdown calculation
            if (totalPnL > _peakPnL)
            {
                _peakPnL = totalPnL;
            }

            // Calculate current drawdown
            var currentDrawdown = _peakPnL - totalPnL;
            if (currentDrawdown > _maxDrawdown)
            {
                _maxDrawdown = currentDrawdown;
            }

            // Check daily loss limit
            if (totalPnL < -_config.MaxDailyLoss && !_isBreached)
            {
                var breach = new RiskBreach(
                    RiskBreachType.MaxDailyLoss,
                    "Daily loss limit exceeded",
                    Math.Abs(totalPnL),
                    _config.MaxDailyLoss
                );
                
                await HandleRiskBreachAsync(breach);
            }

            // Check drawdown limit
            if (currentDrawdown > _config.DrawdownLimit && !_isBreached)
            {
                var breach = new RiskBreach(
                    RiskBreachType.DrawdownLimit,
                    "Maximum drawdown limit exceeded",
                    currentDrawdown,
                    _config.DrawdownLimit
                );
                
                await HandleRiskBreachAsync(breach);
            }

            _logger.LogDebug("[RISK] Updated P&L: {PnL}, Peak: {Peak}, Drawdown: {DD}", 
                totalPnL, _peakPnL, currentDrawdown);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RISK] Error updating daily P&L");
        }
    }

    private Task HandleRiskBreachAsync(RiskBreach breach)
    {
        _isBreached = true;
        
        _logger.LogCritical("[RISK] 🚨 RISK BREACH: {Type} - {Message}. Current: {Current}, Limit: {Limit}",
            breach.Type, breach.Message, breach.CurrentValue, breach.Limit);

        try
        {
            // Log breach to persistent state
            var stateFile = Path.Combine(Directory.GetCurrentDirectory(), "state", "risk_breaches.log");
            Directory.CreateDirectory(Path.GetDirectoryName(stateFile)!);
            
            var logEntry = $"{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss UTC} - {breach.Type}: {breach.Message} " +
                          $"(Current: {breach.CurrentValue}, Limit: {breach.Limit}){Environment.NewLine}";
            File.AppendAllText(stateFile, logEntry);

            // Notify subscribers for safe unwind
            var abstractionsBreach = new TradingBot.Abstractions.RiskBreach
            {
                Type = breach.Type.ToString(),
                Description = breach.Message,
                Message = breach.Message,
                CurrentValue = breach.CurrentValue,
                Limit = breach.Limit,
                Severity = 1.0m, // High severity for any breach
                Timestamp = DateTime.UtcNow,
                Details = new Dictionary<string, object>
                {
                    ["BreachType"] = breach.Type.ToString(),
                    ["OriginalMessage"] = breach.Message
                }
            };
            OnRiskBreach?.Invoke(abstractionsBreach);
            RiskBreachDetected?.Invoke(abstractionsBreach);
            
            _logger.LogInformation("[RISK] Risk breach logged and notifications sent");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RISK] Error handling risk breach");
        }
        
        return Task.CompletedTask;
    }

    public RiskMetrics GetCurrentMetrics()
    {
        return new RiskMetrics(_dailyPnL, _maxDrawdown, _largestPosition, _isBreached);
    }
}