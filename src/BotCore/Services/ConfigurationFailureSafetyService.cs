using System;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Configuration failure safety service with circuit breaker fallbacks
    /// Provides conservative defaults when configuration services fail
    /// </summary>
    public class ConfigurationFailureSafetyService
    {
        private readonly ILogger<ConfigurationFailureSafetyService> _logger;
        private bool _circuitBreakerTripped = false;
        private DateTime _lastFailure = DateTime.MinValue;
        private readonly TimeSpan _circuitBreakerTimeout = TimeSpan.FromMinutes(5);

        // Trading schedule constants
        private const int TradingStartHour = 14;
        private const int TradingStartMinute = 30;
        private const int TradingEndHour = 20;
        private const int TradingEndMinute = 30;

        public ConfigurationFailureSafetyService(ILogger<ConfigurationFailureSafetyService> logger)
        {
            _logger = logger;
        }

        /// <summary>
        /// Execute configuration operation with circuit breaker fallback
        /// </summary>
        public T ExecuteWithFallback<T>(Func<T> configOperation, T conservativeDefault, string operationName)
        {
            if (configOperation is null) throw new ArgumentNullException(nameof(configOperation));
            if (operationName is null) throw new ArgumentNullException(nameof(operationName));
            
            try
            {
                // Check if circuit breaker is active
                if (_circuitBreakerTripped && DateTime.UtcNow - _lastFailure < _circuitBreakerTimeout)
                {
                    _logger.LogWarning("🚨 [CONFIG-SAFETY] Circuit breaker ACTIVE for {Operation}, using conservative default: {Default}", 
                        operationName, conservativeDefault);
                    return conservativeDefault;
                }

                // Attempt configuration operation
                var result = configOperation();
                
                // Reset circuit breaker on success
                if (_circuitBreakerTripped)
                {
                    _logger.LogInformation("✅ [CONFIG-SAFETY] Circuit breaker RESET for {Operation}", operationName);
                    _circuitBreakerTripped = false;
                }
                
                return result;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "🚨 [CONFIG-SAFETY] Configuration failure for {Operation}: {Error}", operationName, ex.Message);
                
                // Trip circuit breaker
                _circuitBreakerTripped = true;
                _lastFailure = DateTime.UtcNow;
                
                _logger.LogWarning("🚨 [CONFIG-SAFETY] Circuit breaker TRIPPED for {Operation}, using conservative default: {Default}", 
                    operationName, conservativeDefault);
                    
                // Raise alert
                RaiseConfigurationAlert(operationName, ex);
                
                return conservativeDefault;
            }
        }

        /// <summary>
        /// Get conservative trading defaults for emergency fallback
        /// </summary>
        public static ConservativeDefaults GetConservativeDefaults()
        {
            return new ConservativeDefaults();
        }

        private void RaiseConfigurationAlert(string operationName, Exception ex)
        {
            var alertMessage = $"CRITICAL CONFIG FAILURE: {operationName} - {ex.Message}";
            
            try
            {
                // Write to critical alert file
                var alertPath = "CRITICAL_ALERT_CONFIG_" + DateTime.UtcNow.ToString("yyyyMMdd_HHmmss") + ".txt";
                System.IO.File.WriteAllText(alertPath, $"{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC - {alertMessage}\n{ex}");
                
                _logger.LogCritical("🚨 [CONFIG-SAFETY] CRITICAL ALERT RAISED: {AlertPath}", alertPath);
            }
            catch (Exception alertEx)
            {
                _logger.LogError(alertEx, "Failed to raise configuration alert for {Operation}", operationName);
            }
        }
    }

    /// <summary>
    /// Conservative default values for emergency fallback
    /// All values are intentionally conservative for safety
    /// </summary>
    public class ConservativeDefaults
    {
        // Trading schedule constants  
        private const int TradingStartHour = 14;
        private const int TradingStartMinute = 30;
        private const int TradingEndHour = 20;
        private const int TradingEndMinute = 30;
        
        // Risk Management - Very conservative
        public decimal MaxPositionSizeMultiplier { get; } = 0.5m;
        public decimal RiskPerTrade { get; } = 100m; // $100 max
        public decimal MaxDailyLoss { get; } = 500m; // $500 max daily loss
        public double AIConfidenceThreshold { get; } = 0.8; // High confidence required
        public double RewardRiskRatioThreshold { get; } = 2.0; // 2:1 minimum R/R

        // Execution - Conservative parameters
        public double MaxSpreadTicks { get; } = 3.0; // Wide spread protection
        public int OrderTimeoutSeconds { get; } = 30; // Quick timeout
        public decimal MaxSlippageTolerance { get; } = 0.5m; // Low slippage tolerance

        // Trading Schedule - Restricted hours
        public TimeSpan TradingStartUtc { get; } = new(TradingStartHour, TradingStartMinute, 0); // 30 min after market open
        public TimeSpan TradingEndUtc { get; } = new(TradingEndHour, TradingEndMinute, 0); // 30 min before market close
        public bool EnablePreMarketTrading { get; } = false;
        public bool EnableAfterHoursTrading { get; } = false;

        // Session Management
        public int MaxConcurrentPositions { get; } = 1; // One position at a time
        public int MaxTradesPerDay { get; } = 5; // Limited trades
        public TimeSpan CooldownBetweenTrades { get; } = TimeSpan.FromMinutes(15);
    }
}