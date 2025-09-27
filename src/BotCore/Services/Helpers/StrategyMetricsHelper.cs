using System;
using System.Collections.Generic;
using System.Linq;
using TradingBot.Abstractions;
using Microsoft.Extensions.DependencyInjection;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Shared helper for strategy metrics calculations
    /// Eliminates duplication of strategy-specific switch statements
    /// </summary>
    internal static class StrategyMetricsHelper
    {
        /// <summary>
        /// Get strategy base win rate
        /// Consolidates all strategy-specific win rate logic
        /// </summary>
        public static decimal GetStrategyBaseWinRate(string strategyId)
        {
            return strategyId switch
            {
                "S2" => 0.58m,  // Mean reversion strategy
                "S3" => 0.45m,  // Volatility breakout strategy  
                "S6" => 0.52m,  // Momentum strategy
                "S11" => 0.48m, // Trend following strategy
                _ => 0.50m      // Default balanced win rate
            };
        }

        /// <summary>
        /// Get strategy risk/reward ratio
        /// Consolidates risk/reward calculations
        /// </summary>
        public static decimal GetStrategyRiskRewardRatio(string strategyId)
        {
            return strategyId switch
            {
                "S2" => 1.3m,   // Mean reversion modest R:R
                "S3" => 1.8m,   // Breakout higher R:R
                "S6" => 1.5m,   // Momentum moderate R:R
                "S11" => 2.2m,  // Trend following highest R:R
                _ => 1.5m       // Default moderate R:R
            };
        }

        /// <summary>
        /// Get strategy maximum drawdown ratio
        /// Consolidates drawdown calculations
        /// </summary>
        public static decimal GetStrategyMaxDrawdownRatio(string strategyId)
        {
            return strategyId switch
            {
                "S2" => 0.15m,  // Mean reversion lower drawdown
                "S3" => 0.25m,  // Breakout moderate drawdown
                "S6" => 0.20m,  // Momentum moderate drawdown
                "S11" => 0.30m, // Trend following higher drawdown
                _ => 0.20m      // Default moderate drawdown
            };
        }

        /// <summary>
        /// Get strategy multiplier for performance calculations with null safety
        /// </summary>
        public static decimal GetStrategyMultiplier(string strategyId, IReadOnlyList<TradingBotTuningRunner.ParameterConfig> parameters)
        {
            if (parameters == null || !parameters.Any())
            {
                return 1.0m;
            }

            return strategyId switch
            {
                "S2" => GetS2StrategyMultiplier(parameters),
                "S3" => GetS3StrategyMultiplier(parameters),
                "S6" => GetS6StrategyMultiplier(parameters),
                "S11" => GetS11StrategyMultiplier(parameters),
                _ => 1.0m
            };
        }

        /// <summary>
        /// Get configuration impact on performance with null safety
        /// </summary>
        public static decimal GetConfigurationImpact(IReadOnlyList<TradingBotTuningRunner.ParameterConfig> parameters)
        {
            if (parameters == null || !parameters.Any())
            {
                return 0m;
            }

            decimal impact = 0m;
            
            foreach (var param in parameters)
            {
                if (param?.Key == null) continue;
                
                impact += param.Key switch
                {
                    "sigma_enter" when param.DecimalValue.HasValue => 
                        Math.Max(-0.05m, Math.Min(0.05m, (2.0m - param.DecimalValue.Value) * 0.02m)),
                    "width_rank_threshold" when param.DecimalValue.HasValue => 
                        Math.Max(-0.03m, Math.Min(0.03m, (0.25m - param.DecimalValue.Value) * 0.1m)),
                    _ => 0m
                };
            }
            
            return impact;
        }

        /// <summary>
        /// S2 strategy multiplier calculation with null safety
        /// </summary>
        private static decimal GetS2StrategyMultiplier(IReadOnlyList<TradingBotTuningRunner.ParameterConfig> parameters)
        {
            var sigmaEnter = GetParameterValue(parameters, "sigma_enter", 2.0m);
            var widthRank = GetParameterValue(parameters, "width_rank_threshold", 0.25m);
            return Math.Max(0.5m, Math.Min(2.0m, (sigmaEnter / 2.0m) * (1.0m - widthRank)));
        }

        /// <summary>
        /// S3 strategy multiplier calculation with null safety
        /// </summary>
        private static decimal GetS3StrategyMultiplier(IReadOnlyList<TradingBotTuningRunner.ParameterConfig> parameters)
        {
            var squeezeThreshold = GetParameterValue(parameters, "squeeze_threshold", 0.8m);
            var breakoutConfidence = GetParameterValue(parameters, "breakout_confidence", 0.7m);
            return Math.Max(0.3m, Math.Min(1.8m, squeezeThreshold * breakoutConfidence));
        }

        /// <summary>
        /// S6 strategy multiplier calculation with null safety
        /// </summary>
        private static decimal GetS6StrategyMultiplier(IReadOnlyList<TradingBotTuningRunner.ParameterConfig> parameters)
        {
            var momentumLookback = GetParameterValue(parameters, "momentum_lookback", 20);
            var momentumThreshold = GetParameterValue(parameters, "momentum_threshold", 0.6m);
            return Math.Max(0.4m, Math.Min(1.6m, (momentumLookback / 20.0m) * momentumThreshold));
        }

        /// <summary>
        /// S11 strategy multiplier calculation with null safety
        /// </summary>
        private static decimal GetS11StrategyMultiplier(IReadOnlyList<TradingBotTuningRunner.ParameterConfig> parameters)
        {
            var trendLength = GetParameterValue(parameters, "trend_length", 30);
            var trendStrength = GetParameterValue(parameters, "trend_strength", 0.8m);
            return Math.Max(0.2m, Math.Min(2.5m, (trendLength / 30.0m) * trendStrength));
        }

        /// <summary>
        /// Helper method to safely extract parameter values from configuration with null safety
        /// </summary>
        private static decimal GetParameterValue(IReadOnlyList<TradingBotTuningRunner.ParameterConfig> parameters, string parameterName, decimal defaultValue)
        {
            if (parameters == null || string.IsNullOrEmpty(parameterName))
            {
                return defaultValue;
            }

            foreach (var param in parameters)
            {
                if (param?.Key == parameterName)
                {
                    return param.DecimalValue ?? param.IntValue ?? defaultValue;
                }
            }
            return defaultValue;
        }

        /// <summary>
        /// Helper method to safely extract integer parameter values with null safety
        /// </summary>
        private static int GetParameterValue(IReadOnlyList<TradingBotTuningRunner.ParameterConfig> parameters, string parameterName, int defaultValue)
        {
            if (parameters == null || string.IsNullOrEmpty(parameterName))
            {
                return defaultValue;
            }

            foreach (var param in parameters)
            {
                if (param?.Key == parameterName)
                {
                    return param.IntValue ?? (int)(param.DecimalValue ?? defaultValue);
                }
            }
            return defaultValue;
        }
    }

    /// <summary>
    /// Shared helper for service provider access patterns
    /// Eliminates duplication and improves null safety
    /// </summary>
    internal static class ServiceProviderHelper
    {
        /// <summary>
        /// Safely access a service from the service provider with proper disposal
        /// </summary>
        public static T? GetService<T>(IServiceProvider? serviceProvider) where T : class
        {
            if (serviceProvider == null)
            {
                return null;
            }

            try
            {
                using var scope = serviceProvider.CreateScope();
                return scope.ServiceProvider.GetService<T>();
            }
            catch (ObjectDisposedException)
            {
                // Service provider has been disposed
                return null;
            }
            catch (Exception)
            {
                // Any other exception during service resolution
                return null;
            }
        }

        /// <summary>
        /// Safely execute an action with a service from the service provider
        /// </summary>
        public static TResult ExecuteWithService<TService, TResult>(
            IServiceProvider? serviceProvider, 
            Func<TService, TResult> action, 
            TResult fallbackResult) where TService : class
        {
            if (serviceProvider == null || action == null)
            {
                return fallbackResult;
            }

            try
            {
                using var scope = serviceProvider.CreateScope();
                var service = scope.ServiceProvider.GetService<TService>();
                return service != null ? action(service) : fallbackResult;
            }
            catch (ObjectDisposedException)
            {
                return fallbackResult;
            }
            catch (Exception)
            {
                return fallbackResult;
            }
        }
    }
}