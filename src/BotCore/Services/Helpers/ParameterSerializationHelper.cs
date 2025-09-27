using System;
using System.Collections.Generic;
using System.Text.Json;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Services
{
    /// <summary>
    /// Helper for parameter serialization to eliminate duplication
    /// Consolidates JsonSerializer.SerializeToElement patterns
    /// </summary>
    internal static class ParameterSerializationHelper
    {
        /// <summary>
        /// Apply parameter to strategy parameters dictionary with type safety
        /// </summary>
        public static void ApplyParameter(Dictionary<string, JsonElement> strategyParameters, string key, object? value)
        {
            if (strategyParameters == null)
                throw new ArgumentNullException(nameof(strategyParameters));
            
            if (string.IsNullOrEmpty(key))
                throw new ArgumentException("Parameter key cannot be null or empty", nameof(key));

            if (value == null)
                return;

            try
            {
                strategyParameters[key] = value switch
                {
                    decimal decimalValue => JsonSerializer.SerializeToElement(decimalValue),
                    int intValue => JsonSerializer.SerializeToElement(intValue),
                    bool boolValue => JsonSerializer.SerializeToElement(boolValue),
                    string stringValue => JsonSerializer.SerializeToElement(stringValue),
                    double doubleValue => JsonSerializer.SerializeToElement(doubleValue),
                    float floatValue => JsonSerializer.SerializeToElement(floatValue),
                    long longValue => JsonSerializer.SerializeToElement(longValue),
                    _ => JsonSerializer.SerializeToElement(value.ToString())
                };
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to serialize parameter '{key}' with value '{value}'", ex);
            }
        }

        /// <summary>
        /// Apply multiple parameters from a parameter config with null safety
        /// </summary>
        public static void ApplyParametersFromConfig(
            Dictionary<string, JsonElement> strategyParameters, 
            IEnumerable<TradingBotTuningRunner.ParameterConfig> parameters)
        {
            if (strategyParameters == null)
                throw new ArgumentNullException(nameof(strategyParameters));

            if (parameters == null)
                return;

            foreach (var param in parameters)
            {
                if (param?.Key == null)
                    continue;

                try
                {
                    if (param.DecimalValue.HasValue)
                        ApplyParameter(strategyParameters, param.Key, param.DecimalValue.Value);
                    else if (param.IntValue.HasValue)
                        ApplyParameter(strategyParameters, param.Key, param.IntValue.Value);
                    else if (param.BoolValue.HasValue)
                        ApplyParameter(strategyParameters, param.Key, param.BoolValue.Value);
                    else if (param.StringValue != null)
                        ApplyParameter(strategyParameters, param.Key, param.StringValue);
                }
                catch (Exception ex)
                {
                    // Log but continue with other parameters
                    throw new InvalidOperationException($"Failed to apply parameter config for key '{param.Key}'", ex);
                }
            }
        }

        /// <summary>
        /// Create strategy parameters dictionary from parameter configs with validation
        /// </summary>
        public static Dictionary<string, JsonElement> CreateStrategyParameters(IEnumerable<TradingBotTuningRunner.ParameterConfig> parameters)
        {
            var strategyParameters = new Dictionary<string, JsonElement>();
            
            if (parameters != null)
            {
                ApplyParametersFromConfig(strategyParameters, parameters);
            }
            
            return strategyParameters;
        }
    }

    /// <summary>
    /// Helper for configuration key generation to eliminate duplication
    /// </summary>
    internal static class ConfigurationKeyHelper
    {
        /// <summary>
        /// Generate configuration key for symbol-session combination with validation
        /// </summary>
        public static string GetConfigurationKey(string symbol, MarketSession sessionType)
        {
            if (string.IsNullOrWhiteSpace(symbol))
                throw new ArgumentException("Symbol cannot be null or empty", nameof(symbol));

            var sessionCode = GetSessionCode(sessionType);
            return $"{symbol.ToUpperInvariant()}_{sessionCode}";
        }

        /// <summary>
        /// Get session code from MarketSession enum with all cases handled
        /// </summary>
        public static string GetSessionCode(MarketSession sessionType)
        {
            return sessionType switch
            {
                MarketSession.RegularHours => "RTH",
                MarketSession.PostMarket => "AH", 
                MarketSession.PreMarket => "PM",
                MarketSession.Closed => "CLOSED",
                _ => "RTH" // Default to regular hours for unknown session types
            };
        }

        /// <summary>
        /// Parse configuration key back to symbol and session
        /// </summary>
        public static (string Symbol, MarketSession SessionType) ParseConfigurationKey(string configurationKey)
        {
            if (string.IsNullOrWhiteSpace(configurationKey))
                throw new ArgumentException("Configuration key cannot be null or empty", nameof(configurationKey));

            var parts = configurationKey.Split('_');
            if (parts.Length != 2)
                throw new ArgumentException($"Invalid configuration key format: {configurationKey}", nameof(configurationKey));

            var symbol = parts[0];
            var sessionCode = parts[1];

            var sessionType = sessionCode switch
            {
                "RTH" => MarketSession.RegularHours,
                "AH" => MarketSession.PostMarket,
                "PM" => MarketSession.PreMarket,
                "CLOSED" => MarketSession.Closed,
                _ => MarketSession.RegularHours
            };

            return (symbol, sessionType);
        }
    }
}