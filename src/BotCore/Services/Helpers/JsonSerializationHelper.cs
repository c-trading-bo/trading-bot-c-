using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace TradingBot.BotCore.Services.Helpers
{
    /// <summary>
    /// Centralized JSON serialization helper to eliminate duplication
    /// Provides standardized JsonSerializerOptions and serialization methods
    /// </summary>
    internal static class JsonSerializationHelper
    {
        /// <summary>
        /// Standard JSON options for compact serialization (logging, network)
        /// </summary>
        public static readonly JsonSerializerOptions CompactOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            PropertyNameCaseInsensitive = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        /// <summary>
        /// Standard JSON options for pretty-printed serialization (files, debugging)
        /// </summary>
        public static readonly JsonSerializerOptions PrettyOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            PropertyNameCaseInsensitive = true,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            WriteIndented = true
        };

        /// <summary>
        /// Serialize object to JSON string using compact format
        /// </summary>
        public static string SerializeCompact<T>(T obj)
        {
            return JsonSerializer.Serialize(obj, CompactOptions);
        }

        /// <summary>
        /// Serialize object to JSON string using pretty format
        /// </summary>
        public static string SerializePretty<T>(T obj)
        {
            return JsonSerializer.Serialize(obj, PrettyOptions);
        }

        /// <summary>
        /// Deserialize JSON string to object
        /// </summary>
        public static T? Deserialize<T>(string json)
        {
            return JsonSerializer.Deserialize<T>(json, CompactOptions);
        }

        /// <summary>
        /// Try to deserialize JSON string to object, returning default on failure
        /// </summary>
        public static T? TryDeserialize<T>(string json, T? defaultValue = default)
        {
            try
            {
                return JsonSerializer.Deserialize<T>(json, CompactOptions);
            }
            catch (JsonException)
            {
                return defaultValue;
            }
        }
    }
}