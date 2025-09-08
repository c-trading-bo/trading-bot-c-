using System;

namespace BotCore.Utilities
{
    /// <summary>
    /// Generates deterministic, human-readable strategy and execution IDs
    /// Replaces GetHashCode() usage with predictable, traceable identifiers
    /// </summary>
    public static class DeterministicIdGenerator
    {
        /// <summary>
        /// Generate a deterministic strategy ID based on strategy name and date
        /// Replaces: strategy.GetHashCode() usage
        /// </summary>
        public static string GenerateStrategyId(string strategyName, DateTime? date = null)
        {
            var targetDate = date ?? DateTime.UtcNow;
            return $"{strategyName}_{targetDate:yyyyMMdd}";
        }
        
        /// <summary>
        /// Generate a deterministic execution ID for order tracking
        /// Format: STRATEGY_SYMBOL_YYYYMMDD_HHMMSS_NNN
        /// </summary>
        public static string GenerateExecutionId(string strategyId, string symbol, DateTime? timestamp = null)
        {
            var targetTime = timestamp ?? DateTime.UtcNow;
            var sequence = targetTime.Millisecond % 1000; // 3-digit sequence
            
            return $"{strategyId}_{symbol}_{targetTime:yyyyMMdd}_{targetTime:HHmmss}_{sequence:D3}";
        }
        
        /// <summary>
        /// Generate a deterministic session ID for trading sessions
        /// </summary>
        public static string GenerateSessionId(string sessionType, DateTime? startTime = null)
        {
            var targetTime = startTime ?? DateTime.UtcNow;
            return $"{sessionType}_{targetTime:yyyyMMdd_HHmm}";
        }
        
        /// <summary>
        /// Generate deterministic feature hash for ML models
        /// Replaces: features.GetHashCode() for consistent model input tracking
        /// </summary>
        public static string GenerateFeatureHash(params object[] features)
        {
            var combined = string.Join("|", features);
            
            // Use a simple but deterministic hash
            uint hash = 2166136261u; // FNV-1a initial value
            
            foreach (byte b in System.Text.Encoding.UTF8.GetBytes(combined))
            {
                hash ^= b;
                hash *= 16777619u; // FNV-1a prime
            }
            
            return $"FEAT_{hash:X8}";
        }
        
        /// <summary>
        /// Generate deterministic order tag for TopstepX
        /// Format: STRAT_SYM_YYYYMMDD_HHMMSS
        /// </summary>
        public static string GenerateOrderTag(string strategyId, string symbol, DateTime? timestamp = null)
        {
            var targetTime = timestamp ?? DateTime.UtcNow;
            return $"{strategyId}_{symbol}_{targetTime:yyyyMMdd}_{targetTime:HHmmss}".ToUpperInvariant();
        }
    }
}