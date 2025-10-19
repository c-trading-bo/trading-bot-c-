using System;
using System.Collections.Generic;
using TradingBot.Abstractions;

namespace TradingBot.BotCore.Models
{
    /// <summary>
    /// Result of seed apply operation.
    /// </summary>
    public class SeedApplyResult
    {
        public bool Success { get; set; }
        public string ErrorMessage { get; set; } = string.Empty;
        public List<HistoricalBar> Bars { get; set; } = new();
        public SeedValidationResult? ValidationResult { get; set; }

        public static SeedApplyResult CreateSuccess(List<HistoricalBar> bars, SeedValidationResult validation)
        {
            return new SeedApplyResult
            {
                Success = true,
                Bars = bars,
                ValidationResult = validation
            };
        }

        public static SeedApplyResult Failed(string errorMessage)
        {
            return new SeedApplyResult
            {
                Success = false,
                ErrorMessage = errorMessage
            };
        }
    }

    /// <summary>
    /// Seed data container.
    /// </summary>
    public class SeedData
    {
        public List<HistoricalBar> Bars { get; set; } = new();
    }

    /// <summary>
    /// Seed validation result.
    /// </summary>
    public class SeedValidationResult
    {
        public bool Passed { get; set; }
        public List<string> Errors { get; set; } = new();
        public int BarCount { get; set; }
        public DateTime OldestBar { get; set; }
        public DateTime NewestBar { get; set; }
        public bool HasGaps { get; set; }
        public bool HasDuplicates { get; set; }
        public bool VolumeValid { get; set; }
        
        // Backward compatibility properties (aliases)
        public int DuplicateTimestamps => HasDuplicates ? 1 : 0;
        public int InvalidVolumes => VolumeValid ? 0 : 1;
        public int TimeGaps => HasGaps ? 1 : 0;
    }
}
