using System.ComponentModel.DataAnnotations;

namespace BotCore.Configuration
{
    /// <summary>
    /// Configuration for model versioning and validation system
    /// Ensures each training run produces different weights and proper versioning
    /// </summary>
    public class ModelVersioningConfiguration
    {
        /// <summary>
        /// Enable model version tracking
        /// </summary>
        public bool EnableVersionTracking { get; set; } = true;

        /// <summary>
        /// Require version differences between training runs
        /// </summary>
        public bool RequireVersionDifference { get; set; } = true;

        /// <summary>
        /// Minimum weight change percentage to consider models different
        /// </summary>
        [Range(0.01, 10.0)]
        public double MinWeightChangePct { get; set; } = 0.1;

        /// <summary>
        /// Enable model hash validation for integrity checking
        /// </summary>
        public bool ModelHashValidation { get; set; } = true;

        /// <summary>
        /// Enable detailed training metadata logging
        /// </summary>
        public bool TrainingMetadataLogging { get; set; } = true;

        /// <summary>
        /// Enable version comparison logging
        /// </summary>
        public bool VersionComparisonLogging { get; set; } = true;

        /// <summary>
        /// Model registry configuration
        /// </summary>
        public ModelRegistryConfiguration ModelRegistry { get; set; } = new();
    }

    /// <summary>
    /// Model registry configuration for versioning and storage
    /// </summary>
    public class ModelRegistryConfiguration
    {
        /// <summary>
        /// Enable registry validation
        /// </summary>
        public bool EnableRegistryValidation { get; set; } = true;

        /// <summary>
        /// Automatically backup models before overwriting
        /// </summary>
        public bool AutoBackupModels { get; set; } = true;

        /// <summary>
        /// Maximum number of model versions to keep in history
        /// </summary>
        [Range(10, 1000)]
        public int MaxVersionHistory { get; set; } = 50;

        /// <summary>
        /// Base directory for model storage
        /// </summary>
        public string BaseDirectory { get; set; } = "./models";

        /// <summary>
        /// Backup directory for model versions
        /// </summary>
        public string BackupDirectory { get; set; } = "./models/backup";
    }

    /// <summary>
    /// Data flow enhancement configuration for robust market data handling
    /// </summary>
    public class DataFlowEnhancementConfiguration
    {
        /// <summary>
        /// Enable snapshot data requests after subscriptions
        /// </summary>
        public bool EnableSnapshotRequests { get; set; } = true;

        /// <summary>
        /// Delay before requesting snapshot data (milliseconds)
        /// </summary>
        [Range(1000, 30000)]
        public int SnapshotRequestDelay { get; set; } = 5000;

        /// <summary>
        /// Enable automatic contract rollover detection
        /// </summary>
        public bool EnableContractRollover { get; set; } = true;

        /// <summary>
        /// Days before expiration to switch to front month
        /// </summary>
        [Range(1, 30)]
        public int ContractRolloverDays { get; set; } = 7;

        /// <summary>
        /// Health monitoring configuration
        /// </summary>
        public HealthMonitoringConfiguration HealthMonitoring { get; set; } = new();

        /// <summary>
        /// Front month contract mapping
        /// </summary>
        public Dictionary<string, string> FrontMonthMapping { get; set; } = new()
        {
            {"ES", "ESZ25"},
            {"NQ", "NQZ25"},
            {"MES", "MESZ25"},
            {"MNQ", "MNQZ25"}
        };

        /// <summary>
        /// Get current front month contract for a symbol
        /// </summary>
        public string GetFrontMonthContract(string symbol)
        {
            return FrontMonthMapping.TryGetValue(symbol.ToUpper(), out var contract) ? contract : symbol;
        }
    }

    /// <summary>
    /// Health monitoring configuration for data flow reliability
    /// </summary>
    public class HealthMonitoringConfiguration
    {
        /// <summary>
        /// Enable data flow health monitoring
        /// </summary>
        public bool EnableDataFlowMonitoring { get; set; } = true;

        /// <summary>
        /// Timeout for silent feed detection (seconds)
        /// </summary>
        [Range(30, 600)]
        public int SilentFeedTimeoutSeconds { get; set; } = 60;

        /// <summary>
        /// Enable automatic recovery from data flow issues
        /// </summary>
        public bool AutoRecoveryEnabled { get; set; } = true;

        /// <summary>
        /// Maximum number of recovery attempts
        /// </summary>
        [Range(1, 10)]
        public int MaxRecoveryAttempts { get; set; } = 3;

        /// <summary>
        /// Delay between recovery attempts (seconds)
        /// </summary>
        [Range(10, 300)]
        public int RecoveryDelaySeconds { get; set; } = 30;

        /// <summary>
        /// Health check interval (seconds)
        /// </summary>
        [Range(10, 300)]
        public int HealthCheckIntervalSeconds { get; set; } = 60;
    }

    /// <summary>
    /// Walk-forward validation configuration for robust ML model validation
    /// </summary>
    public class WalkForwardValidationConfiguration
    {
        /// <summary>
        /// Enable walk-forward validation
        /// </summary>
        public bool EnableWalkForward { get; set; } = true;

        /// <summary>
        /// Validation window size in days
        /// </summary>
        [Range(7, 365)]
        public int ValidationWindowDays { get; set; } = 30;

        /// <summary>
        /// Training window size in days
        /// </summary>
        [Range(30, 1095)]
        public int TrainingWindowDays { get; set; } = 90;

        /// <summary>
        /// Step size for walk-forward in days
        /// </summary>
        [Range(1, 30)]
        public int StepSizeDays { get; set; } = 7;

        /// <summary>
        /// Minimum validation samples required
        /// </summary>
        [Range(100, 10000)]
        public int MinValidationSamples { get; set; } = 1000;

        /// <summary>
        /// Seed rotation configuration
        /// </summary>
        public SeedRotationConfiguration SeedRotation { get; set; } = new();

        /// <summary>
        /// Performance threshold configuration
        /// </summary>
        public PerformanceThresholdsConfiguration PerformanceThresholds { get; set; } = new();
    }

    /// <summary>
    /// Seed rotation configuration for proper randomization
    /// </summary>
    public class SeedRotationConfiguration
    {
        /// <summary>
        /// Enable seed rotation to ensure different training runs
        /// </summary>
        public bool EnableSeedRotation { get; set; } = true;

        /// <summary>
        /// Frequency of seed changes
        /// </summary>
        public string SeedChangeFrequency { get; set; } = "Daily"; // Daily, Hourly, PerRun

        /// <summary>
        /// Base seed value
        /// </summary>
        [Range(1, 1000000)]
        public int BaseSeed { get; set; } = 42;

        /// <summary>
        /// Maximum range for seed variation
        /// </summary>
        [Range(100, 100000)]
        public int MaxSeedRange { get; set; } = 10000;

        /// <summary>
        /// Generate a new seed based on configuration
        /// </summary>
        public int GenerateNewSeed()
        {
            var random = new Random(BaseSeed + DateTime.UtcNow.DayOfYear);
            return random.Next(BaseSeed, BaseSeed + MaxSeedRange);
        }
    }

    /// <summary>
    /// Performance thresholds for model validation
    /// </summary>
    public class PerformanceThresholdsConfiguration
    {
        /// <summary>
        /// Minimum Sharpe ratio required
        /// </summary>
        [Range(0.0, 5.0)]
        public double MinSharpeRatio { get; set; } = 0.5;

        /// <summary>
        /// Maximum drawdown percentage allowed
        /// </summary>
        [Range(1.0, 50.0)]
        public double MaxDrawdownPct { get; set; } = 5.0;

        /// <summary>
        /// Minimum win rate required
        /// </summary>
        [Range(0.1, 0.9)]
        public double MinWinRate { get; set; } = 0.45;

        /// <summary>
        /// Minimum number of trades for validation
        /// </summary>
        [Range(10, 10000)]
        public int MinTrades { get; set; } = 50;
    }
}