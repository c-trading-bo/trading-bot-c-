using System.ComponentModel.DataAnnotations;

namespace BotCore.Configuration
{
    /// <summary>
    /// Configuration for model versioning and validation system
    /// Ensures each training run produces different weights and proper versioning
    /// </summary>
    public class ModelVersioningConfiguration
    {
            public bool EnableVersionTracking { get; set; } = true;

        /// <summary>
        /// Require version differences between training runs
        /// </summary>
        public bool RequireVersionDifference { get; set; } = true;

        // Validation constants for S109 compliance
        private const double MinWeightChangePercent = 0.01;
        private const double MaxWeightChangePercent = 10.0;

        /// <summary>
        /// Minimum weight change percentage to consider models different
        /// </summary>
        [Range(MinWeightChangePercent, MaxWeightChangePercent)]
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
        // Validation constants for S109 compliance
        private const int MinVersionHistoryCount = 10;
        private const int MaxVersionHistoryCount = 1000;

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
        [Range(MinVersionHistoryCount, MaxVersionHistoryCount)]
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
        // Validation constants for S109 compliance
        private const int MinSnapshotDelayMs = 1000;
        private const int MaxSnapshotDelayMs = 30000;
        private const int MinRolloverDays = 1;
        private const int MaxRolloverDays = 30;

        /// <summary>
        /// Enable snapshot data requests after subscriptions
        /// </summary>
        public bool EnableSnapshotRequests { get; set; } = true;

        /// <summary>
        /// Delay before requesting snapshot data (milliseconds)
        /// </summary>
        [Range(MinSnapshotDelayMs, MaxSnapshotDelayMs)]
        public int SnapshotRequestDelay { get; set; } = 5000;

        /// <summary>
        /// Enable automatic contract rollover detection
        /// </summary>
        public bool EnableContractRollover { get; set; } = true;

        /// <summary>
        /// Days before expiration to switch to front month
        /// </summary>
        [Range(MinRolloverDays, MaxRolloverDays)]
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
            {"NQ", "NQZ25"}
        };

        /// <summary>
        /// Get current front month contract for a symbol
        /// </summary>
        public string GetFrontMonthContract(string symbol)
        {
            if (symbol is null) throw new ArgumentNullException(nameof(symbol));
            
            return FrontMonthMapping.TryGetValue(symbol.ToUpper(), out var contract) ? contract : symbol;
        }
    }

    /// <summary>
    /// Health monitoring configuration for data flow reliability
    /// </summary>
    public class HealthMonitoringConfiguration
    {
        // Validation constants for S109 compliance
        private const int MinSilentFeedTimeoutSeconds = 30;
        private const int MaxSilentFeedTimeoutSeconds = 600;
        private const int MinHeartbeatTimeoutSeconds = 5;
        private const int MaxHeartbeatTimeoutSeconds = 60;
        private const int MinRecoveryAttempts = 1;
        private const int MaxRecoveryAttempts = 10;
        private const int MinRecoveryDelaySeconds = 10;
        private const int MaxRecoveryDelaySeconds = 300;
        private const int MinHealthCheckIntervalSeconds = 10;
        private const int MaxHealthCheckIntervalSeconds = 300;

        /// <summary>
        /// Enable data flow health monitoring
        /// </summary>
        public bool EnableDataFlowMonitoring { get; set; } = true;

        /// <summary>
        /// Timeout for silent feed detection (seconds)
        /// </summary>
        [Range(MinSilentFeedTimeoutSeconds, MaxSilentFeedTimeoutSeconds)]
        public int SilentFeedTimeoutSeconds { get; set; } = 60;

        /// <summary>
        /// Heartbeat timeout for immediate recovery (seconds)
        /// </summary>
        [Range(MinHeartbeatTimeoutSeconds, MaxHeartbeatTimeoutSeconds)]
        public int HeartbeatTimeoutSeconds { get; set; } = 15;

        /// <summary>
        /// Enable automatic recovery from data flow issues
        /// </summary>
        public bool AutoRecoveryEnabled { get; set; } = true;

        /// <summary>
        /// Maximum number of recovery attempts
        /// </summary>
        [Range(MinRecoveryAttempts, MaxRecoveryAttempts)]
        public int MaxRecoveryAttempts { get; set; } = 3;

        /// <summary>
        /// Delay between recovery attempts (seconds)
        /// </summary>
        [Range(MinRecoveryDelaySeconds, MaxRecoveryDelaySeconds)]
        public int RecoveryDelaySeconds { get; set; } = 30;

        /// <summary>
        /// Health check interval (seconds)
        /// </summary>
        [Range(MinHealthCheckIntervalSeconds, MaxHealthCheckIntervalSeconds)]
        public int HealthCheckIntervalSeconds { get; set; } = 60;
    }

    /// <summary>
    /// Walk-forward validation configuration for robust ML model validation
    /// </summary>
    public class WalkForwardValidationConfiguration
    {
        // Validation constants for S109 compliance
        private const int MinValidationWindowDays = 7;
        private const int MaxValidationWindowDays = 365;
        private const int MinTrainingWindowDays = 30;
        private const int MaxTrainingWindowDays = 1095;
        private const int MinStepSizeDays = 1;
        private const int MaxStepSizeDays = 30;
        private const int MinValidationSamples = 100;
        private const int MaxValidationSamples = 10000;

        /// <summary>
        /// Enable walk-forward validation
        /// </summary>
        public bool EnableWalkForward { get; set; } = true;

        /// <summary>
        /// Validation window size in days
        /// </summary>
        [Range(MinValidationWindowDays, MaxValidationWindowDays)]
        public int ValidationWindowDays { get; set; } = 30;

        /// <summary>
        /// Training window size in days
        /// </summary>
        [Range(MinTrainingWindowDays, MaxTrainingWindowDays)]
        public int TrainingWindowDays { get; set; } = 90;

        /// <summary>
        /// Step size for walk-forward in days
        /// </summary>
        [Range(MinStepSizeDays, MaxStepSizeDays)]
        public int StepSizeDays { get; set; } = 7;

        /// <summary>
        /// Minimum validation samples required
        /// </summary>
        [Range(MinValidationSamples, MaxValidationSamples)]
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
        // Validation constants for S109 compliance
        private const int MinBaseSeed = 1;
        private const int MaxBaseSeed = 1000000;
        private const int MinSeedRange = 100;
        private const int MaxSeedRange = 100000;

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
        [Range(MinBaseSeed, MaxBaseSeed)]
        public int BaseSeed { get; set; } = 42;

        /// <summary>
        /// Maximum range for seed variation
        /// </summary>
        [Range(MinSeedRange, MaxSeedRange)]
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
        // Validation constants for S109 compliance
        private const double MinSharpeRatioThreshold = 0.0;
        private const double MaxSharpeRatioThreshold = 5.0;
        private const double MinDrawdownPctThreshold = 1.0;
        private const double MaxDrawdownPctThreshold = 50.0;
        private const double MinWinRateThreshold = 0.1;
        private const double MaxWinRateThreshold = 0.9;
        private const int MinTradesThreshold = 10;
        private const int MaxTradesThreshold = 10000;

        /// <summary>
        /// Minimum Sharpe ratio required
        /// </summary>
        [Range(MinSharpeRatioThreshold, MaxSharpeRatioThreshold)]
        public double MinSharpeRatio { get; set; } = 0.5;

        /// <summary>
        /// Maximum drawdown percentage allowed
        /// </summary>
        [Range(MinDrawdownPctThreshold, MaxDrawdownPctThreshold)]
        public double MaxDrawdownPct { get; set; } = 5.0;

        /// <summary>
        /// Minimum win rate required
        /// </summary>
        [Range(MinWinRateThreshold, MaxWinRateThreshold)]
        public double MinWinRate { get; set; } = 0.45;

        /// <summary>
        /// Minimum number of trades for validation
        /// </summary>
        [Range(MinTradesThreshold, MaxTradesThreshold)]
        public int MinTrades { get; set; } = 50;
    }
}