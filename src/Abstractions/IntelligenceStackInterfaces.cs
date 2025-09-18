using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Abstractions;

/// <summary>
/// Core interfaces for the complete intelligence stack
/// </summary>

#region Regime Detection & Hysteresis

public interface IRegimeDetector
{
    Task<RegimeState> DetectCurrentRegimeAsync(CancellationToken cancellationToken = default);
    Task<RegimeTransition> CheckTransitionAsync(RegimeState currentState, CancellationToken cancellationToken = default);
    bool IsInDwellPeriod(RegimeState state);
}

public class RegimeState
{
    public RegimeType Type { get; set; }
    public double Confidence { get; set; }
    public DateTime DetectedAt { get; set; } = DateTime.UtcNow;
    public TimeSpan DwellTime { get; set; }
    public Dictionary<string, double> Indicators { get; } = new();
}

public class RegimeTransition
{
    public bool ShouldTransition { get; set; }
    public RegimeType FromRegime { get; set; }
    public RegimeType ToRegime { get; set; }
    public double TransitionConfidence { get; set; }
    public TimeSpan BlendDuration { get; set; }
}

public enum RegimeType
{
    Range = 0,
    Trend = 1, 
    Volatility = 2,
    LowVol = 3,
    HighVol = 4
}

#endregion

#region Feature Store & Training

public interface IFeatureStore
{
    Task<FeatureSet> GetFeaturesAsync(string symbol, DateTime fromTime, DateTime toTime, CancellationToken cancellationToken = default);
    Task SaveFeaturesAsync(FeatureSet features, CancellationToken cancellationToken = default);
    Task<bool> ValidateSchemaAsync(FeatureSet features, CancellationToken cancellationToken = default);
    Task<FeatureSchema> GetSchemaAsync(string version, CancellationToken cancellationToken = default);
}

public class FeatureSet
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public string Version { get; set; } = string.Empty;
    public string SchemaChecksum { get; set; } = string.Empty;
    public Dictionary<string, double> Features { get; } = new();
    public Dictionary<string, object> Metadata { get; } = new();
}

public class FeatureSchema
{
    public string Version { get; set; } = string.Empty;
    public string Checksum { get; set; } = string.Empty;
    public Dictionary<string, FeatureDefinition> Features { get; set; } = new();
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}

public class FeatureDefinition
{
    public string Name { get; set; } = string.Empty;
    public Type DataType { get; set; } = typeof(double);
    public double? MinValue { get; set; }
    public double? MaxValue { get; set; }
    public bool Required { get; set; } = true;
}

#endregion

#region Model Registry

public interface IModelRegistry
{
    Task<ModelArtifact> GetModelAsync(string familyName, string version = "latest", CancellationToken cancellationToken = default);
    Task<ModelArtifact> RegisterModelAsync(ModelRegistration registration, CancellationToken cancellationToken = default);
    Task<bool> PromoteModelAsync(string modelId, PromotionCriteria criteria, CancellationToken cancellationToken = default);
    Task<ModelMetrics> GetModelMetricsAsync(string modelId, CancellationToken cancellationToken = default);
}

public class ModelArtifact
{
    public string Id { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public TimeSpan TrainingWindow { get; set; }
    public string FeaturesVersion { get; set; } = string.Empty;
    public string SchemaChecksum { get; set; } = string.Empty;
    public ModelMetrics Metrics { get; set; } = new();
    public string CalibrationMapId { get; set; } = string.Empty;
    public string RuntimeSignature { get; set; } = string.Empty;
    public string Checksum { get; set; } = string.Empty;
    public byte[]? ModelData { get; set; }
}

public class ModelRegistration
{
    public string FamilyName { get; set; } = string.Empty;
    public TimeSpan TrainingWindow { get; set; }
    public string FeaturesVersion { get; set; } = string.Empty;
    public ModelMetrics Metrics { get; set; } = new();
    public byte[] ModelData { get; set; } = Array.Empty<byte>();
    public Dictionary<string, object> Metadata { get; set; } = new();
}

public class ModelMetrics
{
    public double AUC { get; set; }
    public double PrAt10 { get; set; }
    public double ECE { get; set; }
    public double EdgeBps { get; set; }
    public int SampleSize { get; set; }
    public DateTime ComputedAt { get; set; } = DateTime.UtcNow;
}

public class PromotionCriteria
{
    public double MinAuc { get; set; } = 0.62;
    public double MinPrAt10 { get; set; } = 0.12;
    public double MaxEce { get; set; } = 0.05;
    public double MinEdgeBps { get; set; } = 3.0;
}

#endregion

#region Calibration

public interface ICalibrationManager
{
    Task<CalibrationMap> LoadCalibrationMapAsync(string modelId, CancellationToken cancellationToken = default);
    Task<CalibrationMap> FitCalibrationAsync(string modelId, IEnumerable<CalibrationPoint> points, CancellationToken cancellationToken = default);
    Task<double> CalibrateConfidenceAsync(string modelId, double rawConfidence, CancellationToken cancellationToken = default);
    Task PerformNightlyCalibrationAsync(CancellationToken cancellationToken = default);
}

public class CalibrationMap
{
    public string ModelId { get; set; } = string.Empty;
    public CalibrationMethod Method { get; set; }
    public Dictionary<string, double> Parameters { get; set; } = new();
    public double BrierScore { get; set; }
    public double LogLoss { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}

public class CalibrationPoint
{
    public double RawConfidence { get; set; }
    public bool Outcome { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public double Weight { get; set; } = 1.0;
}

public enum CalibrationMethod
{
    Platt,
    Isotonic,
    Beta
}

#endregion

#region Online Learning

public interface IOnlineLearningSystem
{
    Task UpdateWeightsAsync(string regimeType, Dictionary<string, double> weights, CancellationToken cancellationToken = default);
    Task<Dictionary<string, double>> GetCurrentWeightsAsync(string regimeType, CancellationToken cancellationToken = default);
    Task AdaptToPerformanceAsync(string modelId, ModelPerformance performance, CancellationToken cancellationToken = default);
    Task DetectDriftAsync(string modelId, FeatureSet features, CancellationToken cancellationToken = default);
    Task UpdateModelAsync(TradeRecord tradeRecord, CancellationToken cancellationToken = default);
}

public class ModelPerformance
{
    public string ModelId { get; set; } = string.Empty;
    public double BrierScore { get; set; }
    public double HitRate { get; set; }
    public double Accuracy { get; set; }
    public double Precision { get; set; }
    public double Recall { get; set; }
    public double F1Score { get; set; }
    public double Latency { get; set; }
    public int SampleSize { get; set; }
    public DateTime WindowStart { get; set; }
    public DateTime WindowEnd { get; set; } = DateTime.UtcNow;
    public DateTime LastUpdated { get; set; } = DateTime.UtcNow;
}

public class TradeRecord
{
    public string TradeId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
    public double Quantity { get; set; }
    public double FillPrice { get; set; }
    public DateTime FillTime { get; set; } = DateTime.UtcNow;
    public string StrategyId { get; set; } = string.Empty;
    public Dictionary<string, object> Metadata { get; set; } = new();
}

#endregion

#region Quarantine & Health

public interface IQuarantineManager
{
    Task<QuarantineStatus> CheckModelHealthAsync(string modelId, CancellationToken cancellationToken = default);
    Task QuarantineModelAsync(string modelId, QuarantineReason reason, CancellationToken cancellationToken = default);
    Task<bool> TryRestoreModelAsync(string modelId, CancellationToken cancellationToken = default);
    Task<List<string>> GetQuarantinedModelsAsync(CancellationToken cancellationToken = default);
}

public class QuarantineStatus
{
    public HealthState State { get; set; }
    public string ModelId { get; set; } = string.Empty;
    public QuarantineReason? Reason { get; set; }
    public DateTime? QuarantinedAt { get; set; }
    public int ShadowDecisionCount { get; set; }
    public double BlendWeight { get; set; } = 1.0;
}

public enum HealthState
{
    Healthy,
    Watch,
    Degrade,
    Quarantine
}

public enum QuarantineReason
{
    BrierDeltaTooHigh,
    LatencyTooHigh,
    ExceptionRateTooHigh,
    HitRateTooLow
}

#endregion

#region Decision Logging & Observability

public interface IDecisionLogger
{
    Task LogDecisionAsync(IntelligenceDecision decision, CancellationToken cancellationToken = default);
    Task<List<IntelligenceDecision>> GetDecisionHistoryAsync(DateTime fromTime, DateTime toTime, CancellationToken cancellationToken = default);
}

public class IntelligenceDecision
{
    public string DecisionId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string Symbol { get; set; } = string.Empty;
    public string Timeframe { get; set; } = string.Empty;
    public RegimeType Regime { get; set; }
    public string ModelId { get; set; } = string.Empty;
    public string ModelVersion { get; set; } = string.Empty;
    public string RegimeHead { get; set; } = string.Empty;
    public double Confidence { get; set; }
    public double Uncertainty { get; set; }
    public string FeaturesVersion { get; set; } = string.Empty;
    public string FeaturesHash { get; set; } = string.Empty;
    public RiskState RiskState { get; set; } = new();
    public string Action { get; set; } = string.Empty;
    public double Size { get; set; }
    public string OrderKey { get; set; } = string.Empty;
    public string Route { get; set; } = string.Empty;
    public double LatencyMs { get; set; }
    public List<string> QuarantineFlags { get; } = new();
    public Dictionary<string, object> Metadata { get; } = new();
}

public class RiskState
{
    public double Exposure { get; set; }
    public double DrawdownForecast { get; set; }
    public double Volatility { get; set; }
}

#endregion

#region Startup Self-Tests

public interface IStartupValidator
{
    Task<StartupValidationResult> ValidateSystemAsync(CancellationToken cancellationToken = default);
    Task<bool> ValidateDIGraphAsync(CancellationToken cancellationToken = default);
    Task<bool> ValidateFeatureStoreAsync(CancellationToken cancellationToken = default);
    Task<bool> ValidateModelRegistryAsync(CancellationToken cancellationToken = default);
    Task<bool> ValidateCalibrationAsync(CancellationToken cancellationToken = default);
    Task<bool> ValidateIdempotencyAsync(CancellationToken cancellationToken = default);
    Task<bool> ValidateKillSwitchAsync(CancellationToken cancellationToken = default);
    Task<bool> ValidateLeaderElectionAsync(CancellationToken cancellationToken = default);
}

public class StartupValidationResult
{
    public bool AllTestsPassed { get; set; }
    public bool IsValid { get; set; }
    public Dictionary<string, TestResult> TestResults { get; } = new();
    public List<string> FailureReasons { get; } = new();
    public List<string> ValidationErrors { get; } = new();
    public TimeSpan TotalDuration { get; set; }
}

public class TestResult
{
    public bool Passed { get; set; }
    public string TestName { get; set; } = string.Empty;
    public TimeSpan Duration { get; set; }
    public string? ErrorMessage { get; set; }
    public DateTime ExecutedAt { get; set; } = DateTime.UtcNow;
}

#endregion

#region Leader Election & Distributed Coordination

public interface ILeaderElectionService
{
    Task<bool> TryAcquireLeadershipAsync(CancellationToken cancellationToken = default);
    Task ReleaseLeadershipAsync(CancellationToken cancellationToken = default);
    Task<bool> IsLeaderAsync(CancellationToken cancellationToken = default);
    Task<bool> RenewLeadershipAsync(CancellationToken cancellationToken = default);
    event EventHandler<LeadershipChangedEventArgs> LeadershipChanged;
}

public class LeadershipChangedEventArgs : EventArgs
{
    public bool IsLeader { get; set; }
    public DateTime ChangedAt { get; set; } = DateTime.UtcNow;
    public string? Reason { get; set; }
}

#endregion

#region Idempotent Orders

public interface IIdempotentOrderService
{
    Task<string> GenerateOrderKeyAsync(OrderRequest request, CancellationToken cancellationToken = default);
    Task<bool> IsDuplicateOrderAsync(string orderKey, CancellationToken cancellationToken = default);
    Task RegisterOrderAsync(string orderKey, string orderId, CancellationToken cancellationToken = default);
    Task<OrderDeduplicationResult> CheckDeduplicationAsync(OrderRequest request, CancellationToken cancellationToken = default);
}

public class OrderRequest
{
    public string ModelId { get; set; } = string.Empty;
    public string StrategyId { get; set; } = string.Empty;
    public string SignalId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
    public double Price { get; set; }
    public double Quantity { get; set; }
}

public class OrderDeduplicationResult
{
    public bool IsDuplicate { get; set; }
    public string OrderKey { get; set; } = string.Empty;
    public string? ExistingOrderId { get; set; }
    public DateTime? FirstSeenAt { get; set; }
}

#endregion