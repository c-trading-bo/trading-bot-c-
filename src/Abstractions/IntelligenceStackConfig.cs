using System.Text.Json.Serialization;

namespace TradingBot.Abstractions;

/// <summary>
/// Configuration for the complete intelligence stack
/// </summary>
public class IntelligenceStackConfig
{
    public bool EnableTrading { get; set; }
    public MLConfig ML { get; set; } = new();
    public OnlineConfig Online { get; set; } = new();
    public RLConfig RL { get; set; } = new();
    public OrdersConfig Orders { get; set; } = new();
    public OrchestratorConfig Orchestrator { get; set; } = new();
    public SloConfig SLO { get; set; } = new();
    public ObservabilityConfig Observability { get; set; } = new();
    public PromotionsConfig Promotions { get; set; } = new();
    public HistoricalConfig Historical { get; set; } = new();
    public NetworkConfig Network { get; set; } = new();
}

public class MLConfig
{
    public ConfidenceConfig Confidence { get; set; } = new();
    public RegimeConfig Regime { get; set; } = new();
    public EnsembleConfig Ensemble { get; set; } = new();
    public QuarantineConfig Quarantine { get; set; } = new();
    public CalibrationConfig Calibration { get; set; } = new();
    public ShadowConfig Shadow { get; set; } = new();
    public CanaryConfig Canary { get; set; } = new();
}

public class ConfidenceConfig
{
    public bool Enabled { get; set; } = true;
    public double MinConfidence { get; set; } = 0.52;
    public double NeutralConfidenceMin { get; set; } = 0.45;
    public double NeutralConfidenceMax { get; set; } = 0.55;
    public double KellyClip { get; set; } = 0.35;
    
    // Configurable scoring functions to replace hardcoded formulas
    public double EdgeConversionOffset { get; set; } = 0.5;
    public double EdgeConversionMultiplier { get; set; } = 2.0;
    public double ConfidenceMultiplierOffset { get; set; } = 0.5;
    public double ConfidenceMultiplierScale { get; set; } = 4.0;
}

public class RegimeConfig
{
    public HysteresisConfig Hysteresis { get; set; } = new();
}

public class HysteresisConfig
{
    public bool Enabled { get; set; } = true;
    public double VolLowIn { get; set; } = 0.8;
    public double VolLowOut { get; set; } = 0.9;
    public double VolHighIn { get; set; } = 1.4;
    public double VolHighOut { get; set; } = 1.3;
    public double TrendIn { get; set; } = 0.8;
    public double TrendOut { get; set; } = 0.6;
    public int DwellSeconds { get; set; } = 180;
    public double MomentumBiasPct { get; set; } = 0.2;
}

public class EnsembleConfig
{
    public MetaPerRegimeConfig MetaPerRegime { get; set; } = new();
}

public class MetaPerRegimeConfig
{
    public bool Enabled { get; set; } = true;
    public int TransitionBlendSeconds { get; set; } = 60;
}

public class QuarantineConfig
{
    public bool Enabled { get; set; } = true;
    public double WatchBrierDelta { get; set; } = 0.02;
    public double DegradeBrierDelta { get; set; } = 0.03;
    public double QuarantineBrierDelta { get; set; } = 0.05;
    public int LatencyP99Ms { get; set; } = 200;
    public double ExceptionRatePerMin { get; set; } = 0.005;
    public int ShadowDecisionsForReentry { get; set; } = 500;
    
    // Configurable latency multipliers for degradation thresholds
    public double LatencyDegradeMultiplier { get; set; } = 2.0;
}

public class CalibrationConfig
{
    public bool Enabled { get; set; } = true;
    public string Nightly { get; set; } = "02:30";
}

public class ShadowConfig
{
    public bool Enabled { get; set; } = true;
}

public class CanaryConfig
{
    public int Percent { get; set; }
    public int MaxPercent { get; set; } = 20;
}

public class OnlineConfig
{
    public MetaLearningConfig MetaLearning { get; set; } = new();
    public DriftConfig Drift { get; set; } = new();
    public TuningConfig Tuning { get; set; } = new();
}

public class MetaLearningConfig
{
    public bool Enabled { get; set; } = true;
    public int MaxWeightChangePctPer5Min { get; set; } = 10;
    public double RollbackVarMultiplier { get; set; } = 2.5;
}

public class DriftConfig
{
    public string Detector { get; set; } = "adwin";
    public double Delta { get; set; } = 0.002;
    public int PostDriftVetoSignals { get; set; } = 10;
}

public class TuningConfig
{
    public bool Enabled { get; set; } = true;
    public int Trials { get; set; } = 50;
    public int EarlyStopNoImprove { get; set; } = 15;
}

public class RLConfig
{
    public AdvisorConfig Advisor { get; set; } = new();
    public SacConfig SAC { get; set; } = new();
}

public class AdvisorConfig
{
    public bool Enabled { get; set; }
    public int ShadowMinDecisions { get; set; } = 1000;
    public int MinEdgeBps { get; set; } = 3;
    
    // Configurable confidence calculation parameters
    public double ExplorationConfidence { get; set; } = 0.3;
    public double ConfidenceOffset { get; set; } = 1.0;
    public double ConfidenceScale { get; set; } = 2.0;
    public double MinConfidence { get; set; } = 0.1;
    public double MaxConfidence { get; set; } = 1.0;
}

public class SacConfig
{
    public bool Enabled { get; set; } = true;
    public string RetrainCron { get; set; } = "weekly";
}

public class OrdersConfig
{
    public IdempotentConfig Idempotent { get; set; } = new();
}

public class IdempotentConfig
{
    public bool Enabled { get; set; } = true;
    public string KeySchema { get; set; } = "sha1(modelId|strategyId|signalId|ts|symbol|side|priceBucket)";
    public int DedupeTtlHours { get; set; } = 24;
    public RetryConfig Retry { get; set; } = new();
}

public class RetryConfig
{
    public int InitialMs { get; set; } = 250;
    public int MaxMs { get; set; } = 4000;
    public int MaxAttempts { get; set; } = 5;
    public int MaxDelaySeconds { get; set; } = 30;
}

public class OrchestratorConfig
{
    public LeaderElectionConfig LeaderElection { get; set; } = new();
}

public class LeaderElectionConfig
{
    public bool Enabled { get; set; } = true;
    public int TtlSeconds { get; set; } = 15;
    public int RenewSeconds { get; set; } = 5;
    public int TakeoverDeadlineSeconds { get; set; } = 10;
}

public class SloConfig
{
    public int DecisionLatencyP99Ms { get; set; } = 120;
    public int E2eOrderP99Ms { get; set; } = 400;
    public int ReconnectSeconds { get; set; } = 5;
    public double DailyErrorBudgetPct { get; set; } = 0.5;
}

public class ObservabilityConfig
{
    public bool Enabled { get; set; } = true;
    public bool Dashboards { get; set; } = true;
    public bool LineageTracking { get; set; } = true;
    public DecisionLineConfig DecisionLine { get; set; } = new();
    public DriftMonitoringConfig DriftMonitoring { get; set; } = new();
    
    // Configurable parameters for dashboard calculations
    public double CalibrationScoreOffset { get; set; } = 0.5;
    public double CalibrationScoreMultiplier { get; set; } = 2.0;
}

public class HistoricalConfig
{
    public TrainerConfig Trainer { get; set; } = new();
}

public class TrainerConfig
{
    public bool Enabled { get; set; } = true;
    public bool WalkForwardCV { get; set; } = true;
    public bool AutoPromote { get; set; } = true;
}

public class DecisionLineConfig
{
    public bool Enabled { get; set; } = true;
}

public class DriftMonitoringConfig
{
    public bool Enabled { get; set; } = true;
    public double PsiWarn { get; set; } = 0.2;
    public double PsiBlock { get; set; } = 0.3;
    public double KlWarn { get; set; } = 0.1;
    public double KlBlock { get; set; } = 0.2;
}

public class PromotionsConfig
{
    public AutoPromoteConfig Auto { get; set; } = new();
}

public class AutoPromoteConfig
{
    public bool Enabled { get; set; } = true;
    public PromoteIfConfig PromoteIf { get; set; } = new();
    public int CooldownMinutes { get; set; } = 60;
    public AdoptIfBetterConfig AdoptIfBetter { get; set; } = new();
}

public class PromoteIfConfig
{
    public double MinAuc { get; set; } = 0.62;
    public double MinPrAt10 { get; set; } = 0.12;
    public double MaxEce { get; set; } = 0.05;
    public int MinEdgeBps { get; set; } = 3;
}

public class AdoptIfBetterConfig
{
    public double MinAucGain { get; set; } = 0.02;
    public double MinPrAt10Gain { get; set; } = 0.03;
}

/// <summary>
/// Network and connection configuration including retry policies and batch sizes
/// </summary>
public class NetworkConfig
{
    public RetryConfig Retry { get; set; } = new();
    public BatchConfig Batch { get; set; } = new();
}

/// <summary>
/// Batch processing configuration
/// </summary>
public class BatchConfig
{
    public int DefaultBatchSize { get; set; } = 32;
    public int MaxBatchSize { get; set; } = 128;
    public int MinBatchSize { get; set; } = 16;
    public int TelemetryBatchSize { get; set; } = 100;
    public int ModelInferenceBatchSize { get; set; } = 64;
}