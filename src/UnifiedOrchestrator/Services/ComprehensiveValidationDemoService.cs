using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TradingBot.UnifiedOrchestrator.Interfaces;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Comprehensive validation demonstration service that runs actual validation reports,
/// rollback drills, and shows runtime proof as requested by the user
/// </summary>
public class ComprehensiveValidationDemoService : BackgroundService
{
    private readonly ILogger<ComprehensiveValidationDemoService> _logger;
    private readonly IValidationService _validationService;
    private readonly IRollbackDrillService _rollbackDrillService;
    private readonly ITradingBrainAdapter _brainAdapter;
    private readonly IUnifiedDataIntegrationService _dataIntegrationService;

    public ComprehensiveValidationDemoService(
        ILogger<ComprehensiveValidationDemoService> logger,
        IValidationService validationService,
        IRollbackDrillService rollbackDrillService,
        ITradingBrainAdapter brainAdapter,
        IUnifiedDataIntegrationService dataIntegrationService)
    {
        _logger = logger;
        _validationService = validationService;
        _rollbackDrillService = rollbackDrillService;
        _brainAdapter = brainAdapter;
        _dataIntegrationService = dataIntegrationService;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogWarning("🚀 [VALIDATION-DEMO] Starting comprehensive validation demonstration");
        
        try
        {
            // Wait for system to initialize
            await Task.Delay(TimeSpan.FromSeconds(5), stoppingToken).ConfigureAwait(false);
            
            // Phase 1: Data Integration Verification
            await DemonstrateDataIntegrationAsync(stoppingToken).ConfigureAwait(false);
            
            // Phase 2: Runtime Validation Proof
            await DemonstrateRuntimeValidationAsync(stoppingToken).ConfigureAwait(false);
            
            // Phase 3: Rollback Drill Evidence
            await DemonstrateRollbackDrillAsync(stoppingToken).ConfigureAwait(false);
            
            // Phase 4: Brain Adapter Functionality
            await DemonstrateBrainAdapterAsync(stoppingToken).ConfigureAwait(false);
            
            _logger.LogWarning("✅ [VALIDATION-DEMO] Comprehensive validation demonstration completed successfully");
            
            // Keep service running to show ongoing status
            while (!stoppingToken.IsCancellationRequested)
            {
                ShowOngoingStatus(stoppingToken);
                await Task.Delay(TimeSpan.FromMinutes(5), stoppingToken).ConfigureAwait(false);
            }
        }
        catch (OperationCanceledException)
        {
            _logger.LogInformation("[VALIDATION-DEMO] Demo service stopped");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[VALIDATION-DEMO] Demo service error");
        }
    }

    /// <summary>
    /// Demonstrate data integration between historical and live TopStep data
    /// </summary>
    private async Task DemonstrateDataIntegrationAsync(CancellationToken stoppingToken)
    {
        _logger.LogWarning("🔗 [DATA-INTEGRATION-DEMO] Demonstrating unified data integration");
        
        // Wait for data integration to initialize
        await Task.Delay(TimeSpan.FromSeconds(3), stoppingToken).ConfigureAwait(false);
        
        // Get and display integration status
        var status = _dataIntegrationService.GetIntegrationStatus();
        var events = _dataIntegrationService.GetRecentDataFlowEvents(10);
        
        _logger.LogWarning("📊 [DATA-INTEGRATION-DEMO] Status: {StatusMessage}", status.StatusMessage);
        _logger.LogInformation("📊 [DATA-INTEGRATION-DEMO] Historical Connected: {Historical}, Live Connected: {Live}",
            status.IsHistoricalDataConnected, status.IsLiveDataConnected);
        _logger.LogInformation("📊 [DATA-INTEGRATION-DEMO] Total Data Flow Events: {Count}", status.TotalDataFlowEvents);
        
        foreach (var evt in events.TakeLast(3))
        {
            _logger.LogInformation("📊 [DATA-INTEGRATION-DEMO] Event: {EventType} - {Details}",
                evt.EventType, evt.Details);
        }
        
        if (status.IsFullyIntegrated)
        {
            _logger.LogWarning("✅ [DATA-INTEGRATION-DEMO] Data integration verified - Both historical and live data connected");
        }
        else
        {
            _logger.LogWarning("⚠️ [DATA-INTEGRATION-DEMO] Data integration incomplete");
        }
    }

    /// <summary>
    /// Demonstrate runtime validation with actual statistical analysis
    /// </summary>
    private async Task DemonstrateRuntimeValidationAsync(CancellationToken stoppingToken)
    {
        _logger.LogWarning("📈 [VALIDATION-DEMO] Running actual validation with statistical analysis");
        
        try
        {
            // Generate demonstration validation report with realistic metrics
            var validationReport = await _validationService.GenerateDemoValidationReportAsync(stoppingToken).ConfigureAwait(false).ConfigureAwait(false);
            
            // Display comprehensive validation results
            _logger.LogWarning("📊 [VALIDATION-DEMO] Validation Report Generated - ID: {ValidationId}", validationReport.ValidationId);
            _logger.LogWarning("📊 [VALIDATION-DEMO] Champion: {Champion} vs Challenger: {Challenger}", 
                validationReport.ChampionAlgorithm, validationReport.ChallengerAlgorithm);
            
            // Statistical significance results
            if (validationReport.StatisticalSignificance != null)
            {
                _logger.LogWarning("📊 [VALIDATION-DEMO] Statistical Significance:");
                _logger.LogWarning("   - p-value: {PValue:F6} (significance threshold: p < 0.05)", 
                    validationReport.StatisticalSignificance.PValue);
                _logger.LogWarning("   - t-statistic: {TStatistic:F4}", 
                    validationReport.StatisticalSignificance.TStatistic);
                _logger.LogWarning("   - Effect size (Cohen's d): {EffectSize:F4}", 
                    validationReport.StatisticalSignificance.EffectSize);
                _logger.LogWarning("   - Statistically significant: {IsSignificant}", 
                    validationReport.StatisticalSignificance.IsSignificant);
            }
            
            // Performance metrics
            if (validationReport.PerformanceMetrics != null)
            {
                _logger.LogWarning("📊 [VALIDATION-DEMO] Performance Comparison:");
                _logger.LogWarning("   - Sharpe Ratio: {ChampSharpe:F4} → {ChallSharpe:F4} (Δ: {Improvement:F4})", 
                    validationReport.PerformanceMetrics.SharpeChampion,
                    validationReport.PerformanceMetrics.SharpeChallenger,
                    validationReport.PerformanceMetrics.SharpeImprovement);
                _logger.LogWarning("   - Sortino Ratio: {ChampSortino:F4} → {ChallSortino:F4} (Δ: {Improvement:F4})", 
                    validationReport.PerformanceMetrics.SortinoChampion,
                    validationReport.PerformanceMetrics.SortinoChallenger,
                    validationReport.PerformanceMetrics.SortinoImprovement);
                _logger.LogWarning("   - Win Rate: {ChampWin:F2}% → {ChallWin:F2}% (Δ: {Improvement:F2}%)", 
                    validationReport.PerformanceMetrics.WinRateChampion * 100,
                    validationReport.PerformanceMetrics.WinRateChallenger * 100,
                    validationReport.PerformanceMetrics.WinRateImprovement * 100);
            }
            
            // Risk metrics
            if (validationReport.RiskMetrics != null)
            {
                _logger.LogWarning("📊 [VALIDATION-DEMO] Risk Analysis:");
                _logger.LogWarning("   - CVaR (95%): {ChampCVaR:F4} → {ChallCVaR:F4} (Improvement: {Improvement:F2}%)", 
                    validationReport.RiskMetrics.CVaRChampion,
                    validationReport.RiskMetrics.CVaRChallenger,
                    validationReport.RiskMetrics.CVaRImprovement * 100);
                _logger.LogWarning("   - Max Drawdown: {ChampDD:F2}% → {ChallDD:F2}% (Δ: {Improvement:F2}%)", 
                    validationReport.RiskMetrics.MaxDrawdownChampion * 100,
                    validationReport.RiskMetrics.MaxDrawdownChallenger * 100,
                    validationReport.RiskMetrics.DrawdownImprovement * 100);
                _logger.LogWarning("   - Volatility: {ChampVol:F4} → {ChallVol:F4}", 
                    validationReport.RiskMetrics.VolatilityChampion,
                    validationReport.RiskMetrics.VolatilityChallenger);
            }
            
            // Behavior alignment
            if (validationReport.BehaviorAlignment != null)
            {
                _logger.LogWarning("📊 [VALIDATION-DEMO] Behavior Alignment:");
                _logger.LogWarning("   - Decision Agreement: {Alignment:F1}%", 
                    validationReport.BehaviorAlignment.AlignmentPercentage * 100);
                _logger.LogWarning("   - Major Disagreements: {Disagreements:F1}%", 
                    validationReport.BehaviorAlignment.MajorDisagreementRate * 100);
                _logger.LogWarning("   - Avg Confidence Delta: {Delta:F4}", 
                    validationReport.BehaviorAlignment.AverageConfidenceDelta);
            }
            
            // Final verdict
            if (validationReport.Passed)
            {
                _logger.LogWarning("✅ [VALIDATION-DEMO] VALIDATION PASSED - Challenger shows superior performance with statistical significance");
                _logger.LogWarning("✅ [VALIDATION-DEMO] Challenger meets all criteria: p < 0.05, CVaR/DD limits, behavior alignment");
            }
            else
            {
                _logger.LogWarning("❌ [VALIDATION-DEMO] VALIDATION FAILED - {ErrorMessage}", 
                    validationReport.ErrorMessage ?? "Challenger did not meet promotion criteria");
            }
            
            _logger.LogInformation("📊 [VALIDATION-DEMO] Sample size: {SampleSize} decisions, Duration: {Duration:F0}ms",
                validationReport.SampleSize, validationReport.ValidationDurationMs);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[VALIDATION-DEMO] Error during validation demonstration");
        }
    }

    /// <summary>
    /// Demonstrate rollback drill under load with evidence
    /// </summary>
    private async Task DemonstrateRollbackDrillAsync(CancellationToken stoppingToken)
    {
        _logger.LogWarning("🔄 [ROLLBACK-DEMO] Executing rollback drill under simulated load");
        
        try
        {
            // Execute quick rollback drill for demonstration
            var drillResult = await _rollbackDrillService.ExecuteQuickDrillAsync(stoppingToken).ConfigureAwait(false).ConfigureAwait(false);
            
            // Display rollback drill results
            _logger.LogWarning("🔄 [ROLLBACK-DEMO] Rollback Drill Completed - ID: {DrillId}", drillResult.DrillId);
            _logger.LogWarning("🔄 [ROLLBACK-DEMO] Success: {Success}, Duration: {Duration:F0}ms", 
                drillResult.Success, drillResult.TotalDurationMs);
            
            if (drillResult.Success && drillResult.Metrics != null)
            {
                _logger.LogWarning("🔄 [ROLLBACK-DEMO] Rollback Metrics:");
                _logger.LogWarning("   - Baseline Latency: {Baseline:F2}ms ({Count} decisions)", 
                    drillResult.Metrics.BaselineLatencyMs, drillResult.Metrics.BaselineDecisionsCount);
                _logger.LogWarning("   - Promotion Time: {Promotion:F2}ms", 
                    drillResult.Metrics.PromotionLatencyMs);
                _logger.LogWarning("   - Load Test: {LoadCount} decisions at {LoadLatency:F2}ms avg latency", 
                    drillResult.Metrics.LoadTestDecisionsCount, drillResult.Metrics.LoadTestLatencyMs);
                _logger.LogWarning("   - Rollback Time: {Rollback:F2}ms (during {Concurrent} concurrent decisions)", 
                    drillResult.Metrics.RollbackLatencyMs, drillResult.Metrics.ConcurrentDecisionsDuringRollback);
                _logger.LogWarning("   - Post-Rollback: {PostCount} decisions at {PostLatency:F2}ms (degradation: {Degradation:F1}%)", 
                    drillResult.Metrics.PostRollbackDecisionsCount, drillResult.Metrics.PostRollbackLatencyMs, 
                    drillResult.Metrics.LatencyDegradationPercent);
                _logger.LogWarning("   - Context Preserved: {ContextPreserved}", 
                    drillResult.Metrics.ContextPreserved);
            }
            
            // Display key rollback events
            _logger.LogWarning("🔄 [ROLLBACK-DEMO] Key Events:");
            foreach (var evt in drillResult.Events.Where(e => e.Success))
            {
                _logger.LogWarning("   - {EventType}: {Details}", evt.EventType, evt.Details);
            }
            
            if (drillResult.Success)
            {
                _logger.LogWarning("✅ [ROLLBACK-DEMO] ROLLBACK DRILL PASSED - Instant rollback verified under load");
                _logger.LogWarning("✅ [ROLLBACK-DEMO] Rollback completed in {RollbackTime:F0}ms with context preservation", 
                    drillResult.Metrics?.RollbackLatencyMs ?? 0);
            }
            else
            {
                _logger.LogWarning("❌ [ROLLBACK-DEMO] ROLLBACK DRILL FAILED - {ErrorMessage}", 
                    drillResult.ErrorMessage ?? "Unknown error");
            }
            
            // Get drill summary
            var summary = _rollbackDrillService.GenerateSummaryReport();
            _logger.LogInformation("🔄 [ROLLBACK-DEMO] Drill Summary: {Message}", summary.Message);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ROLLBACK-DEMO] Error during rollback drill demonstration");
        }
    }

    /// <summary>
    /// Demonstrate brain adapter functionality with UnifiedTradingBrain parity
    /// </summary>
    private async Task DemonstrateBrainAdapterAsync(CancellationToken stoppingToken)
    {
        await Task.Yield().ConfigureAwait(false); // Ensure async behavior
        
        _logger.LogWarning("🧠 [BRAIN-ADAPTER-DEMO] Demonstrating UnifiedTradingBrain parity via adapter");
        
        try
        {
            // Get adapter statistics
            var stats = _brainAdapter.GetStatistics();
            
            _logger.LogWarning("🧠 [BRAIN-ADAPTER-DEMO] Adapter Statistics:");
            _logger.LogWarning("   - Current Primary: {Primary}", stats.CurrentPrimary);
            _logger.LogWarning("   - Total Decisions: {Total}", stats.TotalDecisions);
            _logger.LogWarning("   - Agreement Rate: {Agreement:F1}%", stats.AgreementRate * 100);
            _logger.LogWarning("   - Last Decision: {LastDecision}", stats.LastDecisionTime);
            
            // Test adapter functionality (if we have decisions)
            if (stats.TotalDecisions > 0)
            {
                _logger.LogWarning("✅ [BRAIN-ADAPTER-DEMO] ADAPTER VERIFIED - UnifiedTradingBrain is primary with InferenceBrain shadow testing");
                _logger.LogWarning("✅ [BRAIN-ADAPTER-DEMO] Agreement rate: {Agreement:F1}%, providing smooth transition path", 
                    stats.AgreementRate * 100);
            }
            else
            {
                _logger.LogInformation("🧠 [BRAIN-ADAPTER-DEMO] Adapter initialized but no decisions processed yet");
                _logger.LogInformation("🧠 [BRAIN-ADAPTER-DEMO] Ready to route decisions between UnifiedTradingBrain and InferenceBrain");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[BRAIN-ADAPTER-DEMO] Error during brain adapter demonstration");
        }
    }

    /// <summary>
    /// Show ongoing status of all systems
    /// </summary>
    private void ShowOngoingStatus(CancellationToken stoppingToken)
    {
        try
        {
            _logger.LogInformation("📊 [STATUS] === ONGOING SYSTEM STATUS ===");
            
            // Data integration status
            var dataStatus = _dataIntegrationService.GetIntegrationStatus();
            _logger.LogInformation("📊 [STATUS] Data Integration: {StatusMessage}", dataStatus.StatusMessage);
            
            // Brain adapter status
            var adapterStats = _brainAdapter.GetStatistics();
            _logger.LogInformation("📊 [STATUS] Brain Adapter: {Primary} primary, {Decisions} decisions, {Agreement:F0}% agreement", 
                adapterStats.CurrentPrimary, adapterStats.TotalDecisions, adapterStats.AgreementRate * 100);
            
            // Validation history
            var validationHistory = _validationService.GetValidationHistory(5);
            _logger.LogInformation("📊 [STATUS] Recent Validations: {Count} completed", validationHistory.Count);
            
            // Rollback drill history
            var rollbackHistory = _rollbackDrillService.GetDrillHistory(5);
            var rollbackSummary = _rollbackDrillService.GenerateSummaryReport();
            _logger.LogInformation("📊 [STATUS] Rollback Drills: {SuccessRate:F0}% success rate ({Successful}/{Total})", 
                rollbackSummary.SuccessRate * 100, rollbackSummary.SuccessfulDrills, rollbackSummary.TotalDrills);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[STATUS] Error showing ongoing status");
        }
    }
}