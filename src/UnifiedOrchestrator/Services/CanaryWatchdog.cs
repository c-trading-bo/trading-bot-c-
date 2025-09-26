using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Canary watchdog for automatic model rollback
/// Monitors performance metrics and auto-demotes underperforming models
/// </summary>
internal sealed class CanaryWatchdog : BackgroundService
{
    private readonly ILogger<CanaryWatchdog> _logger;
    private readonly IConfiguration _configuration;
    private readonly IAdaptiveParameterService _parameterService;
    private DateTime _canaryStart = DateTime.MinValue;
    private int _decisionsCount = 0;
    private string _currentModelSha = string.Empty;
    
    // Rollback thresholds
    private const double PnlDropThreshold = 0.15; // 15% PnL drop vs baseline
    private const double SlippageWorseningThreshold = 2.0; // 2 additional ticks
    private const double LatencyP95Threshold = 300.0; // 300ms SLA
    private const int CanaryDecisionCount = 100; // N=100 decisions
    private const int CanaryMinutes = 30; // T=30 minutes
    
    public CanaryWatchdog(
        ILogger<CanaryWatchdog> logger, 
        IConfiguration configuration,
        IAdaptiveParameterService parameterService)
    {
        _logger = logger;
        _configuration = configuration;
        _parameterService = parameterService;
    }
    
    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("🕊️ Canary watchdog started");
        
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                await MonitorCanaryAsync(stoppingToken).ConfigureAwait(false);
                await Task.Delay(TimeSpan.FromSeconds(15), stoppingToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                break;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Error in canary watchdog");
                await Task.Delay(TimeSpan.FromSeconds(60), stoppingToken).ConfigureAwait(false);
            }
        }
    }
    
    private async Task MonitorCanaryAsync(CancellationToken cancellationToken)
    {
        // Check if we're in canary period
        if (_canaryStart == DateTime.MinValue || !IsInCanaryWindow())
        {
            return;
        }
        
        await Task.CompletedTask.ConfigureAwait(false);
        
        // Get current performance metrics
        var currentPnl = _parameterService.GetParameter("canary.pnl", 0.0);
        var baselinePnl = _parameterService.GetParameter("baseline.pnl", 0.0);
        var slippageP95 = _parameterService.GetParameter("latency.slippage.p95", 0.0);
        var baselineSlippage = _parameterService.GetParameter("baseline.slippage.p95", 1.0);
        var latencyP95 = _parameterService.GetParameter("latency.ms.p95", 0.0);
        
        // Check rollback triggers
        var pnlDrop = baselinePnl > 0 ? (baselinePnl - currentPnl) / baselinePnl : 0;
        var slippageWorsening = slippageP95 - baselineSlippage;
        
        var shouldRollback = 
            pnlDrop > PnlDropThreshold ||
            slippageWorsening >= SlippageWorseningThreshold ||
            latencyP95 > LatencyP95Threshold ||
            _decisionsCount >= CanaryDecisionCount ||
            (DateTime.UtcNow - _canaryStart).TotalMinutes >= CanaryMinutes;
        
        if (shouldRollback)
        {
            await DoRollbackAsync(cancellationToken).ConfigureAwait(false);
        }
    }
    
    private bool IsInCanaryWindow()
    {
        return _canaryStart != DateTime.MinValue && 
               (DateTime.UtcNow - _canaryStart).TotalMinutes <= CanaryMinutes;
    }
    
    private async Task DoRollbackAsync(CancellationToken cancellationToken)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        _logger.LogWarning("🚨 Canary rollback triggered for model {ModelSha}", _currentModelSha);
        
        try
        {
            // Atomic swap: artifacts/current -> artifacts/previous
            var currentPath = Path.Combine("artifacts", "current");
            var previousPath = Path.Combine("artifacts", "previous");
            var tempPath = Path.Combine("artifacts", "rollback_temp");
            
            if (Directory.Exists(currentPath) && Directory.Exists(previousPath))
            {
                // Three-way swap to avoid conflicts
                if (Directory.Exists(tempPath)) Directory.Delete(tempPath, true);
                Directory.Move(currentPath, tempPath);
                Directory.Move(previousPath, currentPath);
                Directory.Move(tempPath, previousPath);
                
                _logger.LogInformation("✅ Model rolled back successfully");
                
                // Emit telemetry
                _parameterService.SetParameter("canary.auto_demote", 1);
                _parameterService.SetParameter("canary.rollback_count", 
                    _parameterService.GetParameter("canary.rollback_count", 0) + 1);
                
                // Optionally disable promotion until manual re-arm
                var disablePromotion = _configuration.GetValue("AUTO_DISABLE_ON_ROLLBACK", true);
                if (disablePromotion)
                {
                    _logger.LogWarning("🔒 Disabling PROMOTE_TUNER until manual re-arm");
                    Environment.SetEnvironmentVariable("PROMOTE_TUNER", "0");
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Failed to rollback model");
        }
        finally
        {
            // Reset canary state
            _canaryStart = DateTime.MinValue;
            _decisionsCount = 0;
            _currentModelSha = string.Empty;
        }
    }
    
    public void StartCanary(string modelSha)
    {
        _canaryStart = DateTime.UtcNow;
        _decisionsCount = 0;
        _currentModelSha = modelSha;
        
        _logger.LogInformation("🕊️ Canary started for model {ModelSha}", modelSha);
        
        // Reset canary metrics
        _parameterService.SetParameter("canary.start", DateTime.UtcNow.Ticks);
        _parameterService.SetParameter("canary.auto_demote", 0);
    }
    
    public void IncrementDecisions()
    {
        if (IsInCanaryWindow())
        {
            _decisionsCount++;
        }
    }
}