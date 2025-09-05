using Microsoft.Extensions.Logging;
using BotCore.ML;
using BotCore.Market;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Services;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Integration service that coordinates ML memory management with workflow orchestration
/// Provides a unified interface for advanced system management
/// </summary>
public class AdvancedSystemIntegrationService : IDisposable
{
    private readonly ILogger<AdvancedSystemIntegrationService> _logger;
    private readonly IMLMemoryManager? _mlMemoryManager;
    private readonly IWorkflowOrchestrationManager? _workflowOrchestrationManager;
    private readonly RedundantDataFeedManager? _dataFeedManager;
    private readonly IEconomicEventManager? _economicEventManager;
    private readonly StrategyMlModelManager? _strategyMlManager;
    private readonly Timer _systemHealthTimer;
    private bool _disposed = false;

    public AdvancedSystemIntegrationService(
        ILogger<AdvancedSystemIntegrationService> logger,
        IMLMemoryManager? mlMemoryManager = null,
        IWorkflowOrchestrationManager? workflowOrchestrationManager = null,
        RedundantDataFeedManager? dataFeedManager = null,
        IEconomicEventManager? economicEventManager = null,
        StrategyMlModelManager? strategyMlManager = null)
    {
        _logger = logger;
        _mlMemoryManager = mlMemoryManager;
        _workflowOrchestrationManager = workflowOrchestrationManager;
        _dataFeedManager = dataFeedManager;
        _economicEventManager = economicEventManager;
        _strategyMlManager = strategyMlManager;
        
        _systemHealthTimer = new Timer(CheckSystemHealth, null, Timeout.Infinite, Timeout.Infinite);
        
        _logger.LogInformation("[Advanced-Integration] AdvancedSystemIntegrationService initialized");
    }

    /// <summary>
    /// Initialize all advanced system components
    /// </summary>
    public async Task InitializeAsync()
    {
        _logger.LogInformation("[Advanced-Integration] Initializing advanced system components");

        try
        {
            // Initialize ML memory management
            if (_mlMemoryManager != null)
            {
                await _mlMemoryManager.InitializeMemoryManagementAsync();
                _logger.LogInformation("[Advanced-Integration] ML memory management initialized");
            }

            // Initialize workflow orchestration
            if (_workflowOrchestrationManager != null)
            {
                await _workflowOrchestrationManager.InitializeAsync();
                _logger.LogInformation("[Advanced-Integration] Workflow orchestration initialized");
            }

            // Initialize data feeds
            if (_dataFeedManager != null)
            {
                await _dataFeedManager.InitializeDataFeedsAsync();
                _logger.LogInformation("[Advanced-Integration] Redundant data feeds initialized");
            }

            // Initialize economic event management
            if (_economicEventManager != null)
            {
                await _economicEventManager.InitializeAsync();
                _logger.LogInformation("[Advanced-Integration] Economic event management initialized");
            }

            // Start system health monitoring
            _systemHealthTimer.Change(TimeSpan.Zero, TimeSpan.FromMinutes(1));
            
            _logger.LogInformation("[Advanced-Integration] All advanced system components initialized successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[Advanced-Integration] Failed to initialize advanced system components");
            throw;
        }
    }

    /// <summary>
    /// Execute a workflow with advanced coordination and memory management
    /// </summary>
    public async Task<bool> ExecuteWorkflowWithAdvancedCoordinationAsync(
        string workflowName, 
        Func<Task> workflowAction,
        List<string>? requiredResources = null)
    {
        if (string.IsNullOrEmpty(workflowName))
            throw new ArgumentException("Workflow name cannot be null or empty", nameof(workflowName));
            
        if (workflowAction == null)
            throw new ArgumentNullException(nameof(workflowAction));

        _logger.LogInformation("[Advanced-Integration] Executing workflow with advanced coordination: {WorkflowName}", workflowName);

        try
        {
            // Check memory before execution
            if (_mlMemoryManager != null)
            {
                var memorySnapshot = _mlMemoryManager.GetMemorySnapshot();
                var memoryUsagePercent = (double)memorySnapshot.UsedMemory / (8L * 1024 * 1024 * 1024) * 100;
                
                if (memoryUsagePercent > 85)
                {
                    _logger.LogWarning("[Advanced-Integration] High memory usage ({MemoryPercent:F1}%) before workflow execution", 
                        memoryUsagePercent);
                }
            }

            // Execute through workflow orchestration if available
            if (_workflowOrchestrationManager != null)
            {
                var success = await _workflowOrchestrationManager.RequestWorkflowExecutionAsync(
                    workflowName, workflowAction, requiredResources);
                    
                if (success)
                {
                    _logger.LogInformation("[Advanced-Integration] Workflow executed successfully through orchestration: {WorkflowName}", 
                        workflowName);
                }
                else
                {
                    _logger.LogWarning("[Advanced-Integration] Workflow queued due to conflicts: {WorkflowName}", workflowName);
                }
                
                return success;
            }
            else
            {
                // Execute directly if orchestration not available
                await workflowAction();
                _logger.LogInformation("[Advanced-Integration] Workflow executed directly: {WorkflowName}", workflowName);
                return true;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[Advanced-Integration] Error executing workflow: {WorkflowName}", workflowName);
            return false;
        }
    }

    /// <summary>
    /// Get comprehensive system status
    /// </summary>
    public async Task<AdvancedSystemStatus> GetSystemStatusAsync()
    {
        var status = new AdvancedSystemStatus
        {
            Timestamp = DateTime.UtcNow,
            IsHealthy = true,
            Components = new Dictionary<string, bool>()
        };

        // Check ML memory management
        if (_mlMemoryManager != null)
        {
            try
            {
                var memorySnapshot = _mlMemoryManager.GetMemorySnapshot();
                status.MemorySnapshot = memorySnapshot;
                status.Components["ML Memory Manager"] = true;
                
                var memoryUsagePercent = (double)memorySnapshot.UsedMemory / (8L * 1024 * 1024 * 1024) * 100;
                if (memoryUsagePercent > 90)
                {
                    status.IsHealthy = false;
                    status.Issues.Add($"Critical memory usage: {memoryUsagePercent:F1}%");
                }
            }
            catch (Exception ex)
            {
                status.Components["ML Memory Manager"] = false;
                status.Issues.Add($"ML Memory Manager error: {ex.Message}");
                status.IsHealthy = false;
            }
        }

        // Check workflow orchestration
        if (_workflowOrchestrationManager != null)
        {
            try
            {
                var orchestrationStatus = _workflowOrchestrationManager.GetStatus();
                status.WorkflowOrchestrationStatus = orchestrationStatus;
                status.Components["Workflow Orchestration"] = true;
                
                if (orchestrationStatus.ActiveConflicts > 5)
                {
                    status.Issues.Add($"High workflow conflicts: {orchestrationStatus.ActiveConflicts}");
                }
            }
            catch (Exception ex)
            {
                status.Components["Workflow Orchestration"] = false;
                status.Issues.Add($"Workflow Orchestration error: {ex.Message}");
                status.IsHealthy = false;
            }
        }

        // Check data feeds
        if (_dataFeedManager != null)
        {
            status.Components["Redundant Data Feeds"] = true;
            // Additional data feed health checks could be added here
        }

        // Check economic event management
        if (_economicEventManager != null)
        {
            try
            {
                // Check for upcoming high-impact events
                var upcomingEvents = await _economicEventManager.GetEventsByImpactAsync(EventImpact.High);
                status.Components["Economic Event Manager"] = true;
                
                var criticalEvents = upcomingEvents.Where(e => e.Impact >= EventImpact.Critical).ToList();
                if (criticalEvents.Any())
                {
                    status.Issues.Add($"Critical economic events approaching: {criticalEvents.Count}");
                }
            }
            catch (Exception ex)
            {
                status.Components["Economic Event Manager"] = false;
                status.Issues.Add($"Economic Event Manager error: {ex.Message}");
                status.IsHealthy = false;
            }
        }

        return status;
    }

    private async void CheckSystemHealth(object? state)
    {
        try
        {
            var status = await GetSystemStatusAsync();
            
            if (!status.IsHealthy)
            {
                _logger.LogWarning("[Advanced-Integration] System health issues detected: {Issues}", 
                    string.Join("; ", status.Issues));
            }
            else
            {
                _logger.LogDebug("[Advanced-Integration] System health check passed");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[Advanced-Integration] Error during system health check");
        }
    }

    /// <summary>
    /// Integrate with existing strategy ML manager
    /// </summary>
    public async Task<decimal> GetOptimizedPositionSizeAsync(
        string strategyId,
        string symbol,
        decimal price,
        decimal atr,
        decimal score,
        decimal qScore,
        IList<BotCore.Models.Bar> bars)
    {
        if (_strategyMlManager == null)
        {
            _logger.LogDebug("[Advanced-Integration] Strategy ML manager not available, using default multiplier");
            return 1.0m;
        }

        try
        {
            // Execute through workflow orchestration for ML operations
            decimal multiplier = 1.0m;
            
            if (_workflowOrchestrationManager != null)
            {
                await _workflowOrchestrationManager.RequestWorkflowExecutionAsync(
                    "ml-position-sizing",
                    async () =>
                    {
                        multiplier = _strategyMlManager.GetPositionSizeMultiplier(
                            strategyId, symbol, price, atr, score, qScore, bars);
                    },
                    new List<string> { "ml_model", "position_sizing" }
                );
            }
            else
            {
                multiplier = _strategyMlManager.GetPositionSizeMultiplier(
                    strategyId, symbol, price, atr, score, qScore, bars);
            }

            _logger.LogDebug("[Advanced-Integration] Optimized position size multiplier: {Multiplier} for {Strategy}-{Symbol}", 
                multiplier, strategyId, symbol);
                
            return multiplier;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[Advanced-Integration] Error getting optimized position size for {Strategy}-{Symbol}", 
                strategyId, symbol);
            return 1.0m;
        }
    }

    /// <summary>
    /// Check if trading is allowed for a symbol considering economic events
    /// </summary>
    public async Task<bool> IsTradingAllowedAsync(string symbol)
    {
        if (_economicEventManager == null)
        {
            _logger.LogDebug("[Advanced-Integration] Economic event manager not available, allowing trading for {Symbol}", symbol);
            return true;
        }

        try
        {
            var shouldRestrict = await _economicEventManager.ShouldRestrictTradingAsync(symbol, TimeSpan.FromHours(1));
            
            if (shouldRestrict)
            {
                var restriction = await _economicEventManager.GetTradingRestrictionAsync(symbol);
                _logger.LogWarning("[Advanced-Integration] Trading restricted for {Symbol}: {Reason}", 
                    symbol, restriction.Reason);
                return false;
            }

            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[Advanced-Integration] Error checking trading allowance for {Symbol}", symbol);
            return true; // Default to allow trading on error
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        
        _logger.LogInformation("[Advanced-Integration] Disposing AdvancedSystemIntegrationService");
        
        _disposed = true;
        
        _systemHealthTimer?.Dispose();
        _mlMemoryManager?.Dispose();
        _workflowOrchestrationManager?.Dispose();
        _dataFeedManager?.Dispose();
        (_economicEventManager as IDisposable)?.Dispose();
        _strategyMlManager?.Dispose();
        
        GC.SuppressFinalize(this);
    }
}

/// <summary>
/// Comprehensive system status information
/// </summary>
public class AdvancedSystemStatus
{
    public DateTime Timestamp { get; set; }
    public bool IsHealthy { get; set; }
    public Dictionary<string, bool> Components { get; set; } = new();
    public List<string> Issues { get; set; } = new();
    public MLMemoryManager.MemorySnapshot? MemorySnapshot { get; set; }
    public WorkflowOrchestrationManager.WorkflowOrchestrationStatus? WorkflowOrchestrationStatus { get; set; }
}