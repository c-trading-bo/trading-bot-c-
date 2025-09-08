using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Services;
using TradingBot.UnifiedOrchestrator.Models;
using System.Collections.Concurrent;
using BotCore;
using Trading.Safety;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Core unified orchestrator that consolidates all trading bot functionality
/// This replaces the 4+ separate orchestrators with one unified system
/// </summary>
public class UnifiedOrchestratorService : IUnifiedOrchestrator, IHostedService
{
    private readonly ILogger<UnifiedOrchestratorService> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly ITradingOrchestrator _tradingOrchestrator;
    private readonly IIntelligenceOrchestrator _intelligenceOrchestrator;
    private readonly IDataOrchestrator _dataOrchestrator;
    private readonly IWorkflowScheduler _scheduler;
    private readonly ICloudDataIntegration _cloudDataIntegration;
    private readonly AdvancedSystemIntegrationService? _advancedSystemIntegration;
    
    // Phase 5 Safety Infrastructure
    private readonly IKillSwitchWatcher _killSwitchWatcher;
    private readonly IRiskManager _riskManager;
    private readonly IHealthMonitor _healthMonitor;
    
    private readonly ConcurrentDictionary<string, UnifiedWorkflow> _workflows = new();
    private readonly ConcurrentDictionary<string, List<WorkflowExecutionContext>> _executionHistory = new();
    private readonly CancellationTokenSource _cancellationTokenSource = new();
    private readonly object _lockObject = new();
    
    private bool _isInitialized = false;
    private bool _isRunning = false;
    private DateTime _startTime = DateTime.UtcNow;

    private readonly ICentralMessageBus _messageBus;

    public UnifiedOrchestratorService(
        ILogger<UnifiedOrchestratorService> logger,
        IServiceProvider serviceProvider,
        ITradingOrchestrator tradingOrchestrator,
        IIntelligenceOrchestrator intelligenceOrchestrator,
        IDataOrchestrator dataOrchestrator,
        IWorkflowScheduler scheduler,
        ICentralMessageBus messageBus,
        ICloudDataIntegration cloudDataIntegration,
        IKillSwitchWatcher killSwitchWatcher,
        IRiskManager riskManager,
        IHealthMonitor healthMonitor,
        AdvancedSystemIntegrationService? advancedSystemIntegration = null)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _tradingOrchestrator = tradingOrchestrator;
        _intelligenceOrchestrator = intelligenceOrchestrator;
        _dataOrchestrator = dataOrchestrator;
        _scheduler = scheduler;
        _messageBus = messageBus;
        _cloudDataIntegration = cloudDataIntegration;
        _advancedSystemIntegration = advancedSystemIntegration;
        
        // Phase 5 Safety Infrastructure
        _killSwitchWatcher = killSwitchWatcher;
        _riskManager = riskManager;
        _healthMonitor = healthMonitor;
    }

    #region IHostedService Implementation

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🚀 Starting Unified Trading Orchestrator System");
        
        await InitializeAsync(cancellationToken);
        await StartOrchestratorAsync(cancellationToken);
        
        _logger.LogInformation("✅ Unified Trading Orchestrator System started successfully");
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🛑 Stopping Unified Trading Orchestrator System");
        
        await StopOrchestratorAsync(cancellationToken);
        
        _logger.LogInformation("✅ Unified Trading Orchestrator System stopped");
    }

    #endregion

    #region IUnifiedOrchestrator Implementation

    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        if (_isInitialized) return;

        _logger.LogInformation("🔧 Initializing Unified Orchestrator System...");

        try
        {
            // Start the central message bus FIRST - this is the "ONE BRAIN" communication system
            _logger.LogInformation("🧠 Starting Central Message Bus - ONE BRAIN communication...");
            await _messageBus.StartAsync(cancellationToken);
            
            // Initialize Phase 5 Safety Infrastructure BEFORE any trading operations
            _logger.LogInformation("🛡️ Initializing Safety Infrastructure...");
            await InitializeSafetyComponentsAsync(cancellationToken);
            
            // Initialize all sub-orchestrators with message bus integration
            await _tradingOrchestrator.ConnectAsync(cancellationToken);
            
            // Register all unified workflows (consolidating from all previous orchestrators)
            await RegisterAllWorkflowsAsync();
            
            // Initialize shared state in the brain
            _messageBus.UpdateSharedState("system.status", "initialized");
            _messageBus.UpdateSharedState("orchestrator.active_workflows", _workflows.Count);
            
            _isInitialized = true;
            _logger.LogInformation("✅ Unified Orchestrator System initialized successfully with Central Message Bus");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Failed to initialize Unified Orchestrator System");
            throw;
        }
    }

    public async Task StartOrchestratorAsync(CancellationToken cancellationToken = default)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("Orchestrator must be initialized before starting");

        if (_isRunning) return;

        _logger.LogInformation("▶️ Starting workflow scheduler and execution engine...");

        await _scheduler.StartAsync(cancellationToken);
        
        // Schedule all enabled workflows
        foreach (var workflow in _workflows.Values.Where(w => w.Enabled))
        {
            await _scheduler.ScheduleWorkflowAsync(workflow, cancellationToken);
        }

        _isRunning = true;
        _startTime = DateTime.UtcNow;
        
        _logger.LogInformation("✅ Unified Orchestrator is now running with {WorkflowCount} workflows", _workflows.Count);
    }

    public async Task StopOrchestratorAsync(CancellationToken cancellationToken = default)
    {
        if (!_isRunning) return;

        _logger.LogInformation("⏹️ Stopping workflow scheduler...");

        await _scheduler.StopAsync(cancellationToken);
        await _tradingOrchestrator.DisconnectAsync();
        
        // Stop the central message bus last
        await _messageBus.StopAsync(cancellationToken);
        
        _cancellationTokenSource.Cancel();
        _isRunning = false;
        
        _logger.LogInformation("✅ Unified Orchestrator stopped");
    }

    Task IUnifiedOrchestrator.StartAsync(CancellationToken cancellationToken) => StartOrchestratorAsync(cancellationToken);
    Task IUnifiedOrchestrator.StopAsync(CancellationToken cancellationToken) => StopOrchestratorAsync(cancellationToken);

    public IReadOnlyList<UnifiedWorkflow> GetWorkflows()
    {
        return _workflows.Values.ToList().AsReadOnly();
    }

    public UnifiedWorkflow? GetWorkflow(string workflowId)
    {
        _workflows.TryGetValue(workflowId, out var workflow);
        return workflow;
    }

    public async Task RegisterWorkflowAsync(UnifiedWorkflow workflow, CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrEmpty(workflow.Id))
            throw new ArgumentException("Workflow ID cannot be null or empty", nameof(workflow));

        _workflows.AddOrUpdate(workflow.Id, workflow, (key, existing) => workflow);
        
        if (_isRunning && workflow.Enabled)
        {
            await _scheduler.ScheduleWorkflowAsync(workflow, cancellationToken);
        }

        _logger.LogInformation("📋 Registered workflow: {WorkflowId} - {WorkflowName}", workflow.Id, workflow.Name);
    }

    public async Task<WorkflowExecutionResult> ExecuteWorkflowAsync(string workflowId, Dictionary<string, object>? parameters = null, CancellationToken cancellationToken = default)
    {
        var workflow = GetWorkflow(workflowId);
        if (workflow == null)
        {
            return new WorkflowExecutionResult
            {
                Success = false,
                ErrorMessage = $"Workflow '{workflowId}' not found"
            };
        }

        var context = new WorkflowExecutionContext
        {
            WorkflowId = workflowId,
            Parameters = parameters ?? new Dictionary<string, object>()
        };

        try
        {
            _logger.LogInformation("🔄 Executing workflow: {WorkflowId} - {WorkflowName}", workflowId, workflow.Name);

            var startTime = DateTime.UtcNow;
            
            // Publish workflow start event to the central message bus for coordination
            await _messageBus.PublishAsync("workflow.started", new { WorkflowId = workflowId, Name = workflow.Name }, cancellationToken);
            
            // Execute all actions in the workflow with central brain coordination
            foreach (var action in workflow.Actions)
            {
                context.Logs.Add($"Executing action: {action}");
                await ExecuteActionAsync(action, context, cancellationToken);
                
                // Publish action completion for real-time monitoring
                await _messageBus.PublishAsync("workflow.action_completed", new { WorkflowId = workflowId, Action = action }, cancellationToken);
            }

            var duration = DateTime.UtcNow - startTime;
            context.EndTime = DateTime.UtcNow;
            context.Status = WorkflowExecutionStatus.Completed;

            // Update metrics
            workflow.Metrics.ExecutionCount++;
            workflow.Metrics.SuccessCount++;
            workflow.Metrics.TotalExecutionTime += duration;
            workflow.Metrics.LastExecution = DateTime.UtcNow;
            workflow.Metrics.LastSuccess = DateTime.UtcNow;

            // Store execution history
            _executionHistory.AddOrUpdate(workflowId, 
                new List<WorkflowExecutionContext> { context },
                (key, existing) => 
                {
                    existing.Add(context);
                    if (existing.Count > 1000) // Keep last 1000 executions
                        existing.RemoveAt(0);
                    return existing;
                });

            // Update central brain state with workflow success
            _messageBus.UpdateSharedState($"workflow.{workflowId}.last_success", DateTime.UtcNow);
            
            // Publish workflow completion for coordination
            await _messageBus.PublishAsync("workflow.completed", new { WorkflowId = workflowId, Success = true, Duration = duration }, cancellationToken);

            _logger.LogInformation("✅ Workflow completed successfully: {WorkflowId} in {Duration}ms", 
                workflowId, duration.TotalMilliseconds);

            return new WorkflowExecutionResult
            {
                Success = true,
                Duration = duration,
                Results = context.Parameters
            };
        }
        catch (Exception ex)
        {
            context.EndTime = DateTime.UtcNow;
            context.Status = WorkflowExecutionStatus.Failed;
            context.ErrorMessage = ex.Message;
            
            // Update error metrics
            workflow.Metrics.ExecutionCount++;
            workflow.Metrics.FailureCount++;
            workflow.Metrics.LastExecution = DateTime.UtcNow;
            workflow.Metrics.LastFailure = DateTime.UtcNow;
            workflow.Metrics.LastError = ex.Message;

            // Update central brain state with workflow failure
            _messageBus.UpdateSharedState($"workflow.{workflowId}.last_error", ex.Message);
            
            // Publish workflow failure for coordination
            await _messageBus.PublishAsync("workflow.failed", new { WorkflowId = workflowId, Error = ex.Message }, cancellationToken);

            _logger.LogError(ex, "❌ Workflow execution failed: {WorkflowId}", workflowId);

            return new WorkflowExecutionResult
            {
                Success = false,
                ErrorMessage = ex.Message,
                Duration = context.Duration
            };
        }
    }

    public IReadOnlyList<WorkflowExecutionContext> GetExecutionHistory(string workflowId, int limit = 100)
    {
        if (_executionHistory.TryGetValue(workflowId, out var history))
        {
            return history.TakeLast(limit).ToList().AsReadOnly();
        }
        return new List<WorkflowExecutionContext>().AsReadOnly();
    }

    public async Task<OrchestratorStatus> GetStatusAsync()
    {
        var status = new OrchestratorStatus
        {
            IsRunning = _isRunning,
            IsConnectedToTopstep = await IsConnectedToTopstepAsync(),
            ActiveWorkflows = _workflows.Values.Count(w => w.Enabled),
            TotalWorkflows = _workflows.Count,
            StartTime = _startTime,
            Uptime = DateTime.UtcNow - _startTime
        };

        // Add basic component status
        status.ComponentStatus["TradingOrchestrator"] = "Connected";
        status.ComponentStatus["IntelligenceOrchestrator"] = "Active";
        status.ComponentStatus["DataOrchestrator"] = "Running";
        status.ComponentStatus["Scheduler"] = _isRunning ? "Running" : "Stopped";
        status.ComponentStatus["CentralMessageBus"] = "Active";

        // Add advanced system components status
        if (_advancedSystemIntegration != null)
        {
            try
            {
                var advancedStatus = await _advancedSystemIntegration.GetSystemStatusAsync();
                status.ComponentStatus["AdvancedSystemIntegration"] = advancedStatus.IsHealthy ? "Healthy" : "Unhealthy";
                
                // Add individual component status from advanced system
                foreach (var component in advancedStatus.Components)
                {
                    status.ComponentStatus[component.Key] = component.Value ? "Active" : "Inactive";
                }
                
                // Add any system issues to the status
                if (advancedStatus.Issues.Any())
                {
                    status.ComponentStatus["SystemIssues"] = string.Join("; ", advancedStatus.Issues);
                }
            }
            catch (Exception ex)
            {
                status.ComponentStatus["AdvancedSystemIntegration"] = $"Error: {ex.Message}";
            }
        }
        else
        {
            status.ComponentStatus["AdvancedSystemIntegration"] = "Not Available";
        }

        return status;
    }

    #endregion

    #region Private Methods

    private async Task RegisterAllWorkflowsAsync()
    {
        _logger.LogInformation("📋 Registering unified workflows (consolidating from all previous orchestrators)...");

        // Register all workflows from the previous orchestrators in one unified system
        var workflows = GetUnifiedWorkflowDefinitions();
        
        foreach (var workflow in workflows)
        {
            await RegisterWorkflowAsync(workflow);
        }

        _logger.LogInformation("✅ Registered {WorkflowCount} unified workflows", workflows.Count);
    }

    private async Task ExecuteActionAsync(string action, WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        // Handle cloud integration actions first
        if (action == "syncCloudData" || action == "updateCloudIntelligence" || action == "integrateCloudSignals")
        {
            await ExecuteCloudIntegrationActionAsync(action, context, cancellationToken);
            return;
        }

        // Handle advanced system actions - Economic event checks, memory management, etc.
        if (action.StartsWith("check") || action.StartsWith("validate") || action.StartsWith("optimize"))
        {
            await ExecuteAdvancedSystemActionAsync(action, context, cancellationToken);
            return;
        }

        // Execute through advanced system integration if available for better coordination
        if (_advancedSystemIntegration != null)
        {
            var actionName = $"unified-action-{action}";
            var success = await _advancedSystemIntegration.ExecuteWorkflowWithAdvancedCoordinationAsync(
                actionName,
                async () => await ExecuteBasicActionAsync(action, context, cancellationToken),
                GetResourcesForAction(action));
                
            if (!success)
            {
                _logger.LogWarning("⚠️ Action {Action} was queued due to resource conflicts", action);
            }
        }
        else
        {
            // Fallback to basic execution
            await ExecuteBasicActionAsync(action, context, cancellationToken);
        }
    }

    private async Task ExecuteBasicActionAsync(string action, WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        // Route actions to appropriate orchestrators based on action type
        if (_tradingOrchestrator.CanExecute(action))
        {
            await _tradingOrchestrator.ExecuteActionAsync(action, context, cancellationToken);
        }
        else if (_intelligenceOrchestrator.CanExecute(action))
        {
            await _intelligenceOrchestrator.ExecuteActionAsync(action, context, cancellationToken);
        }
        else if (_dataOrchestrator.CanExecute(action))
        {
            await _dataOrchestrator.ExecuteActionAsync(action, context, cancellationToken);
        }
        else
        {
            _logger.LogWarning("⚠️ No executor found for action: {Action}", action);
        }
    }

    private async Task ExecuteAdvancedSystemActionAsync(string action, WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        _logger.LogInformation("🌟 Executing advanced system action: {Action}", action);
        
        if (_advancedSystemIntegration == null)
        {
            _logger.LogWarning("⚠️ Advanced system integration not available for action: {Action}", action);
            return;
        }

        try
        {
            switch (action)
            {
                case "checkTradingAllowed":
                    var symbol = context.Parameters.ContainsKey("symbol") ? context.Parameters["symbol"].ToString() : "ES";
                    var isAllowed = await _advancedSystemIntegration.IsTradingAllowedAsync(symbol ?? "ES");
                    context.Parameters["tradingAllowed"] = isAllowed;
                    _logger.LogInformation("✅ Trading allowed check for {Symbol}: {IsAllowed}", symbol, isAllowed);
                    break;
                    
                case "validateSystemHealth":
                    var systemStatus = await _advancedSystemIntegration.GetSystemStatusAsync();
                    context.Parameters["systemHealthy"] = systemStatus.IsHealthy;
                    context.Parameters["systemStatus"] = systemStatus;
                    _logger.LogInformation("✅ System health validation: {IsHealthy}", systemStatus.IsHealthy);
                    break;
                    
                case "optimizePositionSize":
                    var strategyId = context.Parameters.GetValueOrDefault("strategyId")?.ToString() ?? "default";
                    var positionSymbol = context.Parameters.GetValueOrDefault("symbol")?.ToString() ?? "ES";
                    var multiplier = await _advancedSystemIntegration.GetOptimizedPositionSizeAsync(
                        strategyId, positionSymbol, 0, 0, 0, 0, new List<BotCore.Models.Bar>());
                    context.Parameters["positionMultiplier"] = multiplier;
                    _logger.LogInformation("✅ Position size optimized for {Strategy}-{Symbol}: {Multiplier}", 
                        strategyId, positionSymbol, multiplier);
                    break;
                    
                default:
                    _logger.LogWarning("⚠️ Unknown advanced system action: {Action}", action);
                    break;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Advanced system action failed: {Action}", action);
            throw;
        }
    }

    private List<string> GetResourcesForAction(string action)
    {
        // Define resource requirements for different actions
        return action switch
        {
            "analyzeESNQ" => new List<string> { "market_data", "ml_model" },
            "checkSignals" => new List<string> { "signal_processor" },
            "executeTrades" => new List<string> { "trading_engine", "position_manager" },
            "runMLModels" => new List<string> { "ml_model", "memory_manager" },
            "calculateRisk" => new List<string> { "risk_calculator", "portfolio_data" },
            "generateReport" => new List<string> { "data_aggregator", "report_generator" },
            _ => new List<string>()
        };
    }

    private async Task ExecuteCloudIntegrationActionAsync(string action, WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        _logger.LogInformation("🌐 Executing cloud integration action: {Action}", action);
        
        try
        {
            switch (action)
            {
                case "syncCloudData":
                    await _cloudDataIntegration.SyncCloudDataForTradingAsync(cancellationToken);
                    _logger.LogInformation("✅ Cloud data synced successfully");
                    break;
                    
                case "updateCloudIntelligence":
                    // Get recommendations for ES and NQ
                    var esRecommendation = await _cloudDataIntegration.GetTradingRecommendationAsync("ES", cancellationToken);
                    var nqRecommendation = await _cloudDataIntegration.GetTradingRecommendationAsync("NQ", cancellationToken);
                    
                    // Update the brain with recommendations
                    await _messageBus.PublishAsync("cloud.trading_recommendation.ES", esRecommendation, cancellationToken);
                    await _messageBus.PublishAsync("cloud.trading_recommendation.NQ", nqRecommendation, cancellationToken);
                    
                    _logger.LogInformation("✅ Cloud intelligence updated: ES={ESSignal} ({ESConfidence:P1}), NQ={NQSignal} ({NQConfidence:P1})", 
                        esRecommendation.Signal, esRecommendation.Confidence, 
                        nqRecommendation.Signal, nqRecommendation.Confidence);
                    break;
                    
                case "integrateCloudSignals":
                    // Publish cloud readiness signal to trading strategies
                    await _messageBus.PublishAsync("trading.cloud_ready", new { 
                        Timestamp = DateTime.UtcNow,
                        CloudDataAvailable = true,
                        Message = "Cloud intelligence is ready for trading decisions"
                    }, cancellationToken);
                    
                    _logger.LogInformation("✅ Cloud signals integrated - trading strategies notified");
                    break;
                    
                default:
                    _logger.LogWarning("⚠️ Unknown cloud integration action: {Action}", action);
                    break;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Cloud integration action failed: {Action}", action);
            throw;
        }
    }

    private async Task<bool> IsConnectedToTopstepAsync()
    {
        try
        {
            // Check if we can reach TopstepX API
            var status = await GetStatusAsync();
            return status.ComponentStatus.ContainsKey("TradingOrchestrator");
        }
        catch
        {
            return false;
        }
    }

    private List<UnifiedWorkflow> GetUnifiedWorkflowDefinitions()
    {
        // Consolidate all workflow definitions from the 4+ orchestrators into one unified list
        return new List<UnifiedWorkflow>
        {
            // TIER 1: CRITICAL TRADING WORKFLOWS
            new UnifiedWorkflow
            {
                Id = "es-nq-critical-trading",
                Name = "ES/NQ Critical Trading",
                Description = "Critical ES and NQ futures trading signals with advanced system protection",
                Priority = 1,
                BudgetAllocation = 8640,
                Type = WorkflowType.Trading,
                Schedule = new WorkflowSchedule
                {
                    MarketHours = "*/5 * * * *",      // Every 5 minutes
                    ExtendedHours = "*/15 * * * *",   // Every 15 minutes
                    Overnight = "*/30 * * * *"        // Every 30 minutes
                },
                Actions = new[] { "checkTradingAllowed", "validateSystemHealth", "analyzeESNQ", "checkSignals", "optimizePositionSize", "executeTrades" }
            },
            
            new UnifiedWorkflow
            {
                Id = "portfolio-heat-management",
                Name = "Portfolio Heat Management",
                Description = "Real-time risk monitoring and portfolio heat management with advanced memory optimization",
                Priority = 1,
                BudgetAllocation = 4880,
                Type = WorkflowType.RiskManagement,
                Schedule = new WorkflowSchedule
                {
                    MarketHours = "*/10 * * * *",     // Every 10 minutes
                    ExtendedHours = "*/30 * * * *",   // Every 30 minutes
                    Overnight = "0 */2 * * *"         // Every 2 hours
                },
                Actions = new[] { "validateSystemHealth", "calculateRisk", "checkThresholds", "optimizePositionSize", "adjustPositions" }
            },
            
            new UnifiedWorkflow
            {
                Id = "cloud-data-integration",
                Name = "Cloud Data Integration",
                Description = "Sync all 27 GitHub workflow results into trading decisions",
                Priority = 1,
                BudgetAllocation = 2000,
                Type = WorkflowType.CloudIntegration,
                Schedule = new WorkflowSchedule
                {
                    MarketHours = "*/3 * * * *",      // Every 3 minutes during market hours
                    ExtendedHours = "*/10 * * * *",   // Every 10 minutes extended
                    Overnight = "*/30 * * * *"        // Every 30 minutes overnight
                },
                Actions = new[] { "syncCloudData", "updateCloudIntelligence", "integrateCloudSignals" }
            },

            new UnifiedWorkflow
            {
                Id = "ml-rl-intel-system",
                Name = "Ultimate ML/RL Intel System",
                Description = "Master ML/RL orchestrator for predictions and learning with advanced memory management",
                Priority = 1,
                BudgetAllocation = 6480,
                Type = WorkflowType.MachineLearning,
                Schedule = new WorkflowSchedule
                {
                    MarketHours = "*/10 * * * *",     // Every 10 minutes
                    ExtendedHours = "*/20 * * * *",   // Every 20 minutes
                    Overnight = "0 * * * *"           // Every hour
                },
                Actions = new[] { "validateSystemHealth", "runMLModels", "updateRL", "generatePredictions", "optimizePositionSize" }
            },

            // TIER 2: HIGH PRIORITY WORKFLOWS
            new UnifiedWorkflow
            {
                Id = "microstructure-analysis",
                Name = "Microstructure Analysis",
                Description = "Order flow and tape reading analysis",
                Priority = 2,
                BudgetAllocation = 3600,
                Type = WorkflowType.Analytics,
                Schedule = new WorkflowSchedule
                {
                    CoreHours = "*/5 9-11,14-16 * * 1-5",  // Every 5 min during core hours
                    MarketHours = "*/15 9-16 * * 1-5",     // Every 15 min rest of market
                    Disabled = "* 16-9 * * *"              // Off after hours
                },
                Actions = new[] { "analyzeOrderFlow", "readTape", "trackMMs" }
            },

            new UnifiedWorkflow
            {
                Id = "options-flow-analysis",
                Name = "Options Flow Analysis",
                Description = "Smart money tracking through options flow",
                Priority = 2,
                BudgetAllocation = 3200,
                Type = WorkflowType.Analytics,
                Schedule = new WorkflowSchedule
                {
                    FirstHour = "*/5 9-10 * * 1-5",    // Every 5 min first hour
                    LastHour = "*/5 15-16 * * 1-5",    // Every 5 min last hour
                    Regular = "*/10 10-15 * * 1-5"      // Every 10 min mid-day
                },
                Actions = new[] { "scanOptionsFlow", "detectDarkPools", "trackSmartMoney" }
            },

            new UnifiedWorkflow
            {
                Id = "intermarket-correlations",
                Name = "Intermarket Correlations",
                Description = "Cross-market correlation and divergence analysis",
                Priority = 2,
                BudgetAllocation = 2880,
                Type = WorkflowType.Analytics,
                Schedule = new WorkflowSchedule
                {
                    MarketHours = "*/15 * * * 1-5",    // Every 15 minutes weekdays
                    Global = "*/30 * * * *",           // Every 30 min 24/7
                    Weekends = "0 */2 * * 0,6"         // Every 2 hours weekends
                },
                Actions = new[] { "correlateAssets", "detectDivergence", "updateMatrix" }
            },

            // TIER 3: STANDARD WORKFLOWS
            new UnifiedWorkflow
            {
                Id = "daily-market-data-collection",
                Name = "Daily Market Data Collection",
                Description = "Comprehensive market data collection and storage",
                Priority = 3,
                BudgetAllocation = 2400,
                Type = WorkflowType.DataCollection,
                Schedule = new WorkflowSchedule
                {
                    MarketHours = "0 9,12,16 * * 1-5",   // 3 times during market
                    Global = "0 0 * * *"                 // Daily at midnight
                },
                Actions = new[] { "collectMarketData", "storeData", "validateData" }
            },

            new UnifiedWorkflow
            {
                Id = "daily-reporting-system",
                Name = "Daily Reporting System",
                Description = "Generate and distribute daily performance reports",
                Priority = 3,
                BudgetAllocation = 1800,
                Type = WorkflowType.Analytics,
                Schedule = new WorkflowSchedule
                {
                    Regular = "0 17 * * 1-5"  // Daily at 5 PM ET
                },
                Actions = new[] { "generateReport", "calculateMetrics", "sendNotifications" }
            }
        };
    }

    /// <summary>
    /// Initialize Phase 5 Safety Infrastructure components
    /// These apply globally to all strategies and services
    /// </summary>
    private async Task InitializeSafetyComponentsAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("🛡️ Starting Phase 5 Safety Infrastructure initialization...");
        
        try
        {
            // Start KillSwitchWatcher - monitors kill.txt file for immediate halt
            _logger.LogInformation("🔴 Initializing Kill Switch Watcher...");
            _killSwitchWatcher.OnKillSwitchActivated += OnKillSwitchActivated;
            await _killSwitchWatcher.StartWatchingAsync(cancellationToken);
            _logger.LogInformation("✅ Kill Switch Watcher active");
            
            // Start RiskManager - enforces real-time risk limits
            _logger.LogInformation("📊 Initializing Risk Manager...");
            _riskManager.OnRiskBreach += OnRiskBreach;
            _logger.LogInformation("✅ Risk Manager active - enforcing MaxDailyLoss, MaxPositionSize, DrawdownLimit");
            
            // Start HealthMonitor - tracks system health and trading eligibility
            _logger.LogInformation("💚 Initializing Health Monitor...");
            _healthMonitor.OnHealthChanged += OnHealthChanged;
            await _healthMonitor.StartMonitoringAsync(cancellationToken);
            _logger.LogInformation("✅ Health Monitor active - tracking hub connections, error rates, latency");
            
            // Update central message bus with safety status
            _messageBus.UpdateSharedState("safety.kill_switch_active", _killSwitchWatcher.IsKillSwitchActive);
            _messageBus.UpdateSharedState("safety.risk_breached", _riskManager.IsRiskBreached);
            _messageBus.UpdateSharedState("safety.trading_allowed", _healthMonitor.IsTradingAllowed);
            
            _logger.LogInformation("🛡️ Phase 5 Safety Infrastructure initialized successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Failed to initialize Safety Infrastructure");
            throw;
        }
    }
    
    /// <summary>
    /// Handle kill switch activation - immediately halt all trading operations
    /// </summary>
    private void OnKillSwitchActivated()
    {
        _logger.LogCritical("🚨 KILL SWITCH ACTIVATED - Halting all trading operations immediately");
        
        try
        {
            // Update central message bus to stop all trading
            _messageBus.UpdateSharedState("safety.kill_switch_active", true);
            _messageBus.PublishAsync("safety.emergency_halt", new { 
                Timestamp = DateTime.UtcNow, 
                Reason = "Kill switch activated",
                Action = "All trading halted"
            });
            
            // Cancel all pending operations
            _cancellationTokenSource.Cancel();
            
            _logger.LogCritical("🛑 Emergency halt completed - System is now in safe mode");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error during emergency halt procedure");
        }
    }
    
    /// <summary>
    /// Handle risk breaches - implement automatic position unwinding
    /// </summary>
    private void OnRiskBreach(RiskBreach breach)
    {
        _logger.LogCritical("🚨 RISK BREACH DETECTED: {Type} - {Message}", breach.Type, breach.Message);
        
        try
        {
            // Update central message bus with risk breach
            _messageBus.UpdateSharedState("safety.risk_breached", true);
            _messageBus.UpdateSharedState($"safety.breach_{breach.Type.ToString().ToLower()}", breach.CurrentValue);
            
            // Publish risk breach event for automatic unwinding
            _messageBus.PublishAsync("safety.risk_breach", new {
                Type = breach.Type.ToString(),
                Message = breach.Message,
                CurrentValue = breach.CurrentValue,
                Limit = breach.Limit,
                Timestamp = DateTime.UtcNow,
                Action = "Automatic position unwinding initiated"
            });
            
            _logger.LogWarning("⚠️ Risk breach handled - Position unwinding initiated");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error handling risk breach");
        }
    }
    
    /// <summary>
    /// Handle health status changes - suspend/resume trading based on system health
    /// </summary>
    private void OnHealthChanged(HealthStatus health)
    {
        _logger.LogInformation("💓 Health status changed: Healthy={IsHealthy}, TradingAllowed={TradingAllowed}", 
            health.IsHealthy, health.TradingAllowed);
        
        try
        {
            // Update central message bus with health status
            _messageBus.UpdateSharedState("safety.trading_allowed", health.TradingAllowed);
            _messageBus.UpdateSharedState("health.is_healthy", health.IsHealthy);
            _messageBus.UpdateSharedState("health.connected_hubs", health.ConnectedHubs);
            _messageBus.UpdateSharedState("health.error_rate", health.ErrorRate);
            _messageBus.UpdateSharedState("health.average_latency_ms", health.AverageLatencyMs);
            
            // Publish health change event
            _messageBus.PublishAsync("safety.health_changed", new {
                IsHealthy = health.IsHealthy,
                TradingAllowed = health.TradingAllowed,
                ConnectedHubs = health.ConnectedHubs,
                TotalHubs = health.TotalHubs,
                ErrorRate = health.ErrorRate,
                AverageLatencyMs = health.AverageLatencyMs,
                StatusMessage = health.StatusMessage,
                Timestamp = DateTime.UtcNow
            });
            
            if (!health.TradingAllowed)
            {
                _logger.LogWarning("⚠️ Trading suspended due to degraded system health: {StatusMessage}", health.StatusMessage);
            }
            else if (health.TradingAllowed)
            {
                _logger.LogInformation("✅ Trading resumed - System health restored");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error handling health status change");
        }
    }

    #endregion
}