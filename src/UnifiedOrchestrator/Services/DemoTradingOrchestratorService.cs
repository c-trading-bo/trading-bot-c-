using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;
using TradingBot.Abstractions;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Demo version of the trading orchestrator that can run without TopstepX credentials
/// </summary>
public class DemoTradingOrchestratorService : TradingBot.Abstractions.ITradingOrchestrator
{
    private readonly ILogger<DemoTradingOrchestratorService> _logger;
    private readonly ICentralMessageBus _messageBus;
    private bool _isConnected = false;

    public IReadOnlyList<string> SupportedActions { get; } = new[]
    {
        "analyzeESNQ", "checkSignals", "executeTrades",
        "calculateRisk", "checkThresholds", "adjustPositions",
        "analyzeOrderFlow", "readTape", "trackMMs",
        "scanOptionsFlow", "detectDarkPools", "trackSmartMoney"
    };

    public DemoTradingOrchestratorService(
        ILogger<DemoTradingOrchestratorService> logger,
        ICentralMessageBus messageBus)
    {
        _logger = logger;
        _messageBus = messageBus;
    }

    public async Task ConnectAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("🔌 [DEMO MODE] Simulating TopstepX connection...");
        await Task.Delay(1000, cancellationToken); // Simulate connection time
        _isConnected = true;
        _logger.LogInformation("✅ [DEMO MODE] Connected to simulated TopstepX environment");
    }

    public async Task DisconnectAsync()
    {
        _logger.LogInformation("🔌 [DEMO MODE] Disconnecting from simulated TopstepX...");
        await Task.Delay(500);
        _isConnected = false;
        _logger.LogInformation("✅ [DEMO MODE] Disconnected from simulated environment");
    }

    public async Task ExecuteESNQTradingAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("📊 [DEMO MODE] Executing ES/NQ trading analysis with cloud intelligence...");
        
        await Task.Delay(200, cancellationToken);
        
        // 🌐 GET CLOUD INTELLIGENCE IN DEMO MODE TOO
        var esCloudRecommendation = _messageBus.GetSharedState<CloudTradingRecommendation>("cloud.trading_recommendation.ES");
        var nqCloudRecommendation = _messageBus.GetSharedState<CloudTradingRecommendation>("cloud.trading_recommendation.NQ");
        
        if (esCloudRecommendation != null)
        {
            _logger.LogInformation("🧠 [DEMO] ES Cloud Intelligence: {Signal} (confidence: {Confidence:P1})", 
                esCloudRecommendation.Signal, esCloudRecommendation.Confidence);
            context.Logs.Add($"[DEMO] ES Cloud Signal: {esCloudRecommendation.Signal} ({esCloudRecommendation.Confidence:P1})");
        }
        
        if (nqCloudRecommendation != null)
        {
            _logger.LogInformation("🧠 [DEMO] NQ Cloud Intelligence: {Signal} (confidence: {Confidence:P1})", 
                nqCloudRecommendation.Signal, nqCloudRecommendation.Confidence);
            context.Logs.Add($"[DEMO] NQ Cloud Signal: {nqCloudRecommendation.Signal} ({nqCloudRecommendation.Confidence:P1})");
        }
        
        context.Logs.Add("[DEMO] ES Price: $5,025.75, NQ Price: $17,485.25");
        context.Logs.Add("[DEMO] Bullish signal detected on ES");
        context.Logs.Add("[DEMO] Neutral signal on NQ");
        context.Parameters["DemoESPrice"] = 5025.75m;
        context.Parameters["DemoNQPrice"] = 17485.25m;
        
        _logger.LogInformation("✅ [DEMO MODE] ES/NQ analysis completed");
    }

    public async Task ManagePortfolioRiskAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("⚖️ [DEMO MODE] Managing portfolio risk...");
        
        await Task.Delay(150, cancellationToken);
        
        context.Logs.Add("[DEMO] Portfolio PnL: $245.50");
        context.Logs.Add("[DEMO] Risk exposure: 15% of account");
        context.Logs.Add("[DEMO] All risk thresholds within limits");
        context.Parameters["DemoPnL"] = 245.50m;
        context.Parameters["DemoRiskExposure"] = 0.15m;
        
        _logger.LogInformation("✅ [DEMO MODE] Risk management completed");
    }

    public async Task AnalyzeMicrostructureAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("🔬 [DEMO MODE] Analyzing market microstructure...");
        
        await Task.Delay(100, cancellationToken);
        
        context.Logs.Add("[DEMO] Order flow: Bullish ES, Neutral NQ");
        context.Logs.Add("[DEMO] Market maker activity detected");
        context.Parameters["DemoOrderFlow"] = "Bullish";
        context.Parameters["DemoMMActivity"] = true;
        
        _logger.LogInformation("✅ [DEMO MODE] Microstructure analysis completed");
    }

    public async Task<WorkflowExecutionResult> ExecuteActionAsync(string action, WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        var startTime = DateTime.UtcNow;
        
        try
        {
            switch (action)
            {
                case "analyzeESNQ":
                case "checkSignals":
                case "executeTrades":
                    await ExecuteESNQTradingAsync(context, cancellationToken);
                    break;
                    
                case "calculateRisk":
                case "checkThresholds":
                case "adjustPositions":
                    await ManagePortfolioRiskAsync(context, cancellationToken);
                    break;
                    
                case "analyzeOrderFlow":
                case "readTape":
                case "trackMMs":
                    await AnalyzeMicrostructureAsync(context, cancellationToken);
                    break;
                    
                case "scanOptionsFlow":
                case "detectDarkPools":
                case "trackSmartMoney":
                    _logger.LogInformation("📈 [DEMO MODE] Analyzing options flow...");
                    await Task.Delay(100, cancellationToken);
                    context.Logs.Add("[DEMO] Smart money detected: Bullish bias");
                    break;
                    
                default:
                    throw new NotSupportedException($"Action '{action}' is not supported by DemoTradingOrchestrator");
            }

            return new WorkflowExecutionResult
            {
                Success = true,
                Duration = DateTime.UtcNow - startTime
            };
        }
        catch (Exception ex)
        {
            return new WorkflowExecutionResult
            {
                Success = false,
                ErrorMessage = ex.Message,
                Duration = DateTime.UtcNow - startTime
            };
        }
    }

    public bool CanExecute(string action)
    {
        return SupportedActions.Contains(action);
    }
}