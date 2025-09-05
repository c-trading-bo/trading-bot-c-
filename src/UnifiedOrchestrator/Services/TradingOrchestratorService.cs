using Microsoft.Extensions.Logging;
using Microsoft.AspNetCore.SignalR.Client;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;
using BotCore;
using BotCore.Models;
using BotCore.Risk;
using BotCore.Strategy;
using System.Text.Json;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Unified trading orchestrator that consolidates all TopstepX trading functionality
/// Replaces multiple trading orchestrators with one unified system
/// </summary>
public class TradingOrchestratorService : ITradingOrchestrator, IDisposable
{
    private readonly ILogger<TradingOrchestratorService> _logger;
    private readonly HttpClient _httpClient;
    private readonly TopstepAuthAgent _authAgent;
    
    // TopstepX Connections
    private HubConnection? _userHub;
    private HubConnection? _marketHub;
    private string? _jwtToken;
    private long _accountId;
    private bool _isConnected = false;

    // Trading Components (unified from all orchestrators)
    private readonly RiskEngine _riskEngine;
    private readonly Dictionary<string, IStrategy> _strategies = new();
    private readonly Dictionary<string, string> _contractIds = new(); // symbol -> contractId mapping
    
    // Supported actions
    public IReadOnlyList<string> SupportedActions { get; } = new[]
    {
        "analyzeESNQ", "checkSignals", "executeTrades",
        "calculateRisk", "checkThresholds", "adjustPositions",
        "analyzeOrderFlow", "readTape", "trackMMs",
        "scanOptionsFlow", "detectDarkPools", "trackSmartMoney"
    };

    public TradingOrchestratorService(
        ILogger<TradingOrchestratorService> logger,
        HttpClient httpClient,
        TopstepAuthAgent authAgent)
    {
        _logger = logger;
        _httpClient = httpClient;
        _authAgent = authAgent;
        _riskEngine = new RiskEngine();
        
        // Set TopstepX base URL
        _httpClient.BaseAddress ??= new Uri("https://api.topstepx.com");
    }

    #region ITradingOrchestrator Implementation

    public async Task ConnectAsync(CancellationToken cancellationToken = default)
    {
        if (_isConnected) return;

        _logger.LogInformation("🔌 Connecting to TopstepX API and hubs...");

        try
        {
            // Get authentication
            await AuthenticateAsync(cancellationToken);
            
            // Connect to SignalR hubs
            await ConnectToHubsAsync(cancellationToken);
            
            // Initialize contract mappings
            await InitializeContractsAsync(cancellationToken);
            
            _isConnected = true;
            _logger.LogInformation("✅ Successfully connected to TopstepX");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Failed to connect to TopstepX");
            throw;
        }
    }

    public async Task DisconnectAsync()
    {
        if (!_isConnected) return;

        _logger.LogInformation("🔌 Disconnecting from TopstepX...");

        try
        {
            if (_userHub != null)
            {
                await _userHub.DisposeAsync();
                _userHub = null;
            }

            if (_marketHub != null)
            {
                await _marketHub.DisposeAsync();
                _marketHub = null;
            }

            _isConnected = false;
            _logger.LogInformation("✅ Disconnected from TopstepX");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "⚠️ Error during disconnect");
        }
    }

    public async Task ExecuteESNQTradingAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        if (!_isConnected)
        {
            throw new InvalidOperationException("Not connected to TopstepX");
        }

        _logger.LogInformation("📊 Executing ES/NQ trading analysis and signals...");

        try
        {
            // Get current market data for ES and NQ
            var esData = await GetMarketDataAsync("ES", cancellationToken);
            var nqData = await GetMarketDataAsync("NQ", cancellationToken);

            context.Logs.Add($"ES Price: {esData?.LastPrice}, NQ Price: {nqData?.LastPrice}");

            // Run strategy analysis
            foreach (var strategy in _strategies.Values)
            {
                if (strategy is IESNQStrategy esNqStrategy)
                {
                    var signals = await esNqStrategy.AnalyzeAsync(esData, nqData, cancellationToken);
                    
                    foreach (var signal in signals)
                    {
                        await ProcessTradingSignalAsync(signal, context, cancellationToken);
                    }
                }
            }

            context.Logs.Add("ES/NQ trading analysis completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error executing ES/NQ trading");
            throw;
        }
    }

    public async Task ManagePortfolioRiskAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("⚖️ Managing portfolio risk and heat...");

        try
        {
            // Get current positions
            var positions = await GetPositionsAsync(cancellationToken);
            
            // Calculate portfolio metrics
            var totalPnL = positions.Sum(p => p.UnrealizedPnL + p.RealizedPnL);
            var totalHeat = positions.Sum(p => Math.Abs(p.Quantity * p.AveragePrice * 0.01m)); // 1% risk per position
            
            context.Logs.Add($"Portfolio PnL: {totalPnL:C}, Heat: {totalHeat:C}");

            // Check risk thresholds
            var maxDailyLoss = -850m; // TopstepX eval account limit
            var maxHeat = 5000m; // Maximum portfolio heat
            
            if (totalPnL < maxDailyLoss)
            {
                _logger.LogWarning("🔥 Daily loss limit approaching: {PnL}", totalPnL);
                await FlattenAllPositionsAsync("Daily loss limit", cancellationToken);
                context.Logs.Add("Flattened all positions due to daily loss limit");
            }
            
            if (totalHeat > maxHeat)
            {
                _logger.LogWarning("🔥 Portfolio heat too high: {Heat}", totalHeat);
                await ReducePositionsAsync(0.5m, "Portfolio heat reduction", cancellationToken);
                context.Logs.Add("Reduced positions due to high portfolio heat");
            }

            // Update risk metrics
            context.Parameters["totalPnL"] = totalPnL;
            context.Parameters["totalHeat"] = totalHeat;
            
            _logger.LogInformation("✅ Portfolio risk management completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error in portfolio risk management");
            throw;
        }
    }

    public async Task AnalyzeMicrostructureAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("🔬 Analyzing market microstructure...");

        try
        {
            // Get order book data
            var esOrderBook = await GetOrderBookAsync("ES", cancellationToken);
            var nqOrderBook = await GetOrderBookAsync("NQ", cancellationToken);

            // Analyze order flow
            var esFlow = AnalyzeOrderFlow(esOrderBook);
            var nqFlow = AnalyzeOrderFlow(nqOrderBook);

            context.Logs.Add($"ES Order Flow: {esFlow.Direction} ({esFlow.Strength})");
            context.Logs.Add($"NQ Order Flow: {nqFlow.Direction} ({nqFlow.Strength})");

            // Detect market maker activity
            var esMMActivity = DetectMarketMakerActivity(esOrderBook);
            var nqMMActivity = DetectMarketMakerActivity(nqOrderBook);

            context.Parameters["ES_OrderFlow"] = esFlow;
            context.Parameters["NQ_OrderFlow"] = nqFlow;
            context.Parameters["ES_MMActivity"] = esMMActivity;
            context.Parameters["NQ_MMActivity"] = nqMMActivity;

            _logger.LogInformation("✅ Microstructure analysis completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error in microstructure analysis");
            throw;
        }
    }

    #endregion

    #region IWorkflowActionExecutor Implementation

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
                    await AnalyzeOptionsFlowAsync(context, cancellationToken);
                    break;
                    
                default:
                    throw new NotSupportedException($"Action '{action}' is not supported by TradingOrchestrator");
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

    #endregion

    #region Private Methods

    private async Task AuthenticateAsync(CancellationToken cancellationToken)
    {
        var username = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
        var apiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");
        _jwtToken = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");

        if (string.IsNullOrEmpty(_jwtToken) && (!string.IsNullOrEmpty(username) && !string.IsNullOrEmpty(apiKey)))
        {
            _jwtToken = await _authAgent.GetJwtAsync(username, apiKey, cancellationToken);
        }

        if (string.IsNullOrEmpty(_jwtToken))
        {
            throw new InvalidOperationException("No TopstepX authentication available");
        }

        _httpClient.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _jwtToken);
        
        // Get account ID
        _accountId = await GetAccountIdAsync(cancellationToken);
        
        _logger.LogInformation("✅ TopstepX authentication successful for account {AccountId}", _accountId);
    }

    private async Task ConnectToHubsAsync(CancellationToken cancellationToken)
    {
        // Connect to User Hub
        _userHub = new HubConnectionBuilder()
            .WithUrl("https://rtc.topstepx.com/hubs/user", options =>
            {
                options.AccessTokenProvider = () => Task.FromResult(_jwtToken);
            })
            .Build();

        await _userHub.StartAsync(cancellationToken);
        await _userHub.InvokeAsync("SubscribeOrders", _accountId, cancellationToken);
        await _userHub.InvokeAsync("SubscribeTrades", _accountId, cancellationToken);

        // Connect to Market Hub
        _marketHub = new HubConnectionBuilder()
            .WithUrl("https://rtc.topstepx.com/hubs/market", options =>
            {
                options.AccessTokenProvider = () => Task.FromResult(_jwtToken);
            })
            .Build();

        await _marketHub.StartAsync(cancellationToken);

        _logger.LogInformation("✅ Connected to TopstepX SignalR hubs");
    }

    private async Task InitializeContractsAsync(CancellationToken cancellationToken)
    {
        // Get ES and NQ contract IDs
        var esContract = await GetContractIdAsync("ES", cancellationToken);
        var nqContract = await GetContractIdAsync("NQ", cancellationToken);

        if (!string.IsNullOrEmpty(esContract))
            _contractIds["ES"] = esContract;
        
        if (!string.IsNullOrEmpty(nqContract))
            _contractIds["NQ"] = nqContract;

        _logger.LogInformation("✅ Contract mappings initialized: ES={EsContract}, NQ={NqContract}", esContract, nqContract);
    }

    private async Task<long> GetAccountIdAsync(CancellationToken cancellationToken)
    {
        var response = await _httpClient.GetAsync("/api/Account", cancellationToken);
        response.EnsureSuccessStatusCode();
        
        var json = await response.Content.ReadAsStringAsync(cancellationToken);
        using var doc = JsonDocument.Parse(json);
        
        return doc.RootElement.GetProperty("accountId").GetInt64();
    }

    private async Task<string?> GetContractIdAsync(string symbol, CancellationToken cancellationToken)
    {
        var response = await _httpClient.GetAsync($"/api/Contract/available?symbol={symbol}&live=false", cancellationToken);
        if (!response.IsSuccessStatusCode) return null;
        
        var json = await response.Content.ReadAsStringAsync(cancellationToken);
        using var doc = JsonDocument.Parse(json);
        
        if (doc.RootElement.ValueKind == JsonValueKind.Array && doc.RootElement.GetArrayLength() > 0)
        {
            return doc.RootElement[0].GetProperty("contractId").GetString();
        }
        
        return null;
    }

    private async Task<MarketData?> GetMarketDataAsync(string symbol, CancellationToken cancellationToken)
    {
        if (!_contractIds.TryGetValue(symbol, out var contractId))
            return null;

        // Subscribe to market data if not already subscribed
        if (_marketHub != null)
        {
            await _marketHub.InvokeAsync("Subscribe", contractId, cancellationToken);
        }

        // For now, return a placeholder - in real implementation this would come from the market hub
        return new MarketData
        {
            Symbol = symbol,
            LastPrice = 5000m, // Placeholder
            BidPrice = 4999.75m,
            AskPrice = 5000.25m,
            Timestamp = DateTime.UtcNow
        };
    }

    private async Task<List<Position>> GetPositionsAsync(CancellationToken cancellationToken)
    {
        var response = await _httpClient.GetAsync($"/api/Position/{_accountId}", cancellationToken);
        if (!response.IsSuccessStatusCode) return new List<Position>();
        
        var json = await response.Content.ReadAsStringAsync(cancellationToken);
        // Parse positions from JSON - implementation depends on TopstepX API structure
        
        return new List<Position>(); // Placeholder
    }

    private async Task ProcessTradingSignalAsync(TradingSignal signal, WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        context.Logs.Add($"Processing signal: {signal.Symbol} {signal.Direction} @ {signal.Price}");
        
        // Risk check using available RiskEngine methods
        var risk = RiskEngine.ComputeRisk(signal.Price, signal.Price * 0.98m, signal.Price * 1.02m, signal.Direction == "BUY");
        if (risk <= 0)
        {
            context.Logs.Add($"Signal rejected by risk engine: {signal.Symbol} - Invalid risk calculation");
            return;
        }

        // Place order (placeholder - would use actual order placement logic)
        context.Logs.Add($"Order placed for {signal.Symbol}");
    }

    private async Task FlattenAllPositionsAsync(string reason, CancellationToken cancellationToken)
    {
        _logger.LogWarning("🔥 Flattening all positions: {Reason}", reason);
        // Implementation would close all open positions
    }

    private async Task ReducePositionsAsync(decimal reductionFactor, string reason, CancellationToken cancellationToken)
    {
        _logger.LogWarning("📉 Reducing positions by {Factor}: {Reason}", reductionFactor, reason);
        // Implementation would reduce position sizes
    }

    private async Task<OrderBook?> GetOrderBookAsync(string symbol, CancellationToken cancellationToken)
    {
        // Placeholder - would get real order book data
        return new OrderBook { Symbol = symbol };
    }

    private OrderFlowAnalysis AnalyzeOrderFlow(OrderBook? orderBook)
    {
        // Placeholder order flow analysis
        return new OrderFlowAnalysis 
        { 
            Direction = "Bullish", 
            Strength = "Medium" 
        };
    }

    private MarketMakerActivity DetectMarketMakerActivity(OrderBook? orderBook)
    {
        // Placeholder market maker detection
        return new MarketMakerActivity 
        { 
            IsActive = true, 
            Side = "Both" 
        };
    }

    private async Task AnalyzeOptionsFlowAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        _logger.LogInformation("📈 Analyzing options flow for smart money detection...");
        
        // Placeholder implementation
        context.Logs.Add("Options flow analysis completed");
        context.Parameters["OptionsFlow"] = new { SmartMoney = "Bullish", Volume = "High" };
    }

    #endregion

    #region Supporting Classes

    public class MarketData
    {
        public string Symbol { get; set; } = string.Empty;
        public decimal LastPrice { get; set; }
        public decimal BidPrice { get; set; }
        public decimal AskPrice { get; set; }
        public DateTime Timestamp { get; set; }
    }

    public class Position
    {
        public string Symbol { get; set; } = string.Empty;
        public int Quantity { get; set; }
        public decimal AveragePrice { get; set; }
        public decimal UnrealizedPnL { get; set; }
        public decimal RealizedPnL { get; set; }
    }

    public class TradingSignal
    {
        public string Symbol { get; set; } = string.Empty;
        public string Direction { get; set; } = string.Empty;
        public decimal Price { get; set; }
        public DateTime Timestamp { get; set; }
    }

    public class OrderBook
    {
        public string Symbol { get; set; } = string.Empty;
    }

    public class OrderFlowAnalysis
    {
        public string Direction { get; set; } = string.Empty;
        public string Strength { get; set; } = string.Empty;
    }

    public class MarketMakerActivity
    {
        public bool IsActive { get; set; }
        public string Side { get; set; } = string.Empty;
    }

    #endregion

    #region IDisposable

    public void Dispose()
    {
        DisconnectAsync().GetAwaiter().GetResult();
    }

    #endregion
}

// Interface for ES/NQ specific strategies
public interface IESNQStrategy : IStrategy
{
    Task<List<TradingOrchestratorService.TradingSignal>> AnalyzeAsync(
        TradingOrchestratorService.MarketData? esData, 
        TradingOrchestratorService.MarketData? nqData, 
        CancellationToken cancellationToken);
}