using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using global::BotCore.Helpers;
using global::BotCore.Configuration;
using global::BotCore.Services;
using TradingBot.Abstractions;

namespace TradingBot.UnifiedOrchestrator.Services;

internal record HealthScoreResult(
    int HealthScore,
    string Status,
    Dictionary<string, object> InstrumentHealth,
    Dictionary<string, object> SuiteStats,
    DateTime LastCheck,
    bool Initialized);

internal record PortfolioStatusResult(
    Dictionary<string, object> Portfolio,
    Dictionary<string, PositionInfo> Positions,
    DateTime Timestamp);

internal record PositionInfo(
    int Size,
    decimal AveragePrice,
    decimal UnrealizedPnL,
    decimal RealizedPnL);

internal record BarEventData(
    string Type,
    string Instrument,
    DateTime Timestamp,
    decimal Open,
    decimal High,
    decimal Low,
    decimal Close,
    long Volume);

internal class TopstepXAdapterService : TradingBot.Abstractions.ITopstepXAdapterService, IAsyncDisposable, IDisposable
{
    private readonly ILogger<TopstepXAdapterService> _logger;
    private readonly TopstepXConfiguration _config;
    private readonly string[] _instruments;
    private Process? _pythonProcess;
    private bool _isInitialized;
    private double _connectionHealth;
    private readonly object _processLock = new();
    private readonly SemaphoreSlim _initLock = new SemaphoreSlim(1, 1);
    private bool _disposed;
    
    // PERSISTENT PROCESS: stdin/stdout communication
    private StreamWriter? _processStdin;
    private Task? _stdoutReaderTask;
    private readonly CancellationTokenSource _processCts = new();
    private readonly ConcurrentDictionary<string, TaskCompletionSource<JsonElement>> _pendingResponses = new();
    
    // PHASE 2: Fill event subscription infrastructure
    private readonly CancellationTokenSource _fillEventCts = new();
    private Task? _fillEventListenerTask;
    private readonly ConcurrentQueue<FillEventData> _fillEventQueue = new();

    // PHASE 2: Fill event callback for OrderExecutionService
    public event EventHandler<FillEventData>? FillEventReceived;
    
    // BAR EVENT STREAMING: Bar event subscription infrastructure
    private readonly CancellationTokenSource _barEventCts = new();
    private Task? _barEventListenerTask;
    private readonly ConcurrentQueue<BarEventData> _barEventQueue = new();
    
    // BAR EVENT STREAMING: Bar event callback for data integration
    public event EventHandler<BarEventData>? BarEventReceived;

    public TopstepXAdapterService(
        ILogger<TopstepXAdapterService> logger,
        IOptions<TopstepXConfiguration> config)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        
        // Log constructor entry for debugging DI issues
        _logger.LogInformation("🏗️ [TopstepXAdapter] Constructor invoked - initializing service");
        
        try
        {
            if (config == null)
            {
                _logger.LogError("❌ [TopstepXAdapter] IOptions<TopstepXConfiguration> is null");
                throw new ArgumentNullException(nameof(config), "Configuration options cannot be null");
            }
            
            _config = config.Value;
            
            if (_config == null)
            {
                _logger.LogError("❌ [TopstepXAdapter] TopstepXConfiguration.Value is null - check appsettings.json");
                throw new InvalidOperationException("TopstepX configuration is missing or invalid in appsettings.json");
            }
            
            _logger.LogInformation("✅ [TopstepXAdapter] Configuration loaded successfully");
            _logger.LogInformation("   📍 ApiBaseUrl: {ApiBase}", _config.ApiBaseUrl);
            _logger.LogInformation("   🔌 UserHubUrl: {UserHub}", _config.UserHubUrl);
            _logger.LogInformation("   📊 MarketHubUrl: {MarketHub}", _config.MarketHubUrl);
        }
        catch (Exception ex)
        {
            _logger.LogCritical(ex, "💥 [TopstepXAdapter] FATAL ERROR in constructor - service cannot be created");
            throw;
        }
        
        _instruments = new[] { "ES", "NQ" }; // Support ES and NQ as specified
        _isInitialized = false;
        _connectionHealth = 0.0;
        
        _logger.LogInformation("✅ [TopstepXAdapter] Constructor completed successfully");
    }

    public bool IsConnected => _isInitialized && _connectionHealth >= 80.0;
    
    // Interface expects string, provide both for compatibility
    public string ConnectionHealth => _connectionHealth.ToString("F1");
    public double ConnectionHealthScore => _connectionHealth;

    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        // CRITICAL: Lab Mode should NEVER connect to TopstepX API
        // Lab Mode uses pre-loaded JSON files only (NO API calls)
        var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
        if (labMode == "1" || labMode?.ToLowerInvariant() == "true")
        {
            _logger.LogInformation("🧪 [LAB-MODE] TopstepX adapter initialization SKIPPED - Lab Mode uses offline data only");
            _logger.LogInformation("📊 [LAB-MODE] Training will use pre-loaded JSON files (NO API calls)");
            _isInitialized = false;
            _connectionHealth = 0.0;
            return;
        }
        
        // Prevent multiple concurrent initialization attempts
        await _initLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_isInitialized)
            {
                _logger.LogWarning("[ADAPTER] Adapter already initialized");
                return;
            }

            try
            {
                _logger.LogInformation("🚀 Initializing TopstepX Python SDK adapter in PERSISTENT mode...");
                
                // Validate Python SDK is available
                await ValidatePythonSDKAsync(cancellationToken).ConfigureAwait(false);

                // Start persistent Python process
                await StartPersistentPythonProcessAsync(cancellationToken).ConfigureAwait(false);
                
                _logger.LogInformation("✅ TopstepX adapter initialized successfully in PERSISTENT mode - single Python process running");
                
                _isInitialized = true;
                _connectionHealth = 100.0;
            }
            catch (Exception ex) when (ex is not InvalidOperationException)
            {
                _logger.LogError(ex, "Exception during TopstepX adapter initialization");
                throw new InvalidOperationException("Failed to initialize TopstepX adapter", ex);
            }
        }
        finally
        {
            _initLock.Release();
        }
    }

    public async Task<decimal> GetPriceAsync(string symbol, CancellationToken cancellationToken = default)
    {
        // CRITICAL: Lab Mode should NEVER poll live prices
        var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
        if (labMode == "1" || labMode?.ToLowerInvariant() == "true")
        {
            throw new InvalidOperationException("GetPriceAsync not available in Lab Mode - use historical data only");
        }
        
        if (!_isInitialized)
        {
            throw new InvalidOperationException("Adapter not initialized. Call InitializeAsync first.");
        }

        if (!Array.Exists(_instruments, i => i == symbol))
        {
            throw new ArgumentException($"Symbol {symbol} not supported. Supported: {string.Join(", ", _instruments)}");
        }

        try
        {
            var parameters = new { symbol };
            var result = await SendPersistentCommandAsync("get_price", parameters, cancellationToken).ConfigureAwait(false);
            
            if (result.TryGetProperty("price", out var priceElement))
            {
                var price = priceElement.GetDecimal();
                _logger.LogDebug("💰 [PRICE] {Symbol}: ${Price:F2}", symbol, price);
                return price;
            }
            
            throw new InvalidOperationException($"Failed to get price for {symbol}: No price in response");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [PRICE] Error getting price for {Symbol}", symbol);
            throw;
        }
    }

    public async Task<OrderExecutionResult> PlaceOrderAsync(
        string symbol, 
        int size, 
        decimal stopLoss, 
        decimal takeProfit, 
        CancellationToken cancellationToken = default)
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException("Adapter not initialized. Call InitializeAsync first.");
        }

        try
        {
            // Validate and round prices to valid tick increments before placing order
            try
            {
                stopLoss = PriceHelper.RoundToTick(stopLoss, symbol);
                takeProfit = PriceHelper.RoundToTick(takeProfit, symbol);
            }
            catch (ArgumentException ex)
            {
                _logger.LogError("Price validation failed: {Error}", ex.Message);
                return new OrderExecutionResult(
                    false,
                    null,
                    ex.Message,
                    symbol,
                    size,
                    0m,
                    stopLoss,
                    takeProfit,
                    DateTime.UtcNow);
            }
            
            var currentPrice = await GetPriceAsync(symbol, cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation(
                "[ORDER] Placing bracket order: {Symbol} size={Size} entry=${EntryPrice:F2} stop=${StopLoss:F2} target=${TakeProfit:F2}",
                symbol, size, currentPrice, stopLoss, takeProfit);

            var parameters = new
            {
                symbol,
                size,
                stop_loss = stopLoss,
                take_profit = takeProfit,
                max_risk_percent = 0.01
            };
            
            var result = await SendPersistentCommandAsync("place_order", parameters, cancellationToken).ConfigureAwait(false);
            
            var success = result.TryGetProperty("success", out var successElement) && successElement.GetBoolean();
            var orderId = result.TryGetProperty("order_id", out var orderIdElement) ? orderIdElement.GetString() : null;
            var error = result.TryGetProperty("error", out var errorElement) ? errorElement.GetString() : null;
            var timestamp = result.TryGetProperty("timestamp", out var tsElement) ? 
                DateTime.Parse(tsElement.GetString()!) : DateTime.UtcNow;

            var orderResult = new OrderExecutionResult(
                success,
                orderId,
                error,
                symbol,
                size,
                currentPrice,
                stopLoss,
                takeProfit,
                timestamp);

            if (success)
            {
                _logger.LogInformation("✅ Order placed successfully: {OrderId}", orderId);
            }
            else
            {
                _logger.LogError("❌ Order placement failed: {Error}", error);
            }

            return orderResult;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error placing order for {Symbol}", symbol);
            return new OrderExecutionResult(
                false,
                null,
                ex.Message,
                symbol,
                size,
                0m,
                stopLoss,
                takeProfit,
                DateTime.UtcNow);
        }
    }

    public async Task<HealthScoreResult> GetHealthScoreAsync(CancellationToken cancellationToken = default)
    {
        // CRITICAL: Lab Mode should NEVER poll health scores from API
        var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
        if (labMode == "1" || labMode?.ToLowerInvariant() == "true")
        {
            // Return offline health in Lab Mode
            return new HealthScoreResult(0, "lab_mode", new(), new(), DateTime.UtcNow, false);
        }
        
        try
        {
            JsonElement result;
            
            if (_pythonProcess != null && !_pythonProcess.HasExited)
            {
                // Use persistent process via stdin/stdout communication
                result = await SendPersistentCommandAsync("get_health_score", cancellationToken: cancellationToken).ConfigureAwait(false);
            }
            else
            {
                // Fallback: If no persistent process exists, return degraded health
                _logger.LogWarning("⚠️ [HEALTH] No persistent Python process - returning degraded health");
                return new HealthScoreResult(0, "no_process", new(), new(), DateTime.UtcNow, false);
            }
            
            // Parse response
            var healthScore = result.TryGetProperty("health_score", out var scoreElement) ? scoreElement.GetInt32() : 0;
            var status = result.TryGetProperty("status", out var statusElement) ? statusElement.GetString()! : "unknown";
            var lastCheck = result.TryGetProperty("last_check", out var checkElement) ? 
                DateTime.Parse(checkElement.GetString()!) : DateTime.UtcNow;
            var initialized = result.TryGetProperty("initialized", out var initElement) && initElement.GetBoolean();

            // Extract instrument health
            var instrumentHealth = new Dictionary<string, object>();
            if (result.TryGetProperty("instruments", out var instrumentsElement))
            {
                foreach (var property in instrumentsElement.EnumerateObject())
                {
                    instrumentHealth[property.Name] = property.Value.GetDouble();
                }
            }

            // Extract suite stats
            var suiteStats = new Dictionary<string, object>();
            if (result.TryGetProperty("suite_stats", out var statsElement))
            {
                foreach (var property in statsElement.EnumerateObject())
                {
                    suiteStats[property.Name] = property.Value.ToString()!;
                }
            }

            // Update internal health tracking
            _connectionHealth = healthScore;

            var healthResult = new HealthScoreResult(
                healthScore,
                status,
                instrumentHealth,
                suiteStats,
                lastCheck,
                initialized);

            if (healthScore >= 80)
            {
                _logger.LogDebug("✅ [HEALTH] System healthy: {HealthScore}%", healthScore);
            }
            else
            {
                _logger.LogWarning("⚠️ [HEALTH] System health degraded: {HealthScore}% - Status: {Status}", healthScore, status);
            }

            return healthResult;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [HEALTH] Error getting health score");
            return new HealthScoreResult(0, "error", new(), new(), DateTime.UtcNow, false);
        }
    }

    public async Task<PortfolioStatusResult> GetPortfolioStatusAsync(CancellationToken cancellationToken = default)
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException("Adapter not initialized");
        }

        try
        {
            JsonElement result;
            
            // Use persistent connection
            result = await SendPersistentCommandAsync("get_portfolio_status", cancellationToken: cancellationToken).ConfigureAwait(false);
            
            var portfolio = new Dictionary<string, object>();
            var positions = new Dictionary<string, PositionInfo>();
            var timestamp = DateTime.UtcNow;

            if (result.TryGetProperty("portfolio", out var portfolioElement))
            {
                foreach (var property in portfolioElement.EnumerateObject())
                {
                    portfolio[property.Name] = property.Value.ToString()!;
                }
            }

            if (result.TryGetProperty("positions", out var positionsElement))
            {
                foreach (var property in positionsElement.EnumerateObject())
                {
                    var posData = property.Value;
                    if (!posData.TryGetProperty("error", out _)) // Skip positions with errors
                    {
                        var size = posData.TryGetProperty("size", out var sizeElement) ? sizeElement.GetInt32() : 0;
                        var avgPrice = posData.TryGetProperty("average_price", out var priceElement) ? priceElement.GetDecimal() : 0m;
                        var unrealizedPnl = posData.TryGetProperty("unrealized_pnl", out var unrealizedElement) ? unrealizedElement.GetDecimal() : 0m;
                        var realizedPnl = posData.TryGetProperty("realized_pnl", out var realizedElement) ? realizedElement.GetDecimal() : 0m;

                        positions[property.Name] = new PositionInfo(size, avgPrice, unrealizedPnl, realizedPnl);
                    }
                }
            }

            if (result.TryGetProperty("timestamp", out var tsElement))
            {
                timestamp = DateTime.Parse(tsElement.GetString()!);
            }

            return new PortfolioStatusResult(portfolio, positions, timestamp);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting portfolio status");
            throw;
        }
    }

    /// <summary>
    /// Close a position (full or partial) via TopstepX API
    /// </summary>
    public async Task<bool> ClosePositionAsync(string symbol, int quantity, CancellationToken cancellationToken = default)
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException("Adapter not initialized. Call InitializeAsync first.");
        }

        try
        {
            _logger.LogInformation("[CLOSE] Closing position: {Symbol} quantity={Qty}", symbol, quantity);

            var parameters = new
            {
                symbol,
                quantity
            };

            var result = await SendPersistentCommandAsync("close_position", parameters, cancellationToken).ConfigureAwait(false);
            
            var success = result.TryGetProperty("success", out var successElement) && successElement.GetBoolean();
            
            if (success)
            {
                _logger.LogInformation("✅ Position closed successfully: {Symbol} {Qty} contracts", symbol, quantity);
            }
            else
            {
                var error = result.TryGetProperty("error", out var errorElement) ? errorElement.GetString() : "Unknown error";
                _logger.LogError("❌ Position close failed: {Error}", error);
            }
            
            return success;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error closing position {Symbol}", symbol);
            return false;
        }
    }

    /// <summary>
    /// Modify stop loss for a position via TopstepX API
    /// </summary>
    public async Task<bool> ModifyStopLossAsync(string symbol, decimal stopPrice, CancellationToken cancellationToken = default)
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException("Adapter not initialized. Call InitializeAsync first.");
        }

        try
        {
            // Round to valid tick increment
            stopPrice = PriceHelper.RoundToTick(stopPrice, symbol);
            
            _logger.LogInformation("[MODIFY-STOP] Modifying stop loss: {Symbol} stop=${StopPrice:F2}", symbol, stopPrice);

            var parameters = new
            {
                symbol,
                stop_price = stopPrice
            };

            var command = new
            {
                action = "modify_stop_loss",
                symbol,
                stop_price = stopPrice
            };
            var cmdResult = await ExecutePythonCommandAsync(JsonSerializer.Serialize(command), cancellationToken).ConfigureAwait(false);
            if (!cmdResult.Success || !cmdResult.Data.HasValue)
            {
                _logger.LogError("❌ Invalid response from Python adapter for modify stop loss");
                return false;
            }
            var result = cmdResult.Data.Value;
            
            var success = result.TryGetProperty("success", out var successElement) && successElement.GetBoolean();
            
            if (success)
            {
                _logger.LogInformation("✅ Stop loss modified successfully: {Symbol} stop=${StopPrice:F2}", symbol, stopPrice);
            }
            else
            {
                var error = result.TryGetProperty("error", out var errorElement) ? errorElement.GetString() : "Unknown error";
                _logger.LogError("❌ Stop loss modification failed: {Error}", error);
            }
            
            return success;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error modifying stop loss for {Symbol}", symbol);
            return false;
        }
    }

    /// <summary>
    /// Modify take profit for a position via TopstepX API
    /// </summary>
    public async Task<bool> ModifyTakeProfitAsync(string symbol, decimal takeProfitPrice, CancellationToken cancellationToken = default)
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException("Adapter not initialized. Call InitializeAsync first.");
        }

        try
        {
            // Round to valid tick increment
            takeProfitPrice = PriceHelper.RoundToTick(takeProfitPrice, symbol);
            
            _logger.LogInformation("[MODIFY-TARGET] Modifying take profit: {Symbol} target=${TargetPrice:F2}", symbol, takeProfitPrice);

            var parameters = new
            {
                symbol,
                take_profit_price = takeProfitPrice
            };

            var command = new
            {
                action = "modify_take_profit",
                symbol,
                take_profit_price = takeProfitPrice
            };
            var cmdResult = await ExecutePythonCommandAsync(JsonSerializer.Serialize(command), cancellationToken).ConfigureAwait(false);
            if (!cmdResult.Success || !cmdResult.Data.HasValue)
            {
                _logger.LogError("❌ Invalid response from Python adapter for modify take profit");
                return false;
            }
            var result = cmdResult.Data.Value;
            
            var success = result.TryGetProperty("success", out var successElement) && successElement.GetBoolean();
            
            if (success)
            {
                _logger.LogInformation("✅ Take profit modified successfully: {Symbol} target=${TargetPrice:F2}", symbol, takeProfitPrice);
            }
            else
            {
                var error = result.TryGetProperty("error", out var errorElement) ? errorElement.GetString() : "Unknown error";
                _logger.LogError("❌ Take profit modification failed: {Error}", error);
            }
            
            return success;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error modifying take profit for {Symbol}", symbol);
            return false;
        }
    }

    /// <summary>
    /// Cancel an order via TopstepX API
    /// </summary>
    public async Task<bool> CancelOrderAsync(string orderId, CancellationToken cancellationToken = default)
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException("Adapter not initialized. Call InitializeAsync first.");
        }

        try
        {
            _logger.LogInformation("[CANCEL] Cancelling order: {OrderId}", orderId);

            var parameters = new
            {
                order_id = orderId
            };

            var command = new
            {
                action = "cancel_order",
                order_id = orderId
            };
            var cmdResult = await ExecutePythonCommandAsync(JsonSerializer.Serialize(command), cancellationToken).ConfigureAwait(false);
            if (!cmdResult.Success || !cmdResult.Data.HasValue)
            {
                _logger.LogError("❌ Invalid response from Python adapter for cancel order");
                return false;
            }
            var result = cmdResult.Data.Value;
            
            var success = result.TryGetProperty("success", out var successElement) && successElement.GetBoolean();
            
            if (success)
            {
                _logger.LogInformation("✅ Order cancelled successfully: {OrderId}", orderId);
            }
            else
            {
                var error = result.TryGetProperty("error", out var errorElement) ? errorElement.GetString() : "Unknown error";
                _logger.LogError("❌ Order cancellation failed: {Error}", error);
            }
            
            return success;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error cancelling order {OrderId}", orderId);
            return false;
        }
    }

    /// <summary>
    /// Fetch historical bar data from TopstepX for backtesting and learning
    /// </summary>
    public async Task<List<HistoricalBar>> GetHistoricalBarsAsync(
        string symbol, 
        int days = 1, 
        int intervalMinutes = 5,
        CancellationToken cancellationToken = default)
    {
        if (!_isInitialized)
        {
            throw new InvalidOperationException("Adapter not initialized. Call InitializeAsync first.");
        }

        try
        {
            _logger.LogInformation("[HISTORICAL] Fetching historical bars: {Symbol}, {Days} days, {Interval}min intervals", 
                symbol, days, intervalMinutes);

            // Calculate start and end dates
            var endDate = DateTime.UtcNow;
            var startDate = endDate.AddDays(-days);

            var parameters = new
            {
                symbol,
                timeframe = intervalMinutes,
                start_date = startDate.ToString("o"),  // ISO 8601 format
                end_date = endDate.ToString("o"),
                limit = 10000  // Allow large result sets for backtesting
            };

            var result = await SendPersistentCommandAsync("fetch_historical_bars", parameters, cancellationToken).ConfigureAwait(false);
            
            var bars = new List<HistoricalBar>();
            
            if (result.TryGetProperty("bars", out var barsElement) && barsElement.ValueKind == JsonValueKind.Array)
            {
                foreach (var barElement in barsElement.EnumerateArray())
                {
                    var bar = new HistoricalBar
                    {
                        Symbol = symbol,
                        Timestamp = barElement.TryGetProperty("timestamp", out var ts) ? DateTime.Parse(ts.GetString()!) : DateTime.UtcNow,
                        Open = barElement.TryGetProperty("open", out var o) ? (decimal)o.GetDouble() : 0m,
                        High = barElement.TryGetProperty("high", out var h) ? (decimal)h.GetDouble() : 0m,
                        Low = barElement.TryGetProperty("low", out var l) ? (decimal)l.GetDouble() : 0m,
                        Close = barElement.TryGetProperty("close", out var c) ? (decimal)c.GetDouble() : 0m,
                        Volume = barElement.TryGetProperty("volume", out var v) ? v.GetInt64() : 0
                    };
                    bars.Add(bar);
                }
                
                _logger.LogInformation("✅ [HISTORICAL] Fetched {Count} bars for {Symbol} from TopstepX", bars.Count, symbol);
            }
            else
            {
                _logger.LogWarning("⚠️ [HISTORICAL] No bars data in response for {Symbol}", symbol);
            }
            
            return bars;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [HISTORICAL] Error fetching historical bars for {Symbol}", symbol);
            throw;
        }
    }

    public async Task DisconnectAsync(CancellationToken cancellationToken = default)
    {
        if (!_isInitialized)
        {
            _logger.LogDebug("Adapter already disconnected");
            return;
        }

        try
        {
            _logger.LogInformation("Disconnecting TopstepX adapter...");
            
            var command = new { action = "disconnect" };
            await ExecutePythonCommandAsync(JsonSerializer.Serialize(command), cancellationToken).ConfigureAwait(false);
            
            _isInitialized = false;
            _connectionHealth = 0.0;
            
            lock (_processLock)
            {
                _pythonProcess?.Kill();
                _pythonProcess?.Dispose();
                _pythonProcess = null;
            }
            
            _logger.LogInformation("✅ TopstepX adapter disconnected successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during disconnect");
            throw;
        }
    }

    // Implement interface methods from TradingBot.Abstractions.ITopstepXAdapterService
    public async Task<bool> IsConnectedAsync()
    {
        return await Task.FromResult(IsConnected).ConfigureAwait(false);
    }

    public async Task<string> GetAccountStatusAsync()
    {
        try
        {
            var portfolioStatus = await GetPortfolioStatusAsync().ConfigureAwait(false);
            return $"Connected: {IsConnected}, Health: {_connectionHealth:F1}%, Positions: {portfolioStatus.Positions.Count}";
        }
        catch
        {
            return $"Connected: {IsConnected}, Health: {_connectionHealth:F1}%";
        }
    }

    public async Task<bool> IsHealthyAsync()
    {
        try
        {
            var healthScore = await GetHealthScoreAsync().ConfigureAwait(false);
            return healthScore.HealthScore >= 80;
        }
        catch
        {
            return false;
        }
    }

    async Task<double> TradingBot.Abstractions.ITopstepXAdapterService.GetHealthScoreAsync(CancellationToken cancellationToken)
    {
        var result = await GetHealthScoreAsync(cancellationToken).ConfigureAwait(false);
        return result.HealthScore;
    }

    string TradingBot.Abstractions.ITopstepXAdapterService.ConnectionHealth => $"{_connectionHealth:F1}%";

#pragma warning disable CS0067 // Event is never used - required by ITopstepXAdapterService interface
    public event EventHandler<TradingBot.Abstractions.StatusChangedEventArgs>? StatusChanged;
#pragma warning restore CS0067

    private async Task ValidatePythonSDKAsync(CancellationToken cancellationToken)
    {
        try
        {
            // Log environment configuration
            var pythonExecutable = Environment.GetEnvironmentVariable("PYTHON_EXECUTABLE") ?? "python";
            var isWsl = pythonExecutable.Equals("wsl", StringComparison.OrdinalIgnoreCase);
            
            _logger.LogInformation("🔍 [SDK-VALIDATION] Checking Python SDK installation...");
            _logger.LogInformation("   🐍 PYTHON_EXECUTABLE: {PythonExec}", pythonExecutable);
            _logger.LogInformation("   🖥️ Platform: {Platform}", isWsl ? "WSL (Ubuntu 24.04)" : "Native");
            
            // Validate credentials are present
            var apiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");
            var username = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
            
            if (string.IsNullOrEmpty(apiKey))
            {
                _logger.LogError("❌ [SDK-VALIDATION] TOPSTEPX_API_KEY environment variable is not set");
                throw new InvalidOperationException("Missing required environment variable: TOPSTEPX_API_KEY");
            }
            
            if (string.IsNullOrEmpty(username))
            {
                _logger.LogError("❌ [SDK-VALIDATION] TOPSTEPX_USERNAME environment variable is not set");
                throw new InvalidOperationException("Missing required environment variable: TOPSTEPX_USERNAME");
            }
            
            _logger.LogInformation("   ✅ TOPSTEPX_API_KEY: [SET]");
            _logger.LogInformation("   ✅ TOPSTEPX_USERNAME: {Username}", username);
            
            if (isWsl)
            {
                _logger.LogInformation("🐧 [WSL-MODE] Validating WSL environment...");
                
                // Check if WSL is actually available
                try
                {
                    var testProcess = new ProcessStartInfo
                    {
                        FileName = "wsl",
                        ArgumentList = { "--status" },
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    };
                    
                    using var proc = Process.Start(testProcess);
                    if (proc != null)
                    {
                        await proc.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
                        if (proc.ExitCode != 0)
                        {
                            _logger.LogError("❌ [WSL-MODE] WSL is not properly installed or configured");
                            throw new InvalidOperationException("WSL is not available. Install WSL with: wsl --install");
                        }
                        _logger.LogInformation("   ✅ WSL is available");
                    }
                }
                catch (Exception wslEx)
                {
                    _logger.LogError(wslEx, "❌ [WSL-MODE] Failed to verify WSL installation");
                    throw new InvalidOperationException("WSL validation failed. Ensure WSL is installed and Ubuntu-24.04 is available.", wslEx);
                }
            }
            
            // Check if project-x-py is installed
            _logger.LogInformation("📦 [SDK-VALIDATION] Checking project-x-py SDK...");
            var result = await ExecutePythonCommandAsync("validate_sdk", cancellationToken).ConfigureAwait(false);
            
            if (!result.Success)
            {
                _logger.LogError("❌ [SDK-VALIDATION] project-x-py SDK not found or validation failed");
                _logger.LogError("   Error: {Error}", result.Error ?? "Unknown error");
                throw new InvalidOperationException(
                    "project-x-py SDK not found. Install with: pip install 'project-x-py[all]'");
            }
            
            _logger.LogInformation("✅ [SDK-VALIDATION] Python SDK validated successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [SDK-VALIDATION] Python SDK validation failed");
            throw new InvalidOperationException("Failed to validate Python SDK installation", ex);
        }
    }

    /// <summary>
    /// Find an executable in the system PATH
    /// </summary>
    private static string? FindExecutableInPath(string executableName)
    {
        var pathVar = Environment.GetEnvironmentVariable("PATH");
        if (string.IsNullOrEmpty(pathVar))
            return null;
        
        var paths = pathVar.Split(RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? ';' : ':');
        
        foreach (var path in paths)
        {
            var fullPath = Path.Combine(path.Trim(), executableName);
            if (File.Exists(fullPath))
                return fullPath;
            
            // Try with .exe on Windows
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                var exePath = fullPath + ".exe";
                if (File.Exists(exePath))
                    return exePath;
            }
        }
        
        return null;
    }

    // ========================================================================
    // HTTP MODE MANAGEMENT
    // ========================================================================
    
    // PERSISTENT PROCESS MANAGEMENT
    // ========================================================================
    
    /// <summary>
    /// Start persistent Python process in "stream" mode with stdin/stdout communication
    /// </summary>
    private async Task StartPersistentPythonProcessAsync(CancellationToken cancellationToken)
    {
        lock (_processLock)
        {
            if (_pythonProcess != null)
            {
                _logger.LogWarning("[PERSISTENT] Python process already running");
                return;
            }
        }
        
        try
        {
            // Find workspace root
            var currentDir = AppContext.BaseDirectory;
            string? workspaceRoot = null;
            
            while (currentDir != null)
            {
                if (Directory.Exists(Path.Combine(currentDir, ".git")))
                {
                    workspaceRoot = currentDir;
                    break;
                }
                currentDir = Directory.GetParent(currentDir)?.FullName;
            }
            
            if (workspaceRoot == null)
            {
                workspaceRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));
            }
            
            var adapterPath = Path.Combine(workspaceRoot, "src", "adapters", "topstep_x_adapter.py");
            if (!File.Exists(adapterPath))
            {
                throw new FileNotFoundException($"TopstepX adapter not found at {adapterPath}");
            }

            var pythonExecutable = Environment.GetEnvironmentVariable("PYTHON_EXECUTABLE") ?? "python";
            var isWsl = pythonExecutable.Equals("wsl", StringComparison.OrdinalIgnoreCase);
            
            var processInfo = new ProcessStartInfo
            {
                RedirectStandardInput = true,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            
            // CRITICAL: Copy parent environment variables first, or Python gets EMPTY environment!
            foreach (System.Collections.DictionaryEntry envVar in Environment.GetEnvironmentVariables())
            {
                var key = envVar.Key?.ToString();
                var value = envVar.Value?.ToString();
                if (!string.IsNullOrEmpty(key) && value != null)
                {
                    processInfo.Environment[key] = value;
                }
            }
            
            // Get environment variables for credentials and retry policy
            var apiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");
            var username = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
            var accountId = Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID");
            
            // CRITICAL FIX: If credentials are empty, reload from .env file
            // This handles the case where the workflow sets empty env vars that override the .env file
            if (string.IsNullOrWhiteSpace(apiKey) || string.IsNullOrWhiteSpace(username) || string.IsNullOrWhiteSpace(accountId))
            {
                _logger.LogWarning("[ENV-FIX] Credentials are empty, reloading from .env file...");
                
                // Reload .env file to ensure we have credentials from file
                TradingBot.UnifiedOrchestrator.Infrastructure.EnvironmentLoader.LoadEnvironmentFiles();
                
                // Re-read after loading
                apiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");
                username = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
                accountId = Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID");
                
                if (string.IsNullOrWhiteSpace(apiKey) || string.IsNullOrWhiteSpace(username) || string.IsNullOrWhiteSpace(accountId))
                {
                    var msg = "TOPSTEPX credentials missing after .env reload. Ensure .env file exists with TOPSTEPX_API_KEY, TOPSTEPX_USERNAME, and TOPSTEPX_ACCOUNT_ID.";
                    _logger.LogError("[ENV-FIX] {Message}", msg);
                    throw new InvalidOperationException(msg);
                }
                
                _logger.LogInformation("[ENV-FIX] ✅ Successfully loaded credentials from .env file");
            }
            
            _logger.LogInformation("[ENV-DEBUG] Parent has TOPSTEPX_API_KEY: {HasKey}, Username: {Username}", 
                !string.IsNullOrEmpty(apiKey), username);
            _logger.LogInformation("[ENV-DEBUG] ProcessInfo.Environment count: {Count}, has API key: {HasInProc}", 
                processInfo.Environment.Count,
                processInfo.Environment.ContainsKey("TOPSTEPX_API_KEY"));
            var maxRetries = Environment.GetEnvironmentVariable("ADAPTER_MAX_RETRIES") ?? "3";
            var baseDelay = Environment.GetEnvironmentVariable("ADAPTER_BASE_DELAY") ?? "1.0";
            var maxDelay = Environment.GetEnvironmentVariable("ADAPTER_MAX_DELAY") ?? "8.0";
            var timeout = Environment.GetEnvironmentVariable("ADAPTER_TIMEOUT") ?? "30.0";
            
            if (isWsl)
            {
                var wslAdapterPath = adapterPath.Replace("\\", "/").Replace("C:", "/mnt/c").Replace("c:", "/mnt/c");
                
                // WSL mode: Pass environment variables via bash command (processInfo.Environment doesn't work for WSL)
                var bashCommand = $"PROJECT_X_API_KEY='{apiKey}' PROJECT_X_USERNAME='{username}' " +
                                  $"TOPSTEPX_API_KEY='{apiKey}' TOPSTEPX_USERNAME='{username}' TOPSTEPX_ACCOUNT_ID='{accountId}' " +
                                  $"ADAPTER_MAX_RETRIES={maxRetries} ADAPTER_BASE_DELAY={baseDelay} ADAPTER_MAX_DELAY={maxDelay} ADAPTER_TIMEOUT={timeout} " +
                                  $"python3 {wslAdapterPath} stream";
                
                processInfo.FileName = "wsl";
                processInfo.ArgumentList.Add("-d");
                processInfo.ArgumentList.Add("Ubuntu-24.04");
                processInfo.ArgumentList.Add("-e");
                processInfo.ArgumentList.Add("bash");
                processInfo.ArgumentList.Add("-c");
                processInfo.ArgumentList.Add(bashCommand);
            }
            else
            {
                // Resolve full path to Python executable for non-WSL mode
                var resolvedPython = FindExecutableInPath(pythonExecutable);
                if (resolvedPython == null)
                {
                    // Fallback: try common Python locations
                    var commonPaths = new[] { "/usr/bin/python3", "/usr/local/bin/python3", "/usr/bin/python", "/usr/local/bin/python" };
                    foreach (var path in commonPaths)
                    {
                        if (File.Exists(path))
                        {
                            resolvedPython = path;
                            _logger.LogInformation("🐍 [Native] Found Python at fallback location: {PythonPath}", resolvedPython);
                            break;
                        }
                    }
                }
                
                // If still not found, use the original executable name and let the OS find it
                if (resolvedPython == null)
                {
                    resolvedPython = pythonExecutable;
                    _logger.LogWarning("⚠️ [Native] Could not resolve Python path, using: {PythonPath}", resolvedPython);
                }
                else
                {
                    _logger.LogInformation("🐍 [Native] Resolved Python path: {PythonPath}", resolvedPython);
                }
                
                processInfo.FileName = resolvedPython;
                processInfo.Arguments = $"\"{adapterPath}\" stream";
                processInfo.Environment["PROJECT_X_API_KEY"] = apiKey ?? "";
                processInfo.Environment["PROJECT_X_USERNAME"] = username ?? "";
                processInfo.Environment["TOPSTEPX_API_KEY"] = apiKey ?? "";
                processInfo.Environment["TOPSTEPX_USERNAME"] = username ?? "";
                processInfo.Environment["TOPSTEPX_ACCOUNT_ID"] = accountId ?? "";
                processInfo.Environment["ADAPTER_MAX_RETRIES"] = maxRetries;
                processInfo.Environment["ADAPTER_BASE_DELAY"] = baseDelay;
                processInfo.Environment["ADAPTER_MAX_DELAY"] = maxDelay;
                processInfo.Environment["ADAPTER_TIMEOUT"] = timeout;
            }
            
            _logger.LogInformation("[PERSISTENT] Starting Python process in stream mode: {FileName} {Arguments}", 
                processInfo.FileName, processInfo.Arguments);
            
            var process = Process.Start(processInfo);
            if (process == null)
            {
                throw new InvalidOperationException("Failed to start Python process");
            }
            
            lock (_processLock)
            {
                _pythonProcess = process;
                _processStdin = process.StandardInput;
            }
            
            // Wait for initialization message FIRST (before starting continuous reader)
            _logger.LogInformation("[PERSISTENT] Waiting for Python adapter initialization...");
            var initTimeout = TimeSpan.FromSeconds(120); // Increased to 120s - TopstepX WebSocket can be slow on Windows with retries
            var initCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            initCts.CancelAfter(initTimeout);
            
            var initialized = false;
            try
            {
                while (!initCts.Token.IsCancellationRequested)
                {
                    var line = await process.StandardOutput.ReadLineAsync().ConfigureAwait(false);
                    if (line != null)
                    {
                        _logger.LogInformation("[PERSISTENT] Init phase received: {Line}", line);
                        try
                        {
                            var msg = JsonSerializer.Deserialize<JsonElement>(line);
                            if (msg.TryGetProperty("type", out var typeElement) && 
                                typeElement.GetString() == "init" &&
                                msg.TryGetProperty("success", out var successElement) &&
                                successElement.GetBoolean())
                            {
                                _logger.LogInformation("[PERSISTENT] ✅ Python adapter initialized successfully");
                                initialized = true;
                                break;
                            }
                        }
                        catch (Exception ex)
                        {
                            _logger.LogWarning(ex, "[PERSISTENT] Failed to parse init message: {Line}", line);
                        }
                    }
                    else
                    {
                        // Process stdout closed unexpectedly
                        _logger.LogError("[PERSISTENT] Python process stdout closed before initialization");
                        break;
                    }
                }
            }
            catch (OperationCanceledException)
            {
                _logger.LogError("[PERSISTENT] Initialization timed out after {Timeout}s", initTimeout.TotalSeconds);
            }
            
            if (!initialized)
            {
                throw new TimeoutException($"Python adapter did not initialize within {initTimeout.TotalSeconds} seconds");
            }
            
            _logger.LogInformation("[PERSISTENT] ✅ Persistent Python process started successfully (PID: {ProcessId})", process.Id);
            
            // NOW start stdout reader task for ongoing communication
            _stdoutReaderTask = Task.Run(async () => await ReadStdoutContinuouslyAsync(_processCts.Token).ConfigureAwait(false), _processCts.Token);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PERSISTENT] Failed to start persistent Python process");
            await StopPersistentPythonProcessAsync().ConfigureAwait(false);
            throw;
        }
    }
    
    /// <summary>
    /// Read stdout continuously to handle both command responses and async events (bars, fills)
    /// </summary>
    private async Task ReadStdoutContinuouslyAsync(CancellationToken cancellationToken)
    {
        Process? process;
        lock (_processLock)
        {
            process = _pythonProcess;
        }
        
        if (process == null)
            return;
        
        _logger.LogInformation("[PERSISTENT] Started stdout reader task");
        
        try
        {
            while (!cancellationToken.IsCancellationRequested && !process.HasExited)
            {
                var line = await process.StandardOutput.ReadLineAsync().ConfigureAwait(false);
                if (line == null)
                    break;
                
                try
                {
                    var message = JsonSerializer.Deserialize<JsonElement>(line);
                    var messageType = message.TryGetProperty("type", out var typeElement) ? typeElement.GetString() : null;
                    
                    if (messageType == "response")
                    {
                        // Command response - complete pending task
                        var action = message.TryGetProperty("action", out var actionElement) ? actionElement.GetString() : null;
                        if (action != null && _pendingResponses.TryRemove(action, out var tcs))
                        {
                            tcs.SetResult(message);
                        }
                    }
                    else if (messageType == "bar")
                    {
                        // Bar event - parse and distribute
                        var barEvent = ParseBarEventFromMessage(message);
                        if (barEvent != null)
                        {
                            _barEventQueue.Enqueue(barEvent);
                            BarEventReceived?.Invoke(this, barEvent);
                            
                            _logger.LogDebug("[PERSISTENT] 📊 Bar event: {Instrument} C={Close}", 
                                barEvent.Instrument, barEvent.Close);
                        }
                    }
                    else if (messageType == "fill")
                    {
                        // Fill event - parse and distribute
                        var fillEvent = ParseFillEventFromMessage(message);
                        if (fillEvent != null)
                        {
                            _fillEventQueue.Enqueue(fillEvent);
                            FillEventReceived?.Invoke(this, fillEvent);
                            
                            _logger.LogInformation("[PERSISTENT] 📥 Fill event: {OrderId}", fillEvent.OrderId);
                        }
                    }
                    else if (messageType == "error")
                    {
                        var error = message.TryGetProperty("error", out var errorElement) ? errorElement.GetString() : "Unknown error";
                        _logger.LogWarning("[PERSISTENT] Error from Python: {Error}", error);
                    }
                }
                catch (JsonException ex)
                {
                    _logger.LogWarning(ex, "[PERSISTENT] Failed to parse message from stdout: {Line}", line);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PERSISTENT] Error in stdout reader task");
        }
        
        _logger.LogInformation("[PERSISTENT] Stdout reader task stopped");
    }
    
    /// <summary>
    /// Send command to persistent Python process via stdin
    /// </summary>
    private async Task<JsonElement> SendPersistentCommandAsync(string action, object? parameters = null, CancellationToken cancellationToken = default)
    {
        StreamWriter? stdin;
        lock (_processLock)
        {
            if (_pythonProcess == null || _pythonProcess.HasExited)
            {
                throw new InvalidOperationException("Python process is not running");
            }
            stdin = _processStdin;
        }
        
        if (stdin == null)
        {
            throw new InvalidOperationException("Process stdin is not available");
        }
        
        // Create task completion source for response
        var tcs = new TaskCompletionSource<JsonElement>(TaskCreationOptions.RunContinuationsAsynchronously);
        _pendingResponses[action] = tcs;
        
        try
        {
            // Build command JSON
            var command = new Dictionary<string, object?> { ["action"] = action };
            if (parameters != null)
            {
                foreach (var prop in parameters.GetType().GetProperties())
                {
                    command[prop.Name] = prop.GetValue(parameters);
                }
            }
            
            var commandJson = JsonSerializer.Serialize(command);
            
            // Send command
            await stdin.WriteLineAsync(commandJson).ConfigureAwait(false);
            await stdin.FlushAsync().ConfigureAwait(false);
            
            _logger.LogDebug("[PERSISTENT] Sent command: {Action}", action);
            
            // Wait for response with timeout
            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
            timeoutCts.CancelAfter(TimeSpan.FromSeconds(30));
            
            var responseTask = tcs.Task;
            var completedTask = await Task.WhenAny(responseTask, Task.Delay(Timeout.Infinite, timeoutCts.Token)).ConfigureAwait(false);
            
            if (completedTask != responseTask)
            {
                throw new TimeoutException($"Command '{action}' timed out after 30 seconds");
            }
            
            return await responseTask.ConfigureAwait(false);
        }
        finally
        {
            _pendingResponses.TryRemove(action, out _);
        }
    }
    
    /// <summary>
    /// Stop persistent Python process gracefully
    /// </summary>
    private async Task StopPersistentPythonProcessAsync()
    {
        Process? process;
        lock (_processLock)
        {
            process = _pythonProcess;
            _pythonProcess = null;
            _processStdin = null;
        }
        
        if (process == null)
            return;
        
        try
        {
            // Send shutdown command
            try
            {
                await SendPersistentCommandAsync("shutdown", cancellationToken: CancellationToken.None).ConfigureAwait(false);
                _logger.LogInformation("[PERSISTENT] Sent shutdown command");
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[PERSISTENT] Failed to send shutdown command");
            }
            
            // Wait for process to exit
            var exited = process.WaitForExit(5000);
            if (!exited)
            {
                _logger.LogWarning("[PERSISTENT] Process did not exit gracefully, killing...");
                process.Kill();
            }
            
            process.Dispose();
            _logger.LogInformation("[PERSISTENT] Python process stopped");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PERSISTENT] Error stopping Python process");
        }
    }
    
    private BarEventData? ParseBarEventFromMessage(JsonElement message)
    {
        try
        {
            var instrument = message.TryGetProperty("instrument", out var instrElement) ? instrElement.GetString() : null;
            var timestampStr = message.TryGetProperty("timestamp", out var tsElement) ? tsElement.GetString() : null;
            var open = message.TryGetProperty("open", out var openElement) ? openElement.GetDecimal() : 0m;
            var high = message.TryGetProperty("high", out var highElement) ? highElement.GetDecimal() : 0m;
            var low = message.TryGetProperty("low", out var lowElement) ? lowElement.GetDecimal() : 0m;
            var close = message.TryGetProperty("close", out var closeElement) ? closeElement.GetDecimal() : 0m;
            var volume = message.TryGetProperty("volume", out var volumeElement) ? volumeElement.GetInt64() : 0L;
            
            if (string.IsNullOrEmpty(instrument))
                return null;
            
            var timestamp = !string.IsNullOrEmpty(timestampStr) && DateTime.TryParse(timestampStr, out var parsedTs)
                ? parsedTs
                : DateTime.UtcNow;
            
            return new BarEventData("bar", instrument, timestamp, open, high, low, close, volume);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PERSISTENT] Error parsing bar event");
            return null;
        }
    }
    
    private FillEventData? ParseFillEventFromMessage(JsonElement message)
    {
        try
        {
            // Parse fill event fields as needed
            var orderId = message.TryGetProperty("orderId", out var orderIdElement) ? orderIdElement.GetString() : null;
            var symbol = message.TryGetProperty("symbol", out var symbolElement) ? symbolElement.GetString() : null;
            var quantity = message.TryGetProperty("quantity", out var qtyElement) ? qtyElement.GetInt32() : 0;
            var fillPrice = message.TryGetProperty("fillPrice", out var priceElement) ? priceElement.GetDecimal() : 0m;
            
            if (string.IsNullOrEmpty(orderId) || string.IsNullOrEmpty(symbol))
                return null;
            
            return new FillEventData
            {
                OrderId = orderId,
                Symbol = symbol,
                Quantity = quantity,
                FillPrice = fillPrice,
                Price = fillPrice,
                Timestamp = DateTime.UtcNow
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PERSISTENT] Error parsing fill event");
            return null;
        }
    }

    private async Task<(bool Success, JsonElement? Data, string? Error)> ExecutePythonCommandAsync(
        string command, 
        CancellationToken cancellationToken)
    {
        try
        {
            // Find workspace root by looking for .git directory
            var currentDir = AppContext.BaseDirectory;
            string? workspaceRoot = null;
            
            while (currentDir != null)
            {
                if (Directory.Exists(Path.Combine(currentDir, ".git")))
                {
                    workspaceRoot = currentDir;
                    break;
                }
                currentDir = Directory.GetParent(currentDir)?.FullName;
            }
            
            if (workspaceRoot == null)
            {
                workspaceRoot = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", ".."));
            }
            
            var adapterPath = Path.Combine(workspaceRoot, "src", "adapters", "topstep_x_adapter.py");
            if (!File.Exists(adapterPath))
            {
                throw new FileNotFoundException($"TopstepX adapter not found at {adapterPath}");
            }

            var pythonExecutable = Environment.GetEnvironmentVariable("PYTHON_EXECUTABLE") ?? "python";
            var isWsl = pythonExecutable.Equals("wsl", StringComparison.OrdinalIgnoreCase);
            
            // Validate WSL is only used on Windows
            if (isWsl && !RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                _logger.LogError("❌ [ExecutePython] PYTHON_EXECUTABLE=wsl is only valid on Windows. Current OS: {OS}",
                    RuntimeInformation.OSDescription);
                _logger.LogInformation("💡 [ExecutePython] Hint: On Linux, use PYTHON_EXECUTABLE=python3 or leave unset");
                throw new PlatformNotSupportedException(
                    "WSL mode (PYTHON_EXECUTABLE=wsl) is only supported on Windows. " +
                    $"Current platform: {RuntimeInformation.OSDescription}. " +
                    "Use PYTHON_EXECUTABLE=python3 on Linux.");
            }
            
            var processInfo = new ProcessStartInfo
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };
            
            // CRITICAL: Copy parent environment variables first, or Python gets EMPTY environment!
            foreach (System.Collections.DictionaryEntry envVar in Environment.GetEnvironmentVariables())
            {
                var key = envVar.Key?.ToString();
                var value = envVar.Value?.ToString();
                if (!string.IsNullOrEmpty(key) && value != null)
                {
                    processInfo.Environment[key] = value;
                }
            }
            
            // Get environment variables for credentials
            var apiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");
            var username = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
            var accountId = Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID");
            var maxRetries = Environment.GetEnvironmentVariable("ADAPTER_MAX_RETRIES") ?? "3";
            var baseDelay = Environment.GetEnvironmentVariable("ADAPTER_BASE_DELAY") ?? "1.0";
            var maxDelay = Environment.GetEnvironmentVariable("ADAPTER_MAX_DELAY") ?? "30.0";
            var timeout = Environment.GetEnvironmentVariable("ADAPTER_TIMEOUT") ?? "60.0";
            
            if (isWsl)
            {
                // WSL mode: Use bash to set environment variables and run Python
                // Convert Windows path to WSL format: C:\Users\... -> /mnt/c/Users/...
                var wslAdapterPath = adapterPath.Replace("\\", "/").Replace("C:", "/mnt/c").Replace("c:", "/mnt/c");
                
                _logger.LogInformation("🐍 [WSL] Preparing to execute Python command: {Command}", command);
                _logger.LogInformation("🐍 [WSL] Adapter path: {Path}", wslAdapterPath);
                
                // Escape command for bash: replace internal single quotes with '\'' and wrap in single quotes
                var escapedCommand = command.Replace("'", "'\\''");
                
                // Build bash command with environment variables
                // Pass BOTH PROJECT_X_* (for SDK) and TOPSTEPX_* (for adapter) variables
                // Wrap command in single quotes to prevent bash from interpreting JSON special chars
                var bashCommand = $"PROJECT_X_API_KEY='{apiKey}' PROJECT_X_USERNAME='{username}' " +
                                  $"TOPSTEPX_API_KEY='{apiKey}' TOPSTEPX_USERNAME='{username}' TOPSTEPX_ACCOUNT_ID='{accountId}' " +
                                  $"ADAPTER_MAX_RETRIES={maxRetries} ADAPTER_BASE_DELAY={baseDelay} ADAPTER_MAX_DELAY={maxDelay} ADAPTER_TIMEOUT={timeout} " +
                                  $"python3 {wslAdapterPath} '{escapedCommand}'";
                
                _logger.LogInformation("🐍 [WSL] Starting WSL process...");
                
                processInfo.FileName = "wsl";
                processInfo.ArgumentList.Add("-d");
                processInfo.ArgumentList.Add("Ubuntu-24.04");
                processInfo.ArgumentList.Add("-e");
                processInfo.ArgumentList.Add("bash");
                processInfo.ArgumentList.Add("-c");
                processInfo.ArgumentList.Add(bashCommand);
            }
            else
            {
                // Native Windows/Linux Python
                // Resolve full Python path
                var resolvedPythonPath = pythonExecutable;
                if (!Path.IsPathRooted(pythonExecutable))
                {
                    // Try to find python3 in PATH
                    var pythonInPath = FindExecutableInPath(pythonExecutable);
                    if (pythonInPath != null)
                    {
                        resolvedPythonPath = pythonInPath;
                        _logger.LogInformation("🐍 Resolved Python: {Path}", resolvedPythonPath);
                    }
                    else
                    {
                        // Try common locations
                        var commonPaths = new[] { 
                            "/usr/bin/python3", 
                            "/usr/local/bin/python3",
                            "C:\\Python312\\python.exe",
                            "C:\\Python311\\python.exe",
                            "C:\\Python310\\python.exe"
                        };
                        
                        foreach (var path in commonPaths)
                        {
                            if (File.Exists(path))
                            {
                                resolvedPythonPath = path;
                                _logger.LogInformation("🐍 Found Python at: {Path}", resolvedPythonPath);
                                break;
                            }
                        }
                    }
                }
                
                processInfo.FileName = resolvedPythonPath;
                processInfo.ArgumentList.Add(adapterPath);
                processInfo.ArgumentList.Add(command);
                
                _logger.LogInformation("🐍 [Native] Starting Python process: {Python} {Args}", 
                    resolvedPythonPath, string.Join(" ", processInfo.ArgumentList));
                
                // Set environment variables for non-WSL Python processes
                if (!string.IsNullOrEmpty(apiKey))
                    processInfo.Environment["TOPSTEPX_API_KEY"] = apiKey;
                if (!string.IsNullOrEmpty(username))
                    processInfo.Environment["TOPSTEPX_USERNAME"] = username;
                if (!string.IsNullOrEmpty(accountId))
                    processInfo.Environment["TOPSTEPX_ACCOUNT_ID"] = accountId;
                
                processInfo.Environment["ADAPTER_MAX_RETRIES"] = maxRetries;
                processInfo.Environment["ADAPTER_BASE_DELAY"] = baseDelay;
                processInfo.Environment["ADAPTER_MAX_DELAY"] = maxDelay;
                processInfo.Environment["ADAPTER_TIMEOUT"] = timeout;
            }

            using var process = Process.Start(processInfo);
            if (process == null)
            {
                throw new InvalidOperationException("Failed to start Python process");
            }

            _logger.LogInformation("🐍 Process started (PID: {ProcessId}), waiting for output...", process.Id);

            var outputTask = process.StandardOutput.ReadToEndAsync(cancellationToken);
            var errorTask = process.StandardError.ReadToEndAsync(cancellationToken);
            
            await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
            
            var output = await outputTask.ConfigureAwait(false);
            var error = await errorTask.ConfigureAwait(false);

            _logger.LogInformation("🐍 Process exited with code {ExitCode}", process.ExitCode);
            if (!string.IsNullOrEmpty(output))
            {
                _logger.LogInformation("🐍 Output: {Output}", output.Length > 200 ? output.Substring(0, 200) + "..." : output);
            }
            if (!string.IsNullOrEmpty(error))
            {
                _logger.LogWarning("🐍 Error output: {Error}", error);
            }

            if (process.ExitCode == 0 && !string.IsNullOrEmpty(output))
            {
                try
                {
                    var data = JsonSerializer.Deserialize<JsonElement>(output);
                    return (true, data, null);
                }
                catch (JsonException)
                {
                    // Output might not be JSON for simple commands
                    return (true, null, null);
                }
            }
            else
            {
                var errorMsg = !string.IsNullOrEmpty(error) ? error : 
                    !string.IsNullOrEmpty(output) ? output : 
                    $"Process exited with code {process.ExitCode}";
                _logger.LogError("Python command failed: {Error}", errorMsg);
                return (false, null, errorMsg);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error executing Python command: {Command}", command);
            return (false, null, ex.Message);
        }
    }

    // ========================================================================
    // PHASE 2: FILL EVENT LISTENER INFRASTRUCTURE
    // ========================================================================
    
    /// <summary>
    /// Start background task to listen for fill events from TopstepX SDK
    /// </summary>
    private void StartFillEventListener()
    {
        _fillEventListenerTask = Task.Run(async () =>
        {
            _logger.LogInformation("🎧 [FILL-LISTENER] Starting fill event listener...");
            
            while (!_fillEventCts.Token.IsCancellationRequested)
            {
                try
                {
                    // PHASE 2: Poll for fill events from Python SDK
                    // In production, this would use WebSocket or streaming API
                    await PollForFillEventsAsync(_fillEventCts.Token).ConfigureAwait(false);
                    
                    // Poll every 2 seconds to avoid excessive API calls
                    await Task.Delay(TimeSpan.FromSeconds(2), _fillEventCts.Token).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    // Expected when shutting down
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error in fill event listener");
                    
                    // Reconnection logic: wait 5 seconds before retrying
                    try
                    {
                        await Task.Delay(TimeSpan.FromSeconds(5), _fillEventCts.Token).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException)
                    {
                        break;
                    }
                }
            }
            
            _logger.LogInformation("🛑 [FILL-LISTENER] Fill event listener stopped");
        }, _fillEventCts.Token);
    }
    
    /// <summary>
    /// Poll Python SDK for new fill events
    /// </summary>
    private async Task PollForFillEventsAsync(CancellationToken cancellationToken)
    {
        try
        {
            var command = new { action = "get_fill_events" };
            var result = await ExecutePythonCommandAsync(JsonSerializer.Serialize(command), cancellationToken).ConfigureAwait(false);
            
            if (result.Success && result.Data.HasValue)
            {
                // Check if we got fill events
                var data = result.Data.Value;
                if (data.TryGetProperty("fills", out var fillsElement) && fillsElement.ValueKind == JsonValueKind.Array)
                {
                    foreach (var fillElement in fillsElement.EnumerateArray())
                    {
                        var fillEvent = ParseFillEvent(fillElement);
                        if (fillEvent != null)
                        {
                            // Add to queue and notify subscribers
                            _fillEventQueue.Enqueue(fillEvent);
                            
                            _logger.LogInformation("📥 [FILL-LISTENER] Received fill: {OrderId} {Symbol} {Qty} @ {Price}",
                                fillEvent.OrderId, fillEvent.Symbol, fillEvent.Quantity, fillEvent.FillPrice);
                            
                            // Notify subscribers
                            FillEventReceived?.Invoke(this, fillEvent);
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error polling for fill events");
            throw;
        }
    }
    
    /// <summary>
    /// Parse fill event from JSON
    /// </summary>
    private FillEventData? ParseFillEvent(JsonElement fillElement)
    {
        try
        {
            var orderId = fillElement.TryGetProperty("order_id", out var orderIdElement) ? orderIdElement.GetString() : null;
            var symbol = fillElement.TryGetProperty("symbol", out var symbolElement) ? symbolElement.GetString() : null;
            var quantity = fillElement.TryGetProperty("quantity", out var quantityElement) ? quantityElement.GetInt32() : 0;
            var price = fillElement.TryGetProperty("price", out var priceElement) ? priceElement.GetDecimal() : 0m;
            var fillPrice = fillElement.TryGetProperty("fill_price", out var fillPriceElement) ? fillPriceElement.GetDecimal() : price;
            var timestamp = fillElement.TryGetProperty("timestamp", out var timestampElement) && DateTime.TryParse(timestampElement.GetString(), out var ts)
                ? ts
                : DateTime.UtcNow;
            var commission = fillElement.TryGetProperty("commission", out var commissionElement) ? commissionElement.GetDecimal() : 0m;
            var exchange = fillElement.TryGetProperty("exchange", out var exchangeElement) ? exchangeElement.GetString() : string.Empty;
            var liquidityType = fillElement.TryGetProperty("liquidity_type", out var liquidityElement) ? liquidityElement.GetString() : string.Empty;
            
            if (string.IsNullOrEmpty(orderId) || string.IsNullOrEmpty(symbol))
                return null;
            
            return new FillEventData
            {
                OrderId = orderId ?? "",
                Symbol = symbol ?? "",
                Quantity = quantity,
                Price = price,
                FillPrice = fillPrice,
                Timestamp = timestamp,
                Commission = commission,
                Exchange = exchange ?? "",
                LiquidityType = liquidityType ?? ""
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing fill event");
            return null;
        }
    }
    
    /// <summary>
    /// Subscribe to fill events with a callback
    /// </summary>
    public void SubscribeToFillEvents(Action<FillEventData> callback)
    {
        if (callback == null)
        {
            throw new ArgumentNullException(nameof(callback));
        }
        
        FillEventReceived += (sender, fillData) => callback(fillData);
        
        _logger.LogInformation("✅ [FILL-LISTENER] Callback subscribed to fill events");
    }
    
    // ========================================================================
    // BAR EVENT STREAMING: BAR EVENT LISTENER INFRASTRUCTURE
    // ========================================================================
    
    /// <summary>
    /// Start background task to listen for bar events from TopstepX SDK
    /// </summary>
    private void StartBarEventListener()
    {
        _barEventListenerTask = Task.Run(async () =>
        {
            _logger.LogInformation("🎧 [BAR-LISTENER] Starting bar event listener...");
            
            while (!_barEventCts.Token.IsCancellationRequested)
            {
                try
                {
                    // Poll for bar events from Python SDK
                    await PollForBarEventsAsync(_barEventCts.Token).ConfigureAwait(false);
                    
                    // Poll every 5 seconds for bar updates (bars complete every minute)
                    await Task.Delay(TimeSpan.FromSeconds(5), _barEventCts.Token).ConfigureAwait(false);
                }
                catch (OperationCanceledException)
                {
                    // Expected when shutting down
                    break;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "Error in bar event listener");
                    
                    // Reconnection logic: wait 5 seconds before retrying
                    try
                    {
                        await Task.Delay(TimeSpan.FromSeconds(5), _barEventCts.Token).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException)
                    {
                        break;
                    }
                }
            }
            
            _logger.LogInformation("🛑 [BAR-LISTENER] Bar event listener stopped");
        }, _barEventCts.Token);
    }
    
    /// <summary>
    /// Poll Python SDK for new bar events
    /// </summary>
    private async Task PollForBarEventsAsync(CancellationToken cancellationToken)
    {
        try
        {
            var command = new { action = "get_bar_events" };
            var result = await ExecutePythonCommandAsync(JsonSerializer.Serialize(command), cancellationToken).ConfigureAwait(false);
            
            if (result.Success && result.Data.HasValue)
            {
                // Check if we got bar events
                var data = result.Data.Value;
                if (data.TryGetProperty("bars", out var barsElement) && barsElement.ValueKind == JsonValueKind.Array)
                {
                    foreach (var barElement in barsElement.EnumerateArray())
                    {
                        var barEvent = ParseBarEvent(barElement);
                        if (barEvent != null)
                        {
                            // Add to queue and notify subscribers
                            _barEventQueue.Enqueue(barEvent);
                            
                            _logger.LogInformation("📊 [BAR-LISTENER] Received 1m bar for {Instrument}: O={Open:F2} H={High:F2} L={Low:F2} C={Close:F2} V={Volume}",
                                barEvent.Instrument, barEvent.Open, barEvent.High, barEvent.Low, barEvent.Close, barEvent.Volume);
                            
                            // Notify subscribers
                            BarEventReceived?.Invoke(this, barEvent);
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error polling for bar events");
            throw;
        }
    }
    
    /// <summary>
    /// Parse bar event from JSON
    /// </summary>
    private BarEventData? ParseBarEvent(JsonElement barElement)
    {
        try
        {
            var type = barElement.TryGetProperty("type", out var typeElement) ? typeElement.GetString() : "bar";
            var instrument = barElement.TryGetProperty("instrument", out var instrElement) ? instrElement.GetString() : null;
            var timestampStr = barElement.TryGetProperty("timestamp", out var tsElement) ? tsElement.GetString() : null;
            var open = barElement.TryGetProperty("open", out var openElement) ? openElement.GetDecimal() : 0m;
            var high = barElement.TryGetProperty("high", out var highElement) ? highElement.GetDecimal() : 0m;
            var low = barElement.TryGetProperty("low", out var lowElement) ? lowElement.GetDecimal() : 0m;
            var close = barElement.TryGetProperty("close", out var closeElement) ? closeElement.GetDecimal() : 0m;
            var volume = barElement.TryGetProperty("volume", out var volumeElement) ? volumeElement.GetInt64() : 0L;
            
            if (string.IsNullOrEmpty(instrument))
            {
                _logger.LogWarning("Received bar event with missing instrument");
                return null;
            }
            
            var timestamp = !string.IsNullOrEmpty(timestampStr) && DateTime.TryParse(timestampStr, out var parsedTs)
                ? parsedTs
                : DateTime.UtcNow;
            
            return new BarEventData(
                Type: type ?? "bar",
                Instrument: instrument,
                Timestamp: timestamp,
                Open: open,
                High: high,
                Low: low,
                Close: close,
                Volume: volume
            );
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error parsing bar event");
            return null;
        }
    }
    
    /// <summary>
    /// Subscribe to bar events with a callback
    /// </summary>
    public void SubscribeToBarEvents(Action<BarEventData> callback)
    {
        if (callback == null)
        {
            throw new ArgumentNullException(nameof(callback));
        }
        
        BarEventReceived += (sender, barData) => callback(barData);
        
        _logger.LogInformation("✅ [BAR-LISTENER] Callback subscribed to bar events");
    }

    public async ValueTask DisposeAsync()
    {
        if (!_disposed)
        {
            try
            {
                // Stop Python process and cleanup resources
                _processCts.Cancel();
                
                // Cleanup stdin/stdout mode
                if (_stdoutReaderTask != null)
                {
                    await _stdoutReaderTask.ConfigureAwait(false);
                }
                await StopPersistentPythonProcessAsync().ConfigureAwait(false);
                
                _processCts.Dispose();
                
                // PHASE 2: Stop fill event listener (no longer needed with persistent mode)
                _fillEventCts.Cancel();
                if (_fillEventListenerTask != null)
                {
                    await _fillEventListenerTask.ConfigureAwait(false);
                }
                _fillEventCts.Dispose();
                
                // BAR EVENT STREAMING: Stop bar event listener (no longer needed with persistent mode)
                _barEventCts.Cancel();
                if (_barEventListenerTask != null)
                {
                    await _barEventListenerTask.ConfigureAwait(false);
                }
                _barEventCts.Dispose();
                
                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
                await DisconnectAsync(cts.Token).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                _logger.LogWarning("Disconnect timed out during async disposal after 5 seconds");
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error during async disposal");
            }

            lock (_processLock)
            {
                _pythonProcess?.Dispose();
                _pythonProcess = null;
            }

            _disposed = true;
        }
    }

    public void Dispose()
    {
        // Synchronous dispose calls async dispose with a bounded helper
        // This is the recommended pattern when IAsyncDisposable is implemented
        if (!_disposed)
        {
            try
            {
                // Use GetAwaiter().GetResult() in a controlled manner with timeout
                var disposeTask = DisposeAsync().AsTask();
                if (!disposeTask.Wait(TimeSpan.FromSeconds(5)))
                {
                    _logger.LogWarning("Async dispose timed out in synchronous Dispose() after 5 seconds");
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error calling async dispose from synchronous Dispose()");
            }

            _disposed = true;
        }
    }
}