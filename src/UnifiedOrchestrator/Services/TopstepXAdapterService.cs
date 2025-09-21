using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;
using TradingBot.BotCore.Configuration;
using TradingBot.UnifiedOrchestrator.Runtime;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Service that manages the TopstepX Python SDK adapter
/// Provides C# integration with the Python project-x-py SDK
/// </summary>
public interface ITopstepXAdapterService
{
    Task<bool> InitializeAsync(CancellationToken cancellationToken = default);
    Task<decimal> GetPriceAsync(string symbol, CancellationToken cancellationToken = default);
    Task<OrderExecutionResult> PlaceOrderAsync(string symbol, int size, decimal stopLoss, decimal takeProfit, CancellationToken cancellationToken = default);
    Task<HealthScoreResult> GetHealthScoreAsync(CancellationToken cancellationToken = default);
    Task<PortfolioStatusResult> GetPortfolioStatusAsync(CancellationToken cancellationToken = default);
    Task DisconnectAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Record a fill event for order evidence tracking
    /// </summary>
    bool RecordFillEvent(string gatewayOrderId, string fillId, decimal fillQuantity, decimal fillPrice, DateTime fillTime);
    
    bool IsConnected { get; }
    double ConnectionHealth { get; }
}

public record OrderExecutionResult(
    bool Success,
    string? OrderId,
    string? Error,
    string Symbol,
    int Size,
    decimal EntryPrice,
    decimal StopLoss,
    decimal TakeProfit,
    DateTime Timestamp);

public record HealthScoreResult(
    int HealthScore,
    string Status,
    Dictionary<string, object> InstrumentHealth,
    Dictionary<string, object> SuiteStats,
    DateTime LastCheck,
    bool Initialized);

public record PortfolioStatusResult(
    Dictionary<string, object> Portfolio,
    Dictionary<string, PositionInfo> Positions,
    DateTime Timestamp);

public record PositionInfo(
    int Size,
    decimal AveragePrice,
    decimal UnrealizedPnL,
    decimal RealizedPnL);

public class TopstepXAdapterService : ITopstepXAdapterService, IDisposable
{
    private readonly ILogger<TopstepXAdapterService> _logger;
    private readonly TopstepXConfiguration _config;
    private readonly OrderLedger _orderLedger;
    private readonly string[] _instruments;
    private Process? _pythonProcess;
    private bool _isInitialized;
    private double _connectionHealth;
    private readonly object _processLock = new();
    private bool _disposed;

    public TopstepXAdapterService(
        ILogger<TopstepXAdapterService> logger,
        IOptions<TopstepXConfiguration> config,
        OrderLedger orderLedger)
    {
        _logger = logger;
        _config = config.Value;
        _orderLedger = orderLedger;
        _instruments = new[] { "MNQ", "ES" }; // Support MNQ and ES as specified
        _isInitialized = false;
        _connectionHealth = 0.0;
    }

    public bool IsConnected => _isInitialized && _connectionHealth >= 80.0;
    public double ConnectionHealth => _connectionHealth;

    public async Task<bool> InitializeAsync(CancellationToken cancellationToken = default)
    {
        if (_isInitialized)
        {
            _logger.LogWarning("TopstepX adapter already initialized");
            return true;
        }

        try
        {
            _logger.LogInformation("🚀 Initializing TopstepX Python SDK adapter...");

            // Validate Python SDK is available
            await ValidatePythonSDKAsync(cancellationToken).ConfigureAwait(false);

            // Initialize adapter through Python process
            var result = await ExecutePythonCommandAsync("initialize", cancellationToken).ConfigureAwait(false);
            
            if (result.Success)
            {
                _isInitialized = true;
                _connectionHealth = 100.0;
                _logger.LogInformation("✅ TopstepX adapter initialized successfully");
                return true;
            }
            else
            {
                _logger.LogError("❌ Failed to initialize TopstepX adapter: {Error}", result.Error);
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Exception during TopstepX adapter initialization");
            return false;
        }
    }

    public async Task<decimal> GetPriceAsync(string symbol, CancellationToken cancellationToken = default)
    {
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
            var command = new { action = "get_price", symbol };
            var result = await ExecutePythonCommandAsync(JsonSerializer.Serialize(command), cancellationToken).ConfigureAwait(false);
            
            if (result.Success && result.Data != null)
            {
                if (result.Data.TryGetProperty("price", out var priceElement))
                {
                    var price = priceElement.GetDecimal();
                    _logger.LogDebug("[PRICE] {Symbol}: ${Price:F2}", symbol, price);
                    return price;
                }
            }
            
            throw new InvalidOperationException($"Failed to get price for {symbol}: {result.Error}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting price for {Symbol}", symbol);
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

        // Generate unique client ID for order tracking
        var clientId = _orderLedger.NewClientId(symbol, "TOPSTEP");
        
        // Check for duplicate orders
        if (_orderLedger.IsDuplicate(clientId))
        {
            _logger.LogError("❌ [ORDER-LEDGER] Duplicate order attempt blocked: {ClientId}", clientId);
            return new OrderExecutionResult(
                false,
                null,
                "Duplicate order attempt blocked by OrderLedger",
                symbol,
                size,
                0m,
                stopLoss,
                takeProfit,
                DateTime.UtcNow);
        }

        try
        {
            var currentPrice = await GetPriceAsync(symbol, cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation(
                "[ORDER] Placing bracket order: {Symbol} size={Size} entry=${EntryPrice:F2} stop=${StopLoss:F2} target=${TakeProfit:F2} ClientId={ClientId}",
                symbol, size, currentPrice, stopLoss, takeProfit, clientId);

            var command = new
            {
                action = "place_order",
                symbol,
                size,
                stop_loss = stopLoss,
                take_profit = takeProfit,
                max_risk_percent = 0.01, // 1% risk as specified
                client_id = clientId // Pass client ID to Python adapter
            };

            var result = await ExecutePythonCommandAsync(JsonSerializer.Serialize(command), cancellationToken).ConfigureAwait(false);
            
            if (result.Success && result.Data != null)
            {
                var success = result.Data.TryGetProperty("success", out var successElement) && successElement.GetBoolean();
                var orderId = result.Data.TryGetProperty("order_id", out var orderIdElement) ? orderIdElement.GetString() : null;
                var error = result.Data.TryGetProperty("error", out var errorElement) ? errorElement.GetString() : null;
                var timestamp = result.Data.TryGetProperty("timestamp", out var tsElement) ? 
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

                if (success && !string.IsNullOrEmpty(orderId))
                {
                    // Record the order in the ledger for evidence tracking
                    var recorded = _orderLedger.TryRecord(clientId, orderId, symbol, size, currentPrice, "BRACKET");
                    if (recorded)
                    {
                        _logger.LogInformation("✅ Order placed and recorded: ClientId={ClientId} → GatewayId={OrderId}", clientId, orderId);
                    }
                    else
                    {
                        _logger.LogWarning("⚠️ Order placed but failed to record in ledger: {OrderId}", orderId);
                    }
                }
                else
                {
                    _logger.LogError("❌ Order placement failed: {Error}", error);
                }

                return orderResult;
            }
            
            throw new InvalidOperationException($"Invalid response from Python adapter: {result.Error}");
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
        try
        {
            var command = new { action = "get_health_score" };
            var result = await ExecutePythonCommandAsync(JsonSerializer.Serialize(command), cancellationToken).ConfigureAwait(false);
            
            if (result.Success && result.Data != null)
            {
                var healthScore = result.Data.TryGetProperty("health_score", out var scoreElement) ? scoreElement.GetInt32() : 0;
                var status = result.Data.TryGetProperty("status", out var statusElement) ? statusElement.GetString()! : "unknown";
                var lastCheck = result.Data.TryGetProperty("last_check", out var checkElement) ? 
                    DateTime.Parse(checkElement.GetString()!) : DateTime.UtcNow;
                var initialized = result.Data.TryGetProperty("initialized", out var initElement) && initElement.GetBoolean();

                // Extract instrument health
                var instrumentHealth = new Dictionary<string, object>();
                if (result.Data.TryGetProperty("instruments", out var instrumentsElement))
                {
                    foreach (var property in instrumentsElement.EnumerateObject())
                    {
                        instrumentHealth[property.Name] = property.Value.GetDouble();
                    }
                }

                // Extract suite stats
                var suiteStats = new Dictionary<string, object>();
                if (result.Data.TryGetProperty("suite_stats", out var statsElement))
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
                    _logger.LogDebug("System healthy: {HealthScore}%", healthScore);
                }
                else
                {
                    _logger.LogWarning("System health degraded: {HealthScore}% - Status: {Status}", healthScore, status);
                }

                return healthResult;
            }
            
            throw new InvalidOperationException($"Failed to get health score: {result.Error}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting health score");
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
            var command = new { action = "get_portfolio_status" };
            var result = await ExecutePythonCommandAsync(JsonSerializer.Serialize(command), cancellationToken).ConfigureAwait(false);
            
            if (result.Success && result.Data != null)
            {
                var portfolio = new Dictionary<string, object>();
                var positions = new Dictionary<string, PositionInfo>();
                var timestamp = DateTime.UtcNow;

                if (result.Data.TryGetProperty("portfolio", out var portfolioElement))
                {
                    foreach (var property in portfolioElement.EnumerateObject())
                    {
                        portfolio[property.Name] = property.Value.ToString()!;
                    }
                }

                if (result.Data.TryGetProperty("positions", out var positionsElement))
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

                if (result.Data.TryGetProperty("timestamp", out var tsElement))
                {
                    timestamp = DateTime.Parse(tsElement.GetString()!);
                }

                return new PortfolioStatusResult(portfolio, positions, timestamp);
            }
            
            throw new InvalidOperationException($"Failed to get portfolio status: {result.Error}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting portfolio status");
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
            
            _isInitialized;
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

    private async Task ValidatePythonSDKAsync(CancellationToken cancellationToken)
    {
        try
        {
            // Check if project-x-py is installed
            var result = await ExecutePythonCommandAsync("validate_sdk", cancellationToken).ConfigureAwait(false);
            if (!result.Success)
            {
                throw new InvalidOperationException(
                    "project-x-py SDK not found. Install with: pip install 'project-x-py[all]'");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Python SDK validation failed");
            throw new InvalidOperationException("Failed to validate Python SDK installation", ex);
        }
    }

    private async Task<(bool Success, JsonElement? Data, string? Error)> ExecutePythonCommandAsync(
        string command, 
        CancellationToken cancellationToken)
    {
        try
        {
            var adapterPath = Path.Combine(AppContext.BaseDirectory, "src", "adapters", "topstep_x_adapter.py");
            if (!File.Exists(adapterPath))
            {
                // Try relative path from current directory
                adapterPath = Path.Combine("src", "adapters", "topstep_x_adapter.py");
                if (!File.Exists(adapterPath))
                {
                    throw new FileNotFoundException($"TopstepX adapter not found at {adapterPath}");
                }
            }

            var processInfo = new ProcessStartInfo
            {
                FileName = "python3",
                Arguments = $"\"{adapterPath}\" \"{command}\"",
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true
            };

            // Set environment variables for credentials if available
            var apiKey = Environment.GetEnvironmentVariable("PROJECT_X_API_KEY");
            var username = Environment.GetEnvironmentVariable("PROJECT_X_USERNAME");
            
            if (!string.IsNullOrEmpty(apiKey))
                processInfo.Environment["PROJECT_X_API_KEY"] = apiKey;
            if (!string.IsNullOrEmpty(username))
                processInfo.Environment["PROJECT_X_USERNAME"] = username;

            using var process = Process.Start(processInfo);
            if (process == null)
            {
                throw new InvalidOperationException("Failed to start Python process");
            }

            var outputTask = process.StandardOutput.ReadToEndAsync(cancellationToken);
            var errorTask = process.StandardError.ReadToEndAsync(cancellationToken);
            
            await process.WaitForExitAsync(cancellationToken).ConfigureAwait(false);
            
            var output = await outputTask.ConfigureAwait(false);
            var error = await errorTask.ConfigureAwait(false);

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
                var errorMsg = !string.IsNullOrEmpty(error) ? error : $"Process exited with code {process.ExitCode}";
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
    
    /// <summary>
    /// Record a fill event for order evidence tracking
    /// </summary>
    public bool RecordFillEvent(string gatewayOrderId, string fillId, decimal fillQuantity, decimal fillPrice, DateTime fillTime)
    {
        try
        {
            var recorded = _orderLedger.RecordFill(gatewayOrderId, fillId, fillQuantity, fillPrice, fillTime);
            if (recorded)
            {
                _logger.LogInformation("💰 [FILL-EVENT] Recorded fill: OrderId={OrderId} FillId={FillId} Qty={Quantity} Price={Price}",
                    gatewayOrderId, fillId, fillQuantity, fillPrice);
            }
            else
            {
                _logger.LogWarning("⚠️ [FILL-EVENT] Failed to record fill for unknown order: {OrderId}", gatewayOrderId);
            }
            return recorded;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ [FILL-EVENT] Error recording fill event for order {OrderId}", gatewayOrderId);
            return false;
        }
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            try
            {
                DisconnectAsync().Wait(TimeSpan.FromSeconds(5));
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Error during disposal");
            }

            lock (_processLock)
            {
                _pythonProcess?.Dispose();
                _pythonProcess = null;
            }

            _disposed = true;
        }
    }
}