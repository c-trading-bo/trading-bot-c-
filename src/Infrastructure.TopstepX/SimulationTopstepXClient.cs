using System;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

#pragma warning disable CS0067 // Event is never used - mock events for interface compliance

namespace TradingBot.Infrastructure.TopstepX;

/// <summary>
/// Simulation TopstepX client for CI/testing purposes
/// Provides realistic responses without connecting to real TopstepX servers
/// Used only when TOPSTEPX_SIMULATION_MODE=true environment variable is set
/// </summary>
public class SimulationTopstepXClient : ITopstepXClient, IDisposable
{
    private readonly ILogger<SimulationTopstepXClient> _logger;
    private volatile bool _disposed;
    private volatile bool _isConnected;

    public bool IsConnected => _isConnected;

    // Events - mock implementations
    public event Action<TradingBot.Abstractions.GatewayUserOrder>? OnOrderUpdate;
    public event Action<TradingBot.Abstractions.GatewayUserTrade>? OnTradeUpdate;
    public event Action<TradingBot.Abstractions.MarketData>? OnMarketDataUpdate;
    public event Action<TradingBot.Abstractions.OrderBookData>? OnLevel2Update;
    public event Action<TradingBot.Abstractions.TradeConfirmation>? OnTradeConfirmed;
    public event Action<string>? OnError;
    public event Action<bool>? OnConnectionStateChanged;

    public SimulationTopstepXClient(ILogger<SimulationTopstepXClient> logger)
    {
        _logger = logger;
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation TopstepX client initialized for CI/testing");
    }

    // ====================================================================
    // CONNECTION MANAGEMENT
    // ====================================================================

    public async Task<bool> ConnectAsync(CancellationToken cancellationToken = default)
    {
        await Task.Delay(100, cancellationToken); // Simulate connection delay
        _isConnected = true;
        OnConnectionStateChanged?.Invoke(true);
        _logger.LogInformation("[SIM-TOPSTEPX] Connected to mock TopstepX services");
        return true;
    }

    public async Task<bool> DisconnectAsync(CancellationToken cancellationToken = default)
    {
        await Task.Delay(50, cancellationToken); // Simulate disconnection delay
        _isConnected = false;
        OnConnectionStateChanged?.Invoke(false);
        _logger.LogInformation("[SIM-TOPSTEPX] Disconnected from mock TopstepX services");
        return true;
    }

    // ====================================================================
    // AUTHENTICATION
    // ====================================================================

    public async Task<(string jwt, DateTimeOffset expiresUtc)> AuthenticateAsync(
        string username, string password, string apiKey, CancellationToken cancellationToken = default)
    {
        await Task.Delay(200, cancellationToken); // Simulate auth delay
        var jwt = "mock.jwt.token.for.testing";
        var expires = DateTimeOffset.UtcNow.AddHours(1);
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation authentication successful for user: {Username}", username);
        return (jwt, expires);
    }

    public async Task<(string jwt, DateTimeOffset expiresUtc)> RefreshTokenAsync(
        string refreshToken, CancellationToken cancellationToken = default)
    {
        await Task.Delay(100, cancellationToken); // Simulate refresh delay
        var jwt = "mock.refreshed.jwt.token";
        var expires = DateTimeOffset.UtcNow.AddHours(1);
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation token refresh successful");
        return (jwt, expires);
    }

    // ====================================================================
    // ACCOUNT MANAGEMENT
    // ====================================================================

    public async Task<JsonElement> GetAccountAsync(string accountId, CancellationToken cancellationToken = default)
    {
        await Task.Delay(150, cancellationToken);
        var mockResponse = JsonSerializer.Serialize(new
        {
            accountId = accountId,
            accountName = "Simulation Trading Account",
            accountType = "Evaluation",
            status = "Active",
            created = DateTimeOffset.UtcNow.AddDays(-30),
            lastUpdated = DateTimeOffset.UtcNow
        });
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation account data retrieved for: {AccountId}", accountId);
        return JsonSerializer.Deserialize<JsonElement>(mockResponse);
    }

    public async Task<JsonElement> GetAccountBalanceAsync(string accountId, CancellationToken cancellationToken = default)
    {
        await Task.Delay(120, cancellationToken);
        var mockResponse = JsonSerializer.Serialize(new
        {
            accountId = accountId,
            balance = 50000.00m,
            equity = 49850.00m,
            availableBalance = 48000.00m,
            marginUsed = 1850.00m,
            unrealizedPnL = -150.00m,
            dayPnL = 250.00m,
            currency = "USD",
            lastUpdated = DateTimeOffset.UtcNow
        });
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation balance data retrieved for: {AccountId}", accountId);
        return JsonSerializer.Deserialize<JsonElement>(mockResponse);
    }

    public async Task<JsonElement> GetAccountPositionsAsync(string accountId, CancellationToken cancellationToken = default)
    {
        await Task.Delay(100, cancellationToken);
        var mockResponse = JsonSerializer.Serialize(new
        {
            accountId = accountId,
            positions = new[]
            {
                new
                {
                    symbol = "ES",
                    quantity = 1,
                    avgPrice = 4125.25m,
                    marketPrice = 4120.50m,
                    unrealizedPnL = -237.50m,
                    side = "Long",
                    contractId = "ES-DEC2024"
                }
            },
            totalUnrealizedPnL = -237.50m,
            lastUpdated = DateTimeOffset.UtcNow
        });
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation positions retrieved for: {AccountId}", accountId);
        return JsonSerializer.Deserialize<JsonElement>(mockResponse);
    }

    public async Task<JsonElement> SearchAccountsAsync(object searchRequest, CancellationToken cancellationToken = default)
    {
        await Task.Delay(180, cancellationToken);
        var mockResponse = JsonSerializer.Serialize(new
        {
            accounts = new[]
            {
                new { accountId = "SIM-001", accountName = "Simulation Evaluation Account", status = "Active" },
                new { accountId = "SIM-002", accountName = "Simulation Live Account", status = "Active" }
            },
            totalCount = 2
        });
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation account search completed");
        return JsonSerializer.Deserialize<JsonElement>(mockResponse);
    }

    // ====================================================================
    // ORDER MANAGEMENT
    // ====================================================================

    public async Task<JsonElement> PlaceOrderAsync(object orderRequest, CancellationToken cancellationToken = default)
    {
        await Task.Delay(300, cancellationToken); // Simulate order placement delay
        var orderId = Guid.NewGuid().ToString();
        var mockResponse = JsonSerializer.Serialize(new
        {
            orderId = orderId,
            status = "NEW",
            timestamp = DateTimeOffset.UtcNow,
            symbol = "ES",
            side = "BUY",
            quantity = 1,
            price = 4125.25m,
            orderType = "LIMIT",
            customTag = $"SIM-{DateTimeOffset.UtcNow:yyyyMMdd-HHmmss}"
        });
        
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation order placed successfully. OrderId: {OrderId}", orderId);
        
        // Simulate order update event after a short delay
        _ = Task.Run(async () =>
        {
            await Task.Delay(500, cancellationToken);
            OnOrderUpdate?.Invoke(new GatewayUserOrder
            {
                OrderId = orderId,
                Status = "FILLED",
                Symbol = "ES",
                Side = "BUY",
                Quantity = 1,
                Price = 4125.25m,
                Timestamp = DateTime.UtcNow
            });
        });
        
        return JsonSerializer.Deserialize<JsonElement>(mockResponse);
    }

    public async Task<bool> CancelOrderAsync(string orderId, CancellationToken cancellationToken = default)
    {
        await Task.Delay(150, cancellationToken);
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation order cancelled. OrderId: {OrderId}", orderId);
        return true;
    }

    public async Task<JsonElement> GetOrderStatusAsync(string orderId, CancellationToken cancellationToken = default)
    {
        await Task.Delay(100, cancellationToken);
        var mockResponse = JsonSerializer.Serialize(new
        {
            orderId = orderId,
            status = "FILLED",
            symbol = "ES",
            side = "BUY",
            quantity = 1,
            filledQuantity = 1,
            price = 4125.25m,
            fillPrice = 4125.25m,
            timestamp = DateTimeOffset.UtcNow
        });
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation order status retrieved. OrderId: {OrderId}", orderId);
        return JsonSerializer.Deserialize<JsonElement>(mockResponse);
    }

    public async Task<JsonElement> SearchOrdersAsync(object searchRequest, CancellationToken cancellationToken = default)
    {
        await Task.Delay(200, cancellationToken);
        var mockResponse = JsonSerializer.Serialize(new
        {
            orders = new[]
            {
                new
                {
                    orderId = Guid.NewGuid().ToString(),
                    status = "FILLED",
                    symbol = "ES",
                    side = "BUY",
                    quantity = 1,
                    price = 4125.25m,
                    timestamp = DateTimeOffset.UtcNow.AddMinutes(-5)
                }
            },
            totalCount = 1
        });
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation order search completed");
        return JsonSerializer.Deserialize<JsonElement>(mockResponse);
    }

    public async Task<JsonElement> SearchOpenOrdersAsync(object searchRequest, CancellationToken cancellationToken = default)
    {
        await Task.Delay(150, cancellationToken);
        var mockResponse = JsonSerializer.Serialize(new
        {
            orders = new object[0], // No open orders in mock
            totalCount = 0
        });
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation open orders search completed (no open orders)");
        return JsonSerializer.Deserialize<JsonElement>(mockResponse);
    }

    // ====================================================================
    // TRADE MANAGEMENT
    // ====================================================================

    public async Task<JsonElement> SearchTradesAsync(object searchRequest, CancellationToken cancellationToken = default)
    {
        await Task.Delay(180, cancellationToken);
        var mockResponse = JsonSerializer.Serialize(new
        {
            trades = new[]
            {
                new
                {
                    tradeId = Guid.NewGuid().ToString(),
                    orderId = Guid.NewGuid().ToString(),
                    symbol = "ES",
                    side = "BUY",
                    quantity = 1,
                    fillPrice = 4125.25m,
                    commission = 2.50m,
                    timestamp = DateTimeOffset.UtcNow.AddMinutes(-3)
                }
            },
            totalCount = 1
        });
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation trade search completed");
        return JsonSerializer.Deserialize<JsonElement>(mockResponse);
    }

    public async Task<JsonElement> GetTradeAsync(string tradeId, CancellationToken cancellationToken = default)
    {
        await Task.Delay(120, cancellationToken);
        var mockResponse = JsonSerializer.Serialize(new
        {
            tradeId = tradeId,
            orderId = Guid.NewGuid().ToString(),
            symbol = "ES",
            side = "BUY",
            quantity = 1,
            fillPrice = 4125.25m,
            commission = 2.50m,
            timestamp = DateTimeOffset.UtcNow.AddMinutes(-3)
        });
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation trade details retrieved. TradeId: {TradeId}", tradeId);
        return JsonSerializer.Deserialize<JsonElement>(mockResponse);
    }

    // ====================================================================
    // MARKET DATA
    // ====================================================================

    public async Task<JsonElement> GetContractAsync(string contractId, CancellationToken cancellationToken = default)
    {
        await Task.Delay(100, cancellationToken);
        var mockResponse = JsonSerializer.Serialize(new
        {
            contractId = contractId,
            symbol = "ES",
            name = "E-mini S&P 500",
            tickSize = 0.25m,
            tickValue = 12.50m,
            contractSize = 50,
            currency = "USD",
            exchange = "CME",
            expirationDate = "2024-12-20",
            tradingHours = "17:00-16:00 CT",
            lastUpdated = DateTimeOffset.UtcNow
        });
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation contract details retrieved. ContractId: {ContractId}", contractId);
        return JsonSerializer.Deserialize<JsonElement>(mockResponse);
    }

    public async Task<JsonElement> SearchContractsAsync(object searchRequest, CancellationToken cancellationToken = default)
    {
        await Task.Delay(150, cancellationToken);
        var mockResponse = JsonSerializer.Serialize(new
        {
            contracts = new[]
            {
                new { contractId = "ES-DEC2024", symbol = "ES", name = "E-mini S&P 500", tickSize = 0.25m },
                new { contractId = "NQ-DEC2024", symbol = "NQ", name = "E-mini NASDAQ 100", tickSize = 0.25m },
                new { contractId = "YM-DEC2024", symbol = "YM", name = "E-mini Dow", tickSize = 1.0m }
            },
            totalCount = 3
        });
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation contract search completed");
        return JsonSerializer.Deserialize<JsonElement>(mockResponse);
    }

    public async Task<JsonElement> GetMarketDataAsync(string symbol, CancellationToken cancellationToken = default)
    {
        await Task.Delay(80, cancellationToken);
        var basePrice = symbol.ToUpper() switch
        {
            "ES" => 4125.25m,
            "NQ" => 14250.50m,
            "YM" => 33875.00m,
            _ => 1000.00m
        };
        
        var mockResponse = JsonSerializer.Serialize(new
        {
            symbol = symbol,
            lastPrice = basePrice,
            bid = basePrice - 0.25m,
            ask = basePrice + 0.25m,
            bidSize = 10,
            askSize = 8,
            volume = 125430,
            openInterest = 2564890,
            high = basePrice + 15.00m,
            low = basePrice - 12.50m,
            open = basePrice - 2.25m,
            change = 2.75m,
            changePercent = 0.067m,
            timestamp = DateTimeOffset.UtcNow
        });
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation market data retrieved for: {Symbol}", symbol);
        return JsonSerializer.Deserialize<JsonElement>(mockResponse);
    }

    // ====================================================================
    // REAL-TIME SUBSCRIPTIONS
    // ====================================================================

    public async Task<bool> SubscribeOrdersAsync(string accountId, CancellationToken cancellationToken = default)
    {
        await Task.Delay(100, cancellationToken);
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation subscription to order updates for account: {AccountId}", accountId);
        return true;
    }

    public async Task<bool> SubscribeTradesAsync(string accountId, CancellationToken cancellationToken = default)
    {
        await Task.Delay(100, cancellationToken);
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation subscription to trade updates for account: {AccountId}", accountId);
        return true;
    }

    public async Task<bool> SubscribeMarketDataAsync(string symbol, CancellationToken cancellationToken = default)
    {
        await Task.Delay(100, cancellationToken);
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation subscription to market data for: {Symbol}", symbol);
        
        // Simulate periodic market data updates
        _ = Task.Run(async () =>
        {
            while (_isConnected && !cancellationToken.IsCancellationRequested)
            {
                await Task.Delay(1000, cancellationToken);
                OnMarketDataUpdate?.Invoke(new MarketData
                {
                    Symbol = symbol,
#pragma warning disable SCS0005 // Weak random generator acceptable for mock data
                    Close = (double)(4125.25m + (decimal)(new Random().NextDouble() - 0.5) * 2),
#pragma warning restore SCS0005
                    Bid = (double)4125.00m,
                    Ask = (double)4125.50m,
                    Volume = 125430,
                    Timestamp = DateTime.UtcNow
                });
            }
        });
        
        return true;
    }

    public async Task<bool> SubscribeLevel2DataAsync(string symbol, CancellationToken cancellationToken = default)
    {
        await Task.Delay(120, cancellationToken);
        _logger.LogInformation("[SIM-TOPSTEPX] Simulation subscription to Level 2 data for: {Symbol}", symbol);
        return true;
    }

    // ====================================================================
    // DISPOSAL
    // ====================================================================

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (disposing && !_disposed)
        {
            _disposed = true;
            _isConnected = false;
            _logger.LogInformation("[SIM-TOPSTEPX] Simulation TopstepX client disposed");
        }
    }
}