using System;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;

namespace TradingBot.Infrastructure.TopstepX;

/// <summary>
/// Mock TopstepX client for testing and development
/// Simulates all TopstepX functionality with configurable scenarios
/// </summary>
public class MockTopstepXClient : ITopstepXClient, IDisposable
{
    private readonly ILogger<MockTopstepXClient> _logger;
    private readonly TopstepXClientConfiguration _config;
    private readonly Random _random;
    private volatile bool _isConnected;
    private volatile bool _disposed;
    private int _orderIdCounter = 1000;

    public bool IsConnected => _isConnected;

    // Events
#pragma warning disable CS0067 // Event is never used - these are mock events for interface compliance
    public event Action<TradingBot.Abstractions.GatewayUserOrder>? OnOrderUpdate;
    public event Action<TradingBot.Abstractions.GatewayUserTrade>? OnTradeUpdate;
    public event Action<TradingBot.Abstractions.MarketData>? OnMarketDataUpdate;
    public event Action<TradingBot.Abstractions.OrderBookData>? OnLevel2Update;
    public event Action<TradingBot.Abstractions.TradeConfirmation>? OnTradeConfirmed;
    public event Action<string>? OnError;
    public event Action<bool>? OnConnectionStateChanged;
#pragma warning restore CS0067

    public MockTopstepXClient(ILogger<MockTopstepXClient> logger, IOptions<TopstepXClientConfiguration> config)
    {
        _logger = logger;
        _config = config.Value;
        _random = new Random();
        
        LogMockCall("Constructor", new { scenario = _config.MockScenario, latencyMs = _config.MockLatencyMs });
    }

    // ====================================================================
    // CONNECTION MANAGEMENT
    // ====================================================================

    public async Task<bool> ConnectAsync(CancellationToken cancellationToken = default)
    {
        LogMockCall("ConnectAsync", new { scenario = _config.MockScenario });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            var error = "Mock connection failed";
            LogMockCall("ConnectAsync", new { success = false, error });
            OnError?.Invoke(error);
            return false;
        }

        _isConnected = true;
        OnConnectionStateChanged?.Invoke(true);
        
        LogMockCall("ConnectAsync", new { success = true, connected = _isConnected });
        
        // Start simulating market data if connected
        _ = Task.Run(SimulateMarketDataAsync, cancellationToken);
        
        return true;
    }

    public async Task<bool> DisconnectAsync(CancellationToken cancellationToken = default)
    {
        LogMockCall("DisconnectAsync", new { currentlyConnected = _isConnected });
        
        await SimulateLatency(cancellationToken);
        
        _isConnected = false;
        OnConnectionStateChanged?.Invoke(false);
        
        LogMockCall("DisconnectAsync", new { success = true, connected = _isConnected });
        return true;
    }

    // ====================================================================
    // AUTHENTICATION
    // ====================================================================

    public async Task<(string jwt, DateTimeOffset expiresUtc)> AuthenticateAsync(
        string username, string password, string apiKey, CancellationToken cancellationToken = default)
    {
        LogMockCall("AuthenticateAsync", new { username = MaskCredential(username), hasPassword = !string.IsNullOrEmpty(password), hasApiKey = !string.IsNullOrEmpty(apiKey) });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            throw new InvalidOperationException("Mock authentication failed");
        }

        var jwt = GenerateMockJwt();
        var expires = DateTimeOffset.UtcNow.AddHours(24);
        
        LogMockCall("AuthenticateAsync", new { success = true, jwtLength = jwt.Length, expiresUtc = expires });
        
        return (jwt, expires);
    }

    public async Task<(string jwt, DateTimeOffset expiresUtc)> RefreshTokenAsync(
        string refreshToken, CancellationToken cancellationToken = default)
    {
        LogMockCall("RefreshTokenAsync", new { hasRefreshToken = !string.IsNullOrEmpty(refreshToken) });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            throw new InvalidOperationException("Mock token refresh failed");
        }

        var jwt = GenerateMockJwt();
        var expires = DateTimeOffset.UtcNow.AddHours(24);
        
        LogMockCall("RefreshTokenAsync", new { success = true, jwtLength = jwt.Length, expiresUtc = expires });
        
        return (jwt, expires);
    }

    // ====================================================================
    // ACCOUNT MANAGEMENT
    // ====================================================================

    public async Task<JsonElement> GetAccountAsync(string accountId, CancellationToken cancellationToken = default)
    {
        LogMockCall("GetAccountAsync", new { accountId = MaskAccountId(accountId) });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            throw new InvalidOperationException("Mock GetAccount failed");
        }

        var accountData = CreateMockAccountData();
        var json = JsonSerializer.Serialize(accountData);
        var element = JsonSerializer.Deserialize<JsonElement>(json);
        
        LogMockCall("GetAccountAsync", new { success = true, accountType = _config.MockAccount.AccountType });
        
        return element;
    }

    public async Task<JsonElement> GetAccountBalanceAsync(string accountId, CancellationToken cancellationToken = default)
    {
        LogMockCall("GetAccountBalanceAsync", new { accountId = MaskAccountId(accountId) });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            throw new InvalidOperationException("Mock GetAccountBalance failed");
        }

        var balanceData = new
        {
            accountId,
            balance = _config.MockAccount.Balance,
            dayTradingBuyingPower = _config.MockAccount.DayTradingBuyingPower,
            currentDrawdown = _config.MockAccount.CurrentDrawdown,
            maxTrailingDrawdown = _config.MockAccount.MaxTrailingDrawdown,
            isRiskBreached = _config.MockAccount.IsRiskBreached,
            isTradingAllowed = _config.MockAccount.IsTradingAllowed,
            timestamp = DateTime.UtcNow
        };
        
        var json = JsonSerializer.Serialize(balanceData);
        var element = JsonSerializer.Deserialize<JsonElement>(json);
        
        LogMockCall("GetAccountBalanceAsync", new { success = true, balance = _config.MockAccount.Balance });
        
        return element;
    }

    public async Task<JsonElement> GetAccountPositionsAsync(string accountId, CancellationToken cancellationToken = default)
    {
        LogMockCall("GetAccountPositionsAsync", new { accountId = MaskAccountId(accountId) });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            throw new InvalidOperationException("Mock GetAccountPositions failed");
        }

        var positions = CreateMockPositions();
        var json = JsonSerializer.Serialize(positions);
        var element = JsonSerializer.Deserialize<JsonElement>(json);
        
        LogMockCall("GetAccountPositionsAsync", new { success = true, positionCount = positions.Length });
        
        return element;
    }

    public async Task<JsonElement> SearchAccountsAsync(object searchRequest, CancellationToken cancellationToken = default)
    {
        LogMockCall("SearchAccountsAsync", new { hasSearchRequest = searchRequest != null });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            throw new InvalidOperationException("Mock SearchAccounts failed");
        }

        var accounts = new[] { CreateMockAccountData() };
        var json = JsonSerializer.Serialize(accounts);
        var element = JsonSerializer.Deserialize<JsonElement>(json);
        
        LogMockCall("SearchAccountsAsync", new { success = true, accountCount = accounts.Length });
        
        return element;
    }

    // ====================================================================
    // ORDER MANAGEMENT
    // ====================================================================

    public async Task<JsonElement> PlaceOrderAsync(object orderRequest, CancellationToken cancellationToken = default)
    {
        LogMockCall("PlaceOrderAsync", new { hasOrderRequest = orderRequest != null });
        
        await SimulateLatency(cancellationToken);
        
        if (_config.MockScenario == "RiskBreach" && _config.MockAccount.IsRiskBreached)
        {
            throw new InvalidOperationException("Mock order rejected - risk breach");
        }
        
        if (ShouldSimulateError())
        {
            throw new InvalidOperationException("Mock PlaceOrder failed");
        }

        var orderId = Guid.NewGuid().ToString();
        var orderResult = new
        {
            success = true,
            orderId,
            message = "Mock order placed successfully",
            timestamp = DateTime.UtcNow
        };
        
        var json = JsonSerializer.Serialize(orderResult);
        var element = JsonSerializer.Deserialize<JsonElement>(json);
        
        LogMockCall("PlaceOrderAsync", new { success = true, orderId });
        
        // Simulate order update event
        _ = Task.Run(async () =>
        {
            await Task.Delay(500, cancellationToken);
            var orderUpdate = new TradingBot.Abstractions.GatewayUserOrder
            {
                AccountId = _config.MockAccount.AccountId,
                OrderId = orderId,
                CustomTag = $"MOCK-{_orderIdCounter++}",
                Status = "New",
                Reason = "Mock order received",
                Symbol = "ES",
                Side = "Buy",
                Quantity = 1,
                Price = 4500m,
                Timestamp = DateTime.UtcNow
            };
            OnOrderUpdate?.Invoke(orderUpdate);
        }, cancellationToken);
        
        return element;
    }

    public async Task<bool> CancelOrderAsync(string orderId, CancellationToken cancellationToken = default)
    {
        LogMockCall("CancelOrderAsync", new { orderId });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            LogMockCall("CancelOrderAsync", new { success = false, orderId });
            return false;
        }

        LogMockCall("CancelOrderAsync", new { success = true, orderId });
        
        // Simulate order cancellation event
        _ = Task.Run(async () =>
        {
            await Task.Delay(200, cancellationToken);
            var orderUpdate = new TradingBot.Abstractions.GatewayUserOrder
            {
                AccountId = _config.MockAccount.AccountId,
                OrderId = orderId,
                CustomTag = $"MOCK-{orderId}",
                Status = "Cancelled",
                Reason = "Mock order cancelled",
                Symbol = "ES",
                Side = "Buy",
                Quantity = 1,
                Price = 4500m,
                Timestamp = DateTime.UtcNow
            };
            OnOrderUpdate?.Invoke(orderUpdate);
        }, cancellationToken);
        
        return true;
    }

    public async Task<JsonElement> GetOrderStatusAsync(string orderId, CancellationToken cancellationToken = default)
    {
        LogMockCall("GetOrderStatusAsync", new { orderId });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            throw new InvalidOperationException("Mock GetOrderStatus failed");
        }

        var orderStatus = new
        {
            orderId,
            status = "Filled",
            reason = "Mock order filled",
            lastUpdated = DateTime.UtcNow,
            symbol = "ES",
            side = "Buy",
            quantity = 1,
            price = 4500m
        };
        
        var json = JsonSerializer.Serialize(orderStatus);
        var element = JsonSerializer.Deserialize<JsonElement>(json);
        
        LogMockCall("GetOrderStatusAsync", new { success = true, orderId, status = "Filled" });
        
        return element;
    }

    public async Task<JsonElement> SearchOrdersAsync(object searchRequest, CancellationToken cancellationToken = default)
    {
        LogMockCall("SearchOrdersAsync", new { hasSearchRequest = searchRequest != null });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            throw new InvalidOperationException("Mock SearchOrders failed");
        }

        var orders = CreateMockOrders();
        var json = JsonSerializer.Serialize(orders);
        var element = JsonSerializer.Deserialize<JsonElement>(json);
        
        LogMockCall("SearchOrdersAsync", new { success = true, orderCount = orders.Length });
        
        return element;
    }

    public async Task<JsonElement> SearchOpenOrdersAsync(object searchRequest, CancellationToken cancellationToken = default)
    {
        LogMockCall("SearchOpenOrdersAsync", new { hasSearchRequest = searchRequest != null });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            throw new InvalidOperationException("Mock SearchOpenOrders failed");
        }

        var openOrders = CreateMockOrders().Where(o => o.Status == "Open").ToArray();
        var result = new { orders = openOrders };
        var json = JsonSerializer.Serialize(result);
        var element = JsonSerializer.Deserialize<JsonElement>(json);
        
        LogMockCall("SearchOpenOrdersAsync", new { success = true, openOrderCount = openOrders.Length });
        
        return element;
    }

    // ====================================================================
    // TRADE MANAGEMENT
    // ====================================================================

    public async Task<JsonElement> SearchTradesAsync(object searchRequest, CancellationToken cancellationToken = default)
    {
        LogMockCall("SearchTradesAsync", new { hasSearchRequest = searchRequest != null });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            throw new InvalidOperationException("Mock SearchTrades failed");
        }

        var trades = CreateMockTrades();
        var json = JsonSerializer.Serialize(trades);
        var element = JsonSerializer.Deserialize<JsonElement>(json);
        
        LogMockCall("SearchTradesAsync", new { success = true, tradeCount = trades.Length });
        
        return element;
    }

    public async Task<JsonElement> GetTradeAsync(string tradeId, CancellationToken cancellationToken = default)
    {
        LogMockCall("GetTradeAsync", new { tradeId });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            throw new InvalidOperationException("Mock GetTrade failed");
        }

        var trade = CreateMockTrades().First();
        trade.TradeId = tradeId;
        
        var json = JsonSerializer.Serialize(trade);
        var element = JsonSerializer.Deserialize<JsonElement>(json);
        
        LogMockCall("GetTradeAsync", new { success = true, tradeId });
        
        return element;
    }

    // ====================================================================
    // MARKET DATA
    // ====================================================================

    public async Task<JsonElement> GetContractAsync(string contractId, CancellationToken cancellationToken = default)
    {
        LogMockCall("GetContractAsync", new { contractId });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            throw new InvalidOperationException("Mock GetContract failed");
        }

        var contract = CreateMockContract(contractId);
        var json = JsonSerializer.Serialize(contract);
        var element = JsonSerializer.Deserialize<JsonElement>(json);
        
        LogMockCall("GetContractAsync", new { success = true, contractId });
        
        return element;
    }

    public async Task<JsonElement> SearchContractsAsync(object searchRequest, CancellationToken cancellationToken = default)
    {
        LogMockCall("SearchContractsAsync", new { hasSearchRequest = searchRequest != null });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            throw new InvalidOperationException("Mock SearchContracts failed");
        }

        var contracts = new[] { CreateMockContract("ES"), CreateMockContract("NQ") };
        var json = JsonSerializer.Serialize(contracts);
        var element = JsonSerializer.Deserialize<JsonElement>(json);
        
        LogMockCall("SearchContractsAsync", new { success = true, contractCount = contracts.Length });
        
        return element;
    }

    public async Task<JsonElement> GetMarketDataAsync(string symbol, CancellationToken cancellationToken = default)
    {
        LogMockCall("GetMarketDataAsync", new { symbol });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            throw new InvalidOperationException("Mock GetMarketData failed");
        }

        var marketData = CreateMockMarketData(symbol);
        var json = JsonSerializer.Serialize(marketData);
        var element = JsonSerializer.Deserialize<JsonElement>(json);
        
        LogMockCall("GetMarketDataAsync", new { success = true, symbol, bid = marketData.Bid, ask = marketData.Ask });
        
        return element;
    }

    // ====================================================================
    // REAL-TIME SUBSCRIPTIONS
    // ====================================================================

    public async Task<bool> SubscribeOrdersAsync(string accountId, CancellationToken cancellationToken = default)
    {
        LogMockCall("SubscribeOrdersAsync", new { accountId = MaskAccountId(accountId) });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            LogMockCall("SubscribeOrdersAsync", new { success = false, accountId = MaskAccountId(accountId) });
            return false;
        }

        LogMockCall("SubscribeOrdersAsync", new { success = true, accountId = MaskAccountId(accountId) });
        return true;
    }

    public async Task<bool> SubscribeTradesAsync(string accountId, CancellationToken cancellationToken = default)
    {
        LogMockCall("SubscribeTradesAsync", new { accountId = MaskAccountId(accountId) });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            LogMockCall("SubscribeTradesAsync", new { success = false, accountId = MaskAccountId(accountId) });
            return false;
        }

        LogMockCall("SubscribeTradesAsync", new { success = true, accountId = MaskAccountId(accountId) });
        return true;
    }

    public async Task<bool> SubscribeMarketDataAsync(string symbol, CancellationToken cancellationToken = default)
    {
        LogMockCall("SubscribeMarketDataAsync", new { symbol });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            LogMockCall("SubscribeMarketDataAsync", new { success = false, symbol });
            return false;
        }

        LogMockCall("SubscribeMarketDataAsync", new { success = true, symbol });
        return true;
    }

    public async Task<bool> SubscribeLevel2DataAsync(string symbol, CancellationToken cancellationToken = default)
    {
        LogMockCall("SubscribeLevel2DataAsync", new { symbol });
        
        await SimulateLatency(cancellationToken);
        
        if (ShouldSimulateError())
        {
            LogMockCall("SubscribeLevel2DataAsync", new { success = false, symbol });
            return false;
        }

        LogMockCall("SubscribeLevel2DataAsync", new { success = true, symbol });
        return true;
    }

    // ====================================================================
    // PRIVATE HELPER METHODS
    // ====================================================================

    private async Task SimulateLatency(CancellationToken cancellationToken)
    {
        if (_config.MockLatencyMs > 0)
        {
            var jitter = _random.Next(-50, 51); // ±50ms jitter
            var latency = Math.Max(1, _config.MockLatencyMs + jitter);
            await Task.Delay(latency, cancellationToken);
        }
    }

    private bool ShouldSimulateError()
    {
        if (_config.MockScenario == "ApiError")
        {
            return _random.NextDouble() < _config.MockErrorRate;
        }
        return false;
    }

    private void LogMockCall(string operation, object parameters)
    {
        if (!_config.EnableMockAuditLogging) return;
        
        var logData = new
        {
            timestamp = DateTime.UtcNow,
            component = "mock_topstepx_client",
            operation = operation,
            scenario = _config.MockScenario,
            parameters = parameters
        };

        _logger.LogInformation("[MOCK-TOPSTEPX] {LogData}", JsonSerializer.Serialize(logData));
    }

    private string MaskCredential(string credential)
    {
        if (string.IsNullOrEmpty(credential) || credential.Length <= 4)
            return "****";
        
        return credential[..2] + "****" + credential[^2..];
    }

    private string MaskAccountId(string accountId)
    {
        if (string.IsNullOrEmpty(accountId) || accountId.Length <= 6)
            return "****";
        
        return accountId[..3] + "****" + accountId[^3..];
    }

    private string GenerateMockJwt()
    {
        // Generate a mock JWT-like token
        var header = Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes("{\"alg\":\"HS256\",\"typ\":\"JWT\"}"));
        var payload = Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes($"{{\"sub\":\"mock\",\"exp\":{DateTimeOffset.UtcNow.AddHours(24).ToUnixTimeSeconds()}}}"));
        var signature = Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes("mock_signature"));
        
        return $"{header}.{payload}.{signature}";
    }

    private object CreateMockAccountData()
    {
        return new
        {
            accountId = _config.MockAccount.AccountId,
            accountType = _config.MockAccount.AccountType,
            balance = _config.MockAccount.Balance,
            dayTradingBuyingPower = _config.MockAccount.DayTradingBuyingPower,
            maxTrailingDrawdown = _config.MockAccount.MaxTrailingDrawdown,
            currentDrawdown = _config.MockAccount.CurrentDrawdown,
            isRiskBreached = _config.MockAccount.IsRiskBreached,
            isTradingAllowed = _config.MockAccount.IsTradingAllowed,
            status = _config.MockScenario == "RiskBreach" ? "Breached" : "Active",
            timestamp = DateTime.UtcNow
        };
    }

    private object[] CreateMockPositions()
    {
        return new[]
        {
            new
            {
                symbol = "ES",
                quantity = 2,
                averagePrice = 4495.50m,
                marketValue = 224775m,
                unrealizedPnL = 175m,
                timestamp = DateTime.UtcNow
            },
            new
            {
                symbol = "NQ",
                quantity = -1,
                averagePrice = 15420.25m,
                marketValue = -308050m,
                unrealizedPnL = -125m,
                timestamp = DateTime.UtcNow
            }
        };
    }

    private MockOrder[] CreateMockOrders()
    {
        return new[]
        {
            new MockOrder
            {
                OrderId = Guid.NewGuid().ToString(),
                Symbol = "ES",
                Side = "Buy",
                Quantity = 1,
                Price = 4500m,
                Status = "Filled",
                Timestamp = DateTime.UtcNow.AddMinutes(-30)
            },
            new MockOrder
            {
                OrderId = Guid.NewGuid().ToString(),
                Symbol = "NQ",
                Side = "Sell",
                Quantity = 1,
                Price = 15400m,
                Status = "Open",
                Timestamp = DateTime.UtcNow.AddMinutes(-10)
            }
        };
    }

    private MockTrade[] CreateMockTrades()
    {
        return new[]
        {
            new MockTrade
            {
                TradeId = Guid.NewGuid().ToString(),
                OrderId = Guid.NewGuid().ToString(),
                Symbol = "ES",
                Side = "Buy",
                Quantity = 1,
                Price = 4500m,
                Timestamp = DateTime.UtcNow.AddMinutes(-25)
            },
            new MockTrade
            {
                TradeId = Guid.NewGuid().ToString(),
                OrderId = Guid.NewGuid().ToString(),
                Symbol = "ES",
                Side = "Sell",
                Quantity = 1,
                Price = 4502.25m,
                Timestamp = DateTime.UtcNow.AddMinutes(-15)
            }
        };
    }

    private object CreateMockContract(string symbol)
    {
        var basePrice = symbol == "ES" ? 4500m : 15400m;
        
        return new
        {
            contractId = symbol,
            symbol = symbol,
            description = $"{symbol} Future Contract",
            tickSize = symbol == "ES" ? 0.25m : 0.25m,
            pointValue = symbol == "ES" ? 50m : 20m,
            marginRequirement = symbol == "ES" ? 400m : 1200m,
            isActive = true,
            expirationDate = DateTime.UtcNow.AddMonths(3),
            currentPrice = basePrice + (decimal)(_random.NextDouble() * 20 - 10)
        };
    }

    private TradingBot.Abstractions.MarketData CreateMockMarketData(string symbol)
    {
        var basePrice = symbol == "ES" ? 4500m : 15400m;
        var spread = symbol == "ES" ? 0.25m : 0.50m;
        var mid = basePrice + (decimal)(_random.NextDouble() * 20 - 10);
        
        return new TradingBot.Abstractions.MarketData
        {
            Symbol = symbol,
            Bid = (double)(mid - spread / 2),
            Ask = (double)(mid + spread / 2),
            Close = (double)mid,
            Volume = _random.Next(10000, 100000),
            Timestamp = DateTime.UtcNow,
            Open = (double)mid,
            High = (double)mid,
            Low = (double)mid
        };
    }

    private async Task SimulateMarketDataAsync()
    {
        var symbols = new[] { "ES", "NQ" };
        
        while (_isConnected && !_disposed)
        {
            try
            {
                foreach (var symbol in symbols)
                {
                    var marketData = CreateMockMarketData(symbol);
                    OnMarketDataUpdate?.Invoke(marketData);
                    
                    // Simulate Level 2 data
                    var orderBook = new TradingBot.Abstractions.OrderBookData
                    {
                        Symbol = symbol,
                        BidPrice = (decimal)marketData.Bid,
                        BidSize = _random.Next(10, 100),
                        AskPrice = (decimal)marketData.Ask,
                        AskSize = _random.Next(10, 100),
                        Timestamp = DateTime.UtcNow
                    };
                    OnLevel2Update?.Invoke(orderBook);
                }
                
                await Task.Delay(1000); // 1 second interval
            }
            catch (Exception ex)
            {
                if (!_disposed)
                {
                    _logger.LogError(ex, "[MOCK-TOPSTEPX] Error in market data simulation");
                }
                break;
            }
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        
        _isConnected = false;
        LogMockCall("Dispose", new { });
    }
}

// Helper classes for mock data
public class MockOrder
{
    public string OrderId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
    public int Quantity { get; set; }
    public decimal Price { get; set; }
    public string Status { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
}

public class MockTrade
{
    public string TradeId { get; set; } = string.Empty;
    public string OrderId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
    public int Quantity { get; set; }
    public decimal Price { get; set; }
    public DateTime Timestamp { get; set; }
}