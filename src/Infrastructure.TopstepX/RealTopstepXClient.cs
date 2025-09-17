using System;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;
using BotCore.Services;

namespace TradingBot.Infrastructure.TopstepX;

/// <summary>
/// Real TopstepX client implementation that wraps existing services
/// Provides interface parity with MockTopstepXClient for hot-swapping
/// </summary>
public class RealTopstepXClient : ITopstepXClient, IDisposable
{
    private readonly ILogger<RealTopstepXClient> _logger;
    private readonly ITopstepXService _topstepXService;
    private readonly IOrderService _orderService;
    private readonly IAccountService _accountService;
    private readonly HttpClient _httpClient;
    private volatile bool _disposed;

    public bool IsConnected => _topstepXService?.IsConnected ?? false;

    // Events - forward from TopstepXService
    public event Action<TradingBot.Abstractions.GatewayUserOrder>? OnOrderUpdate;
    public event Action<TradingBot.Abstractions.GatewayUserTrade>? OnTradeUpdate;
    public event Action<TradingBot.Abstractions.MarketData>? OnMarketDataUpdate;
    public event Action<TradingBot.Abstractions.OrderBookData>? OnLevel2Update;
    public event Action<TradingBot.Abstractions.TradeConfirmation>? OnTradeConfirmed;
    public event Action<string>? OnError;
    public event Action<bool>? OnConnectionStateChanged;

    public RealTopstepXClient(
        ILogger<RealTopstepXClient> logger,
        ITopstepXService topstepXService,
        IOrderService orderService,
        IAccountService accountService,
        HttpClient httpClient)
    {
        _logger = logger;
        _topstepXService = topstepXService;
        _orderService = orderService;
        _accountService = accountService;
        _httpClient = httpClient;

        // Wire up events from TopstepXService
        WireUpEvents();
        
        _logger.LogInformation("[REAL-TOPSTEPX] Client initialized with real TopstepX services");
    }

    // ====================================================================
    // CONNECTION MANAGEMENT
    // ====================================================================

    public async Task<bool> ConnectAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] ConnectAsync called");
        
        try
        {
            var result = await _topstepXService.ConnectAsync();
            _logger.LogInformation("[REAL-TOPSTEPX] ConnectAsync result: {Result}", result);
            
            if (result)
            {
                OnConnectionStateChanged?.Invoke(true);
            }
            
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] ConnectAsync failed");
            return false;
        }
    }

    public async Task<bool> DisconnectAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] DisconnectAsync called");
        
        try
        {
            var result = await _topstepXService.DisconnectAsync();
            _logger.LogInformation("[REAL-TOPSTEPX] DisconnectAsync result: {Result}", result);
            
            if (result)
            {
                OnConnectionStateChanged?.Invoke(false);
            }
            
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] DisconnectAsync failed");
            return false;
        }
    }

    // ====================================================================
    // AUTHENTICATION
    // ====================================================================

    public async Task<(string jwt, DateTimeOffset expiresUtc)> AuthenticateAsync(
        string username, string password, string apiKey, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] AuthenticateAsync called for user: {Username}", MaskCredential(username));
        
        try
        {
            // Use HTTP client to authenticate
            var loginPayload = new
            {
                username,
                password,
                apiKey
            };

            var json = JsonSerializer.Serialize(loginPayload);
            var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync("auth/token", content, cancellationToken);
            
            if (response.IsSuccessStatusCode)
            {
                var responseContent = await response.Content.ReadAsStringAsync(cancellationToken);
                var tokenResponse = JsonSerializer.Deserialize<JsonElement>(responseContent);
                
                if (tokenResponse.TryGetProperty("token", out var tokenElement))
                {
                    var jwt = tokenElement.GetString() ?? throw new InvalidOperationException("Token is null");
                    var expiresUtc = DateTimeOffset.UtcNow.AddHours(24); // Default expiration
                    
                    if (tokenResponse.TryGetProperty("expiresIn", out var expiresElement) && expiresElement.TryGetInt32(out var expiresIn))
                    {
                        expiresUtc = DateTimeOffset.UtcNow.AddSeconds(expiresIn);
                    }
                    
                    _logger.LogInformation("[REAL-TOPSTEPX] Authentication successful, token expires: {ExpiresUtc}", expiresUtc);
                    return (jwt, expiresUtc);
                }
            }
            
            throw new InvalidOperationException("Authentication failed - no token in response");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] Authentication failed");
            throw;
        }
    }

    public async Task<(string jwt, DateTimeOffset expiresUtc)> RefreshTokenAsync(
        string refreshToken, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] RefreshTokenAsync called");
        
        try
        {
            var refreshPayload = new { refreshToken };
            var json = JsonSerializer.Serialize(refreshPayload);
            var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync("auth/refresh", content, cancellationToken);
            
            if (response.IsSuccessStatusCode)
            {
                var responseContent = await response.Content.ReadAsStringAsync(cancellationToken);
                var tokenResponse = JsonSerializer.Deserialize<JsonElement>(responseContent);
                
                if (tokenResponse.TryGetProperty("token", out var tokenElement))
                {
                    var jwt = tokenElement.GetString() ?? throw new InvalidOperationException("Token is null");
                    var expiresUtc = DateTimeOffset.UtcNow.AddHours(24); // Default expiration
                    
                    if (tokenResponse.TryGetProperty("expiresIn", out var expiresElement) && expiresElement.TryGetInt32(out var expiresIn))
                    {
                        expiresUtc = DateTimeOffset.UtcNow.AddSeconds(expiresIn);
                    }
                    
                    _logger.LogInformation("[REAL-TOPSTEPX] Token refresh successful, expires: {ExpiresUtc}", expiresUtc);
                    return (jwt, expiresUtc);
                }
            }
            
            throw new InvalidOperationException("Token refresh failed - no token in response");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] Token refresh failed");
            throw;
        }
    }

    // ====================================================================
    // ACCOUNT MANAGEMENT
    // ====================================================================

    public async Task<JsonElement> GetAccountAsync(string accountId, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] GetAccountAsync called for account: {AccountId}", MaskAccountId(accountId));
        
        try
        {
            // Real implementation using account service
            var account = await _accountService.GetAccountAsync(accountId, cancellationToken);
            if (account == null)
            {
                throw new InvalidOperationException($"Account {MaskAccountId(accountId)} not found");
            }
            
            var accountData = new 
            { 
                accountId = account.AccountId,
                status = account.Status,
                type = account.Type,
                balance = account.Balance,
                equity = account.Equity,
                unrealizedPnL = account.UnrealizedPnL,
                isActive = account.IsActive,
                riskLevel = account.RiskLevel,
                lastUpdated = account.LastUpdated
            };
            
            var json = JsonSerializer.Serialize(accountData);
            var element = JsonSerializer.Deserialize<JsonElement>(json);
            
            _logger.LogInformation("[REAL-TOPSTEPX] Account data retrieved for: {AccountId}, Status: {Status}", 
                MaskAccountId(accountId), account.Status);
            
            return element;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] GetAccountAsync failed for account: {AccountId}", MaskAccountId(accountId));
            throw;
        }
    }

    public async Task<JsonElement> GetAccountBalanceAsync(string accountId, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] GetAccountBalanceAsync called for account: {AccountId}", MaskAccountId(accountId));
        
        try
        {
            // Real implementation using account service
            var balance = await _accountService.GetAccountBalanceAsync(accountId, cancellationToken);
            if (balance == null)
            {
                throw new InvalidOperationException($"Balance information not available for account {MaskAccountId(accountId)}");
            }
            
            var balanceData = new 
            { 
                accountId,
                balance = balance.CurrentBalance,
                equity = balance.Equity,
                unrealizedPnL = balance.UnrealizedPnL,
                realizedPnL = 0m, // Not available in BalanceInfo
                buyingPower = balance.BuyingPower,
                netLiquidatingValue = balance.TotalValue, // Use TotalValue as approximation
                isRiskBreached = balance.RiskPercentage > 0.8m, // Calculate from risk percentage
                riskPercentage = balance.RiskPercentage,
                maxDrawdown = 0m, // Not available in BalanceInfo
                lastUpdated = DateTime.UtcNow, // Use current time
                currency = balance.Currency ?? "USD"
            };
            
            var json = JsonSerializer.Serialize(balanceData);
            var element = JsonSerializer.Deserialize<JsonElement>(json);
            
            _logger.LogInformation("[REAL-TOPSTEPX] Balance retrieved for: {AccountId}, Balance: {Balance:C}, Risk: {Risk}%", 
                MaskAccountId(accountId), balance.CurrentBalance, balance.RiskPercentage);
            
            return element;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] GetAccountBalanceAsync failed for account: {AccountId}", MaskAccountId(accountId));
            throw;
        }
    }

    public async Task<JsonElement> GetAccountPositionsAsync(string accountId, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] GetAccountPositionsAsync called for account: {AccountId}", MaskAccountId(accountId));
        
        try
        {
            // Real implementation using account service
            var positions = await _accountService.GetAccountPositionsAsync(accountId, cancellationToken);
            if (positions == null)
            {
                throw new InvalidOperationException($"Position information not available for account {MaskAccountId(accountId)}");
            }
            
            var positionData = positions.Select(pos => new
            {
                symbol = pos.Symbol,
                quantity = pos.Quantity,
                side = pos.Side,
                averagePrice = pos.AveragePrice,
                marketPrice = pos.MarketPrice,
                unrealizedPnL = pos.UnrealizedPnL,
                realizedPnL = pos.RealizedPnL,
                netValue = pos.NetValue,
                openTime = pos.OpenTime,
                lastUpdated = pos.LastUpdated,
                riskAmount = pos.RiskAmount,
                marginRequirement = pos.MarginRequirement
            }).ToArray();
            
            var json = JsonSerializer.Serialize(positionData);
            var element = JsonSerializer.Deserialize<JsonElement>(json);
            
            _logger.LogInformation("[REAL-TOPSTEPX] Positions retrieved for: {AccountId}, Count: {Count}, Total P&L: {PnL:C}", 
                MaskAccountId(accountId), positions.Count(), positions.Sum(p => p.UnrealizedPnL));
            
            return element;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] GetAccountPositionsAsync failed for account: {AccountId}", MaskAccountId(accountId));
            throw;
        }
    }

    public async Task<JsonElement> SearchAccountsAsync(object searchRequest, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] SearchAccountsAsync called with search criteria");
        
        try
        {
            // Real implementation using account service
            var accounts = await _accountService.SearchAccountsAsync(cancellationToken);
            if (accounts == null)
            {
                throw new InvalidOperationException("Account search service returned null");
            }
            
            var accountsData = accounts.Select(acc => new
            {
                accountId = MaskAccountId(acc.AccountId),
                status = acc.Status,
                type = acc.Type,
                balance = acc.Balance,
                equity = acc.Equity,
                isActive = acc.IsActive,
                riskLevel = acc.RiskLevel,
                lastUpdated = acc.LastUpdated,
                createdDate = acc.LastUpdated.AddDays(-30), // Approximate creation date
                lastLoginDate = acc.LastUpdated, // Use LastUpdated as approximation
                buyingPower = acc.BuyingPower,
                tradingEnabled = acc.IsActive, // Use IsActive as trading enabled
                accountName = $"Account-{MaskAccountId(acc.AccountId)}", // Generate account name
                region = "US" // Default region
            }).ToArray();
            
            var json = JsonSerializer.Serialize(accountsData);
            var element = JsonSerializer.Deserialize<JsonElement>(json);
            
            _logger.LogInformation("[REAL-TOPSTEPX] Account search completed, found {Count} accounts", accounts.Count());
            
            return element;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] SearchAccountsAsync failed");
            throw;
        }
    }

    // ====================================================================
    // ORDER MANAGEMENT
    // ====================================================================

    public async Task<JsonElement> PlaceOrderAsync(object orderRequest, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] PlaceOrderAsync called");
        
        try
        {
            // Convert object to PlaceOrderRequest if needed
            PlaceOrderRequest request;
            if (orderRequest is PlaceOrderRequest placeOrderRequest)
            {
                request = placeOrderRequest;
            }
            else
            {
                // Convert from object to PlaceOrderRequest
                var json = JsonSerializer.Serialize(orderRequest);
                var jsonElement = JsonSerializer.Deserialize<JsonElement>(json);
                
                request = new PlaceOrderRequest(
                    jsonElement.GetProperty("symbol").GetString() ?? throw new ArgumentException("Missing symbol"),
                    jsonElement.GetProperty("side").GetString() ?? throw new ArgumentException("Missing side"),
                    jsonElement.GetProperty("quantity").GetDecimal(),
                    jsonElement.GetProperty("price").GetDecimal(),
                    jsonElement.GetProperty("orderType").GetString() ?? "LIMIT",
                    jsonElement.GetProperty("customTag").GetString() ?? Guid.NewGuid().ToString(),
                    jsonElement.GetProperty("accountId").GetString() ?? throw new ArgumentException("Missing accountId")
                );
            }
            
            var result = await _orderService.PlaceOrderAsync(request);
            var resultJson = JsonSerializer.Serialize(result);
            var element = JsonSerializer.Deserialize<JsonElement>(resultJson);
            
            _logger.LogInformation("[REAL-TOPSTEPX] PlaceOrderAsync successful, OrderId: {OrderId}", result.OrderId);
            return element;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] PlaceOrderAsync failed");
            throw;
        }
    }

    public async Task<bool> CancelOrderAsync(string orderId, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] CancelOrderAsync called for order: {OrderId}", orderId);
        
        try
        {
            var result = await _orderService.CancelOrderAsync(orderId, cancellationToken);
            _logger.LogInformation("[REAL-TOPSTEPX] CancelOrderAsync result for order {OrderId}: {Result}", orderId, result);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] CancelOrderAsync failed for order: {OrderId}", orderId);
            return false;
        }
    }

    public async Task<JsonElement> GetOrderStatusAsync(string orderId, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] GetOrderStatusAsync called for order: {OrderId}", orderId);
        
        try
        {
#pragma warning disable CS0618 // Type or member is obsolete
            var status = await _orderService.GetOrderStatusAsync(orderId);
#pragma warning restore CS0618 // Type or member is obsolete
            var json = JsonSerializer.Serialize(status);
            var element = JsonSerializer.Deserialize<JsonElement>(json);
            
            _logger.LogInformation("[REAL-TOPSTEPX] GetOrderStatusAsync successful for order: {OrderId}", orderId);
            return element;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] GetOrderStatusAsync failed for order: {OrderId}", orderId);
            throw;
        }
    }

    public async Task<JsonElement> SearchOrdersAsync(object searchRequest, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] SearchOrdersAsync called");
        
        try
        {
            var orders = await _orderService.SearchOrdersAsync(searchRequest);
            _logger.LogInformation("[REAL-TOPSTEPX] SearchOrdersAsync successful");
            return orders;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] SearchOrdersAsync failed");
            throw;
        }
    }

    public async Task<JsonElement> SearchOpenOrdersAsync(object searchRequest, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] SearchOpenOrdersAsync called");
        
        try
        {
            var orders = await _orderService.SearchOpenOrdersAsync(searchRequest);
            _logger.LogInformation("[REAL-TOPSTEPX] SearchOpenOrdersAsync successful");
            return orders;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] SearchOpenOrdersAsync failed");
            throw;
        }
    }

    // ====================================================================
    // TRADE MANAGEMENT
    // ====================================================================

    public async Task<JsonElement> SearchTradesAsync(object searchRequest, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] SearchTradesAsync called");
        
        try
        {
            // Use HTTP client for trade search
            var json = JsonSerializer.Serialize(searchRequest);
            var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");
            var response = await _httpClient.PostAsync("api/Trade/search", content, cancellationToken);
            
            if (response.IsSuccessStatusCode)
            {
                var responseContent = await response.Content.ReadAsStringAsync(cancellationToken);
                var result = JsonSerializer.Deserialize<JsonElement>(responseContent);
                _logger.LogInformation("[REAL-TOPSTEPX] SearchTradesAsync successful");
                return result;
            }
            else
            {
                _logger.LogWarning("[REAL-TOPSTEPX] SearchTradesAsync failed with status: {StatusCode}", response.StatusCode);
                return new JsonElement();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] SearchTradesAsync failed");
            throw;
        }
    }

    public async Task<JsonElement> GetTradeAsync(string tradeId, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] GetTradeAsync called for trade: {TradeId}", tradeId);
        
        try
        {
            // Make real HTTP call to TopstepX API for trade data
            var url = $"/v1/trades/{tradeId}";
            var response = await _httpClient.GetAsync(url, cancellationToken);
            
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync(cancellationToken);
                var tradeData = JsonSerializer.Deserialize<JsonElement>(content);
                _logger.LogInformation("[REAL-TOPSTEPX] GetTradeAsync succeeded for trade: {TradeId}", tradeId);
                return tradeData;
            }
            else
            {
                _logger.LogWarning("[REAL-TOPSTEPX] GetTradeAsync returned {StatusCode} for trade: {TradeId}", 
                    response.StatusCode, tradeId);
                return new JsonElement();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] GetTradeAsync failed for trade: {TradeId}", tradeId);
            throw;
        }
    }

    // ====================================================================
    // MARKET DATA
    // ====================================================================

    public async Task<JsonElement> GetContractAsync(string contractId, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] GetContractAsync called for contract: {ContractId}", contractId);
        
        try
        {
            // Make real HTTP call to TopstepX API for contract data
            var url = $"/v1/contracts/{contractId}";
            var response = await _httpClient.GetAsync(url, cancellationToken);
            
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync(cancellationToken);
                var contractData = JsonSerializer.Deserialize<JsonElement>(content);
                _logger.LogInformation("[REAL-TOPSTEPX] GetContractAsync succeeded for contract: {ContractId}", contractId);
                return contractData;
            }
            else
            {
                _logger.LogWarning("[REAL-TOPSTEPX] GetContractAsync returned {StatusCode} for contract: {ContractId}", 
                    response.StatusCode, contractId);
                return new JsonElement();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] GetContractAsync failed for contract: {ContractId}", contractId);
            throw;
        }
    }

    public async Task<JsonElement> SearchContractsAsync(object searchRequest, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] SearchContractsAsync called");
        
        try
        {
            // Make real HTTP call to TopstepX API for contract search
            var url = "/v1/contracts/search";
            var jsonContent = JsonSerializer.Serialize(searchRequest);
            var content = new StringContent(jsonContent, System.Text.Encoding.UTF8, "application/json");
            
            var response = await _httpClient.PostAsync(url, content, cancellationToken);
            
            if (response.IsSuccessStatusCode)
            {
                var responseContent = await response.Content.ReadAsStringAsync(cancellationToken);
                var contractsData = JsonSerializer.Deserialize<JsonElement>(responseContent);
                _logger.LogInformation("[REAL-TOPSTEPX] SearchContractsAsync succeeded");
                return contractsData;
            }
            else
            {
                _logger.LogWarning("[REAL-TOPSTEPX] SearchContractsAsync returned {StatusCode}", response.StatusCode);
                return new JsonElement();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] SearchContractsAsync failed");
            throw;
        }
    }

    public async Task<JsonElement> GetMarketDataAsync(string symbol, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] GetMarketDataAsync called for symbol: {Symbol}", symbol);
        
        try
        {
            // Make real HTTP call to TopstepX API for market data
            var url = $"/v1/market-data/{symbol}";
            var response = await _httpClient.GetAsync(url, cancellationToken);
            
            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync(cancellationToken);
                var marketData = JsonSerializer.Deserialize<JsonElement>(content);
                _logger.LogInformation("[REAL-TOPSTEPX] GetMarketDataAsync succeeded for symbol: {Symbol}", symbol);
                return marketData;
            }
            else
            {
                _logger.LogWarning("[REAL-TOPSTEPX] GetMarketDataAsync returned {StatusCode} for symbol: {Symbol}", 
                    response.StatusCode, symbol);
                return new JsonElement();
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] GetMarketDataAsync failed for symbol: {Symbol}", symbol);
            throw;
        }
    }

    // ====================================================================
    // REAL-TIME SUBSCRIPTIONS
    // ====================================================================

    public async Task<bool> SubscribeOrdersAsync(string accountId, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] SubscribeOrdersAsync called for account: {AccountId}", MaskAccountId(accountId));
        
        try
        {
            var result = await _topstepXService.SubscribeOrdersAsync(accountId);
            _logger.LogInformation("[REAL-TOPSTEPX] SubscribeOrdersAsync result for account {AccountId}: {Result}", MaskAccountId(accountId), result);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] SubscribeOrdersAsync failed for account: {AccountId}", MaskAccountId(accountId));
            return false;
        }
    }

    public async Task<bool> SubscribeTradesAsync(string accountId, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] SubscribeTradesAsync called for account: {AccountId}", MaskAccountId(accountId));
        
        try
        {
            var result = await _topstepXService.SubscribeTradesAsync(accountId);
            _logger.LogInformation("[REAL-TOPSTEPX] SubscribeTradesAsync result for account {AccountId}: {Result}", MaskAccountId(accountId), result);
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] SubscribeTradesAsync failed for account: {AccountId}", MaskAccountId(accountId));
            return false;
        }
    }

    public async Task<bool> SubscribeMarketDataAsync(string symbol, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] SubscribeMarketDataAsync called for symbol: {Symbol}", symbol);
        
        try
        {
            // TopstepXService automatically subscribes to configured symbols
            // This is a no-op for now, but could be extended to support dynamic subscriptions
            await Task.CompletedTask; // Make the method async
            _logger.LogInformation("[REAL-TOPSTEPX] SubscribeMarketDataAsync successful for symbol: {Symbol}", symbol);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] SubscribeMarketDataAsync failed for symbol: {Symbol}", symbol);
            return false;
        }
    }

    public async Task<bool> SubscribeLevel2DataAsync(string symbol, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("[REAL-TOPSTEPX] SubscribeLevel2DataAsync called for symbol: {Symbol}", symbol);
        
        try
        {
            // TopstepXService automatically subscribes to Level 2 data if enabled
            // This is a no-op for now, but could be extended to support dynamic subscriptions
            await Task.CompletedTask; // Make the method async
            _logger.LogInformation("[REAL-TOPSTEPX] SubscribeLevel2DataAsync successful for symbol: {Symbol}", symbol);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[REAL-TOPSTEPX] SubscribeLevel2DataAsync failed for symbol: {Symbol}", symbol);
            return false;
        }
    }

    // ====================================================================
    // PRIVATE HELPER METHODS
    // ====================================================================

    private void WireUpEvents()
    {
        if (_topstepXService != null)
        {
            _topstepXService.OnGatewayUserOrder += (order) => {
                // Convert from BotCore type to Abstractions type
                var abstractOrder = new TradingBot.Abstractions.GatewayUserOrder
                {
                    AccountId = order.AccountId,
                    OrderId = order.OrderId,
                    CustomTag = order.CustomTag,
                    Status = order.Status,
                    Reason = order.Reason,
                    Symbol = order.Symbol,
                    Side = order.Side,
                    Quantity = order.Quantity,
                    Price = order.Price,
                    Timestamp = order.Timestamp
                };
                OnOrderUpdate?.Invoke(abstractOrder);
            };
            
            _topstepXService.OnGatewayUserTrade += (trade) => {
                // Convert from BotCore type to Abstractions type
                var abstractTrade = new TradingBot.Abstractions.GatewayUserTrade
                {
                    AccountId = trade.AccountId,
                    OrderId = trade.OrderId,
                    CustomTag = trade.CustomTag,
                    FillPrice = trade.FillPrice,
                    Quantity = trade.Quantity,
                    Time = trade.Time,
                    Symbol = trade.Symbol,
                    Side = trade.Side
                };
                OnTradeUpdate?.Invoke(abstractTrade);
            };
            
            _topstepXService.OnMarketData += (data) => {
                // Convert from BotCore type to Abstractions type
                var abstractData = new TradingBot.Abstractions.MarketData
                {
                    Symbol = data.Symbol,
                    Timestamp = data.Timestamp,
                    Open = (double)data.Last, // Use Last as Open
                    High = (double)data.Last, // Use Last as High
                    Low = (double)data.Last,  // Use Last as Low
                    Close = (double)data.Last,
                    Volume = (double)data.Volume,
                    Bid = (double)data.Bid,
                    Ask = (double)data.Ask
                };
                OnMarketDataUpdate?.Invoke(abstractData);
            };
            
            _topstepXService.OnLevel2Update += (data) => {
                // Convert from BotCore type to Abstractions type
                var abstractData = new TradingBot.Abstractions.OrderBookData
                {
                    Symbol = data.Symbol,
                    BidPrice = data.BidPrice,
                    BidSize = data.BidSize,
                    AskPrice = data.AskPrice,
                    AskSize = data.AskSize,
                    Timestamp = data.Timestamp
                };
                OnLevel2Update?.Invoke(abstractData);
            };
            
            _topstepXService.OnTradeConfirmed += (trade) => {
                // Convert from BotCore type to Abstractions type
                var abstractTrade = new TradingBot.Abstractions.TradeConfirmation
                {
                    OrderId = trade.OrderId,
                    Symbol = trade.Symbol,
                    Side = trade.Side,
                    Quantity = trade.Quantity,
                    Price = trade.Price,
                    Timestamp = trade.Timestamp
                };
                OnTradeConfirmed?.Invoke(abstractTrade);
            };
            
            _topstepXService.OnError += (error) => OnError?.Invoke(error);
        }
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

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _logger.LogInformation("[REAL-TOPSTEPX] Client disposed");
            }
            _disposed = true;
        }
    }
}