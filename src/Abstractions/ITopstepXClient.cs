using System;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;

namespace TradingBot.Abstractions;

/// <summary>
/// Complete TopstepX client interface covering all functionality
/// Can be implemented by both real and mock clients for testing
/// </summary>
public interface ITopstepXClient
{
    // ====================================================================
    // CONNECTION MANAGEMENT
    // ====================================================================
    
    /// <summary>
    /// Connect to TopstepX services (SignalR, REST API)
    /// </summary>
    Task<bool> ConnectAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Disconnect from TopstepX services
    /// </summary>
    Task<bool> DisconnectAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check if client is connected and ready
    /// </summary>
    bool IsConnected { get; }
    
    // ====================================================================
    // AUTHENTICATION
    // ====================================================================
    
    /// <summary>
    /// Authenticate with TopstepX and obtain JWT token
    /// </summary>
    Task<(string jwt, DateTimeOffset expiresUtc)> AuthenticateAsync(
        string username, 
        string password, 
        string apiKey, 
        CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Refresh existing JWT token
    /// </summary>
    Task<(string jwt, DateTimeOffset expiresUtc)> RefreshTokenAsync(
        string refreshToken, 
        CancellationToken cancellationToken = default);
    
    // ====================================================================
    // ACCOUNT MANAGEMENT
    // ====================================================================
    
    /// <summary>
    /// Get account information
    /// </summary>
    Task<JsonElement> GetAccountAsync(string accountId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get account balance and margin info
    /// </summary>
    Task<JsonElement> GetAccountBalanceAsync(string accountId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get account positions
    /// </summary>
    Task<JsonElement> GetAccountPositionsAsync(string accountId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Search for available accounts
    /// </summary>
    Task<JsonElement> SearchAccountsAsync(object searchRequest, CancellationToken cancellationToken = default);
    
    // ====================================================================
    // ORDER MANAGEMENT
    // ====================================================================
    
    /// <summary>
    /// Place a new order
    /// </summary>
    Task<JsonElement> PlaceOrderAsync(object orderRequest, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Cancel an existing order
    /// </summary>
    Task<bool> CancelOrderAsync(string orderId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get order status
    /// </summary>
    Task<JsonElement> GetOrderStatusAsync(string orderId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Search orders with filtering
    /// </summary>
    Task<JsonElement> SearchOrdersAsync(object searchRequest, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Search open orders
    /// </summary>
    Task<JsonElement> SearchOpenOrdersAsync(object searchRequest, CancellationToken cancellationToken = default);
    
    // ====================================================================
    // TRADE MANAGEMENT
    // ====================================================================
    
    /// <summary>
    /// Search trade history
    /// </summary>
    Task<JsonElement> SearchTradesAsync(object searchRequest, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get trade details
    /// </summary>
    Task<JsonElement> GetTradeAsync(string tradeId, CancellationToken cancellationToken = default);
    
    // ====================================================================
    // MARKET DATA
    // ====================================================================
    
    /// <summary>
    /// Get contract information
    /// </summary>
    Task<JsonElement> GetContractAsync(string contractId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Search available contracts
    /// </summary>
    Task<JsonElement> SearchContractsAsync(object searchRequest, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get market data snapshot
    /// </summary>
    Task<JsonElement> GetMarketDataAsync(string symbol, CancellationToken cancellationToken = default);
    
    // ====================================================================
    // REAL-TIME SUBSCRIPTIONS
    // ====================================================================
    
    /// <summary>
    /// Subscribe to order updates for an account
    /// </summary>
    Task<bool> SubscribeOrdersAsync(string accountId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Subscribe to trade updates for an account
    /// </summary>
    Task<bool> SubscribeTradesAsync(string accountId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Subscribe to market data for a symbol
    /// </summary>
    Task<bool> SubscribeMarketDataAsync(string symbol, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Subscribe to Level 2 market data
    /// </summary>
    Task<bool> SubscribeLevel2DataAsync(string symbol, CancellationToken cancellationToken = default);
    
    // ====================================================================
    // EVENTS
    // ====================================================================
    
    /// <summary>
    /// Order update events
    /// </summary>
    event Action<TradingBot.Abstractions.GatewayUserOrder>? OnOrderUpdate;
    
    /// <summary>
    /// Trade execution events
    /// </summary>
    event Action<TradingBot.Abstractions.GatewayUserTrade>? OnTradeUpdate;
    
    /// <summary>
    /// Market data update events
    /// </summary>
    event Action<TradingBot.Abstractions.MarketData>? OnMarketDataUpdate;
    
    /// <summary>
    /// Level 2 market data events
    /// </summary>
    event Action<TradingBot.Abstractions.OrderBookData>? OnLevel2Update;
    
    /// <summary>
    /// Trade confirmation events
    /// </summary>
    event Action<TradingBot.Abstractions.TradeConfirmation>? OnTradeConfirmed;
    
    /// <summary>
    /// Error events
    /// </summary>
    event Action<string>? OnError;
    
    /// <summary>
    /// Connection state change events
    /// </summary>
    event Action<bool>? OnConnectionStateChanged;
}

/// <summary>
/// Configuration for TopstepX client (production only)
/// </summary>
public class TopstepXClientConfiguration
{
    /// <summary>
    /// Client implementation type - always "Real" for production
    /// </summary>
    public string ClientType { get; set; } = "Real";
}

