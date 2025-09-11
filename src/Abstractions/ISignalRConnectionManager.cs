using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;

namespace TradingBot.Abstractions;

/// <summary>
/// Production-ready SignalR hub connection manager interface
/// Completes the connection state machine for production trading
/// </summary>
public interface ISignalRConnectionManager
{
    Task<HubConnection> GetUserHubConnectionAsync();
    Task<HubConnection> GetMarketHubConnectionAsync();
    bool IsUserHubConnected { get; }
    bool IsMarketHubConnected { get; }
    event Action<string> ConnectionStateChanged;
    
    // TopstepX specification compliant subscription methods
    Task<bool> SubscribeToUserEventsAsync(string accountId);
    Task<bool> SubscribeToMarketEventsAsync(string contractId);
    
    // Retry subscriptions with valid account ID after login
    Task<bool> RetrySubscriptionsWithAccountId(string accountId);
    
    // Data reception events - completing the connection state machine
    event Action<object> OnMarketDataReceived;
    event Action<object> OnContractQuotesReceived;
    event Action<object> OnGatewayUserOrderReceived;
    event Action<object> OnGatewayUserTradeReceived;
    event Action<object> OnFillUpdateReceived;
    event Action<object> OnOrderUpdateReceived;
}