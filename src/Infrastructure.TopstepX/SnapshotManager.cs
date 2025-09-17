using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.Infrastructure.TopstepX;

/// <summary>
/// Snapshot Manager for TopstepX state synchronization
/// Implements Snapshot + Delta pattern to handle missed events during disconnections
/// </summary>
public interface ISnapshotManager
{
    Task<bool> FetchInitialSnapshotAsync(string accountId);
    Task<bool> ReconcileAfterReconnectAsync(string accountId);
    Task<AccountSnapshot?> GetCurrentSnapshotAsync(string accountId);
    event Action<string, AccountSnapshot> SnapshotUpdated;
}

public record AccountSnapshot(
    string AccountId,
    DateTime Timestamp,
    List<PositionSnapshot> Positions,
    List<OrderSnapshot> Orders,
    decimal Balance,
    decimal Equity,
    decimal DayPnL
);

public record PositionSnapshot(
    string Symbol,
    int Quantity,
    decimal AveragePrice,
    decimal MarketValue,
    decimal UnrealizedPnL
);

public record OrderSnapshot(
    string OrderId,
    string Symbol,
    string Side,
    int Quantity,
    decimal Price,
    string Status,
    DateTime CreatedTime
);

public class SnapshotManager : ISnapshotManager
{
    private readonly ILogger<SnapshotManager> _logger;
    private readonly ITradingLogger _tradingLogger;
    private readonly HttpClient _httpClient;
    private readonly Dictionary<string, AccountSnapshot> _snapshots = new();
    private readonly object _snapshotLock = new();

    public event Action<string, AccountSnapshot>? SnapshotUpdated;

    public SnapshotManager(
        ILogger<SnapshotManager> logger,
        ITradingLogger tradingLogger,
        HttpClient httpClient)
    {
        _logger = logger;
        _tradingLogger = tradingLogger;
        _httpClient = httpClient;
    }

    public async Task<bool> FetchInitialSnapshotAsync(string accountId)
    {
        try
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SnapshotManager",
                $"Fetching initial snapshot for account {accountId.Substring(0, Math.Min(4, accountId.Length))}***");

            var snapshot = await FetchAccountSnapshotAsync(accountId);
            if (snapshot != null)
            {
                lock (_snapshotLock)
                {
                    _snapshots[accountId] = snapshot;
                }

                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SnapshotManager",
                    $"✅ Initial snapshot fetched - Positions: {snapshot.Positions.Count}, Orders: {snapshot.Orders.Count}");

                SnapshotUpdated?.Invoke(accountId, snapshot);
                return true;
            }

            return false;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching initial snapshot for account {AccountId}", accountId);
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SnapshotManager",
                $"Failed to fetch initial snapshot: {ex.Message}");
            return false;
        }
    }

    public async Task<bool> ReconcileAfterReconnectAsync(string accountId)
    {
        try
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SnapshotManager",
                $"Reconciling state after reconnect for account {accountId.Substring(0, Math.Min(4, accountId.Length))}***");

            var currentSnapshot = await FetchAccountSnapshotAsync(accountId);
            if (currentSnapshot == null)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SnapshotManager",
                    "Failed to fetch current snapshot for reconciliation");
                return false;
            }

            AccountSnapshot? previousSnapshot = null;
            lock (_snapshotLock)
            {
                _snapshots.TryGetValue(accountId, out previousSnapshot);
                _snapshots[accountId] = currentSnapshot;
            }

            if (previousSnapshot != null)
            {
                await DetectAndLogChangesAsync(previousSnapshot, currentSnapshot);
            }

            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SnapshotManager",
                "✅ State reconciliation completed");

            SnapshotUpdated?.Invoke(accountId, currentSnapshot);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during state reconciliation for account {AccountId}", accountId);
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SnapshotManager",
                $"State reconciliation failed: {ex.Message}");
            return false;
        }
    }

    public async Task<AccountSnapshot?> GetCurrentSnapshotAsync(string accountId)
    {
        await Task.Yield(); // Ensure async behavior for snapshot retrieval
        
        lock (_snapshotLock)
        {
            if (_snapshots.TryGetValue(accountId, out var snapshot))
            {
                // Check if snapshot is still fresh (less than 5 minutes old)
                if (DateTime.UtcNow - snapshot.Timestamp < TimeSpan.FromMinutes(5))
                {
                    _logger.LogDebug("[SNAPSHOT-MANAGER] Returning cached snapshot for account {AccountId}", MaskAccountId(accountId));
                    return snapshot;
                }
                else
                {
                    _logger.LogDebug("[SNAPSHOT-MANAGER] Cached snapshot for account {AccountId} is stale, triggering refresh", MaskAccountId(accountId));
                    // Trigger async refresh in background
                    _ = Task.Run(async () => await RefreshSnapshotAsync(accountId));
                    return snapshot; // Return stale snapshot while refresh is in progress
                }
            }
            
            _logger.LogDebug("[SNAPSHOT-MANAGER] No cached snapshot found for account {AccountId}", MaskAccountId(accountId));
            return null;
        }
    }

    private async Task<AccountSnapshot?> FetchAccountSnapshotAsync(string accountId)
    {
        try
        {
            // Fetch account details
            var accountResponse = await _httpClient.GetAsync($"/account/{accountId}");
            if (!accountResponse.IsSuccessStatusCode)
            {
                _logger.LogWarning("Failed to fetch account details: {StatusCode}", accountResponse.StatusCode);
                return null;
            }

            var accountJson = await accountResponse.Content.ReadAsStringAsync();
            var accountData = JsonSerializer.Deserialize<JsonElement>(accountJson);

            // Fetch positions
            var positionsResponse = await _httpClient.GetAsync($"/account/{accountId}/positions");
            var positions = new List<PositionSnapshot>();
            
            if (positionsResponse.IsSuccessStatusCode)
            {
                var positionsJson = await positionsResponse.Content.ReadAsStringAsync();
                var positionsData = JsonSerializer.Deserialize<JsonElement>(positionsJson);
                
                if (positionsData.TryGetProperty("positions", out var positionsArray))
                {
                    foreach (var pos in positionsArray.EnumerateArray())
                    {
                        positions.Add(new PositionSnapshot(
                            Symbol: pos.GetProperty("symbol").GetString() ?? "",
                            Quantity: pos.GetProperty("quantity").GetInt32(),
                            AveragePrice: pos.GetProperty("averagePrice").GetDecimal(),
                            MarketValue: pos.GetProperty("marketValue").GetDecimal(),
                            UnrealizedPnL: pos.GetProperty("unrealizedPnL").GetDecimal()
                        ));
                    }
                }
            }

            // Fetch orders
            var ordersResponse = await _httpClient.GetAsync($"/account/{accountId}/orders");
            var orders = new List<OrderSnapshot>();
            
            if (ordersResponse.IsSuccessStatusCode)
            {
                var ordersJson = await ordersResponse.Content.ReadAsStringAsync();
                var ordersData = JsonSerializer.Deserialize<JsonElement>(ordersJson);
                
                if (ordersData.TryGetProperty("orders", out var ordersArray))
                {
                    foreach (var order in ordersArray.EnumerateArray())
                    {
                        orders.Add(new OrderSnapshot(
                            OrderId: order.GetProperty("orderId").GetString() ?? "",
                            Symbol: order.GetProperty("symbol").GetString() ?? "",
                            Side: order.GetProperty("side").GetString() ?? "",
                            Quantity: order.GetProperty("quantity").GetInt32(),
                            Price: order.GetProperty("price").GetDecimal(),
                            Status: order.GetProperty("status").GetString() ?? "",
                            CreatedTime: order.GetProperty("createdTime").GetDateTime()
                        ));
                    }
                }
            }

            return new AccountSnapshot(
                AccountId: accountId,
                Timestamp: DateTime.UtcNow,
                Positions: positions,
                Orders: orders,
                Balance: accountData.TryGetProperty("balance", out var balance) ? balance.GetDecimal() : 0,
                Equity: accountData.TryGetProperty("equity", out var equity) ? equity.GetDecimal() : 0,
                DayPnL: accountData.TryGetProperty("dayPnL", out var dayPnL) ? dayPnL.GetDecimal() : 0
            );
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error fetching account snapshot");
            return null;
        }
    }

    private async Task DetectAndLogChangesAsync(AccountSnapshot previous, AccountSnapshot current)
    {
        try
        {
            var changes = new List<string>();

            // Check balance changes
            if (previous.Balance != current.Balance)
            {
                changes.Add($"Balance: {previous.Balance:C} → {current.Balance:C}");
            }

            if (previous.DayPnL != current.DayPnL)
            {
                changes.Add($"DayPnL: {previous.DayPnL:C} → {current.DayPnL:C}");
            }

            // Check position changes
            var previousPositions = previous.Positions.ToDictionary(p => p.Symbol, p => p);
            var currentPositions = current.Positions.ToDictionary(p => p.Symbol, p => p);

            foreach (var symbol in previousPositions.Keys.Union(currentPositions.Keys))
            {
                var hadPosition = previousPositions.TryGetValue(symbol, out var prevPos);
                var hasPosition = currentPositions.TryGetValue(symbol, out var currPos);

                if (!hadPosition && hasPosition)
                {
                    changes.Add($"New position: {currPos!.Symbol} {currPos.Quantity} shares @ {currPos.AveragePrice:C}");
                }
                else if (hadPosition && !hasPosition)
                {
                    changes.Add($"Closed position: {prevPos!.Symbol}");
                }
                else if (hadPosition && hasPosition && (prevPos!.Quantity != currPos!.Quantity || prevPos.AveragePrice != currPos.AveragePrice))
                {
                    changes.Add($"Position changed: {symbol} {prevPos.Quantity} → {currPos.Quantity} shares");
                }
            }

            // Check order changes
            var previousOrders = previous.Orders.ToDictionary(o => o.OrderId, o => o);
            var currentOrders = current.Orders.ToDictionary(o => o.OrderId, o => o);

            foreach (var orderId in previousOrders.Keys.Union(currentOrders.Keys))
            {
                var hadOrder = previousOrders.TryGetValue(orderId, out var prevOrder);
                var hasOrder = currentOrders.TryGetValue(orderId, out var currOrder);

                if (!hadOrder && hasOrder)
                {
                    changes.Add($"New order: {currOrder!.OrderId} {currOrder.Side} {currOrder.Symbol}");
                }
                else if (hadOrder && !hasOrder)
                {
                    changes.Add($"Order removed: {prevOrder!.OrderId}");
                }
                else if (hadOrder && hasOrder && prevOrder!.Status != currOrder!.Status)
                {
                    changes.Add($"Order status changed: {orderId} {prevOrder.Status} → {currOrder.Status}");
                }
            }

            if (changes.Any())
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SnapshotManager",
                    $"Detected {changes.Count} changes during reconnection: {string.Join("; ", changes)}");
            }
            else
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SnapshotManager",
                    "No state changes detected during reconnection");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error detecting changes between snapshots");
        }
    }

    /// <summary>
    /// Refresh snapshot for specific account
    /// </summary>
    private async Task RefreshSnapshotAsync(string accountId)
    {
        try
        {
            _logger.LogDebug("[SNAPSHOT-MANAGER] Refreshing snapshot for account {AccountId}", MaskAccountId(accountId));
            
            var newSnapshot = await FetchAccountSnapshotAsync(accountId);
            if (newSnapshot != null)
            {
                AccountSnapshot? previousSnapshot = null;
                lock (_snapshotLock)
                {
                    _snapshots.TryGetValue(accountId, out previousSnapshot);
                    _snapshots[accountId] = newSnapshot;
                }

                if (previousSnapshot != null)
                {
                    await DetectAndLogChangesAsync(previousSnapshot, newSnapshot);
                }

                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SnapshotManager",
                    $"Successfully refreshed snapshot for account {MaskAccountId(accountId)}");
            }
            else
            {
                _logger.LogWarning("[SNAPSHOT-MANAGER] Failed to fetch updated snapshot for account {AccountId}", MaskAccountId(accountId));
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[SNAPSHOT-MANAGER] Error refreshing snapshot for account {AccountId}", MaskAccountId(accountId));
        }
    }

    /// <summary>
    /// Mask account ID for logging
    /// </summary>
    private static string MaskAccountId(string accountId)
    {
        if (string.IsNullOrEmpty(accountId) || accountId.Length <= 4)
            return "****";
        
        return accountId.Substring(0, 2) + "****" + accountId.Substring(accountId.Length - 2);
    }
}