using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TopstepX.Bot.Core.Services
{
    /// <summary>
    /// Order Fill Confirmation System - Ensures no trades without proof
    /// Integrates with TopstepX API for order verification
    /// </summary>
    public class OrderFillConfirmationSystem
    {
        private readonly ILogger<OrderFillConfirmationSystem> _logger;
        private readonly HttpClient _httpClient;
        private readonly HubConnection _userHubConnection;
        private readonly HubConnection _marketHubConnection;
        private readonly ConcurrentDictionary<string, OrderTrackingRecord> _orderTracking = new();
        private readonly Timer _verificationTimer;
        private readonly PositionTrackingSystem _positionTracker;
        private readonly EmergencyStopSystem _emergencyStop;
        
        public event EventHandler<OrderConfirmedEventArgs>? OrderConfirmed;
        public event EventHandler<OrderRejectedEventArgs>? OrderRejected;
        public event EventHandler<FillConfirmedEventArgs>? FillConfirmed;
        
        public class OrderTrackingRecord
        {
            public string ClientOrderId { get; set; } = string.Empty;
            public string? GatewayOrderId { get; set; }
            public string Symbol { get; set; } = string.Empty;
            public int Quantity { get; set; }
            public decimal Price { get; set; }
            public string Side { get; set; } = string.Empty;
            public string OrderType { get; set; } = string.Empty;
            public DateTime SubmittedTime { get; set; }
            public string Status { get; set; } = "PENDING";
            public List<FillConfirmation> Fills { get; set; } = new();
            public string? RejectReason { get; set; }
            public bool IsVerified { get; set; }
            public int VerificationAttempts { get; set; }
        }
        
        public class FillConfirmation
        {
            public string FillId { get; set; } = string.Empty;
            public DateTime FillTime { get; set; }
            public decimal FillPrice { get; set; }
            public int FillQuantity { get; set; }
            public decimal Commission { get; set; }
            public string Exchange { get; set; } = string.Empty;
            public bool IsVerified { get; set; }
        }
        
        public class PlaceOrderRequest
        {
            public string Symbol { get; set; } = string.Empty;
            public int Quantity { get; set; }
            public decimal Price { get; set; }
            public string Side { get; set; } = string.Empty; // BUY/SELL
            public string OrderType { get; set; } = "LIMIT";
            public string TimeInForce { get; set; } = "DAY";
            public string ClientOrderId { get; set; } = string.Empty;
        }
        
        public OrderFillConfirmationSystem(
            ILogger<OrderFillConfirmationSystem> logger,
            HttpClient httpClient,
            HubConnection userHubConnection,
            HubConnection marketHubConnection,
            PositionTrackingSystem positionTracker,
            EmergencyStopSystem emergencyStop)
        {
            _logger = logger;
            _httpClient = httpClient;
            _userHubConnection = userHubConnection;
            _marketHubConnection = marketHubConnection;
            _positionTracker = positionTracker;
            _emergencyStop = emergencyStop;
            
            // Setup verification timer - runs every 10 seconds
            _verificationTimer = new Timer(VerifyPendingOrders, null, TimeSpan.FromSeconds(10), TimeSpan.FromSeconds(10));
            
            SetupSignalRListeners();
            _logger.LogInformation("📋 Order Fill Confirmation System initialized");
        }
        
        private void SetupSignalRListeners()
        {
            try
            {
                // Listen for order status updates from User Hub
                _userHubConnection.On<GatewayUserOrder>("OrderUpdate", async (orderUpdate) =>
                {
                    await ProcessOrderUpdateAsync(orderUpdate);
                });
                
                // Listen for trade fills from User Hub
                _userHubConnection.On<GatewayUserTrade>("TradeUpdate", async (tradeUpdate) =>
                {
                    await ProcessTradeUpdateAsync(tradeUpdate);
                });
                
                _logger.LogInformation("📡 SignalR listeners configured for order/trade updates");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Failed to setup SignalR listeners");
            }
        }
        
        /// <summary>
        /// Place order with full tracking and verification
        /// </summary>
        public async Task<OrderResult> PlaceOrderAsync(PlaceOrderRequest request, string accountId)
        {
            // Check emergency stop
            if (_emergencyStop.IsEmergencyStop)
            {
                return OrderResult.Failed("Emergency stop is active - no trading allowed");
            }
            
            try
            {
                // Generate unique client order ID if not provided
                if (string.IsNullOrEmpty(request.ClientOrderId))
                {
                    request.ClientOrderId = $"S11L-{DateTime.UtcNow:yyyyMMdd-HHmmss}-{Guid.NewGuid().ToString("N")[..6]}";
                }
                
                // Create tracking record
                var trackingRecord = new OrderTrackingRecord
                {
                    ClientOrderId = request.ClientOrderId,
                    Symbol = request.Symbol,
                    Quantity = request.Quantity,
                    Price = request.Price,
                    Side = request.Side,
                    OrderType = request.OrderType,
                    SubmittedTime = DateTime.UtcNow,
                    Status = "SUBMITTING"
                };
                
                _orderTracking[request.ClientOrderId] = trackingRecord;
                
                // Log order submission
                _logger.LogInformation("[{ClientOrderId}] side={Side} symbol={Symbol} qty={Quantity} price={Price:F2} type={OrderType}",
                    request.ClientOrderId, request.Side, request.Symbol, request.Quantity, request.Price, request.OrderType);
                
                // Submit order to TopstepX API
                var orderResponse = await SubmitOrderToApiAsync(request, accountId);
                
                if (orderResponse.IsSuccess)
                {
                    trackingRecord.GatewayOrderId = orderResponse.OrderId;
                    trackingRecord.Status = "SUBMITTED";
                    
                    // Add to position tracker as pending
                    _positionTracker.AddPendingOrder(new PositionTrackingSystem.PendingOrder
                    {
                        OrderId = orderResponse.OrderId ?? string.Empty,
                        ClientOrderId = request.ClientOrderId,
                        Symbol = request.Symbol,
                        Quantity = request.Side == "BUY" ? request.Quantity : -request.Quantity,
                        Price = request.Price,
                        Side = request.Side,
                        Status = "PENDING",
                        SubmittedTime = DateTime.UtcNow,
                        OrderType = request.OrderType
                    });
                    
                    _logger.LogInformation("✅ ORDER SUBMITTED: account={AccountId} clientOrderId={ClientOrderId} gatewayOrderId={GatewayOrderId}",
                        accountId, request.ClientOrderId, orderResponse.OrderId);
                    
                    return OrderResult.Success(orderResponse.OrderId, request.ClientOrderId);
                }
                else
                {
                    trackingRecord.Status = "REJECTED";
                    trackingRecord.RejectReason = orderResponse.ErrorMessage;
                    
                    _logger.LogWarning("❌ ORDER REJECTED: account={AccountId} clientOrderId={ClientOrderId} reason={Reason}",
                        accountId, request.ClientOrderId, orderResponse.ErrorMessage);
                    
                    return OrderResult.Failed(orderResponse.ErrorMessage ?? "Unknown error");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error placing order {ClientOrderId}", request.ClientOrderId);
                return OrderResult.Failed($"Exception: {ex.Message}");
            }
        }
        
        private async Task<ApiOrderResponse> SubmitOrderToApiAsync(PlaceOrderRequest request, string accountId)
        {
            try
            {
                var orderPayload = new
                {
                    accountId = accountId,
                    symbol = request.Symbol,
                    quantity = request.Quantity,
                    price = request.Price,
                    side = request.Side,
                    orderType = request.OrderType,
                    timeInForce = request.TimeInForce,
                    clientOrderId = request.ClientOrderId
                };
                
                var json = JsonSerializer.Serialize(orderPayload);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                
                var response = await _httpClient.PostAsync("/api/orders", content);
                
                if (response.IsSuccessStatusCode)
                {
                    var responseContent = await response.Content.ReadAsStringAsync();
                    var orderResponse = JsonSerializer.Deserialize<ApiOrderResponse>(responseContent);
                    return orderResponse ?? ApiOrderResponse.Failed("Failed to parse response");
                }
                else
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    _logger.LogWarning("API order rejection: {StatusCode} - {Content}", response.StatusCode, errorContent);
                    return ApiOrderResponse.Failed($"HTTP {response.StatusCode}: {errorContent}");
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "API call failed for order submission");
                return ApiOrderResponse.Failed($"API Exception: {ex.Message}");
            }
        }
        
        private async Task ProcessOrderUpdateAsync(GatewayUserOrder orderUpdate)
        {
            try
            {
                // Find tracking record by gateway order ID
                var trackingRecord = _orderTracking.Values.FirstOrDefault(r => r.GatewayOrderId == orderUpdate.OrderId);
                
                if (trackingRecord != null)
                {
                    trackingRecord.Status = orderUpdate.Status;
                    trackingRecord.IsVerified = true;
                    
                    _logger.LogInformation("ORDER UPDATE: account={AccountId} status={Status} orderId={OrderId} reason={Reason}",
                        SecurityHelpers.MaskAccountId(orderUpdate.AccountId), orderUpdate.Status, SecurityHelpers.MaskOrderId(orderUpdate.OrderId), orderUpdate.Reason ?? "N/A");
                    
                    if (orderUpdate.Status == "FILLED" || orderUpdate.Status == "PARTIALLY_FILLED")
                    {
                        OrderConfirmed?.Invoke(this, new OrderConfirmedEventArgs
                        {
                            TrackingRecord = trackingRecord,
                            GatewayOrderUpdate = orderUpdate
                        });
                    }
                    else if (orderUpdate.Status == "REJECTED" || orderUpdate.Status == "CANCELLED")
                    {
                        trackingRecord.RejectReason = orderUpdate.Reason;
                        OrderRejected?.Invoke(this, new OrderRejectedEventArgs
                        {
                            TrackingRecord = trackingRecord,
                            GatewayOrderUpdate = orderUpdate
                        });
                    }
                }
                else
                {
                    _logger.LogWarning("⚠️ Received order update for unknown order: {OrderId}", orderUpdate.OrderId);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error processing order update");
            }
        }
        
        private async Task ProcessTradeUpdateAsync(GatewayUserTrade tradeUpdate)
        {
            try
            {
                // Find tracking record by gateway order ID
                var trackingRecord = _orderTracking.Values.FirstOrDefault(r => r.GatewayOrderId == tradeUpdate.OrderId);
                
                if (trackingRecord != null)
                {
                    var fillConfirmation = new FillConfirmation
                    {
                        FillId = Guid.NewGuid().ToString(),
                        FillTime = DateTime.UtcNow,
                        FillPrice = tradeUpdate.FillPrice,
                        FillQuantity = trackingRecord.Side == "BUY" ? tradeUpdate.Quantity : -tradeUpdate.Quantity,
                        Commission = tradeUpdate.Commission,
                        Exchange = tradeUpdate.Exchange ?? "TOPSTEPX",
                        IsVerified = true
                    };
                    
                    trackingRecord.Fills.Add(fillConfirmation);
                    
                    _logger.LogInformation("TRADE CONFIRMED: account={AccountId} orderId={OrderId} fillPrice={FillPrice:F2} qty={Quantity} time={Time:yyyy-MM-dd HH:mm:ss}",
                        SecurityHelpers.MaskAccountId(tradeUpdate.AccountId), SecurityHelpers.MaskOrderId(tradeUpdate.OrderId), tradeUpdate.FillPrice, tradeUpdate.Quantity, DateTime.UtcNow);
                    
                    // Update position tracker
                    await _positionTracker.ProcessFillAsync(
                        tradeUpdate.OrderId ?? string.Empty,
                        trackingRecord.Symbol,
                        tradeUpdate.FillPrice,
                        fillConfirmation.FillQuantity,
                        tradeUpdate.Commission);
                    
                    FillConfirmed?.Invoke(this, new FillConfirmedEventArgs
                    {
                        TrackingRecord = trackingRecord,
                        FillConfirmation = fillConfirmation,
                        GatewayTradeUpdate = tradeUpdate
                    });
                }
                else
                {
                    _logger.LogWarning("⚠️ Received trade update for unknown order: {OrderId}", tradeUpdate.OrderId);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error processing trade update");
            }
        }
        
        private void VerifyPendingOrders(object? state)
        {
            try
            {
                var pendingOrders = _orderTracking.Values.Where(r => 
                    r.Status == "SUBMITTED" && 
                    !r.IsVerified && 
                    r.VerificationAttempts < 5 &&
                    DateTime.UtcNow - r.SubmittedTime < TimeSpan.FromMinutes(10)).ToList();
                
                foreach (var order in pendingOrders)
                {
                    order.VerificationAttempts++;
                    _ = Task.Run(async () => await VerifyOrderWithApiAsync(order));
                }
                
                // Clean up old tracking records (older than 1 hour)
                var cutoffTime = DateTime.UtcNow.AddHours(-1);
                var staleRecords = _orderTracking.Values.Where(r => r.SubmittedTime < cutoffTime).ToList();
                
                foreach (var staleRecord in staleRecords)
                {
                    _orderTracking.TryRemove(staleRecord.ClientOrderId, out _);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error during order verification");
            }
        }
        
        private async Task VerifyOrderWithApiAsync(OrderTrackingRecord order)
        {
            try
            {
                if (string.IsNullOrEmpty(order.GatewayOrderId)) return;
                
                var response = await _httpClient.GetAsync($"/api/orders/{order.GatewayOrderId}");
                
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var orderDetails = JsonSerializer.Deserialize<ApiOrderDetails>(content);
                    
                    if (orderDetails != null)
                    {
                        order.Status = orderDetails.Status;
                        order.IsVerified = true;
                        
                        _logger.LogDebug("✅ Order verified via API: {OrderId} status={Status}", 
                            order.GatewayOrderId, orderDetails.Status);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "⚠️ Failed to verify order {OrderId} via API", order.GatewayOrderId);
            }
        }
        
        /// <summary>
        /// Cancel order by client order ID
        /// </summary>
        public async Task<bool> CancelOrderAsync(string clientOrderId, string accountId)
        {
            try
            {
                if (_orderTracking.TryGetValue(clientOrderId, out var trackingRecord) && 
                    !string.IsNullOrEmpty(trackingRecord.GatewayOrderId))
                {
                    var response = await _httpClient.DeleteAsync($"/api/orders/{trackingRecord.GatewayOrderId}?accountId={accountId}");
                    
                    if (response.IsSuccessStatusCode)
                    {
                        trackingRecord.Status = "CANCEL_PENDING";
                        _logger.LogInformation("📝 Cancel request sent for order {ClientOrderId}", clientOrderId);
                        return true;
                    }
                }
                
                return false;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ Error cancelling order {ClientOrderId}", clientOrderId);
                return false;
            }
        }
        
        /// <summary>
        /// Get order status by client order ID
        /// </summary>
        public OrderTrackingRecord? GetOrderStatus(string clientOrderId)
        {
            _orderTracking.TryGetValue(clientOrderId, out var record);
            return record;
        }
        
        /// <summary>
        /// Get all orders with their current status
        /// </summary>
        public Dictionary<string, OrderTrackingRecord> GetAllOrders()
        {
            return new Dictionary<string, OrderTrackingRecord>(_orderTracking);
        }
        
        public void Dispose()
        {
            _verificationTimer?.Dispose();
        }
    }
    
    // Supporting classes for API integration
    public class GatewayUserOrder
    {
        public string? AccountId { get; set; }
        public string? OrderId { get; set; }
        public string Status { get; set; } = string.Empty;
        public string? Reason { get; set; }
    }
    
    public class GatewayUserTrade
    {
        public string? AccountId { get; set; }
        public string? OrderId { get; set; }
        public decimal FillPrice { get; set; }
        public int Quantity { get; set; }
        public decimal Commission { get; set; }
        public string? Exchange { get; set; }
    }
    
    public class ApiOrderResponse
    {
        public bool IsSuccess { get; set; }
        public string? OrderId { get; set; }
        public string? ErrorMessage { get; set; }
        
        public static ApiOrderResponse Success(string orderId) => new() { IsSuccess = true, OrderId = orderId };
        public static ApiOrderResponse Failed(string error) => new() { IsSuccess = false, ErrorMessage = error };
    }
    
    public class ApiOrderDetails
    {
        public string OrderId { get; set; } = string.Empty;
        public string Status { get; set; } = string.Empty;
        public string Symbol { get; set; } = string.Empty;
        public int Quantity { get; set; }
        public decimal Price { get; set; }
    }
    
    public class OrderResult
    {
        public bool IsSuccess { get; set; }
        public string? OrderId { get; set; }
        public string? ClientOrderId { get; set; }
        public string? ErrorMessage { get; set; }
        
        public static OrderResult Success(string? orderId, string clientOrderId) => 
            new() { IsSuccess = true, OrderId = orderId, ClientOrderId = clientOrderId };
        public static OrderResult Failed(string error) => 
            new() { IsSuccess = false, ErrorMessage = error };
    }
    
    // Event argument classes
    public class OrderConfirmedEventArgs : EventArgs
    {
        public OrderFillConfirmationSystem.OrderTrackingRecord TrackingRecord { get; set; } = new();
        public GatewayUserOrder GatewayOrderUpdate { get; set; } = new();
    }
    
    public class OrderRejectedEventArgs : EventArgs
    {
        public OrderFillConfirmationSystem.OrderTrackingRecord TrackingRecord { get; set; } = new();
        public GatewayUserOrder GatewayOrderUpdate { get; set; } = new();
    }
    
    public class FillConfirmedEventArgs : EventArgs
    {
        public OrderFillConfirmationSystem.OrderTrackingRecord TrackingRecord { get; set; } = new();
        public OrderFillConfirmationSystem.FillConfirmation FillConfirmation { get; set; } = new();
        public GatewayUserTrade GatewayTradeUpdate { get; set; } = new();
    }
}