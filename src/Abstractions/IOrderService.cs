using System.Threading.Tasks;
using System.Collections.Generic;

namespace TradingBot.Abstractions
{
    /// <summary>
    /// Interface for order service with full production order lifecycle management
    /// </summary>
    public interface IOrderService
    {
        Task<bool> IsHealthyAsync();
        Task<string> GetStatusAsync();
        
        // Order placement methods
        Task<string> PlaceMarketOrderAsync(string symbol, string side, int quantity, string? tag = null);
        Task<string> PlaceLimitOrderAsync(string symbol, string side, int quantity, decimal price, string? tag = null);
        Task<string> PlaceStopOrderAsync(string symbol, string side, int quantity, decimal stopPrice, string? tag = null);
        
        // Order management methods
        Task<bool> CancelOrderAsync(string orderId);
        Task<bool> ModifyOrderAsync(string orderId, int? quantity = null, decimal? price = null);
        Task<OrderStatus> GetOrderStatusAsync(string orderId);
        
        // Position management methods
        Task<bool> ClosePositionAsync(string positionId);
        Task<bool> ModifyStopLossAsync(string positionId, decimal stopPrice);
        Task<bool> ModifyTakeProfitAsync(string positionId, decimal takeProfitPrice);
        
        // Account and position queries
        Task<List<Position>> GetPositionsAsync();
        Task<Position?> GetPositionAsync(string positionId);
        Task<List<Order>> GetActiveOrdersAsync();
    }

    /// <summary>
    /// Order status enumeration
    /// </summary>
    public enum OrderStatus
    {
        Pending,
        PartiallyFilled,
        Filled,
        Cancelled,
        Rejected,
        Expired
    }

    /// <summary>
    /// Position model for position management
    /// </summary>
    public class Position
    {
        public required string Id { get; set; }
        public required string Symbol { get; set; }
        public required string Side { get; set; }
        public int Quantity { get; set; }
        public decimal AveragePrice { get; set; }
        public decimal UnrealizedPnL { get; set; }
        public decimal RealizedPnL { get; set; }
        public required string ConfigSnapshotId { get; set; }
        public System.DateTimeOffset OpenTime { get; set; }
        public decimal? StopLoss { get; set; }
        public decimal? TakeProfit { get; set; }
    }

    /// <summary>
    /// Order model for order management
    /// </summary>
    public class Order
    {
        public required string Id { get; set; }
        public required string Symbol { get; set; }
        public required string Side { get; set; }
        public int Quantity { get; set; }
        public int FilledQuantity { get; set; }
        public decimal? Price { get; set; }
        public decimal? StopPrice { get; set; }
        public required string OrderType { get; set; }
        public OrderStatus Status { get; set; }
        public string? Tag { get; set; }
        public required string ConfigSnapshotId { get; set; }
        public System.DateTimeOffset CreatedAt { get; set; }
        public System.DateTimeOffset? UpdatedAt { get; set; }
    }
}