using System;

namespace BotCore.Models
{
    /// <summary>
    /// Pending order tracking record
    /// </summary>
    public class PendingOrder
    {
        public string OrderId { get; set; } = string.Empty;
        public string ClientOrderId { get; set; } = string.Empty;
        public string Symbol { get; set; } = string.Empty;
        public int Quantity { get; set; }
        public decimal Price { get; set; }
        public string Side { get; set; } = string.Empty;
        public string Status { get; set; } = string.Empty;
        public DateTime SubmittedTime { get; set; }
        public string OrderType { get; set; } = string.Empty;
    }
}