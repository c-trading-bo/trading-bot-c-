namespace BotCore.Models
{
    /// <summary>
    /// Place order request for order fill confirmation system
    /// </summary>
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
}