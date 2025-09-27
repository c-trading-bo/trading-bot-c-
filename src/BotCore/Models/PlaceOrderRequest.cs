namespace BotCore.Models
{
    /// <summary>
    /// Place order request for order fill confirmation system and trading integration
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
        
        // Additional properties for trading system integration
        public decimal StopPrice { get; set; }
        public decimal TargetPrice { get; set; }
        public string CustomTag { get; set; } = string.Empty;
        public string AccountId { get; set; } = string.Empty;
    }
}