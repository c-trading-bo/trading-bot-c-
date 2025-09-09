using System;

namespace BotCore.Models
{
    /// <summary>
    /// Order confirmation event data from UserHub
    /// </summary>
    public class OrderConfirmation
    {
        public string OrderId { get; set; } = "";
        public string CustomTag { get; set; } = "";
        public string Status { get; set; } = "";
        public string Reason { get; set; } = "";
        public DateTime Timestamp { get; set; }
    }

    /// <summary>
    /// Fill confirmation event data from UserHub
    /// </summary>
    public class FillConfirmation
    {
        public string OrderId { get; set; } = "";
        public string CustomTag { get; set; } = "";
        public decimal FillPrice { get; set; }
        public int Quantity { get; set; }
        public DateTime Timestamp { get; set; }
    }
}