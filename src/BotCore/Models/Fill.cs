using System;

namespace BotCore.Models
{
    /// <summary>
    /// Trade fill record for position tracking
    /// </summary>
    public class Fill
    {
        public string FillId { get; set; } = string.Empty;
        public string OrderId { get; set; } = string.Empty;
        public DateTime Time { get; set; }
        public decimal Price { get; set; }
        public int Quantity { get; set; }
        public decimal Commission { get; set; }
    }
}