using System;
using System.Collections.Generic;
using System.Linq;

namespace BotCore.Models
{
    /// <summary>
    /// Order tracking record for fill confirmation system
    /// </summary>
    public class OrderTrackingRecord
    {
        private readonly List<FillConfirmation> _fills = new();
        
        public string ClientOrderId { get; set; } = string.Empty;
        public string? GatewayOrderId { get; set; }
        public string Symbol { get; set; } = string.Empty;
        public int Quantity { get; set; }
        public decimal Price { get; set; }
        public string Side { get; set; } = string.Empty;
        public string OrderType { get; set; } = string.Empty;
        public DateTime SubmittedTime { get; set; }
        public string Status { get; set; } = "PENDING";
        public IReadOnlyList<FillConfirmation> Fills => _fills;
        public string? RejectReason { get; set; }
        public bool IsVerified { get; set; }
        public int VerificationAttempts { get; set; }
        
        public void ReplaceFills(IEnumerable<FillConfirmation> fills)
        {
            _fills.Clear();
            if (fills != null) _fills.AddRange(fills);
        }
        
        public void AddFill(FillConfirmation fill)
        {
            if (fill != null) _fills.Add(fill);
        }
    }
}