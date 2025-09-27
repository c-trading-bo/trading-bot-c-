using System;
using System.Collections.Generic;
using System.Linq;

namespace BotCore.Models
{
    /// <summary>
    /// Trading position tracking record
    /// </summary>
    public class Position
    {
        private readonly List<Fill> _fills = new();
        
        public string Symbol { get; set; } = string.Empty;
        public int NetQuantity { get; set; }
        public decimal AveragePrice { get; set; }
        public decimal UnrealizedPnL { get; set; }
        public decimal RealizedPnL { get; set; }
        public DateTime LastUpdate { get; set; }
        public IReadOnlyList<Fill> Fills => _fills;
        public decimal MarketValue { get; set; }
        public decimal DailyPnL { get; set; }
        
        public void ReplaceFills(IEnumerable<Fill> fills)
        {
            _fills.Clear();
            if (fills != null) _fills.AddRange(fills);
        }
        
        public void AddFill(Fill fill)
        {
            if (fill != null) _fills.Add(fill);
        }
    }
}