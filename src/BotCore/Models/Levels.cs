namespace BotCore.Models
{
    public sealed class Levels 
    { 
        public decimal Support1 { get; set; }
        public decimal Support2 { get; set; }
        public decimal Support3 { get; set; }
        public decimal Resistance1 { get; set; }
        public decimal Resistance2 { get; set; }
        public decimal Resistance3 { get; set; }
        public decimal VWAP { get; set; }
        public decimal DailyPivot { get; set; }
        public decimal WeeklyPivot { get; set; }
        public decimal MonthlyPivot { get; set; }
        public DateTime CalculatedAt { get; set; } = DateTime.UtcNow;
        
        public Levels()
        {
            // Initialize with default values
            Support1 = Support2 = Support3;
            Resistance1 = Resistance2 = Resistance3;
            VWAP = DailyPivot = WeeklyPivot = MonthlyPivot;
        }
        
        public bool IsValidLevel(decimal price)
        {
            // Check if price is near any significant level (within 0.1%)
            var tolerance = 0.001m;
            var levels = new[] { Support1, Support2, Support3, Resistance1, Resistance2, Resistance3, VWAP, DailyPivot };
            
            return levels.Any(level => level > 0 && Math.Abs(price - level) / level <= tolerance);
        }
        
        public string GetNearestLevel(decimal price)
        {
            var levels = new Dictionary<string, decimal>
            {
                ["Support1"] = Support1,
                ["Support2"] = Support2,
                ["Support3"] = Support3,
                ["Resistance1"] = Resistance1,
                ["Resistance2"] = Resistance2,
                ["Resistance3"] = Resistance3,
                ["VWAP"] = VWAP,
                ["DailyPivot"] = DailyPivot
            };
            
            var nearestLevel = levels
                .Where(kvp => kvp.Value > 0)
                .OrderBy(kvp => Math.Abs(price - kvp.Value))
                .FirstOrDefault();
                
            return nearestLevel.Key ?? "None";
        }
    }
}
