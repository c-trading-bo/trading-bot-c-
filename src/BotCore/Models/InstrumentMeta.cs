namespace BotCore.Models
{
    public static class InstrumentMeta
    {
        public static decimal PointValue(string symbol)
        {
            // Example: ES = 50, NQ = 20
            return symbol == "ES" ? 50m : symbol == "NQ" ? 20m : 1m;
        }
        public static decimal Tick(string symbol)
        {
            // Example: ES = 0.25, NQ = 0.25 (adapt for other products)
            return 0.25m;
        }
        public static int LotStep(string symbol)
        {
            // Futures micros/minis typically step in 1 contract; adapt for FX/CFD if needed
            return 1;
        }
        public static decimal RoundToTick(string symbol, decimal price)
        {
            var t = Tick(symbol);
            if (t <= 0) t = 0.25m;
            return Math.Round(price / t, 0, MidpointRounding.AwayFromZero) * t;
        }
    }
}
