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
            // Example: ES = 0.25, NQ = 0.25
            return 0.25m;
        }
    }
}
