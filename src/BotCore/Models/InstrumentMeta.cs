namespace BotCore.Models
{
    public static class InstrumentMeta
    {
        // Big point value (USD per 1.00 price move per contract)
        public static decimal PointValue(string symbol)
        {
            return symbol.Equals("ES", System.StringComparison.OrdinalIgnoreCase) ? 50m
                 : symbol.Equals("NQ", System.StringComparison.OrdinalIgnoreCase) ? 20m
                 : 1m;
        }
        // Alias often used by other layers
        public static decimal BigPointValue(string symbol) => PointValue(symbol);

        // Minimum price increment
        public static decimal Tick(string symbol)
        {
            // ES and NQ commonly 0.25; adapt if your contract specs differ
            return 0.25m;
        }
        // Display decimals used for formatting
        public static int Decimals(string symbol)
        {
            // ES/NQ two decimals in most UIs
            return 2;
        }
        // Contract lot step
        public static int LotStep(string symbol)
        {
            return 1;
        }
        public static decimal RoundToTick(string symbol, decimal price)
        {
            var t = Tick(symbol);
            if (t <= 0) t = 0.25m;
            return Math.Round(price / t, 0, MidpointRounding.AwayFromZero) * t;
        }
        // Attempt to derive root (ES/NQ) from various symbol forms incl. contractId-like strings
        public static string RootFromName(string name)
        {
            if (string.IsNullOrWhiteSpace(name)) return string.Empty;
            var n = name.ToUpperInvariant();
            if (n.Contains(".ENQ.") || n.StartsWith("NQ") || n.Contains("MNQ")) return "NQ";
            if (n.Contains(".EP.") || n.StartsWith("ES") || n.Contains("MES")) return "ES";
            // default fallback: if already short root, return as-is
            if (n == "ES" || n == "NQ") return n;
            return n;
        }
    }
}
