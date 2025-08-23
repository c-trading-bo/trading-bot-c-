namespace BotCore.Models
{
    public sealed class Env {
        public string Symbol { get; set; } = "";
        public decimal? atr { get; set; }
        public decimal? volz { get; set; }
    }
}
