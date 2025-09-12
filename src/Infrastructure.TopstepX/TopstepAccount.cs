namespace TradingBot.Infrastructure.TopstepX;

public class TopstepAccount
{
    public long Id { get; set; }
    public string? Name { get; set; }
    public string? Type { get; set; }
    public bool CanTrade { get; set; }
    public decimal Balance { get; set; }
}
