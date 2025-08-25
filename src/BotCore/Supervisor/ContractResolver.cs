#nullable enable
namespace BotCore.Supervisor
{
    /// <summary>
    /// Minimal contract rollover helper. In production, replace with real volume/OI checks via API.
    /// Env controls (per symbol):
    ///  - TOPSTEPX_EXPIRING_{SYM}=1  => IsExpiring=true
    ///  - TOPSTEPX_SHOULD_ROLL_{SYM}=1 => ShouldRoll=true
    /// </summary>
    public sealed class ContractResolver
    {
        public bool IsExpiring(string symbol)
        {
            var v = Environment.GetEnvironmentVariable($"TOPSTEPX_EXPIRING_{symbol.ToUpperInvariant()}");
            return v == "1" || string.Equals(v, "true", StringComparison.OrdinalIgnoreCase);
        }
        public bool ShouldRoll(string symbol)
        {
            var v = Environment.GetEnvironmentVariable($"TOPSTEPX_SHOULD_ROLL_{symbol.ToUpperInvariant()}");
            return v == "1" || string.Equals(v, "true", StringComparison.OrdinalIgnoreCase);
        }
    }
}
