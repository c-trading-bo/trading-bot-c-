#nullable enable
using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;
using System.Text.Json;
using System.Linq;

namespace SupervisorAgent
{
    public sealed class StatusService
    {
        private readonly ILogger<StatusService> _log;
        private readonly ConcurrentDictionary<string, object> _vals = new();
        private DateTimeOffset _lastBeat = DateTimeOffset.MinValue;
        private DateTimeOffset _lastEmit = DateTimeOffset.MinValue;
        private string _lastJson = string.Empty;
        private string _lastSig = string.Empty;

        public long AccountId { get; set; }
        public Dictionary<string,string> Contracts { get; set; } = new();

        public StatusService(ILogger<StatusService> log) => _log = log;

        public void Set(string key, object value) => _vals[key] = value;
        public T? Get<T>(string key) => _vals.TryGetValue(key, out var v) ? (T?)v : default;

        private static bool Concise() => (Environment.GetEnvironmentVariable("APP_CONCISE_CONSOLE") ?? "true").Trim().ToLowerInvariant() is "1" or "true" or "yes";

        public void Heartbeat()
        {
            var now = DateTimeOffset.UtcNow;
            // Base throttle to avoid hot loops
            if (!Concise())
            {
                if (now - _lastBeat < TimeSpan.FromSeconds(5)) return; // throttle
            }
            _lastBeat = now;

            var snapshot = new
            {
                whenUtc = now,
                accountId = AccountId,
                contracts = Contracts,
                userHub = Get<string>("user.state"),
                marketHub = Get<string>("market.state"),
                lastTrade = Get<DateTimeOffset?>("last.trade"),
                lastQuote = Get<DateTimeOffset?>("last.quote"),
                strategies = Get<object?>("strategies"),
                openOrders = Get<object?>("orders.open"),
                risk = Get<object?>("risk.state")
            };
            var json = JsonSerializer.Serialize(snapshot);

            // Stable signature excludes whenUtc so we don't emit just because time advanced
            // Deterministic ordering for contracts in signature to avoid spurious emits
            var contractsSig = Contracts is null
                ? new Dictionary<string, string>()
                : Contracts.OrderBy(kv => kv.Key, StringComparer.OrdinalIgnoreCase).ToDictionary(kv => kv.Key, kv => kv.Value);
            var sigObj = new
            {
                accountId = AccountId,
                contracts = contractsSig,
                userHub = Get<string>("user.state"),
                marketHub = Get<string>("market.state"),
                lastTrade = Get<DateTimeOffset?>("last.trade"),
                lastQuote = Get<DateTimeOffset?>("last.quote"),
                strategies = Get<object?>("strategies"),
                openOrders = Get<object?>("orders.open"),
                risk = Get<object?>("risk.state")
            };
            var sig = JsonSerializer.Serialize(sigObj);

            if (Concise())
            {
                // Emit only when state signature changed or every 60s
                if (sig != _lastSig || (now - _lastEmit) >= TimeSpan.FromSeconds(60))
                {
                    _lastSig = sig;
                    _lastJson = json;
                    _lastEmit = now;
                    _log.LogInformation("BOT STATUS => {Json}", json);
                }
                return;
            }

            _log.LogInformation("BOT STATUS => {Json}", json);
        }
    }
}
