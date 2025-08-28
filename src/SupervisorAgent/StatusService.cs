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

        // Snapshot the current status values; when prefix is provided, only keys with that prefix are returned.
        public Dictionary<string, object> Snapshot(string? prefix = null)
        {
            if (string.IsNullOrWhiteSpace(prefix)) return _vals.ToDictionary(kv => kv.Key, kv => kv.Value);
            var p = prefix!;
            return _vals.Where(kv => kv.Key.StartsWith(p, StringComparison.OrdinalIgnoreCase))
                        .ToDictionary(kv => kv.Key, kv => kv.Value);
        }

        // Remove keys by prefix (e.g., to clear veto.* counters)
        public int ClearByPrefix(string prefix)
        {
            if (string.IsNullOrWhiteSpace(prefix)) return 0;
            int n = 0;
            foreach (var k in _vals.Keys)
            {
                try
                {
                    if (k.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                    {
                        if (_vals.TryRemove(k, out _)) n++;
                    }
                }
                catch { }
            }
            return n;
        }

        private static bool Concise() => (Environment.GetEnvironmentVariable("APP_CONCISE_CONSOLE") ?? "true").Trim().ToLowerInvariant() is "1" or "true" or "yes";
        private static bool ShowStatusTick() => (Environment.GetEnvironmentVariable("APP_SHOW_STATUS_TICK") ?? "false").Trim().ToLowerInvariant() is "1" or "true" or "yes";
        private static bool QuietJson() => (Environment.GetEnvironmentVariable("QUIET_JSON_STATUS") ?? "false").Trim().ToLowerInvariant() is "1" or "true" or "yes";

        public void Heartbeat()
        {
            var now = DateTimeOffset.UtcNow;
            if (QuietJson()) return;
            // In concise mode, status tick is opt-in
            if (Concise() && !ShowStatusTick()) return;
            // Base throttle to avoid hot loops
            if (!Concise())
            {
                if (now - _lastBeat < TimeSpan.FromSeconds(5)) return; // throttle
            }
            _lastBeat = now;

            // Compute top veto reasons (best-effort)
            Dictionary<string, int> vetoTop = new();
            try
            {
                vetoTop = _vals
                    .Where(kv => kv.Key.StartsWith("veto.", StringComparison.OrdinalIgnoreCase))
                    .Select(kv => new { kv.Key, Val = kv.Value is int i ? i : (kv.Value is long l ? (int)l : 0) })
                    .OrderByDescending(x => x.Val)
                    .Take(5)
                    .ToDictionary(x => x.Key, x => x.Val);
            }
            catch { }

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
                risk = Get<object?>("risk.state"),
                vetoTop
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
