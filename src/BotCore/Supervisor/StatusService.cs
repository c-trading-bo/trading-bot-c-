#nullable enable
using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace BotCore.Supervisor
{
    public sealed class StatusService
    {
        private readonly ILogger<StatusService> _log;
        private readonly ConcurrentDictionary<string, object> _vals = new();
        private DateTimeOffset _lastBeat = DateTimeOffset.MinValue;

        public long AccountId { get; set; }
        public Dictionary<string,string> Contracts { get; set; } = new();

        public StatusService(ILogger<StatusService> log) => _log = log;

        public void Set(string key, object value) => _vals[key] = value;
        public T? Get<T>(string key) => _vals.TryGetValue(key, out var v) ? (T?)v : default;

        public void Heartbeat()
        {
            var now = DateTimeOffset.UtcNow;
            if (now - _lastBeat < TimeSpan.FromSeconds(5)) return; // throttle
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

            _log.LogInformation("BOT STATUS => {Json}", JsonSerializer.Serialize(snapshot));
        }
    }
}
