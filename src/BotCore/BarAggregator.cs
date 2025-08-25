
#nullable enable
using System.Text.Json;

namespace BotCore
{

    // Use the unified Bar model from BotCore.Models
    using BotCore.Models;

    public sealed class BarAggregator
    {
        private readonly object _lock = new();
        private readonly int _seconds;
        private DateTimeOffset _bucket = DateTimeOffset.MinValue;
        private decimal _o, _h, _l, _c;
        private long _v;

        public event Action<Bar>? OnBar;
        public string Symbol { get; set; } = string.Empty;

        public BarAggregator(int seconds)
        {
            _seconds = Math.Max(5, seconds);
        }

        // ---- Public entry points ----

        public void OnTrade(JsonElement payload)
        {
            if (payload.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in payload.EnumerateArray())
                    ApplyTrade(item);
            }
            else if (payload.ValueKind == JsonValueKind.Object)
            {
                ApplyTrade(payload);
            }
        }

        public void OnQuote(JsonElement payload)
        {
            if (payload.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in payload.EnumerateArray())
                    ApplyQuote(item);
            }
            else if (payload.ValueKind == JsonValueKind.Object)
            {
                ApplyQuote(payload);
            }
        }

        // ---- Internals ----

        private void ApplyTrade(JsonElement trade)
        {
            if (!TryGetDecimal(trade, out var price, "price", "last", "px", "p", "lastPrice")) return;
            TryGetLong(trade, out var size, "size", "qty", "quantity", "volume", "v", "tradeSize");
            var ts = TryGetExchangeTime(trade) ?? DateTimeOffset.UtcNow;
            Upsert(price, size, ts);
        }

        private void ApplyQuote(JsonElement quote)
        {
            // Prefer an explicit last/price; otherwise derive from bid/ask mid if both exist.
            if (!TryGetDecimal(quote, out var price, "last", "lastPrice", "price", "px", "p"))
            {
                var hasBid = TryGetDecimal(quote, out var bid, "bid", "bidPrice", "b");
                var hasAsk = TryGetDecimal(quote, out var ask, "ask", "askPrice", "a");
                if (!(hasBid && hasAsk)) return;
                price = (bid + ask) / 2m;
            }

            var ts = TryGetExchangeTime(quote) ?? DateTimeOffset.UtcNow;
            Upsert(price, addVolume: 0, ts);
        }

        private void Upsert(decimal price, long addVolume, DateTimeOffset ts)
        {
            var bucketStart = Align(ts);

            lock (_lock)
            {
                // New bucket? Flush the previous bar.
                if (_bucket != DateTimeOffset.MinValue && bucketStart != _bucket)
                    Flush();

                if (_bucket == DateTimeOffset.MinValue)
                {
                    _bucket = bucketStart;
                    _o = _h = _l = _c = price;
                    _v = addVolume;
                }
                else
                {
                    _c = price;
                    if (price > _h) _h = price;
                    if (price < _l) _l = price;
                    _v += addVolume;
                }
            }
        }

        private void Flush()
        {
            var bar = new Bar {
                Start = _bucket.UtcDateTime,
                Ts = new DateTimeOffset(_bucket.UtcDateTime).ToUnixTimeMilliseconds(),
                Symbol = this.Symbol,
                Open = _o,
                High = _h,
                Low = _l,
                Close = _c,
                Volume = (int)_v
            };
            _bucket = DateTimeOffset.MinValue;
            _o = _h = _l = _c = 0m; _v = 0;
            OnBar?.Invoke(bar);
        }

        private DateTimeOffset Align(DateTimeOffset ts)
        {
            var s = (ts.Second / _seconds) * _seconds;
            return new DateTimeOffset(ts.Year, ts.Month, ts.Day, ts.Hour, ts.Minute, s, ts.Offset);
        }

        // ---- JSON helpers ----

        private static DateTimeOffset? TryGetExchangeTime(JsonElement e)
        {
            // Try several common fields for exchange timestamps
            if (e.ValueKind != JsonValueKind.Object) return null;
            foreach (var name in new[] { "exchangeTimeUtc", "exchangeTime", "ts", "timestamp", "time" })
            {
                if (e.TryGetProperty(name, out var p))
                {
                    if (p.ValueKind == JsonValueKind.Number && p.TryGetInt64(out var ms))
                        return DateTimeOffset.FromUnixTimeMilliseconds(ms);
                    if (p.ValueKind == JsonValueKind.String && DateTimeOffset.TryParse(p.GetString(), out var dto))
                        return dto.ToUniversalTime();
                }
            }
            return null;
        }

        private static bool TryGetDecimal(JsonElement e, out decimal val, params string[] names)
        {
            foreach (var n in names)
            {
                if (e.TryGetProperty(n, out var p))
                {
                    if (p.ValueKind == JsonValueKind.Number && p.TryGetDecimal(out val)) return true;
                    if (p.ValueKind == JsonValueKind.String && decimal.TryParse(p.GetString(), out val)) return true;
                }
            }
            val = 0m; return false;
        }

        private static bool TryGetLong(JsonElement e, out long val, params string[] names)
        {
            foreach (var n in names)
            {
                if (e.TryGetProperty(n, out var p))
                {
                    if (p.ValueKind == JsonValueKind.Number && p.TryGetInt64(out val)) return true;
                    if (p.ValueKind == JsonValueKind.String && long.TryParse(p.GetString(), out val)) return true;
                }
            }
            val = 0; return false;
        }
    }
}
