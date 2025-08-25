using System;
using System.Text.Json;
using BotCore.Models;

namespace MarketDataAgent
{
    // BarAggregator: builds time-based bars from trades and quotes (JsonElement input)
    public sealed class BarAggregator
    {
        private readonly int _periodSec;
        private long _bucketStartUnixMs;
        private Bar? _cur;
        public string Symbol { get; set; } = string.Empty;
        public event Action<Bar>? OnBar;

        public BarAggregator(int periodSeconds)
        {
            _periodSec = Math.Max(1, periodSeconds);
        }

        public void OnTrade(JsonElement trade)
        {
            try
            {
                var ts = ReadUnixMs(trade, "ts")
                      ?? ReadIsoUtc(trade, "time")
                      ?? DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                var symbol = ReadString(trade, "symbol") ?? Symbol;
                var price = ReadDecimal(trade, "price") ?? 0m;
                var size = (int?)ReadDecimal(trade, "size") ?? 0;
                if (string.IsNullOrWhiteSpace(symbol)) symbol = Symbol;
                if (price <= 0m) return;
                EnsureBucket(ts, symbol, price);
                // update bar
                if (_cur != null)
                {
                    if (price > _cur.High) _cur.High = price;
                    if (price < _cur.Low) _cur.Low = price;
                    _cur.Close = price;
                    _cur.Volume += Math.Max(0, size);
                }
                TryRoll(ts);
            }
            catch { }
        }

        public void OnQuote(JsonElement quote)
        {
            // Optional: update close with mid price if no trades
            try
            {
                var ts = ReadUnixMs(quote, "ts")
                      ?? ReadIsoUtc(quote, "time")
                      ?? DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                var symbol = ReadString(quote, "symbol") ?? Symbol;
                var bid = ReadDecimal(quote, "bid");
                var ask = ReadDecimal(quote, "ask");
                if (bid is null || ask is null) return;
                var mid = (bid.Value + ask.Value) / 2m;
                if (mid <= 0m) return;
                EnsureBucket(ts, symbol, mid);
                if (_cur != null)
                {
                    if (mid > _cur.High) _cur.High = mid;
                    if (mid < _cur.Low) _cur.Low = mid;
                    _cur.Close = mid;
                }
                TryRoll(ts);
            }
            catch { }
        }

        private void EnsureBucket(long tsUnixMs, string symbol, decimal px)
        {
            var bucketStart = AlignToBucketStart(tsUnixMs);
            if (_cur == null || bucketStart != _bucketStartUnixMs)
            {
                // emit previous
                if (_cur != null) OnBar?.Invoke(_cur);
                _bucketStartUnixMs = bucketStart;
                _cur = new Bar
                {
                    Symbol = symbol,
                    Start = DateTimeOffset.FromUnixTimeMilliseconds(bucketStart).UtcDateTime,
                    Ts = bucketStart,
                    Open = px,
                    High = px,
                    Low = px,
                    Close = px,
                    Volume = 0
                };
            }
        }

        private void TryRoll(long tsUnixMs)
        {
            var bucketStart = AlignToBucketStart(tsUnixMs);
            if (bucketStart > _bucketStartUnixMs && _cur != null)
            {
                // finalize current and start a new bucket referencing last price
                var last = _cur.Close;
                var sym = _cur.Symbol;
                OnBar?.Invoke(_cur);
                _bucketStartUnixMs = bucketStart;
                _cur = new Bar
                {
                    Symbol = sym,
                    Start = DateTimeOffset.FromUnixTimeMilliseconds(bucketStart).UtcDateTime,
                    Ts = bucketStart,
                    Open = last,
                    High = last,
                    Low = last,
                    Close = last,
                    Volume = 0
                };
            }
        }

        private long AlignToBucketStart(long tsUnixMs)
        {
            var sec = tsUnixMs / 1000L;
            var bucket = (sec / _periodSec) * _periodSec;
            return bucket * 1000L;
        }

        private static string? ReadString(JsonElement obj, string name)
        {
            try { if (obj.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String) return v.GetString(); } catch { }
            return null;
        }
        private static long? ReadUnixMs(JsonElement obj, string name)
        {
            try
            {
                if (obj.TryGetProperty(name, out var v))
                {
                    if (v.ValueKind == JsonValueKind.Number && v.TryGetInt64(out var l)) return l;
                    if (v.ValueKind == JsonValueKind.String && long.TryParse(v.GetString(), out var l2)) return l2;
                }
            }
            catch { }
            return null;
        }
        private static long? ReadIsoUtc(JsonElement obj, string name)
        {
            try { if (obj.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String && DateTimeOffset.TryParse(v.GetString(), out var dto)) return dto.ToUnixTimeMilliseconds(); }
            catch { }
            return null;
        }
        private static decimal? ReadDecimal(JsonElement obj, string name)
        {
            try
            {
                if (obj.TryGetProperty(name, out var v))
                {
                    if (v.ValueKind == JsonValueKind.Number && v.TryGetDecimal(out var d)) return d;
                    if (v.ValueKind == JsonValueKind.String && decimal.TryParse(v.GetString(), System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var d2)) return d2;
                }
            }
            catch { }
            return null;
        }
    }
}
