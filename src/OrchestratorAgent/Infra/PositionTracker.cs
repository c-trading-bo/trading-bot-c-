#nullable enable
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCore.Infra;

namespace OrchestratorAgent.Infra
{
    /// <summary>
    /// Tracks positions in real-time from user hub snapshots/fills and market trades.
    /// Maintains signed quantity, average price, realized/unrealized PnL (USD) per symbol.
    /// </summary>
    public sealed class PositionTracker
    {
        private readonly ILogger _log;
        private readonly long _accountId;
        private readonly ConcurrentDictionary<string, PositionState> _bySymbol = new(StringComparer.OrdinalIgnoreCase);
        private readonly string _persistKey = "positions";

        public PositionTracker(ILogger log, long accountId)
        {
            _log = log; _accountId = accountId;
            TryLoad();
        }

        public sealed class PositionState
        {
            public string Symbol { get; set; } = "";  // root or contract symbol
            public int Qty { get; set; }               // signed (+ long / - short)
            public decimal AvgPrice { get; set; }      // weighted average entry price
            public decimal RealizedUsd { get; set; }   // running realized PnL for trading day
            public decimal LastPrice { get; set; }     // last market price
            public decimal UnrealizedUsd { get; set; } // computed from last price and avg
            public DateTime UpdatedUtc { get; set; }
        }

        public IReadOnlyDictionary<string, PositionState> Snapshot() => _bySymbol;

        public void OnPosition(JsonElement je)
        {
            try
            {
                var symbol = ReadString(je, "symbol") ?? ReadString(je, "Symbol") ?? "";
                if (string.IsNullOrWhiteSpace(symbol)) return;
                var qty = ReadInt(je, new[] { "qty", "quantity", "Qty", "Quantity" });
                var avg = ReadDecimal(je, new[] { "avgPrice", "averagePrice", "AvgPrice", "AveragePrice" });
                ApplySnapshot(symbol, qty, avg);
            }
            catch { }
        }

        public void OnTrade(JsonElement je)
        {
            try
            {
                var symbol = ReadString(je, "symbol") ?? ReadString(je, "Symbol") ?? "";
                if (string.IsNullOrWhiteSpace(symbol)) return;
                var sideStr = ReadString(je, "side") ?? ReadString(je, "Side") ?? "";
                var sideNum = ReadInt(je, new[] { "side", "Side" });
                bool isSell = sideStr.Equals("SELL", StringComparison.OrdinalIgnoreCase) || sideNum == 1;
                var qty = Math.Abs(ReadInt(je, new[] { "qty", "quantity", "Qty", "Quantity", "size" }));
                var price = ReadDecimal(je, new[] { "price", "avgFillPrice", "fillPrice", "limitPrice", "Price" });
                if (qty <= 0 || price <= 0m) return;
                ApplyFill(symbol, isSell ? -qty : qty, price);
            }
            catch { }
        }

        public void OnMarketTrade(JsonElement je)
        {
            try
            {
                var symbol = ReadString(je, "symbol") ?? "";
                var price = ReadDecimal(je, new[] { "price", "last", "tradePrice" });
                if (string.IsNullOrWhiteSpace(symbol) || price <= 0m) return;
                UpdateLastPrice(symbol, price);
            }
            catch { }
        }

        public async Task SeedFromRestAsync(BotCore.ApiClient api, long accountId, CancellationToken ct)
        {
            try
            {
                var list = await api.GetAsync<List<JsonElement>>($"/positions?accountId={accountId}&status=OPEN", ct) ?? new();
                foreach (var p in list)
                {
                    var symbol = ReadString(p, "symbol") ?? ReadString(p, "Symbol") ?? "";
                    var qty = ReadInt(p, new[] { "qty", "quantity", "Qty", "Quantity" });
                    var avg = ReadDecimal(p, new[] { "avgPrice", "averagePrice", "AvgPrice", "AveragePrice" });
                    if (!string.IsNullOrWhiteSpace(symbol)) ApplySnapshot(symbol, qty, avg);
                }
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[Positions] SeedFromRest failed");
            }
        }

        private void ApplySnapshot(string symbol, int qtySigned, decimal avgPrice)
        {
            var st = _bySymbol.GetOrAdd(symbol, s => new PositionState { Symbol = s });
            st.Qty = qtySigned;
            st.AvgPrice = avgPrice > 0m ? avgPrice : st.AvgPrice;
            st.UpdatedUtc = DateTime.UtcNow;
            RecalcUnreal(symbol, st);
            Persist();
        }

        private void ApplyFill(string symbol, int qtySigned, decimal px)
        {
            var st = _bySymbol.GetOrAdd(symbol, s => new PositionState { Symbol = s });
            var oldQty = st.Qty;
            var newQty = st.Qty + qtySigned;

            if (oldQty == 0 || Math.Sign(oldQty) == Math.Sign(qtySigned))
            {
                // same direction or opening fresh: weighted average
                var totalQty = Math.Abs(oldQty) + Math.Abs(qtySigned);
                st.AvgPrice = totalQty == 0 ? px : ((st.AvgPrice * Math.Abs(oldQty)) + (px * Math.Abs(qtySigned))) / totalQty;
                st.Qty = newQty;
            }
            else
            {
                // opposite direction: closing existing position partially or fully, maybe flipping
                var closeQty = Math.Min(Math.Abs(oldQty), Math.Abs(qtySigned));
                var remaining = Math.Abs(qtySigned) - closeQty;
                // points moved for realized PnL (USD via SymbolMeta)
                var root = OrchestratorAgent.SymbolMeta.RootFromName(symbol);
                var points = (px - st.AvgPrice) * (oldQty > 0 ? 1 : -1); // positive when profitable
                var realized = OrchestratorAgent.SymbolMeta.ToPnlUSD(root, points, closeQty);
                st.RealizedUsd += realized;

                // update qty after closing part/all
                st.Qty = oldQty + qtySigned;
                if (st.Qty == 0)
                {
                    st.AvgPrice = 0m;
                }
                else if (remaining > 0)
                {
                    // flipped to new side; new avg is the fill price for leftover
                    st.AvgPrice = px;
                }
            }

            st.LastPrice = px; // at least have last as fill price
            st.UpdatedUtc = DateTime.UtcNow;
            RecalcUnreal(symbol, st);
            Persist();
        }

        private void UpdateLastPrice(string symbol, decimal px)
        {
            if (!_bySymbol.TryGetValue(symbol, out var st)) return;
            st.LastPrice = px;
            st.UpdatedUtc = DateTime.UtcNow;
            RecalcUnreal(symbol, st);
        }

        private static decimal ReadDecimal(JsonElement je, string[] names)
        {
            foreach (var n in names)
            {
                try
                {
                    if (je.TryGetProperty(n, out var v))
                    {
                        if (v.ValueKind == JsonValueKind.Number && v.TryGetDecimal(out var d)) return d;
                        if (v.ValueKind == JsonValueKind.String && decimal.TryParse(v.GetString(), System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var ds)) return ds;
                    }
                }
                catch { }
            }
            return 0m;
        }
        private static int ReadInt(JsonElement je, string[] names)
        {
            foreach (var n in names)
            {
                try
                {
                    if (je.TryGetProperty(n, out var v))
                    {
                        if (v.ValueKind == JsonValueKind.Number && v.TryGetInt32(out var i)) return i;
                        if (v.ValueKind == JsonValueKind.String && int.TryParse(v.GetString(), out var is2)) return is2;
                    }
                }
                catch { }
            }
            return 0;
        }
        private static string? ReadString(JsonElement je, string name)
        {
            try { if (je.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String) return v.GetString(); } catch { }
            return null;
        }

        private void RecalcUnreal(string symbol, PositionState st)
        {
            if (st.Qty == 0 || st.AvgPrice <= 0m || st.LastPrice <= 0m) { st.UnrealizedUsd = 0m; return; }
            var root = OrchestratorAgent.SymbolMeta.RootFromName(symbol);
            var points = (st.LastPrice - st.AvgPrice) * (st.Qty >= 0 ? 1 : -1); // profit positive
            st.UnrealizedUsd = OrchestratorAgent.SymbolMeta.ToPnlUSD(root, points, Math.Abs(st.Qty));
        }

        private void TryLoad()
        {
            try
            {
                var snap = Persistence.Load<Dictionary<string, PositionState>>(_persistKey);
                if (snap != null)
                {
                    foreach (var kv in snap)
                        _bySymbol[kv.Key] = kv.Value;
                }
            }
            catch { }
        }

        private void Persist()
        {
            try
            {
                Persistence.Save(_persistKey, new Dictionary<string, PositionState>(_bySymbol));
            }
            catch { }
        }
    }
}
