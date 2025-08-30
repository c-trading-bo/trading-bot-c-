#nullable enable
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Text.Json;
using System.Linq;
using Microsoft.Extensions.Logging;
using BotCore.Infra;

namespace OrchestratorAgent.Infra
{
    /// <summary>
    /// Tracks P&L by strategy and symbol for comprehensive performance analysis.
    /// Maintains per-strategy performance metrics for ES, NQ, and other symbols.
    /// </summary>
    public sealed class StrategyPnlTracker
    {
        private readonly ILogger _log;
        private readonly ConcurrentDictionary<string, StrategySymbolPnl> _pnlByKey = new(StringComparer.OrdinalIgnoreCase);
        private readonly string _persistKey = "strategy_pnl";

        public StrategyPnlTracker(ILogger log)
        {
            _log = log;
            TryLoad();
        }

        public sealed class StrategySymbolPnl
        {
            public string StrategyId { get; set; } = "";
            public string Symbol { get; set; } = "";
            public decimal RealizedPnl { get; set; }
            public decimal UnrealizedPnl { get; set; }
            public int TotalTrades { get; set; }
            public int WinningTrades { get; set; }
            public decimal TotalVolume { get; set; }
            public DateTime LastUpdateUtc { get; set; }
            public decimal MaxDrawdown { get; set; }
            public decimal MaxProfit { get; set; }
        }

        public IReadOnlyDictionary<string, StrategySymbolPnl> Snapshot() => _pnlByKey;

        public void OnFill(string strategyId, string symbol, string side, int quantity, decimal fillPrice, decimal? entryPrice = null)
        {
            if (string.IsNullOrWhiteSpace(strategyId) || string.IsNullOrWhiteSpace(symbol)) return;
            
            var key = $"{strategyId}:{symbol}";
            var pnl = _pnlByKey.GetOrAdd(key, k => new StrategySymbolPnl 
            { 
                StrategyId = strategyId, 
                Symbol = symbol 
            });

            // Update trade statistics
            pnl.TotalTrades++;
            pnl.TotalVolume += Math.Abs(quantity);
            pnl.LastUpdateUtc = DateTime.UtcNow;

            // Calculate P&L if we have both entry and fill price
            if (entryPrice.HasValue && entryPrice.Value > 0)
            {
                var root = OrchestratorAgent.SymbolMeta.RootFromName(symbol);
                var isLong = side.Equals("BUY", StringComparison.OrdinalIgnoreCase);
                var points = isLong ? (fillPrice - entryPrice.Value) : (entryPrice.Value - fillPrice);
                var tradePnl = OrchestratorAgent.SymbolMeta.ToPnlUSD(root, points, Math.Abs(quantity));
                
                pnl.RealizedPnl += tradePnl;
                
                // Track win/loss
                if (tradePnl > 0)
                {
                    pnl.WinningTrades++;
                    pnl.MaxProfit = Math.Max(pnl.MaxProfit, pnl.RealizedPnl);
                }
                else if (tradePnl < 0)
                {
                    pnl.MaxDrawdown = Math.Min(pnl.MaxDrawdown, pnl.RealizedPnl);
                }

                _log.LogInformation("[StrategyPnl] {Strategy} {Symbol} {Side} {Qty} @ {Price} => P&L: {TradePnl:F2} (Total: {TotalPnl:F2})",
                    strategyId, symbol, side, quantity, fillPrice, tradePnl, pnl.RealizedPnl);
            }

            Persist();
        }

        public void UpdateUnrealized(string strategyId, string symbol, decimal unrealizedPnl)
        {
            if (string.IsNullOrWhiteSpace(strategyId) || string.IsNullOrWhiteSpace(symbol)) return;
            
            var key = $"{strategyId}:{symbol}";
            if (_pnlByKey.TryGetValue(key, out var pnl))
            {
                pnl.UnrealizedPnl = unrealizedPnl;
                pnl.LastUpdateUtc = DateTime.UtcNow;
            }
        }

        public Dictionary<string, object> GetDashboardData()
        {
            var result = new Dictionary<string, object>();
            
            // Group by strategy
            var byStrategy = _pnlByKey.Values
                .GroupBy(p => p.StrategyId, StringComparer.OrdinalIgnoreCase)
                .ToDictionary(g => g.Key, g => g.ToList(), StringComparer.OrdinalIgnoreCase);

            foreach (var strategyGroup in byStrategy)
            {
                var strategy = strategyGroup.Key;
                var strategyData = new Dictionary<string, object>();

                // Get ES and NQ specific data
                var esData = strategyGroup.Value.FirstOrDefault(p => p.Symbol.Equals("ES", StringComparison.OrdinalIgnoreCase));
                var nqData = strategyGroup.Value.FirstOrDefault(p => p.Symbol.Equals("NQ", StringComparison.OrdinalIgnoreCase));

                strategyData["es"] = new
                {
                    pnl = (esData?.RealizedPnl ?? 0) + (esData?.UnrealizedPnl ?? 0),
                    trades = esData?.TotalTrades ?? 0,
                    wins = esData?.WinningTrades ?? 0
                };

                strategyData["nq"] = new
                {
                    pnl = (nqData?.RealizedPnl ?? 0) + (nqData?.UnrealizedPnl ?? 0),
                    trades = nqData?.TotalTrades ?? 0,
                    wins = nqData?.WinningTrades ?? 0
                };

                result[strategy] = strategyData;
            }

            return result;
        }

        public void Reset()
        {
            _pnlByKey.Clear();
            Persist();
            _log.LogInformation("[StrategyPnl] Reset all strategy P&L data");
        }

        private void TryLoad()
        {
            try
            {
                var data = Persistence.Load<Dictionary<string, StrategySymbolPnl>>(_persistKey);
                if (data != null)
                {
                    foreach (var kvp in data)
                    {
                        _pnlByKey[kvp.Key] = kvp.Value;
                    }
                    _log.LogInformation("[StrategyPnl] Loaded {Count} strategy P&L entries", data.Count);
                }
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[StrategyPnl] Failed to load persisted data");
            }
        }

        private void Persist()
        {
            try
            {
                var data = new Dictionary<string, StrategySymbolPnl>(_pnlByKey);
                Persistence.Save(_persistKey, data);
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[StrategyPnl] Failed to persist data");
            }
        }
    }
}
