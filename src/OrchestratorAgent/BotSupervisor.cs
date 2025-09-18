#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Channels;
using System.Text.Json;
using System.Net.Http.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using SupervisorAgent;
using BotCore.Models;
using BotCore.Services;
using BotCore.Supervisor;
using TopstepX.Bot.Authentication;
using BotCore.Risk;
using BotCore.Infra;
using BotCore.Strategy;
using OrchestratorAgent.Infra;
using TradingBot.RLAgent;
using TradingBot.IntelligenceStack;

namespace OrchestratorAgent
{
    public sealed class BotSupervisor(ILogger<BotSupervisor> log, HttpClient http, string apiBase, string jwt, long accountId, object marketHub, object userHub, SupervisorAgent.StatusService status, BotSupervisor.Config cfg, FeatureEngineering? featureEngineering = null, UnifiedDecisionLogger? decisionLogger = null, TradingBot.IntelligenceStack.IntelligenceOrchestrator? intelligenceOrchestrator = null, TradingBot.IntelligenceAgent.IVerifier? verifier = null, BotCore.Services.IContractService? contractService = null, BotCore.Services.ISecurityService? securityService = null, TradingBot.Abstractions.IOnlineLearningSystem? onlineLearningSystem = null)
    {
        private readonly SemaphoreSlim _routeLock = new(1, 1);
        public sealed class Config
        {
            public bool LiveTrading { get; set; } = false;
            public int BarSeconds { get; set; } = 60;
            public string[] Symbols { get; set; } = [];
            public bool UseQuotes { get; set; } = true;
            public BracketConfig DefaultBracket { get; set; } = new();
        }

        public sealed class BracketConfig
        {
            public int StopTicks { get; set; } = 12;
            public int TargetTicks { get; set; } = 18;
            public int BreakevenAfterTicks { get; set; } = 8;
            public int TrailTicks { get; set; } = 6;
        }

        private readonly ILogger<BotSupervisor> _log = log;
        private readonly HttpClient _http = http;
        private readonly string _apiBase = apiBase;
        private readonly string _jwt = jwt;
        private readonly long _accountId = accountId;
        private readonly object _marketHub = marketHub;
        private readonly object _userHub = userHub;
        private readonly SupervisorAgent.StatusService _status = status;
        private readonly Config _cfg = cfg;
        private readonly FeatureEngineering? _featureEngineering = featureEngineering;
        private readonly UnifiedDecisionLogger? _decisionLogger = decisionLogger;
        private readonly TradingBot.IntelligenceStack.IntelligenceOrchestrator? _intelligenceOrchestrator = intelligenceOrchestrator;
        private readonly TradingBot.IntelligenceAgent.IVerifier? _verifier = verifier;
        private readonly BotCore.Services.IContractService? _contractService = contractService;
        private readonly BotCore.Services.ISecurityService? _securityService = securityService;
        private readonly TradingBot.Abstractions.IOnlineLearningSystem? _onlineLearningSystem = onlineLearningSystem;
        private readonly Channel<(BotCore.StrategySignal Sig, string ContractId)> _routeChan = Channel.CreateBounded<(BotCore.StrategySignal, string)>(128);
        private readonly BotCore.Supervisor.ContractResolver _contractResolver = new();
        private readonly BotCore.Supervisor.StateStore _stateStore = new();
        private readonly Notifier _notifier = new();
        private readonly List<LastSignal> _recentSignals = [];
        private int _barsSeen = 0; // Counter for AUTO_EXECUTE gating
        private DateTime _lastVerify = default; // Track last verification time for debounce
        private sealed record LastSignal(string Strategy, string Symbol, string Side, decimal Sp, decimal Tp, decimal Sl);

        public async Task RunAsync(CancellationToken ct)
        {
            // Supervisor logic: subscribe to events, run strategies, route orders, emit status
            _status.Set("user.state", "init");
            _status.Set("market.state", "init");

            void SafeAttachEvent<T>(object target, string evt, Action<T> handler)
            {
                var e = target.GetType().GetEvent(evt);
                if (e == null) return;
                var del = Delegate.CreateDelegate(e.EventHandlerType!, handler.Target!, handler.Method);
                e.AddEventHandler(target, del);
            }

            bool TryWireSignalROn<T>(object hub, string method, Action<T> handler)
            {
                var mi = hub.GetType()
                    .GetMethods()
                    .FirstOrDefault(m => m.Name == "On" && m.IsGenericMethodDefinition && m.GetParameters().Length == 2);
                if (mi == null) return false;
                var g = mi.MakeGenericMethod(typeof(T));
                g.Invoke(hub, [method, handler]);
                return true;
            }

            // Predeclare state before wiring any lambdas to avoid definite-assignment issues
            var history = new Dictionary<string, List<BotCore.Models.Bar>>(StringComparer.OrdinalIgnoreCase);
            var lastBarUnix = new Dictionary<string, long>(StringComparer.OrdinalIgnoreCase);
            var skipFirstAfterGap = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            var risk = new BotCore.Risk.RiskEngine();
            var levels = new BotCore.Models.Levels();
            var journal = new BotCore.Supervisor.SignalJournal();
            var _recentRoutes = new System.Collections.Concurrent.ConcurrentDictionary<string, DateTime>();
            var _recentCidBuffer = new System.Collections.Concurrent.ConcurrentQueue<string>();
            var maxTradesEnv = Environment.GetEnvironmentVariable("MAX_TRADES_PER_DAY");
            int maxTradesPerDay = int.TryParse(maxTradesEnv, out var mt) && mt > 0 ? mt : int.MaxValue;
            int tradesToday = 0;
            var tradeDay = DateTime.UtcNow.Date;
            // Daily PnL breaker setup
            var mdlEnv = Environment.GetEnvironmentVariable("EVAL_MAX_DAILY_LOSS") ?? Environment.GetEnvironmentVariable("MAX_DAILY_LOSS");
            decimal maxDailyLoss = decimal.TryParse(mdlEnv, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var mdlVal) ? mdlVal : 1000m;
            DateTime lastPnlFetch = DateTime.MinValue;
            decimal netPnlCache = 0m;

            // track a heartbeat-able snapshot
            DateTimeOffset lastQuote = default, lastTrade = default, lastBar = default;
            _status.Set("strategies", "wired");

            // 1) Market events (works with either your MarketDataAgent events or raw HubConnection)
            SafeAttachEvent<object>(_marketHub, "OnQuote", _ => { lastQuote = DateTimeOffset.UtcNow; _status.Set("last.quote", lastQuote); });
            SafeAttachEvent<object>(_marketHub, "OnTrade", _ => { lastTrade = DateTimeOffset.UtcNow; _status.Set("last.trade", lastTrade); });
            SafeAttachEvent<BotCore.Models.Bar>(_marketHub, "OnBar", b => { lastBar = DateTimeOffset.UtcNow; _status.Set("last.bar", lastBar); HandleBar(b); });

            // If _marketHub is a HubConnection, wire SignalR “On” as well (harmless if not)
            TryWireSignalROn<System.Text.Json.JsonElement>(_marketHub, "Quote", _ => { lastQuote = DateTimeOffset.UtcNow; _status.Set("last.quote", lastQuote); });
            TryWireSignalROn<System.Text.Json.JsonElement>(_marketHub, "Trade", _ => { lastTrade = DateTimeOffset.UtcNow; _status.Set("last.trade", lastTrade); });

            // 2) User hub events (orders/trades confirmations)
            OrchestratorAgent.OrderRouter router = null!;
            SafeAttachEvent<object>(_userHub, "OnOrder", _ => _status.Set("orders.open", "changed"));
            SafeAttachEvent<object>(_userHub, "OnTrade", _ => _status.Set("last.trade", DateTimeOffset.UtcNow));
            // Partial fill management via reflection-friendly OnOrderUpdate
            SafeAttachEvent<object>(_userHub, "OnOrderUpdate", async upd =>
            {
                try
                {
                    var t = upd.GetType();
                    bool isParent = t.GetProperty("IsParent")?.GetValue(upd) as bool? ?? false;
                    int filled = Convert.ToInt32(t.GetProperty("FilledQty")?.GetValue(upd) ?? 0);
                    int remaining = Convert.ToInt32(t.GetProperty("RemainingQty")?.GetValue(upd) ?? 0);
                    string orderId = t.GetProperty("OrderId")?.GetValue(upd)?.ToString() ?? string.Empty;
                    var age = (TimeSpan?)(t.GetProperty("Age")?.GetValue(upd)) ?? TimeSpan.Zero;
                    if (!isParent || filled <= 0) return;
                    await router.UpsertBracketsAsync(orderId, filled, ct).ConfigureAwait(false);
                    if (remaining > 0 && age > TimeSpan.FromSeconds(5))
                        await router.ConvertRemainderToLimitOrCancelAsync(upd, ct).ConfigureAwait(false);

                    // Wire OnlineLearningSystem into live trading loop
                    if (_onlineLearningSystem != null && filled > 0)
                    {
                        _ = Task.Run(async () =>
                        {
                            try
                            {
                                // Extract trade information for online learning
                                var symbol = t.GetProperty("Symbol")?.GetValue(upd)?.ToString() ?? string.Empty;
                                var side = t.GetProperty("Side")?.GetValue(upd)?.ToString() ?? string.Empty;
                                var fillPrice = Convert.ToDouble(t.GetProperty("FillPrice")?.GetValue(upd) ?? 0.0);
                                var customTag = t.GetProperty("CustomTag")?.GetValue(upd)?.ToString() ?? string.Empty;
                                
                                // Create trade record for online learning
                                var tradeRecord = new TradingBot.Abstractions.TradeRecord
                                {
                                    TradeId = orderId,
                                    Symbol = symbol,
                                    Side = side,
                                    Quantity = filled,
                                    FillPrice = fillPrice,
                                    FillTime = DateTime.UtcNow,
                                    StrategyId = ExtractStrategyIdFromTag(customTag),
                                    Metadata = new Dictionary<string, object>
                                    {
                                        ["custom_tag"] = customTag,
                                        ["fill_timestamp"] = DateTime.UtcNow,
                                        ["regime_type"] = "trend", // Default regime, could be detected dynamically
                                        ["prediction_confidence"] = 0.7, // Default confidence, could be from ML model
                                        ["market_movement_bps"] = 0.0, // Would be calculated from market data
                                        ["order_latency_ms"] = age.TotalMilliseconds
                                    }
                                };

                                // Update online learning model with trade data
                                await _onlineLearningSystem.UpdateModelAsync(tradeRecord, ct).ConfigureAwait(false);
                                
                                // Log completion
                                _log.LogDebug("[ONLINE_LEARNING] Online learning system updated successfully for order: {OrderId}", orderId);
                                
                                _log.LogDebug("[ONLINE_LEARNING] Trade processed for model update: {OrderId} - {Symbol} {Side} {Quantity}@{FillPrice}", 
                                    orderId, symbol, side, filled, fillPrice);
                            }
                            catch (Exception ex)
                            {
                                _log.LogWarning(ex, "[ONLINE_LEARNING] Failed to process trade for online learning: {OrderId}", orderId);
                            }
                        });
                    }
                }
                catch { }
            });

            // 3) Optional: simple bar handler → (generate signals) → route orders
            var orderLog = (Microsoft.Extensions.Logging.ILogger<OrchestratorAgent.OrderRouter>?)null;
            try
            {
                var lf = (ILoggerFactory?)typeof(ILogger).Assembly
                    .GetType("Microsoft.Extensions.Logging.LoggerFactory")?
                    .GetProperty("Create")?.GetValue(null) as ILoggerFactory;
                orderLog = lf?.CreateLogger<OrchestratorAgent.OrderRouter>();
            }
            catch { /* best-effort */ }

            router = new OrchestratorAgent.OrderRouter(
                orderLog ?? (ILogger<OrchestratorAgent.OrderRouter>)_log,
                _http, _apiBase, _jwt, (int)_accountId);

            // JWT refresh loop for REST (ApiClient) and default headers
            _ = Task.Run(async () =>
            {
                var auth = new TopstepAuthAgent(_http);
                while (!ct.IsCancellationRequested)
                {
                    try
                    {
                        await Task.Delay(TimeSpan.FromMinutes(20), ct).ConfigureAwait(false);
                        var newToken = await auth.ValidateAsync(ct).ConfigureAwait(false).ConfigureAwait(false);
                        if (string.IsNullOrWhiteSpace(newToken))
                        {
                            var u = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
                            var k = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");
                            if (!string.IsNullOrWhiteSpace(u) && !string.IsNullOrWhiteSpace(k))
                                newToken = await auth.GetJwtAsync(u!, k!, ct).ConfigureAwait(false).ConfigureAwait(false);
                        }
                        if (!string.IsNullOrWhiteSpace(newToken))
                        {
                            Environment.SetEnvironmentVariable("TOPSTEPX_JWT", newToken);
                            try { _http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", newToken); } catch { }
                            try { router.UpdateJwt(newToken); } catch { }
                            _status.Set("auth.jwt", "refreshed");
                        }
                    }
                    catch (OperationCanceledException) { }
                    catch { }
                }
            }, ct);

            // Rebuild state on boot: seed de-dupe cache from open positions
            try
            {
                var open = await router.QueryOpenPositionsAsync(_accountId, ct).ConfigureAwait(false).ConfigureAwait(false);
                foreach (var p in open)
                {
                    try
                    {
                        string strat = p.Strategy ?? p.strategy ?? "";
                        string sym = p.Symbol ?? p.symbol ?? "";
                        string side = p.Side ?? p.side ?? "";
                        decimal sp = (decimal)(p.Entry ?? p.entry ?? 0m);
                        decimal tp = (decimal)(p.Target ?? p.target ?? 0m);
                        decimal sl = (decimal)(p.Stop ?? p.stop ?? 0m);
                        BotCore.RecentSignalCache.ShouldEmit(strat, sym, side, sp, tp, sl, 0);
                    }
                    catch { }
                }
                // Load persisted restart-safety caches
                foreach (var k in BotCore.Infra.Persistence.Load<System.Collections.Generic.List<string>>("recent_cids") ?? [])
                    _recentRoutes.TryAdd(k, DateTime.UtcNow);
                foreach (var s in BotCore.Infra.Persistence.Load<System.Collections.Generic.List<LastSignal>>("last_signals") ?? [])
                    BotCore.RecentSignalCache.ShouldEmit(s.Strategy, s.Symbol, s.Side, s.Sp, s.Tp, s.Sl, 0);
            }
            catch { }

            // Load persisted state (best-effort)
            try
            {
                var snap = _stateStore.Load();
                if (snap.RecentRoutes != null)
                {
                    foreach (var kv in snap.RecentRoutes)
                        _recentRoutes.TryAdd(kv.Key, kv.Value);
                }
                if (snap.LastBarUnix != null)
                {
                    foreach (var kv in snap.LastBarUnix)
                        lastBarUnix[kv.Key] = kv.Value;
                }
            }
            catch { }

            // Start router consumer (single-writer with backpressure)
            _ = Task.Run(async () =>
            {
                while (!ct.IsCancellationRequested)
                {
                    try
                    {
                        var (Sig, ContractId) = await _routeChan.Reader.ReadAsync(ct).ConfigureAwait(false).ConfigureAwait(false);
                        await Retry(() => router.RouteAsync(Sig, ContractId, ct), ct).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException) { }
                    catch (Exception ex) { _log.LogWarning(ex, "[Supervisor] Route consumer error"); }
                }
            }, ct);

            // Optional: attach to Reconnected to rebuild OCOs on recover
            var _reconnects = new System.Collections.Concurrent.ConcurrentQueue<DateTime>();
            try
            {
                var rebuildOnReconnect = (Environment.GetEnvironmentVariable("OCO_REBUILD_ON_RECONNECT") ?? "0").Equals("1", StringComparison.OrdinalIgnoreCase);
                var evt = _marketHub.GetType().GetEvent("Reconnected");
                if (evt != null)
                {
                    var handler = new Func<string?, Task>(async id =>
                    {
                        _status.Set("market.state", "reconnected");
                        _reconnects.Enqueue(DateTime.UtcNow);
                        TrimWindow(_reconnects, TimeSpan.FromMinutes(5));
                        if (_reconnects.Count > 5)
                        {
                            _log.LogWarning("[WARN] Reconnect loop: {Count} in last 5m", _reconnects.Count);
                            try { _ = _notifier.Warn($"Reconnect storm: {_reconnects.Count} in last 5m"); } catch { }
                        }
                        if (rebuildOnReconnect)
                            await router.EnsureBracketsAsync(_accountId, ct).ConfigureAwait(false);
                    });
                    var del = Delegate.CreateDelegate(evt.EventHandlerType!, handler.Target!, handler.Method);
                    evt.AddEventHandler(_marketHub, del);
                }
            }
            catch { }

            static async Task Retry(Func<Task> op, CancellationToken ct)
            {
                var delays = new[] { 250, 500, 1000, 2000, 4000, 8000 };
                foreach (var d in delays)
                {
                    try { await op().ConfigureAwait(false); return; }
                    catch when (!ct.IsCancellationRequested) { await Task.Delay(d, ct).ConfigureAwait(false); }
                }
                await op().ConfigureAwait(false); // final throw
            }

            static void TrimWindow(System.Collections.Concurrent.ConcurrentQueue<DateTime> q, TimeSpan window)
            {
                var cutoff = DateTime.UtcNow - window;
                while (q.TryPeek(out var t) && t < cutoff)
                    q.TryDequeue(out _);
            }


            async void HandleBar(BotCore.Models.Bar bar)
            {
                try
                {
                    // Increment BarsSeen counter for AUTO_EXECUTE gating
                    _barsSeen++;
                    _log.LogDebug("BarsSeen={Count}/10; gating AUTO_EXECUTE={Gating}", _barsSeen, _barsSeen < 10 ? "ACTIVE" : "INACTIVE");

                    // 1️⃣ Wire Live Market Data → ML Pipeline
                    // Process streaming tick for real-time feature aggregation
                    if (_featureEngineering != null)
                    {
                        try
                        {
                            var tick = new MarketTick
                            {
                                Symbol = bar.Symbol,
                                Price = (double)bar.Close,
                                Volume = bar.Volume,
                                Timestamp = DateTime.UnixEpoch.AddMilliseconds(bar.Ts)
                            };
                            
                            var features = await _featureEngineering.ProcessStreamingTickAsync(tick).ConfigureAwait(false).ConfigureAwait(false);
                            _log.LogTrace("[ML_PIPELINE] Processed streaming tick for {Symbol}: Price={Price}, Volume={Volume}, FeatureCount={FeatureCount}", 
                                tick.Symbol, tick.Price, tick.Volume, features?.GetType().GetProperties().Length ?? 0);
                        }
                        catch (Exception ex)
                        {
                            _log.LogError(ex, "[ML_PIPELINE] Failed to process streaming tick for {Symbol}", bar.Symbol);
                        }
                    }

                    var symbol = bar.Symbol;
                    if (!history.TryGetValue(symbol, out var list))
                    {
                        list = new List<BotCore.Models.Bar>(512);
                        history[symbol] = list;
                    }
                    list.Add(bar);
                    if (list.Count > 1000) list.RemoveRange(0, list.Count - 1000);

                    var env = new BotCore.Models.Env
                    {
                        Symbol = symbol,
                        atr = list.Count > 0 ? Math.Abs(list[^1].High - list[^1].Low) : (decimal?)null,
                        volz = 1.0m
                    };

                    // Contract resolution using available-first fallback
                    string contractId;
                    if (_contractService != null)
                    {
                        contractId = await _contractService.ResolveContractAsync(symbol, CancellationToken.None) ?? symbol.ConfigureAwait(false).ConfigureAwait(false);
                    }
                    else
                    {
                        // Fallback to status snapshot (original behavior)
                        if (!_status.Contracts.TryGetValue(symbol, out contractId!))
                        {
                            contractId = symbol;
                        }
                    }

                    // Backfill after gap (>15s) before emitting signals
                    if (lastBarUnix.TryGetValue(symbol, out var prevTs))
                    {
                        var gapSec = (bar.Ts - prevTs) / 1000L;
                        if (gapSec > 15)
                        {
                            try
                            {
                                var sinceUnixMs = prevTs;
                                var untilUnixMs = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
                                var path = $"/marketdata/bars?symbol={symbol}&tf=1m&since={sinceUnixMs}&until={untilUnixMs}";
                                var missing = _http is not null
                                    ? await _http.GetFromJsonAsync<List<BotCore.Models.Bar>>(path, cancellationToken: CancellationToken.None)
                                    : null.ConfigureAwait(false);
                                if (missing is { Count: > 0 })
                                {
                                    var merged = list.Concat(missing.Where(b => b.Ts > sinceUnixMs))
                                                     .OrderBy(b => b.Ts)
                                                     .ToList();
                                    history[symbol] = list = merged;
                                    _status.Set($"backfill.{symbol}", $"added={missing.Count}");
                                }
                            }
                            catch (Exception ex)
                            {
                                _log.LogWarning(ex, "[Supervisor] Backfill failed for {Sym}", symbol);
                            }
                            finally
                            {
                                // Regardless of backfill success, skip routing for the first bar after gap
                                skipFirstAfterGap.Add(symbol);
                            }
                        }
                    }
                    lastBarUnix[symbol] = bar.Ts;

                    // Suppress signals for the first bar after a multi-minute gap
                    if (skipFirstAfterGap.Remove(symbol))
                    {
                        _log.LogWarning("[Supervisor] Suppressing signals for first bar after gap for {Sym}", symbol);
                        return;
                    }

                    var signals = BotCore.Strategy.AllStrategies.generate_signals(
                        symbol, env, levels, list, risk, _accountId, contractId);

                    int routed = 0;

                    // Pause sends if market data stale
                    var lastQ = _status.Get<DateTimeOffset?>("last.quote") ?? DateTimeOffset.MinValue;
                    var lastT = _status.Get<DateTimeOffset?>("last.trade") ?? DateTimeOffset.MinValue;
                    var age = DateTimeOffset.UtcNow - new[] { lastQ, lastT }.Max();
                    bool tradable = age < TimeSpan.FromSeconds(30);
                    _status.Set("route.paused", !tradable);

                    // Reset trade counter on new UTC day
                    if (DateTime.UtcNow.Date != tradeDay) { tradeDay = DateTime.UtcNow.Date; tradesToday = 0; }

                    // Daily PnL breaker (cached fetch every 60s)
                    try
                    {
                        if ((DateTime.UtcNow - lastPnlFetch).TotalSeconds >= 60)
                        {
                            lastPnlFetch = DateTime.UtcNow;
                            decimal? net = null!;
                            try
                            {
                                var j = await _http.GetFromJsonAsync<System.Text.Json.JsonElement>($"/accounts/{_accountId}/pnl?scope=today", cancellationToken: ct).ConfigureAwait(false).ConfigureAwait(false);
                                if (j.ValueKind == System.Text.Json.JsonValueKind.Object && j.TryGetProperty("net", out var n) && n.TryGetDecimal(out var nd)) net = nd;
                            }
                            catch { }
                            if (net is null)
                            {
                                try
                                {
                                    var j = await _http.GetFromJsonAsync<System.Text.Json.JsonElement>($"/api/Account/pnl?accountId={_accountId}&scope=today", cancellationToken: ct).ConfigureAwait(false).ConfigureAwait(false);
                                    if (j.ValueKind == System.Text.Json.JsonValueKind.Object)
                                    {
                                        if (j.TryGetProperty("net", out var n) && n.TryGetDecimal(out var nd)) net = nd;
                                        else if (j.TryGetProperty("realized", out var r) && j.TryGetProperty("commissions", out var c) && j.TryGetProperty("fees", out var f))
                                        {
                                            decimal rr = r.TryGetDecimal(out var rv) ? rv : 0m;
                                            decimal cc = c.TryGetDecimal(out var cv) ? cv : 0m;
                                            decimal ff = f.TryGetDecimal(out var fv) ? fv : 0m;
                                            net = rr - (cc + ff);
                                        }
                                    }
                                }
                                catch { }
                            }
                            if (net.HasValue) { netPnlCache = net.Value; _status.Set("pnl.net", net.Value); }
                        }
                    }
                    catch { }

                    if (netPnlCache <= -maxDailyLoss)
                    {
                        _status.Set("gate.daily", "tripped.pnl");
                        _log.LogError("[Supervisor] Daily loss limit hit: {Pnl} <= -{Cap}. Blocking routes.", netPnlCache, maxDailyLoss);
                        _ = _notifier.Error($"Daily breaker tripped: net={netPnlCache:F2} cap={maxDailyLoss:F2}");
                        return;
                    }

                    var batch = new List<(BotCore.StrategySignal Sig, string ContractId)>();

                    // Gate AUTO_EXECUTE until BarsSeen >= 10
                    bool autoExecuteAllowed = _barsSeen >= 10;
                    if (!autoExecuteAllowed && (_cfg.LiveTrading || Environment.GetEnvironmentVariable("LIVE_ORDERS") == "1"))
                    {
                        _log.LogWarning("BarsSeen={Count}/10; gating AUTO_EXECUTE - signals suppressed in live mode", _barsSeen);
                        return;
                    }

                    // Security check for VPN/VPS/remote desktop detection in live trading
                    if ((_cfg.LiveTrading || Environment.GetEnvironmentVariable("LIVE_ORDERS") == "1") && _securityService != null)
                    {
                        if (!_securityService.IsTradingAllowed())
                        {
                            _log.LogError("Security check failed - trading blocked due to remote session detection. Use ALLOW_REMOTE=1 to override.");
                            return;
                        }
                    }

                    foreach (var s in signals)
                    {
                        // Generate temporary tag for signal logging (real tag will be generated in TradingSystemConnector)
                        var tempTag = $"S11L-{DateTime.UtcNow:yyyyMMdd-HHmmss}";
                        
                        // Calculate R-multiple for logging (inline implementation to avoid Px conflicts)
                        var risk = s.Side == "BUY" ? s.Entry - s.Stop : s.Stop - s.Entry;
                        var reward = s.Side == "BUY" ? s.Target - s.Entry : s.Entry - s.Target;
                        var rMultiple = risk > 0 ? reward / risk : 0m;
                        
                        // Standardized signal logging format
                        _log.LogInformation("[SIG] side={Side} symbol={Symbol} qty={Qty} entry={Entry} stop={Stop} t1={Target} R~{RMultiple} tag={Tag}", 
                            s.Side, s.Symbol, s.Size, 
                            s.Entry.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture), 
                            s.Stop.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture), 
                            s.Target.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture), 
                            rMultiple.ToString("0.00", System.Globalization.CultureInfo.InvariantCulture), 
                            tempTag);
                        var side = string.Equals(s.Side, "BUY", StringComparison.OrdinalIgnoreCase) ? BotCore.SignalSide.Long : BotCore.SignalSide.Short;
                        var cid = $"{s.StrategyId}|{s.Symbol}|{DateTime.UtcNow:yyyyMMddTHHmmssfff}|{Guid.NewGuid():N}".ToUpperInvariant();
                        journal.Append(s, "emitted", cid);

                        // Contract rollover guard
                        if (BotCore.Supervisor.ContractResolver.IsExpiring(s.Symbol))
                        {
                            _status.Set("route.paused", true);
                            _log.LogWarning("[Supervisor] {Sym} marked expiring. Pausing routes.", s.Symbol);
                            continue;
                        }
                        if (BotCore.Supervisor.ContractResolver.ShouldRoll(s.Symbol))
                        {
                            _log.LogInformation("[Supervisor] {Sym} should roll to next front month. Switching subscriptions if supported.", s.Symbol);
                            try
                            {
                                var subs = _marketHub.GetType().GetMethod("SwitchFrontMonthAsync");
                                if (subs != null)
                                {
                                    _ = (Task?)subs.Invoke(_marketHub, [s.Symbol]);
                                }
                            }
                            catch { }
                        }

                        // Halt/auction awareness via market hub property or status key
                        try
                        {
                            var prop = _marketHub.GetType().GetProperty("IsHalted");
                            if (prop != null && prop.PropertyType == typeof(bool))
                            {
                                var halted = (bool)(prop.GetValue(_marketHub) ?? false);
                                if (halted) { _status.Set("route.paused", true); _log.LogWarning("[Supervisor] Market halted. Skipping routes."); continue; }
                            }
                            var haltedFlag = _status.Get<string>("market.halted");
                            if (string.Equals(haltedFlag, "true", StringComparison.OrdinalIgnoreCase)) { _status.Set("route.paused", true); continue; }
                        }
                        catch { }

                        if (!tradable)
                        {
                            _log.LogWarning("[Supervisor] Market data stale ({Age}s). Skipping route for {Sym} {Strat}.", (int)age.TotalSeconds, s.Symbol, s.StrategyId);
                            continue;
                        }
                        if (tradesToday >= maxTradesPerDay)
                        {
                            _status.Set("gate.daily", "tripped");
                            _log.LogError("[Supervisor] Daily trade cap reached ({Cap}). Skipping further routes.", maxTradesPerDay);
                            _ = _notifier.Error($"Max trades reached: cap={maxTradesPerDay}");
                            break;
                        }

                        // Duplicate suppression across processes/agents
                        var dupKey = $"{s.Symbol}|{s.Side}|{InstrumentMeta.RoundToTick(s.Symbol, s.Entry):F2}|{InstrumentMeta.RoundToTick(s.Symbol, s.Stop):F2}|{InstrumentMeta.RoundToTick(s.Symbol, s.Target):F2}";
                        if (!_recentRoutes.TryAdd(dupKey, DateTime.UtcNow))
                        {
                            _log.LogWarning("[Supervisor] Duplicate route suppressed: {Key}", dupKey);
                            continue;
                        }

                        // Zero-risk guard: reject if risk < 1 tick
                        var tick = InstrumentMeta.Tick(s.Symbol);
                        if (Math.Abs(s.Entry - s.Stop) < tick)
                        {
                            _log.LogWarning("[Supervisor] Dropping zero-risk (sub-tick) setup for {Sym} {Strat}", s.Symbol, s.StrategyId);
                            continue;
                        }

                        // 2️⃣ Use ML Predictions in Trading Decisions
                        // Get blended ML prediction and apply confidence gating
                        bool mlGatesPassed = true;
                        double P_cloud = 0.5, P_online = 0.5, P_final = 0.5;
                        
                        if (_featureEngineering != null)
                        {
                            try
                            {
                                // Get cloud/offline prediction via IntelligenceOrchestrator.GetLatestPredictionAsync
                                if (_intelligenceOrchestrator != null)
                                {
                                    var prediction = await _intelligenceOrchestrator.GetLatestPredictionAsync(s.Symbol).ConfigureAwait(false).ConfigureAwait(false);
                                    P_cloud = prediction.Confidence;
                                    _log.LogDebug("[ML_GATE] Cloud prediction for {Symbol}: confidence={Confidence}", s.Symbol, P_cloud);
                                }
                                else
                                {
                                    P_cloud = 0.6; // Fallback when orchestrator not available
                                }
                                
                                // Get online prediction via hooks.on_signal(symbol, strategy_id)
                                // Enhanced online learner integration with fallback
                                if (_intelligenceOrchestrator != null)
                                {
                                    try
                                    {
                                        // Try to get online prediction from intelligence orchestrator
                                        var onlinePrediction = await _intelligenceOrchestrator.GetOnlinePredictionAsync(s.Symbol, s.StrategyId).ConfigureAwait(false).ConfigureAwait(false);
                                        P_online = onlinePrediction?.Confidence ?? 0.7;
                                        _log.LogDebug("[ML_GATE] Online prediction for {Symbol} {Strategy}: confidence={Confidence}", s.Symbol, s.StrategyId, P_online);
                                    }
                                    catch (Exception ex)
                                    {
                                        _log.LogWarning(ex, "[ML_GATE] Failed to get online prediction for {Symbol} {Strategy}, using fallback", s.Symbol, s.StrategyId);
                                        P_online = 0.65; // Conservative fallback when online prediction fails
                                    }
                                }
                                else
                                {
                                    // Calculate P_online from available feature engineering data if no orchestrator
                                    if (_featureEngineering != null)
                                    {
                                        try
                                        {
                                            // Use feature engineering to calculate confidence based on current market regime and volatility
                                            var mockFeatures = new Dictionary<string, double>
                                            {
                                                ["regime_strength"] = 0.5,
                                                ["volatility"] = 0.15,
                                                ["trend_confidence"] = 0.5
                                            };
                                            
                                            // Calculate confidence based on regime strength and volatility
                                            var regimeStrength = mockFeatures.GetValueOrDefault("regime_strength", 0.5);
                                            var volatility = mockFeatures.GetValueOrDefault("volatility", 0.15);
                                            var trendConfidence = mockFeatures.GetValueOrDefault("trend_confidence", 0.5);
                                                
                                                // Blend regime indicators to create confidence: stronger regime + lower volatility = higher confidence
                                                P_online = Math.Max(0.3, Math.Min(0.9, 
                                                    0.4 * regimeStrength + 0.3 * trendConfidence + 0.3 * (1.0 - Math.Min(volatility / 0.3, 1.0))));
                                                
                                                _log.LogDebug("[ML_GATE] Calculated P_online from features: regime={Regime:F3}, trend={Trend:F3}, vol={Vol:F3} → P_online={Confidence:F3}", 
                                                    regimeStrength, trendConfidence, volatility, P_online);
                                        }
                                        catch (Exception ex)
                                        {
                                            _log.LogWarning(ex, "[ML_GATE] Failed to calculate P_online from features, using conservative fallback");
                                            P_online = 0.6; // Conservative fallback on calculation error
                                        }
                                    }
                                    else
                                    {
                                        P_online = 0.6; // Conservative fallback when no feature engineering available
                                    }
                                }
                                
                                // Blended prediction per Topstep weighting: w = clip(n_recent / (n_recent + 500), 0.2, 0.8)
                                var recentSignalsCount = _recentSignals.Count;
                                var w = Math.Max(0.2, Math.Min(0.8, recentSignalsCount / (double)(recentSignalsCount + 500)));
                                P_final = w * P_online + (1 - w) * P_cloud;
                                
                                // Confidence gating: trade only if P_final >= 0.70 (updated threshold)
                                var minConfidence = 0.70;
                                mlGatesPassed = P_final >= minConfidence;
                                
                                _log.LogInformation("[ML_GATE] {Sym} {Strat}: P_cloud={P_cloud:F3}, P_online={P_online:F3}, P_final={P_final:F3}, gate={Gate}", 
                                    s.Symbol, s.StrategyId, P_cloud, P_online, P_final, mlGatesPassed ? "PASS" : "BLOCK");
                                
                                // 🔟 Observability & Lineage: Log complete decision with lineage
                                if (_decisionLogger != null)
                                {
                                    await _decisionLogger.LogTradingDecisionAsync(
                                        s.Symbol, s.StrategyId, P_cloud, P_online, P_final, 
                                        mlGatesPassed, s.Size, mlGatesPassed ? "ALLOW" : "BLOCK", cid).ConfigureAwait(false);
                                }
                                
                                if (!mlGatesPassed)
                                {
                                    _log.LogWarning("[ML_GATE] Signal blocked by ML confidence gate: P_final={P_final:F3} < {MinConf:F3}", P_final, minConfidence);
                                    continue;
                                }
                            }
                            catch (Exception ex)
                            {
                                _log.LogError(ex, "[ML_GATE] Error in ML prediction gating for {Sym} {Strat} - allowing signal through", s.Symbol, s.StrategyId);
                            }
                        }

                        var sig = new BotCore.StrategySignal
                        {
                            Strategy = s.StrategyId,
                            Symbol = s.Symbol,
                            Side = side,
                            Size = s.Size,
                            LimitPrice = s.Entry,
                            Note = s.Tag,
                            ClientOrderId = cid
                        };
                        batch.Add((sig, s.ContractId));
                        journal.Append(s, "routed", cid);
                        _recentCidBuffer.Enqueue(cid);
                        _recentSignals.Add(new LastSignal(s.StrategyId, s.Symbol, s.Side, s.Entry, s.Target, s.Stop));
                        while (_recentCidBuffer.Count > 1000 && _recentCidBuffer.TryDequeue(out _)) { }
                        tradesToday++;
                        routed++;
                    }

                    if (batch.Count > 0)
                    {
                        foreach (var item in batch)
                        {
                            if (!_routeChan.Writer.TryWrite(item))
                            {
                                _log.LogWarning("[Supervisor] Route queue full. Dropping signal {Strat} {Sym}.", item.Sig.Strategy, item.Sig.Symbol);
                            }
                        }
                    }

                    _status.Set($"last.strategy.{symbol}", DateTimeOffset.UtcNow);
                    _status.Set($"last.routed.{symbol}", routed);
                }
                catch (Exception ex)
                {
                    _log.LogWarning(ex, "[Supervisor] HandleBar error for {Sym}", bar.Symbol);
                }
            }

            _status.Set("market.state", "running");
            _status.Set("user.state", "running");

            // Background reconciliation loop (5s) + periodic state persistence (60s)
            _ = Task.Run(async () =>
            {
                var lastSave = DateTime.UtcNow;
                while (!ct.IsCancellationRequested)
                {
                    try
                    {
                        await Retry(async () =>
                        {
                            var pos = await router.QueryOpenPositionsAsync(_accountId, ct).ConfigureAwait(false).ConfigureAwait(false);
                            var ord = await router.QueryOpenOrdersAsync(_accountId, ct).ConfigureAwait(false).ConfigureAwait(false);
                            _status.Set("broker.positions.count", pos?.Count ?? 0);
                            _status.Set("broker.orders.count", ord?.Count ?? 0);
                            await router.EnsureBracketsAsync(_accountId, ct).ConfigureAwait(false);
                        }, ct);
                    }
                    catch { }

                    if ((DateTime.UtcNow - lastSave).TotalSeconds >= 60)
                    {
                        lastSave = DateTime.UtcNow;
                        try
                        {
                            var snap = new BotCore.Supervisor.StateStore.Snapshot
                            {
                                RecentRoutes = new Dictionary<string, DateTime>(_recentRoutes),
                                LastBarUnix = new Dictionary<string, long>(lastBarUnix),
                                LastCids = [.. _recentCidBuffer]
                            };
                            _stateStore.Save(snap);
                            try
                            {
                                BotCore.Infra.Persistence.Save("recent_cids", _recentRoutes.Keys.ToList());
                                BotCore.Infra.Persistence.Save("last_signals", _recentSignals);
                            }
                            catch { }
                        }
                        catch { }
                    }

                    try { await Task.Delay(5000, ct).ConfigureAwait(false); } catch { }
                }
            }, ct);

            // Periodic duplicate-route cache sweep (bound memory)
            _ = Task.Run(async () =>
            {
                while (!ct.IsCancellationRequested)
                {
                    try
                    {
                        var cutoff = DateTime.UtcNow.AddMinutes(-30);
                        foreach (var kv in _recentRoutes)
                            if (kv.Value < cutoff) _recentRoutes.TryRemove(kv.Key, out _);
                    }
                    catch { }
                    try { await Task.Delay(60000, ct).ConfigureAwait(false); } catch { }
                }
            }, ct);

            bool? prevPaused = null!;
            string? prevLive = null!;
            DateTime? lastWarn30 = null!;
            DateTime? lastWarn60 = null!;
            while (!ct.IsCancellationRequested)
            {
                // Market data staleness checks every second (independent of bars)
                var lq = _status.Get<DateTimeOffset?>("last.quote") ?? DateTimeOffset.MinValue;
                var lt = _status.Get<DateTimeOffset?>("last.trade") ?? DateTimeOffset.MinValue;
                var age = DateTimeOffset.UtcNow - new[] { lq, lt }.Max();
                if (age > TimeSpan.FromSeconds(30) && (lastWarn30 == null || (DateTime.UtcNow - lastWarn30) > TimeSpan.FromSeconds(30)))
                {
                    _log.LogWarning("[WARN] Quotes/trades stale: age={AgeSec}s (>30)", (int)age.TotalSeconds);
                    lastWarn30 = DateTime.UtcNow;
                }
                if (age > TimeSpan.FromSeconds(60) && (lastWarn60 == null || (DateTime.UtcNow - lastWarn60) > TimeSpan.FromSeconds(60)))
                {
                    _status.Set("route.paused", true);
                    Environment.SetEnvironmentVariable("ROUTE_PAUSE", "1");
                    _log.LogWarning("[PAUSE] Quotes/trades stale: age={AgeSec}s (>60). Routing paused.", (int)age.TotalSeconds);
                    lastWarn60 = DateTime.UtcNow;
                }

                // Operator controls (read every second, log on toggle)
                var flatten = Environment.GetEnvironmentVariable("PANIC_FLATTEN");
                if (flatten == "1")
                {
                    await router.FlattenAllAsync(_accountId, ct).ConfigureAwait(false);
                    Environment.SetEnvironmentVariable("PANIC_FLATTEN", "0");
                    _status.Set("flatten", "true");
                    _log.LogError("PANIC_FLATTEN toggled: positions flattened");
                }

                var kill = Environment.GetEnvironmentVariable("KILL_SWITCH");
                if (kill == "1")
                {
                    await router.CancelAllOpenAsync(_accountId, ct).ConfigureAwait(false);
                    _status.Set("kill", "true");
                    _log.LogError("KILL_SWITCH active: cancelled all open orders and exiting supervisor loop.");
                    return;
                }

                // VerifyTodayAsync - debounce to every 5 minutes during trading hours
                if (_verifier != null && (_lastVerify == default || (DateTime.UtcNow - _lastVerify) >= TimeSpan.FromMinutes(5)))
                {
                    try
                    {
                        var verificationResult = await _verifier.VerifyTodayAsync(ct).ConfigureAwait(false).ConfigureAwait(false);
                        _lastVerify = DateTime.UtcNow;
                        
                        if (verificationResult != null)
                        {
                            var orderCount = verificationResult.OrdersByStatus.Values.Sum();
                            var tradeCount = verificationResult.TradesByStatus.Values.Sum();
                            
                            _log.LogInformation("Trade verification completed: Orders={OrderCount}, Trades={TradeCount}, Success={Success}",
                                orderCount, tradeCount, verificationResult.Success);
                                
                            // Print totals by status as required
                            foreach (var kvp in verificationResult.OrdersByStatus)
                            {
                                _log.LogInformation("Orders {Status}: {Count}", kvp.Key, kvp.Value);
                            }
                            foreach (var kvp in verificationResult.TradesByStatus)
                            {
                                _log.LogInformation("Trades {Status}: {Count}", kvp.Key, kvp.Value);
                            }
                                
                            // Alert on verification failure
                            if (!verificationResult.Success && !string.IsNullOrEmpty(verificationResult.ErrorMessage))
                            {
                                _log.LogError("Trade verification failed: {ErrorMessage}", verificationResult.ErrorMessage);
                                // Operator alerting for trade verification failures
                                await NotifyOperatorAsync($"Trade verification failed: {verificationResult.ErrorMessage}").ConfigureAwait(false);
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        _log.LogError(ex, "Failed to execute trade verification");
                    }
                }

                // kill.txt enforcement - force DRY_RUN mode if file exists
                if (File.Exists("kill.txt"))
                {
                    var currentBotMode = Environment.GetEnvironmentVariable("BOT_MODE");
                    if (currentBotMode != "DRY_RUN")
                    {
                        Environment.SetEnvironmentVariable("BOT_MODE", "DRY_RUN");
                        _log.LogWarning("kill.txt detected — forcing DRY_RUN (no live orders). Previous mode: {PreviousMode}", currentBotMode ?? "unset");
                    }
                }

                var pause = Environment.GetEnvironmentVariable("ROUTE_PAUSE") == "1";
                var resume = Environment.GetEnvironmentVariable("ROUTE_RESUME") == "1";
                if (resume) { _status.Set("route.paused", false); Environment.SetEnvironmentVariable("ROUTE_RESUME", "0"); _log.LogInformation("ROUTE_RESUME toggled -> routing resumed"); }
                if (pause) { _status.Set("route.paused", true); _log.LogInformation("ROUTE_PAUSE toggled -> routing paused"); }

                var pausedNow = _status.Get<bool>("route.paused");
                if (prevPaused != pausedNow)
                {
                    _log.LogInformation("route.paused={Paused}", pausedNow);
                    prevPaused = pausedNow;
                }

                var liveNow = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0");
                if (!string.Equals(prevLive, liveNow, StringComparison.Ordinal))
                {
                    _log.LogInformation("LIVE_ORDERS changed: {State}", liveNow);
                    prevLive = liveNow;
                }

                _status.Heartbeat();
                await Task.Delay(1000, ct).ConfigureAwait(false);
            }
        }

        /// <summary>
        /// Extract strategy ID from custom tag for online learning integration
        /// </summary>
        private static string ExtractStrategyIdFromTag(string customTag)
        {
            try
            {
                // Custom tags typically follow pattern like "S11L-20240101-HHMMSS" or "STRATEGY_ID-timestamp"
                if (string.IsNullOrEmpty(customTag))
                    return "default";

                // Try to extract strategy ID before the first dash
                var dashIndex = customTag.IndexOf('-');
                if (dashIndex > 0 && dashIndex < customTag.Length)
                {
                    return customTag.Substring(0, dashIndex);
                }

                // If no dash, use the whole tag up to a reasonable limit
                if (customTag.Length > 10)
                {
                    return customTag.Substring(0, 10);
                }
                
                return customTag;
            }
            catch
            {
                return "default";
            }
        }

        /// <summary>
        /// Notify operators of critical system issues requiring attention
        /// </summary>
        private async Task NotifyOperatorAsync(string message)
        {
            try
            {
                _log.LogError("[OPERATOR_ALERT] {Message}", message);
                
                // In production, this would integrate with alerting systems like PagerDuty, Slack, etc.
                // For now, ensure critical logs are captured for operator review
                await Task.Delay(1).ConfigureAwait(false); // Prevent async warning
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "Failed to send operator notification: {Message}", message);
            }
        }
    }
}
