#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using SupervisorAgent;
using BotCore;

namespace OrchestratorAgent
{
    public sealed class BotSupervisor
    {
        public sealed class Config
        {
            public bool LiveTrading { get; set; } = false;
            public int BarSeconds { get; set; } = 60;
            public string[] Symbols { get; set; } = Array.Empty<string>();
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

        private readonly ILogger<BotSupervisor> _log;
        private readonly HttpClient _http;
        private readonly string _apiBase;
        private readonly string _jwt;
        private readonly long _accountId;
        private readonly object _marketHub;
        private readonly object _userHub;
    private readonly StatusService _status;
        private readonly Config _cfg;

    public BotSupervisor(ILogger<BotSupervisor> log, HttpClient http, string apiBase, string jwt, long accountId, object marketHub, object userHub, StatusService status, Config cfg)
        {
            _log = log;
            _http = http;
            _apiBase = apiBase;
            _jwt = jwt;
            _accountId = accountId;
            _marketHub = marketHub;
            _userHub = userHub;
            _status = status;
            _cfg = cfg;
        }

        public async Task RunAsync(CancellationToken ct)
        {
            // TODO: Implement supervisor logic: subscribe to events, run strategies, route orders, emit status
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
                g.Invoke(hub, new object[] { method, handler });
                return true;
            }

            // track a heartbeat-able snapshot
            DateTimeOffset lastQuote = default, lastTrade = default, lastBar = default;
            _status.Set("strategies", "wired");

            // 1) Market events (works with either your MarketDataAgent events or raw HubConnection)
            SafeAttachEvent<object>(_marketHub, "OnQuote", _ => { lastQuote = DateTimeOffset.UtcNow; _status.Set("last.quote", lastQuote); });
            SafeAttachEvent<object>(_marketHub, "OnTrade", _ => { lastTrade = DateTimeOffset.UtcNow; _status.Set("last.trade", lastTrade); });
            SafeAttachEvent<BotCore.Models.Bar>(_marketHub, "OnBar", b => { lastBar = DateTimeOffset.UtcNow; _status.Set("last.quote", lastBar); HandleBar(b); });

            // If _marketHub is a HubConnection, wire SignalR “On” as well (harmless if not)
            TryWireSignalROn<System.Text.Json.JsonElement>(_marketHub, "Quote", _ => { lastQuote = DateTimeOffset.UtcNow; _status.Set("last.quote", lastQuote); });
            TryWireSignalROn<System.Text.Json.JsonElement>(_marketHub, "Trade", _ => { lastTrade = DateTimeOffset.UtcNow; _status.Set("last.trade", lastTrade); });

            // 2) User hub events (orders/trades confirmations)
            SafeAttachEvent<object>(_userHub, "OnOrder", _ => _status.Set("orders.open", "changed"));
            SafeAttachEvent<object>(_userHub, "OnTrade", _ => _status.Set("last.trade", DateTimeOffset.UtcNow));

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

            var router = new OrchestratorAgent.OrderRouter(
                orderLog ?? (ILogger<OrchestratorAgent.OrderRouter>)_log,
                _http, _apiBase, _jwt, (int)_accountId);

            void HandleBar(BotCore.Models.Bar bar)
            {
                // If you want to actually run strategies here, call your StrategyAgent and route:
                // var cfg = BotCore.Config.ConfigLoader.FromFile("src/BotCore/Config/high_win_rate_profile.json");
                // var strat = new StrategyAgent.StrategyAgent(cfg);
                // var risk  = new BotCore.Risk.RiskEngine(cfg); // if you have one
                // var signals = strat.GenerateSignals(bar.Symbol, new List<BotCore.Models.Bar> { bar }, risk);
                // foreach (var sig in signals) _ = router.PlaceAsync(sig, ct);
            }

            _status.Set("market.state", "running");
            _status.Set("user.state", "running");

            while (!ct.IsCancellationRequested)
            {
                _status.Heartbeat();
                await Task.Delay(1000, ct);
            }
        }
    }
}
