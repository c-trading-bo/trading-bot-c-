// Agent: SupervisorAgent
// Role: Monitors agent health, diagnostics, and system status.
// Integration: Used by orchestrator for health checks and diagnostics.
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

namespace SupervisorAgent
{
    public sealed class SupervisorAgent
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

        private readonly ILogger<SupervisorAgent> _log;
        private readonly HttpClient _http;
        private readonly string _apiBase;
        private readonly string _jwt;
        private readonly long _accountId;
        private readonly object _marketHub;
        private readonly object _userHub;
        private readonly StatusService _status;
        private readonly Config _cfg;

        public SupervisorAgent(ILogger<SupervisorAgent> log, HttpClient http, string apiBase, string jwt, long accountId, object marketHub, object userHub, StatusService status, Config cfg)
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

            void SafeAttach<T>(object target, string evt, Action<T> handler)
            {
                var e = target.GetType().GetEvent(evt);
                if (e == null) return;
                var del = Delegate.CreateDelegate(e.EventHandlerType!, handler.Target!, handler.Method);
                e.AddEventHandler(target, del);
            }

            bool TryOn<T>(object hub, string method, Action<T> handler)
            {
                var mi = hub.GetType()
                    .GetMethods()
                    .FirstOrDefault(m => m.Name == "On" && m.IsGenericMethodDefinition && m.GetParameters().Length == 2);
                if (mi == null) return false;
                mi.MakeGenericMethod(typeof(T)).Invoke(hub, new object[] { method, handler });
                return true;
            }

            SafeAttach<object>(_marketHub, "OnQuote", _ => _status.Set("last.quote", DateTimeOffset.UtcNow));
            SafeAttach<BotCore.Models.Bar>(_marketHub, "OnBar", _ => _status.Set("last.quote", DateTimeOffset.UtcNow));
            SafeAttach<object>(_userHub, "OnTrade", _ => _status.Set("last.trade", DateTimeOffset.UtcNow));

            TryOn<System.Text.Json.JsonElement>(_marketHub, "Quote", _ => _status.Set("last.quote", DateTimeOffset.UtcNow));
            TryOn<System.Text.Json.JsonElement>(_marketHub, "Trade", _ => _status.Set("last.trade", DateTimeOffset.UtcNow));

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
