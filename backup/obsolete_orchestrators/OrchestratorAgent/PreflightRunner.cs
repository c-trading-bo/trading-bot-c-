using System;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;

namespace OrchestratorAgent
{
    internal sealed class PreflightRunner(ILogger log, HubConnection userHub, HubConnection marketHub, string primaryCid)
    {
        private readonly ILogger _log = log;
        private readonly HubConnection _userHub = userHub;
        private readonly HubConnection _marketHub = marketHub;
        private readonly string _primaryCid = primaryCid;
        private Func<Task<bool>>? _orderTest;

        private readonly List<string> _steps = [];
        private readonly List<string> _fails = [];
        public IReadOnlyList<string> Steps => _steps;
        public IReadOnlyList<string> FailReasons => _fails;

        private void Step(string msg)
        {
            _steps.Add(msg);
            _log.LogInformation("[Precheck] {Msg}", msg);
        }

        private void StepOk(string msg)
        {
            _steps.Add($"OK: {msg}");
            _log.LogInformation("[Precheck] ✅ {Msg}", msg);
        }

        private void StepFail(string msg)
        {
            _fails.Add(msg);
            _steps.Add($"FAIL: {msg}");
            _log.LogError("[Precheck] ❌ {Msg}", msg);
        }

        public PreflightRunner WithOptionalOrderTest(HttpClient http, Func<Task<string?>> tokenProvider, long accountId)
        {
            _orderTest = async () =>
            {
                // Place then cancel using existing smoke tester and verify canceled seen on user stream
                var tcs = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
                void onOrder(JsonElement e)
                {
                    try
                    {
                        if (e.TryGetProperty("status", out var st))
                        {
                            var s = st.GetString();
                            if (string.Equals(s, "Canceled", StringComparison.OrdinalIgnoreCase))
                                tcs.TrySetResult(true);
                        }
                    }
                    catch { }
                }

                // subscribe before placing
                _userHub.On<JsonElement>("GatewayUserOrder", onOrder);
                try
                {
                    Step("Order test: placing far-away limit then canceling (expect Canceled event)…");
                    await OrderSmokeTester.RunAsync(http, tokenProvider, accountId, _primaryCid, _log, CancellationToken.None);
                    // Wait up to 3s for canceled event
                    using var waitCts = new CancellationTokenSource(TimeSpan.FromSeconds(3));
                    try { await Task.WhenAny(tcs.Task, Task.Delay(Timeout.Infinite, waitCts.Token)); } catch { }
                    return tcs.Task.IsCompleted && tcs.Task.Result;
                }
                finally
                {
                    // remove handler
                    try { _userHub.Remove("GatewayUserOrder"); } catch { }
                }
            };
            return this;
        }

        public async Task<bool> RunAsync(TimeSpan timeout, CancellationToken ct)
        {
            Step("Checklist start (DRY-RUN): validating hubs and data flow…");

            // Step 1: Hubs connected
            var userConnected = _userHub.State == HubConnectionState.Connected;
            var marketConnected = _marketHub.State == HubConnectionState.Connected;
            if (!userConnected || !marketConnected)
            {
                StepFail($"Hubs not connected. UserHub={_userHub.State}, MarketHub={_marketHub.State}");
                return false;
            }
            StepOk($"Hubs connected. UserHubId={_userHub.ConnectionId ?? ""}, MarketHubId={_marketHub.ConnectionId ?? ""}");

            // Step 2: Market data first tick
            Step($"Waiting for first quote/trade on {_primaryCid} (<= {(int)timeout.TotalSeconds}s)…");
            var tcs = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            void qh(string cid, JsonElement _) { if (cid == _primaryCid) tcs.TrySetResult(true); }
            void th(string cid, JsonElement _) { if (cid == _primaryCid) tcs.TrySetResult(true); }

            _marketHub.On<string, JsonElement>("GatewayQuote", qh);
            _marketHub.On<string, JsonElement>("GatewayTrade", th);

            using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
            cts.CancelAfter(timeout);
            try { await Task.WhenAny(tcs.Task, Task.Delay(Timeout.Infinite, cts.Token)); } catch { }

            try { _marketHub.Remove("GatewayQuote"); } catch { }
            try { _marketHub.Remove("GatewayTrade"); } catch { }

            if (!tcs.Task.IsCompleted)
            {
                StepFail($"No market data received for {_primaryCid} within {(int)timeout.TotalSeconds}s");
                return false;
            }
            StepOk("Market data OK (tick received)");

            // Step 3: Optional order stream test
            if (_orderTest != null)
            {
                Step("Running order stream test (place + cancel)…");
                var ok = await _orderTest();
                if (!ok)
                {
                    StepFail("Order stream did not emit expected Canceled event");
                    return false;
                }
                StepOk("Order stream OK (Canceled event observed)");
            }

            StepOk("Precheck PASS");
            return true;
        }
    }
}
