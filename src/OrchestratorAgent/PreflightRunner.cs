using System;
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;

namespace OrchestratorAgent
{
    internal sealed class PreflightRunner
    {
        private readonly ILogger _log;
        private readonly HubConnection _userHub;
        private readonly HubConnection _marketHub;
        private readonly string _primaryCid;
        private Func<Task<bool>>? _orderTest;

        public PreflightRunner(ILogger log, HubConnection userHub, HubConnection marketHub, string primaryCid)
        {
            _log = log;
            _userHub = userHub;
            _marketHub = marketHub;
            _primaryCid = primaryCid;
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
            if (_userHub.State != HubConnectionState.Connected || _marketHub.State != HubConnectionState.Connected)
            {
                _log.LogError("PRECHECK FAIL: hub(s) not connected");
                return false;
            }

            _log.LogInformation("PRECHECK: waiting for first quote/trade on {Cid} (<= {Sec}s)…", _primaryCid, (int)timeout.TotalSeconds);
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
                _log.LogError("PRECHECK FAIL: no market data");
                return false;
            }
            _log.LogInformation("PRECHECK PASS: market data OK.");

            if (_orderTest != null)
            {
                var ok = await _orderTest();
                if (!ok) { _log.LogError("PRECHECK FAIL: order test failed"); return false; }
                _log.LogInformation("PRECHECK PASS: order stream OK.");
            }
            return true;
        }
    }
}