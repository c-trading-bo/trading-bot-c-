using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace OrchestratorAgent.Ops
{
    internal sealed class PartialExitService(HttpClient http, Func<Task<string?>> getJwtAsync, ILogger log)
    {
        private readonly HttpClient _http = http; private readonly Func<Task<string?>> _getJwtAsync = getJwtAsync; private readonly ILogger _log = log;

        public async Task TryScaleOutAsync(string symbol, string parentId, int parentQty, decimal parentAvg, bool isLong, int tp1Ticks, int tp1Qty, CancellationToken ct)
        {
            try
            {
                if (tp1Ticks <= 0 || tp1Qty <= 0) return;
                var tick = BotCore.Models.InstrumentMeta.Tick(symbol);
                if (tick <= 0) tick = 0.25m;
                var px = isLong ? parentAvg + tp1Ticks * tick : parentAvg - tp1Ticks * tick;

                var token = await _getJwtAsync().ConfigureAwait(false); if (string.IsNullOrWhiteSpace(token)) { _log.LogWarning("[TP1] Skipped: missing JWT"); return; }

                // Reduce-only IOC limit order at TP1 price. Payload fields may vary across tenants; we try primary then legacy.
                var payload = new { parentId, side = isLong ? 1 : 0, type = "LIMIT", timeInForce = "IOC", quantity = tp1Qty, price = px, reduceOnly = true };
                using var req = new HttpRequestMessage(HttpMethod.Post, "/orders");
                req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                req.Content = new StringContent(JsonSerializer.Serialize(payload), Encoding.UTF8, "application/json");
                using var resp = await _http.SendAsync(req, ct).ConfigureAwait(false);
                if (resp.IsSuccessStatusCode)
                {
                    _log.LogInformation("[TP1] reduce-only OK sym={Sym} qty={Qty} px={Px}", symbol, tp1Qty, px);
                    return;
                }
                // Fallback legacy endpoint
                using var req2 = new HttpRequestMessage(HttpMethod.Post, "/api/Order/place");
                req2.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                var legacy = new { accountId = (long?)null, contractId = (string?)null, type = 1, side = isLong ? 1 : 0, size = tp1Qty, limitPrice = px, customTag = $"TP1-{parentId}", reduceOnly = true, parentId };
                req2.Content = new StringContent(JsonSerializer.Serialize(legacy), Encoding.UTF8, "application/json");
                using var resp2 = await _http.SendAsync(req2, ct).ConfigureAwait(false);
                _log.LogInformation("[TP1] legacy post {Ok} sym={Sym} qty={Qty} px={Px}", resp2.IsSuccessStatusCode, symbol, tp1Qty, px);
            }
            catch (OperationCanceledException) { }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[TP1] reduce-only scale-out failed/suppressed");
            }
        }
    }
}
