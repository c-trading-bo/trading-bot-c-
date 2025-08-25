using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using BotCore.Models;
using Microsoft.Extensions.Logging;

namespace OrchestratorAgent
{
    internal sealed class SimpleOrderRouter
    {
        private readonly HttpClient _http;
        private readonly Func<Task<string?>> _getJwtAsync;
        private readonly ILogger _log;
        private readonly bool _live;

        public SimpleOrderRouter(HttpClient http, Func<Task<string?>> getJwtAsync, ILogger log, bool live)
        {
            _http = http;
            _getJwtAsync = getJwtAsync;
            _log = log;
            _live = live;
        }

        public async Task<bool> RouteAsync(Signal sig, CancellationToken ct)
        {
            if (sig is null) return false;
            if (sig.AccountId <= 0 || string.IsNullOrWhiteSpace(sig.ContractId))
            {
                _log.LogWarning("[Router] Missing accountId/contractId for signal {Tag}; skipping.", sig.Tag);
                return false;
            }

            _log.LogInformation("[Router] {Mode} Route: {Side} {Size} {Contract} @ {Entry} (stop {Stop}, target {Target}) tag={Tag}",
                _live ? "LIVE" : "DRY-RUN", sig.Side, sig.Size, sig.ContractId, sig.Entry, sig.Stop, sig.Target, sig.Tag);

            if (!_live)
                return true;

            try
            {
                var token = await _getJwtAsync();
                if (string.IsNullOrWhiteSpace(token))
                {
                    _log.LogWarning("[Router] Missing JWT; cannot place order.");
                    return false;
                }

                var placeBody = new
                {
                    accountId = sig.AccountId,
                    contractId = sig.ContractId,
                    type = 1, // 1 = Limit
                    side = string.Equals(sig.Side, "SELL", StringComparison.OrdinalIgnoreCase) ? 1 : 0,
                    size = sig.Size > 0 ? sig.Size : 1,
                    limitPrice = sig.Entry,
                    customTag = sig.Tag
                };

                using var req = new HttpRequestMessage(HttpMethod.Post, "/api/Order/place");
                req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                req.Content = new StringContent(JsonSerializer.Serialize(placeBody), Encoding.UTF8, "application/json");

                using var resp = await _http.SendAsync(req, ct);
                var text = await resp.Content.ReadAsStringAsync(ct);
                if (!resp.IsSuccessStatusCode)
                {
                    _log.LogWarning("[Router] Place failed {Status}: {Body}", (int)resp.StatusCode, Trunc(text));
                    return false;
                }

                _log.LogInformation("[Router] Place OK: {Body}", Trunc(text));
                return true;
            }
            catch (OperationCanceledException)
            {
                return false;
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[Router] Unexpected error placing order.");
                return false;
            }
        }

        private static string Trunc(string? s, int max = 256)
        {
            if (string.IsNullOrEmpty(s)) return string.Empty;
            return s.Length <= max ? s : s.Substring(0, max) + "…";
        }
    }
}
