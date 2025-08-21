using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace BotCore
{
    public sealed class ApiClient
    {
        private readonly HttpClient _http;
        private readonly ILogger<ApiClient> _log;
        private string? _jwt;

        public ApiClient(HttpClient http, ILogger<ApiClient> log, string apiBase)
        {
            _http = http;
            _log = log;
            _http.BaseAddress = new Uri(apiBase);
        }

        public void SetJwt(string jwt)
        {
            _jwt = jwt;
            _http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt);
        }

        // Contract search: returns contractId for symbol (e.g. ES, NQ)
        public async Task<string> ResolveContractIdAsync(string symbol, CancellationToken ct)
        {
            // Try /Contract/available { live:false, symbol } first
            var req = new { symbol, live = false };
            string reqJson = System.Text.Json.JsonSerializer.Serialize(req);
            _log.LogInformation("Contract available request: {Req}", reqJson);
            try
            {
                var resp = await _http.PostAsJsonAsync("/api/Contract/available", req, ct);
                var body = await resp.Content.ReadAsStringAsync(ct);
                if (!resp.IsSuccessStatusCode)
                {
                    _log.LogWarning("Contract available {Status} (symbol='{Symbol}'): {Body}", (int)resp.StatusCode, symbol, body);
                    _log.LogWarning("Request payload: {Req}", reqJson);
                }
                var json = await resp.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct);
                if (json.TryGetProperty("data", out var data) && data.ValueKind == JsonValueKind.Array && data.GetArrayLength() > 0)
                {
                    var first = data[0];
                    if (first.TryGetProperty("contractId", out var cid))
                        return cid.GetString() ?? "";
                }
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "Contract search failed for {Symbol}", symbol);
            }
            // Fallback: search endpoint with correct property names
            foreach (var live in new[] { false, true })
            {
                var fallbackReq = new { searchText = symbol, live };
                string fallbackReqJson = System.Text.Json.JsonSerializer.Serialize(fallbackReq);
                _log.LogInformation("Contract search fallback request: {Req}", fallbackReqJson);
                var fallbackResp = await _http.PostAsJsonAsync("/api/Contract/search", fallbackReq, ct);
                var fallbackBody = await fallbackResp.Content.ReadAsStringAsync(ct);
                if (!fallbackResp.IsSuccessStatusCode)
                {
                    _log.LogWarning("Contract search fallback {Status} (symbol='{Symbol}'): {Body}", (int)fallbackResp.StatusCode, symbol, fallbackBody);
                    _log.LogWarning("Request payload: {Req}", fallbackReqJson);
                    continue;
                }
                var fallbackJson = await fallbackResp.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct);
                if (fallbackJson.TryGetProperty("data", out var fdata) && fdata.ValueKind == JsonValueKind.Array && fdata.GetArrayLength() > 0)
                {
                    var first = fdata[0];
                    if (first.TryGetProperty("contractId", out var cid))
                        return cid.GetString() ?? "";
                }
            }
            throw new InvalidOperationException($"No contractId found for symbol: {symbol}");
        }

        // Example: Place limit order (used in orchestration demo)
        public async Task<string> PlaceLimit(int accountId, string contractId, int sideBuy0Sell1, int size, decimal limitPrice, CancellationToken ct)
        {
            var req = new
            {
                accountId,
                contractId,
                side = sideBuy0Sell1,
                size,
                limitPrice,
                customTag = $"DEMO-{DateTime.UtcNow:yyyyMMdd-HHmmss}"
            };
            var resp = await _http.PostAsJsonAsync("/api/Order/place", req, ct);
            resp.EnsureSuccessStatusCode();
            var json = await resp.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct);
            if (json.TryGetProperty("orderId", out var id))
                return id.GetString() ?? "";
            throw new InvalidOperationException("No orderId returned from place order.");
        }
    }
}
