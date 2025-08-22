using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace BotCore
{
    public sealed class ApiClient
    {
    private readonly HttpClient _http;
    private readonly ILogger<ApiClient> _log;
    private readonly string _apiBase;
    private string? _jwt;

        public ApiClient(HttpClient http, ILogger<ApiClient> log, string apiBase)
        {
            _http = http;
            _log = log;
            _apiBase = apiBase;
            _http.BaseAddress = new Uri(apiBase);
        }

        public void SetJwt(string jwt)
        {
            _jwt = jwt;
            _http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt);
        }

        #nullable enable
        private sealed record AvailableReq(bool live);
        private sealed record ContractDto(string id, string name, string? description, string symbolId, bool activeContract);
        private sealed record AvailableResp(List<ContractDto>? contracts, bool success, int errorCode, string? errorMessage);

        private static readonly Dictionary<string,string> SymbolRootToSymbolId = new(StringComparer.OrdinalIgnoreCase)
        {
            ["ES"] = "F.US.EP",   // E-mini S&P 500
            ["NQ"] = "F.US.ENQ",  // E-mini NASDAQ-100
            // Micros if you need them:
            ["MES"] = "F.US.MES",
            ["MNQ"] = "F.US.MNQ",
        };

        public async Task<string> ResolveContractIdAsync(string root, CancellationToken ct = default)
        {
            // 1) Try AVAILABLE (eval = live:false)
            var id = await TryResolveViaAvailableAsync(root, live:false, ct);
            if (id is null)
                id = await TryResolveViaAvailableAsync(root, live:true, ct); // safety fallback

            // 2) Fallback to SEARCH if still nothing
            if (id is null)
                id = await TryResolveViaSearchAsync(root, live:false, ct) ?? await TryResolveViaSearchAsync(root, live:true, ct);

            if (string.IsNullOrWhiteSpace(id))
                throw new InvalidOperationException($"No contractId found for symbol: {root}");

            return id!;
        }

        private Uri U(string path) => new($"{_apiBase}{path}");

        private async Task<string?> TryResolveViaAvailableAsync(string root, bool live, CancellationToken ct)
        {
            SymbolRootToSymbolId.TryGetValue(root, out var wantedSymbolId);

            using var resp = await _http.PostAsJsonAsync(U("/api/Contract/available"), new AvailableReq(live), ct);
            var body = await resp.Content.ReadAsStringAsync(ct);
            if (!resp.IsSuccessStatusCode)
            {
                _log.LogWarning("Contract available {Status} (live={Live}): {Body}", (int)resp.StatusCode, live, body);
                return null;
            }

            var data = JsonSerializer.Deserialize<AvailableResp>(body, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            var list = data?.contracts ?? new();

            // Filter by symbolId if we know it; prefer active front-month (name like ES?U5 / NQ?U5)
            IEnumerable<ContractDto> pool = string.IsNullOrWhiteSpace(wantedSymbolId)
                ? list
                : list.Where(c => string.Equals(c.symbolId, wantedSymbolId, StringComparison.OrdinalIgnoreCase));

            var pick = pool
                .Where(c => c.activeContract)
                .OrderByDescending(c => c.name) // front month tends to sort later
                .FirstOrDefault() ?? pool.FirstOrDefault();

            if (pick is not null)
            {
                _log.LogInformation("Available pick (root={Root}, live={Live}) -> {Id} ({Name}) [{SymId}]",
                    root, live, pick.id, pick.name, pick.symbolId);
                return pick.id;
            }
            return null;
        }

        private sealed record SearchReq(string searchText, bool live);
        private sealed record SearchResp(List<ContractDto>? contracts, bool success, int errorCode, string? errorMessage);

        private async Task<string?> TryResolveViaSearchAsync(string searchText, bool live, CancellationToken ct)
        {
            using var resp = await _http.PostAsJsonAsync(U("/api/Contract/search"), new SearchReq(searchText, live), ct);
            var body = await resp.Content.ReadAsStringAsync(ct);
            if (!resp.IsSuccessStatusCode)
            {
                _log.LogWarning("Contract search {Status} (q='{Q}', live={Live}): {Body}", (int)resp.StatusCode, searchText, live, body);
                return null;
            }

            var data = JsonSerializer.Deserialize<SearchResp>(body, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
            var list = data?.contracts ?? new();

            // Prefer ES*/NQ* exact/starts-with, active first
            var pick = list
                .OrderByDescending(c => c.activeContract)
                .ThenBy(c => c.name)
                .FirstOrDefault();

            return pick?.id;
        }
        // Place an order via REST
        public async Task<string?> PlaceOrderAsync(object req, CancellationToken ct)
        {
            using var resp = await _http.PostAsJsonAsync(U("/api/Order/place"), req, ct);
            resp.EnsureSuccessStatusCode();
            var json = await resp.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct);
            // Extract orderId from json (field name per docs)
            return json.TryGetProperty("orderId", out var id) ? id.GetString() : null;
        }

        // Search for orders via REST
        public async Task<JsonElement> SearchOrdersAsync(object body, CancellationToken ct)
        {
            using var resp = await _http.PostAsJsonAsync(U("/api/Order/search"), body, ct);
            resp.EnsureSuccessStatusCode();
            var json = await resp.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct);
            return json;
        }
    }
}
