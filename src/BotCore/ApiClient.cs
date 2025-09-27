using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace BotCore
{
    /// <summary>
    /// Response structure for searching open positions
    /// </summary>
    public sealed record SearchOpenPositionsResponse(IReadOnlyList<Position> positions, bool success, int errorCode, string? errorMessage);
    
    /// <summary>
    /// Position data structure for API responses
    /// </summary>
    public sealed record Position(long id, long accountId, string contractId, DateTimeOffset creationTimestamp, int type, int size, decimal averagePrice);

    public sealed class ApiClient(HttpClient http, ILogger<ApiClient> log, string apiBase)
    {
        // Retry Configuration Constants
        private const double ExponentialBackoffBase = 2.0;  // Base for exponential backoff calculation
        
        private readonly HttpClient _http = http;
        private readonly ILogger<ApiClient> _log = log;
        private readonly string _apiBase = apiBase;
        private string? _jwt;

        public string? CurrentJwt => _jwt;

        public void SetJwt(string jwt)
        {
            _jwt = jwt;
            _http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt);
        }

#nullable enable
        private sealed record AvailableReq(bool live);
        // These records are used by System.Text.Json for deserialization - CA1812 false positive
        internal sealed record ContractDto(string id, string name, string? description, string symbolId, bool activeContract);
        internal sealed record AvailableResp(IReadOnlyList<ContractDto>? contracts, bool success, int errorCode, string? errorMessage);

        private static readonly Dictionary<string, string> SymbolRootToSymbolId = new(StringComparer.OrdinalIgnoreCase)
        {
            ["ES"] = "F.US.EP",   // E-mini S&P 500
            ["NQ"] = "F.US.ENQ",  // E-mini NASDAQ-100
        };

        public async Task<string> ResolveContractIdAsync(string root, CancellationToken ct = default)
        {
            // 1) Try AVAILABLE (eval = live:false)
            var id = await TryResolveViaAvailableAsync(root, live: false, ct).ConfigureAwait(false);
            id ??= await TryResolveViaAvailableAsync(root, live: true, ct).ConfigureAwait(false); // safety fallback

            // 2) Fallback to SEARCH if still nothing
            id ??= await TryResolveViaSearchAsync(root, live: false, ct).ConfigureAwait(false) ?? await TryResolveViaSearchAsync(root, live: true, ct).ConfigureAwait(false);

            if (string.IsNullOrWhiteSpace(id))
                throw new InvalidOperationException($"No contractId found for symbol: {root}");

            return id!;
        }

        private Uri U(string path) => new($"{_apiBase}{path}");
        
        /// <summary>
        /// Determine if HTTP status code should trigger a retry (5xx/408 only)
        /// </summary>
        private static bool ShouldRetry(System.Net.HttpStatusCode statusCode)
        {
            return statusCode == System.Net.HttpStatusCode.RequestTimeout || // 408
                   statusCode == System.Net.HttpStatusCode.InternalServerError || // 500
                   statusCode == System.Net.HttpStatusCode.BadGateway || // 502
                   statusCode == System.Net.HttpStatusCode.ServiceUnavailable || // 503
                   statusCode == System.Net.HttpStatusCode.GatewayTimeout; // 504
        }

        private async Task<string?> TryResolveViaAvailableAsync(string root, bool live, CancellationToken ct)
        {
            SymbolRootToSymbolId.TryGetValue(root, out var wantedSymbolId);
            try
            {
                using var resp = await _http.PostAsJsonAsync(U("/api/Contract/available"), new AvailableReq(live), ct).ConfigureAwait(false);
                var body = await resp.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
                if (!resp.IsSuccessStatusCode)
                {
                    _log.LogWarning("Contract available {Status} (live={Live}): {Body}", (int)resp.StatusCode, live, body);
                    return null;
                }

                var data = JsonSerializer.Deserialize<AvailableResp>(body, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                var list = data?.contracts ?? [];

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
            catch (Exception ex)
            {
                _log.LogWarning(ex, "Contract available EX (live={Live}) for root={Root}", live, root);
                return null;
            }
        }

        private sealed record SearchReq(string searchText, bool live);
        // This record is used by System.Text.Json for deserialization - CA1812 false positive  
        internal sealed record SearchResp(IReadOnlyList<ContractDto>? contracts, bool success, int errorCode, string? errorMessage);

        private async Task<string?> TryResolveViaSearchAsync(string searchText, bool live, CancellationToken ct)
        {
            try
            {
                using var resp = await _http.PostAsJsonAsync(U("/api/Contract/search"), new SearchReq(searchText, live), ct).ConfigureAwait(false);
                var body = await resp.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
                if (!resp.IsSuccessStatusCode)
                {
                    _log.LogWarning("Contract search {Status} (q='{Q}', live={Live}): {Body}", (int)resp.StatusCode, searchText, live, body);
                    return null;
                }

                var data = JsonSerializer.Deserialize<SearchResp>(body, new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                var list = data?.contracts ?? [];

                // Prefer ES*/NQ* exact/starts-with, active first
                var pick = list
                    .OrderByDescending(c => c.activeContract)
                    .ThenBy(c => c.name)
                    .FirstOrDefault();

                return pick?.id;
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "Contract search EX (q='{Q}', live={Live})", searchText, live);
                return null;
            }
        }
        // Place an order via REST
        public async Task<string?> PlaceOrderAsync(object req, CancellationToken ct)
        {
            const int maxRetries = 3;
            
            for (int attempt = 1; attempt <= maxRetries; attempt++)
            {
                try
                {
                    using var resp = await _http.PostAsJsonAsync(U("/api/Order/place"), req, ct).ConfigureAwait(false);
                    
                    if (resp.IsSuccessStatusCode)
                    {
                        var json = await resp.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct).ConfigureAwait(false);
                        // Extract orderId from json (field name per docs)
                        return json.TryGetProperty("orderId", out var id) ? id.GetString() : null;
                    }
                    else if (ShouldRetry(resp.StatusCode) && attempt < maxRetries)
                    {
                        _log.LogWarning("[APICLIENT] PlaceOrder attempt {Attempt}/{Max} failed: HTTP {StatusCode}, retrying...", 
                            attempt, maxRetries, (int)resp.StatusCode);
                        await Task.Delay(TimeSpan.FromSeconds(Math.Pow(ExponentialBackoffBase, attempt)), ct).ConfigureAwait(false);
                        continue;
                    }
                    else
                    {
                        // Don't retry 4xx errors or final attempt
                        resp.EnsureSuccessStatusCode();
                    }
                }
                catch (HttpRequestException ex) when (attempt < maxRetries)
                {
                    _log.LogWarning(ex, "[APICLIENT] PlaceOrder HTTP request failed on attempt {Attempt}/{Max}, retrying...", 
                        attempt, maxRetries);
                    await Task.Delay(TimeSpan.FromSeconds(Math.Pow(ExponentialBackoffBase, attempt)), ct).ConfigureAwait(false);
                }
            }

            // This should not be reached due to EnsureSuccessStatusCode above
            throw new InvalidOperationException("Failed to place order after all retry attempts");
        }

        // Search for orders via REST
        public async Task<JsonElement> SearchOrdersAsync(object body, CancellationToken ct)
        {
            using var resp = await _http.PostAsJsonAsync(U("/api/Order/search"), body, ct).ConfigureAwait(false);
            resp.EnsureSuccessStatusCode();
            var json = await resp.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct).ConfigureAwait(false);
            return json;
        }

        // Search for trades via REST
        public async Task<JsonElement> SearchTradesAsync(object body, CancellationToken ct)
        {
            using var resp = await _http.PostAsJsonAsync(U("/api/Trade/search"), body, ct).ConfigureAwait(false);
            resp.EnsureSuccessStatusCode();
            var json = await resp.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct).ConfigureAwait(false);
            return json;
        }

        // Generic GET helper
        public async Task<T?> GetAsync<T>(string relativePath, CancellationToken ct)
        {
            using var resp = await _http.GetAsync(U(relativePath), ct).ConfigureAwait(false);
            resp.EnsureSuccessStatusCode();
            return await resp.Content.ReadFromJsonAsync<T>(cancellationToken: ct).ConfigureAwait(false);
        }

        // Generic POST helper without response body
        public async Task PostAsync(string relativePath, object body, CancellationToken ct)
        {
            using var resp = await _http.PostAsJsonAsync(U(relativePath), body, ct).ConfigureAwait(false);
            resp.EnsureSuccessStatusCode();
        }

        // Generic POST helper with typed response
        public async Task<T?> PostAsync<T>(string relativePath, object body, CancellationToken ct)
        {
            using var resp = await _http.PostAsJsonAsync(U(relativePath), body, ct).ConfigureAwait(false);
            resp.EnsureSuccessStatusCode();
            return await resp.Content.ReadFromJsonAsync<T>(cancellationToken: ct).ConfigureAwait(false);
        }

        // New wrapper-aware method (preferred)
        public async Task<IReadOnlyList<Position>> GetOpenPositionsAsync(long accountId, CancellationToken ct)
        {
            var resp = await PostAsync<SearchOpenPositionsResponse>("/api/Position/searchOpen", new { accountId }, ct).ConfigureAwait(false);
            return resp?.positions ?? Array.Empty<Position>();
        }

        // Legacy one kept for backward-compat if any call sites still expect bare arrays
        public Task<List<System.Text.Json.JsonElement>?> GetOpenPositionsLegacyAsync(long accountId, CancellationToken ct)
            => PostAsync<List<System.Text.Json.JsonElement>>("/api/Position/searchOpen", new { accountId }, ct);

        public Task CloseContractAsync(long accountId, string contractId, CancellationToken ct)
            => PostAsync<object>("/api/Position/closeContract", new { accountId, contractId }, ct)!;

        public Task PartialCloseAsync(long accountId, string contractId, int size, CancellationToken ct)
            => PostAsync<object>("/api/Position/partialCloseContract", new { accountId, contractId, size }, ct)!;
    }
}
