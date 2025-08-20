using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore
{
    public class OrderRouterAgent
    {
        private readonly HttpClient _http;
        public OrderRouterAgent(string jwt)
        {
            _http = new HttpClient { BaseAddress = new Uri("https://api.topstepx.com") };
            _http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt);
        }

        public async Task<string?> PlaceOrderAsync(object req, CancellationToken ct)
        {
            using var resp = await _http.PostAsJsonAsync("/api/Order/place", req, ct);
            resp.EnsureSuccessStatusCode();
            var json = await resp.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct);
            return json.TryGetProperty("orderId", out var id) ? id.GetString() : null;
        }

        public async Task<JsonElement> SearchOrdersAsync(object body, CancellationToken ct)
            => await PostJsonAsync("/api/Order/search", body, ct);

        public async Task<JsonElement> SearchTradesAsync(object body, CancellationToken ct)
            => await PostJsonAsync("/api/Trade/search", body, ct);

        private async Task<JsonElement> PostJsonAsync(string path, object body, CancellationToken ct)
        {
            using var resp = await _http.PostAsJsonAsync(path, body, ct);
            resp.EnsureSuccessStatusCode();
            return (await resp.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct)).GetProperty("data");
        }
    }
}
