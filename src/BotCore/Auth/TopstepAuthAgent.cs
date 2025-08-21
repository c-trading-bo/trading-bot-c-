using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TopstepAuthAgent
{
    public sealed class TopstepAuthAgent
    {
        private readonly HttpClient _http;
        private readonly ILogger<TopstepAuthAgent> _log;
        private readonly string _apiBase;

        public TopstepAuthAgent(HttpClient http, ILogger<TopstepAuthAgent> log, string apiBase)
        {
            _http = http;
            _log = log;
            _apiBase = apiBase;
            _http.BaseAddress = new Uri(apiBase);
        }

        public async Task<string> GetJwtAsync(string username, string apiKey, CancellationToken ct)
        {
            var req = new { username, apiKey };
            var resp = await _http.PostAsJsonAsync("/api/Auth/loginKey", req, ct);
            resp.EnsureSuccessStatusCode();
            var json = await resp.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct);
            if (json.TryGetProperty("token", out var token))
                return token.GetString() ?? throw new InvalidOperationException("No JWT returned");
            throw new InvalidOperationException("No token field in loginKey response");
        }

        public async Task<string> ValidateAsync(string jwt, CancellationToken ct)
        {
            var req = new { token = jwt };
            var resp = await _http.PostAsJsonAsync("/api/Auth/validate", req, ct);
            resp.EnsureSuccessStatusCode();
            var json = await resp.Content.ReadFromJsonAsync<JsonElement>(cancellationToken: ct);
            if (json.TryGetProperty("token", out var token))
                return token.GetString() ?? throw new InvalidOperationException("No JWT returned");
            throw new InvalidOperationException("No token field in validate response");
        }
    }
}
