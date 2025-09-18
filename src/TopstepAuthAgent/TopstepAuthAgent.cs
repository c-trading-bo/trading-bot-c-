using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace TopstepX.Bot.Authentication
{
    public sealed class TopstepAuthAgent
    {
        private readonly HttpClient _http;

        public TopstepAuthAgent(HttpClient http)
        {
            _http = http;
            // Use configuration-based URL, requiring environment variable for production
            var apiBase = Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE");
            if (string.IsNullOrEmpty(apiBase))
            {
                throw new InvalidOperationException("TOPSTEPX_API_BASE environment variable is required for TopstepX authentication");
            }
            _http.BaseAddress ??= new Uri(apiBase);
            _http.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
        }

        public async Task<string> GetJwtAsync(string username, string apiKey, CancellationToken ct)
        {
            // IMPORTANT: /api/Auth/loginKey (exact path)
            var req = new HttpRequestMessage(HttpMethod.Post, "/api/Auth/loginKey")
            {
                // Use property names exactly as docs show: userName, apiKey
                Content = new StringContent(
                    JsonSerializer.Serialize(new { userName = username, apiKey }),
                    Encoding.UTF8, "application/json")
            };

            using var resp = await _http.SendAsync(req, ct).ConfigureAwait(false).ConfigureAwait(false);
            if (!resp.IsSuccessStatusCode)
            {
                var body = await resp.Content.ReadAsStringAsync(ct).ConfigureAwait(false).ConfigureAwait(false);
                throw new HttpRequestException($"Auth {(int)resp.StatusCode} {resp.StatusCode}: {body}", null, resp.StatusCode);
            }

            using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync(ct)).ConfigureAwait(false);
            return doc.RootElement.GetProperty("token").GetString()!;
        }

        public async Task<string?> ValidateAsync(CancellationToken ct)
        {
            var req = new HttpRequestMessage(HttpMethod.Post, "/api/Auth/validate");
            using var resp = await _http.SendAsync(req, ct).ConfigureAwait(false).ConfigureAwait(false);
            if (!resp.IsSuccessStatusCode) return null;

            using var doc = JsonDocument.Parse(await resp.Content.ReadAsStringAsync(ct)).ConfigureAwait(false);
            if (doc.RootElement.TryGetProperty("newToken", out var nt)) return nt.GetString();
            return null;
        }
    }
}

namespace TopstepX.Bot.Extensions
{
    // Simple HttpRequestMessage.Clone() so we can resend the content on retries:
    public static class HttpRequestMessageExtensions
    {
        public static async Task<HttpRequestMessage> CloneAsync(this HttpRequestMessage req)
        {
            var clone = new HttpRequestMessage(req.Method, req.RequestUri);
            // Copy headers
            foreach (var h in req.Headers)
                clone.Headers.TryAddWithoutValidation(h.Key, h.Value);
            // Copy content asynchronously
            if (req.Content != null)
            {
                var contentBytes = await req.Content.ReadAsByteArrayAsync().ConfigureAwait(false).ConfigureAwait(false);
                var newContent = new ByteArrayContent(contentBytes);
                foreach (var h in req.Content.Headers)
                    newContent.Headers.TryAddWithoutValidation(h.Key, h.Value);
                clone.Content = newContent;
            }
            return clone;
        }
    }
}
