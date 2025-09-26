using System;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace BotCore.Auth
{
    /// <summary>
    /// Simple, reliable TopstepX authentication - based on working August 22 version
    /// This replaces complex auth chains with proven working logic
    /// </summary>
    public sealed class SimpleTopstepAuth : IAsyncDisposable, IDisposable
    {
        private readonly HttpClient _http;
        private readonly ILogger<SimpleTopstepAuth> _logger;
        private readonly string _username;
        private readonly string _apiKey;
        private readonly object _gate = new();
        
        private string _jwt = string.Empty;
        private DateTimeOffset _expUtc = DateTimeOffset.MinValue;

        public SimpleTopstepAuth(HttpClient httpClient, ILogger<SimpleTopstepAuth> logger)
        {
            _http = httpClient ?? throw new ArgumentNullException(nameof(httpClient));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            
            // Get credentials from environment variables
            _username = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME") ?? 
                       Environment.GetEnvironmentVariable("TOPSTEP_USERNAME") ?? 
                       throw new InvalidOperationException("TOPSTEPX_USERNAME or TOPSTEP_USERNAME environment variable not set.");
            
            _apiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY") ?? 
                     Environment.GetEnvironmentVariable("TOPSTEP_API_KEY") ?? 
                     throw new InvalidOperationException("TOPSTEPX_API_KEY or TOPSTEP_API_KEY environment variable not set.");
            
            // Ensure base address is set - Use TopstepX API
            if (_http.BaseAddress == null)
            {
                _http.BaseAddress = new Uri("https://api.topstepx.com");
            }
        }

        public async Task<(string jwt, DateTimeOffset expiresUtc)> GetFreshJwtAsync(CancellationToken ct = default)
        {
            // Simple refresh logic - if token expires in 2 minutes, refresh it
            if (DateTimeOffset.UtcNow >= _expUtc - TimeSpan.FromSeconds(120))
            {
                await RefreshTokenAsync(ct).ConfigureAwait(false);
            }
            
            lock (_gate)
            {
                return (_jwt, _expUtc);
            }
        }

        public Task EnsureFreshTokenAsync(CancellationToken ct = default)
        {
            return GetFreshJwtAsync(ct);
        }

        public async Task<string> GetTokenAsync(CancellationToken ct = default)
        {
            var (jwt, _) = await GetFreshJwtAsync(ct).ConfigureAwait(false);
            return jwt;
        }

        private async Task RefreshTokenAsync(CancellationToken ct)
        {
            try
            {
                _logger.LogInformation("Refreshing TopstepX JWT token...");

                // Use TopstepX API endpoint for authentication
                var request = new HttpRequestMessage(HttpMethod.Post, "/api/Auth/loginKey")
                {
                    Content = new StringContent(
                        JsonSerializer.Serialize(new { userName = _username, apiKey = _apiKey }),
                        Encoding.UTF8, 
                        "application/json")
                };

                using var response = await _http.SendAsync(request, ct).ConfigureAwait(false);
                
                if (!response.IsSuccessStatusCode)
                {
                    var errorBody = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
                    _logger.LogError("Auth failed: {StatusCode} {ReasonPhrase} - {Body}", 
                        response.StatusCode, response.ReasonPhrase, errorBody);
                    throw new HttpRequestException($"Auth {(int)response.StatusCode} {response.StatusCode}: {errorBody}");
                }

                var json = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
                _logger.LogInformation("JWT received successfully");

                using var doc = JsonDocument.Parse(json);
                if (!doc.RootElement.TryGetProperty("token", out var tokenProp))
                {
                    throw new InvalidOperationException("No 'token' property in auth response");
                }

                var newJwt = tokenProp.GetString();
                if (string.IsNullOrEmpty(newJwt))
                {
                    throw new InvalidOperationException("Empty token received from auth endpoint");
                }

                var expiry = GetJwtExpiryUtc(newJwt);

                lock (_gate)
                {
                    _jwt = newJwt;
                    _expUtc = expiry;
                }

                // Update environment variable so other services can use the fresh token
                Environment.SetEnvironmentVariable("TOPSTEPX_JWT", newJwt);

                _logger.LogInformation("JWT token refreshed successfully, expires at {ExpiryUtc}", expiry);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to refresh JWT token");
                throw;
            }
        }

        private static DateTimeOffset GetJwtExpiryUtc(string jwt)
        {
            var parts = jwt.Split('.');
            if (parts.Length < 2) throw new ArgumentException("Invalid JWT format");
            
            string payloadJson = Encoding.UTF8.GetString(Base64UrlDecode(parts[1]));
            using var doc = JsonDocument.Parse(payloadJson);
            long exp = doc.RootElement.GetProperty("exp").GetInt64();
            return DateTimeOffset.FromUnixTimeSeconds(exp);
        }

        private static byte[] Base64UrlDecode(string s)
        {
            s = s.Replace('-', '+').Replace('_', '/');
            switch (s.Length % 4) 
            { 
                case 2: s += "=="; break; 
                case 3: s += "="; break; 
            }
            return Convert.FromBase64String(s);
        }

        public void Dispose()
        {
            // HttpClient is typically managed by DI container, don't dispose it
            lock (_gate)
            {
                _jwt = string.Empty;
                _expUtc = DateTimeOffset.MinValue;
            }
        }

        public ValueTask DisposeAsync()
        {
            Dispose();
            return ValueTask.CompletedTask;
        }
    }
}
