
using System;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace BotCore.Auth
{
    public interface ITopstepAuth
    {
        Task<(string jwt, DateTimeOffset expiresUtc)> GetFreshJwtAsync(CancellationToken ct = default);
        Task EnsureFreshTokenAsync(CancellationToken ct = default);
    }

    public sealed class CachedTopstepAuth : ITopstepAuth, IDisposable
    {
        private readonly Func<CancellationToken, Task<string>> _fetchJwt;
        private readonly ILogger<CachedTopstepAuth> _logger;
        private readonly SemaphoreSlim _refreshLock = new(1, 1);
        
        private string _jwt = string.Empty;
        private DateTimeOffset _expUtc = DateTimeOffset.MinValue;
        
        public CachedTopstepAuth(Func<CancellationToken, Task<string>> fetchJwt, ILogger<CachedTopstepAuth> logger)
        {
            _fetchJwt = fetchJwt ?? throw new ArgumentNullException(nameof(fetchJwt));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        public async Task<(string jwt, DateTimeOffset expiresUtc)> GetFreshJwtAsync(CancellationToken ct = default)
        {
            await EnsureFreshTokenAsync(ct).ConfigureAwait(false);
            return (_jwt, _expUtc);
        }

        public async Task EnsureFreshTokenAsync(CancellationToken ct = default)
        {
            // First check without lock - most calls will return here
            if (DateTimeOffset.UtcNow < _expUtc - TimeSpan.FromMinutes(5))
                return;

            // Acquire lock for refresh
            await _refreshLock.WaitAsync(ct).ConfigureAwait(false);
            try
            {
                // Double-checked locking pattern - check again inside lock
                if (DateTimeOffset.UtcNow < _expUtc - TimeSpan.FromMinutes(5))
                    return;

                _logger.LogInformation("[AUTH] JWT refresh initiated (expires: {ExpiryUtc})", _expUtc);
                
                var newJwt = await _fetchJwt(ct).ConfigureAwait(false);
                var newExpUtc = GetJwtExpiryUtc(newJwt);
                
                _jwt = newJwt;
                _expUtc = newExpUtc;
                
                var refreshData = new
                {
                    timestamp = DateTimeOffset.UtcNow,
                    component = "cached_topstep_auth",
                    operation = "jwt_refresh",
                    new_expiry_utc = newExpUtc,
                    refresh_window_minutes = 5
                };

                _logger.LogInformation("JWT refreshed: {RefreshData}", System.Text.Json.JsonSerializer.Serialize(refreshData));
            }
            catch (Exception ex)
            {
                var errorData = new
                {
                    timestamp = DateTimeOffset.UtcNow,
                    component = "cached_topstep_auth",
                    operation = "jwt_refresh_failed",
                    error_type = ex.GetType().Name,
                    sanitized_message = SanitizeErrorMessage(ex.Message)
                };

                _logger.LogError("JWT refresh failed: {ErrorData}", System.Text.Json.JsonSerializer.Serialize(errorData));
                throw;
            }
            finally
            {
                _refreshLock.Release();
            }
        }

        private static DateTimeOffset GetJwtExpiryUtc(string jwt)
        {
            var parts = jwt.Split('.');
            if (parts.Length < 2) throw new ArgumentException("Invalid JWT format");
            string payloadJson = System.Text.Encoding.UTF8.GetString(Base64UrlDecode(parts[1]));
            using var doc = JsonDocument.Parse(payloadJson);
            long exp = doc.RootElement.GetProperty("exp").GetInt64();
            return DateTimeOffset.FromUnixTimeSeconds(exp);
        }

        private static byte[] Base64UrlDecode(string s)
        {
            s = s.Replace('-', '+').Replace('_', '/');
            switch (s.Length % 4) { case 2: s += "=="; break; case 3: s += "="; break; }
            return Convert.FromBase64String(s);
        }

        private static string SanitizeErrorMessage(string message)
        {
            if (string.IsNullOrEmpty(message))
                return message;

            // Remove potential token/secret patterns from error messages
            var patterns = new[]
            {
                (@"(token|key|secret|password|auth)[=:]\s*[^\s,}]+", "$1=[REDACTED]"),
                (@"(bearer\s+)[a-zA-Z0-9\.\-_]+", "$1[REDACTED]"),
                (@"[a-zA-Z0-9]{50,}", "[LONG_STRING_REDACTED]") // Potential tokens
            };

            var result = message;
            foreach (var (pattern, replacement) in patterns)
            {
                result = System.Text.RegularExpressions.Regex.Replace(
                    result, pattern, replacement, System.Text.RegularExpressions.RegexOptions.IgnoreCase);
            }

            return result;
        }

        public void Dispose()
        {
            _refreshLock?.Dispose();
        }
    }
}

