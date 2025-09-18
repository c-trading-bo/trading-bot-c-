using System;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;

namespace BotCore
{
    public sealed class JwtCache(Func<Task<string>> acquire)
    {
        private readonly Func<Task<string>> _acquire = acquire;
        private string? _token;
        private DateTimeOffset _exp;

        public async Task<string?> GetAsync()
        {
            if (_token is null || DateTimeOffset.UtcNow >= _exp - TimeSpan.FromMinutes(3))
            {
                var t = await _acquire().ConfigureAwait(false);
                _token = t;
                _exp = ParseExp(t);
            }
            return _token;
        }

        private static DateTimeOffset ParseExp(string jwt)
        {
            var parts = jwt.Split('.');
            if (parts.Length < 2) return DateTimeOffset.UtcNow; // fallback: treat as expiring now
            var payload = parts[1];
            var padLen = (4 - payload.Length % 4) % 4;
            payload = payload.PadRight(payload.Length + padLen, '=');
            var json = Encoding.UTF8.GetString(Convert.FromBase64String(payload));
            using var doc = JsonDocument.Parse(json);
            var exp = doc.RootElement.GetProperty("exp").GetInt64();
            return DateTimeOffset.FromUnixTimeSeconds(exp);
        }
    }
}
