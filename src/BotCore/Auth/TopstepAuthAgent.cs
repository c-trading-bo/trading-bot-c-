
using System;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Auth
{
	public interface ITopstepAuth
	{
		Task<(string jwt, DateTimeOffset expiresUtc)> GetFreshJwtAsync(CancellationToken ct);
	}

	public sealed class CachedTopstepAuth : ITopstepAuth
	{
		private readonly Func<CancellationToken, Task<string>> _fetchJwt;
	private string _jwt = string.Empty;
		private DateTimeOffset _expUtc = DateTimeOffset.MinValue;
		private readonly object _gate = new();

		public CachedTopstepAuth(Func<CancellationToken, Task<string>> fetchJwt) => _fetchJwt = fetchJwt;

		public async Task<(string jwt, DateTimeOffset expiresUtc)> GetFreshJwtAsync(CancellationToken ct)
		{
			if (DateTimeOffset.UtcNow >= _expUtc - TimeSpan.FromSeconds(120))
			{
				var newJwt = await _fetchJwt(ct);
				var exp = GetJwtExpiryUtc(newJwt);
				lock (_gate) { _jwt = newJwt; _expUtc = exp; }
			}
			return (_jwt, _expUtc);
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
	}
}

