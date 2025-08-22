using System.Text.Json;

using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TopstepAuthAgent
{
	/// <summary>
	/// Handles JWT authentication and validation for TopstepX.
	/// </summary>
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
		}

		public async Task<string> GetJwtAsync(string username, string apiKey, CancellationToken ct)
		{
			// POST to /api/Auth/loginKey with userName/apiKey
			var req = new { userName = username, apiKey };
			var resp = await _http.PostAsJsonAsync(_apiBase + "/api/Auth/loginKey", req, ct);
			resp.EnsureSuccessStatusCode();
			var json = await resp.Content.ReadAsStringAsync(ct);
			_log.LogInformation($"JWT received: {json}");
			// Parse JWT from response
			using var doc = JsonDocument.Parse(json);
			if (doc.RootElement.TryGetProperty("token", out var tokenProp))
				return tokenProp.GetString() ?? string.Empty;
			return string.Empty;
		}

		public async Task<string> ValidateAsync(string jwt, CancellationToken ct)
		{
			// Example: POST to /api/Auth/validate with JWT
			var req = new { jwt };
			var resp = await _http.PostAsJsonAsync(_apiBase + "/api/Auth/validate", req, ct);
			resp.EnsureSuccessStatusCode();
			var json = await resp.Content.ReadAsStringAsync(ct);
			_log.LogInformation($"JWT validated: {json}");
			// Parse validation result
			using var doc = JsonDocument.Parse(json);
			if (doc.RootElement.TryGetProperty("success", out var successProp) && successProp.GetBoolean())
				return jwt;
			return string.Empty;
		}
	}
}
