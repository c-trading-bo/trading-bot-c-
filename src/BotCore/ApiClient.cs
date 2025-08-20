using System;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore
{
	public class ApiClient
	{
		private readonly HttpClient _http;

		public ApiClient(string jwt)
		{
			_http = new HttpClient { BaseAddress = new Uri("https://api.topstepx.com") };
			_http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt);
		}

		public async Task<long?> PlaceLimit(int accountId, string contractId, int sideBuy0Sell1, int size, decimal limitPrice)
		{
			var body = new {
				accountId, contractId,
				type = 1,             // 1 = Limit
				side = sideBuy0Sell1, // 0=Buy,1=Sell
				size,
				limitPrice
			};
			var r = await _http.PostAsJsonAsync("/api/Order/place", body);
			var txt = await r.Content.ReadAsStringAsync();
			Console.WriteLine("[Order/place] " + txt);
			if (!r.IsSuccessStatusCode) return null;

			try {
				using var doc = JsonDocument.Parse(txt);
				bool success = doc.RootElement.TryGetProperty("success", out var s) && s.GetBoolean();
				if (!success) return null;
				return doc.RootElement.TryGetProperty("orderId", out var id) ? id.GetInt64() : null;
			} catch { return null; }
		}

			public async Task<long?> ResolveContractIdAsync(string symbol, CancellationToken ct)
			{
				var body = new { symbol = symbol };
				var resp = await _http.PostAsJsonAsync("/api/Contract/available", body, ct);
				resp.EnsureSuccessStatusCode();
				var txt = await resp.Content.ReadAsStringAsync();
				// Simple string search for contractId
				var marker = "contractId";
				var idx = txt.IndexOf(marker);
				if (idx < 0) return null;
				var start = idx + marker.Length;
				var end = txt.IndexOf(',', start);
				if (end < 0) end = txt.Length;
				var val = txt.Substring(start, end - start).Trim(' ', '"', ':');
				if (long.TryParse(val, out var contractId)) return contractId;
				return null;
			}
	public async Task<string?> GetActiveAccountAsync(CancellationToken ct = default)
		{
			var body = new { onlyActiveAccounts = true };
			var resp = await _http.PostAsJsonAsync("/api/Account/search", body, ct);
			resp.EnsureSuccessStatusCode();
			var txt = await resp.Content.ReadAsStringAsync();
			// Return raw string for now
			return string.IsNullOrWhiteSpace(txt) ? null : txt;
		}
	}
}
