// PURPOSE: MarketHubClient for live market data and contract resolution.
using System.Net.Http.Json;
using Microsoft.AspNetCore.SignalR.Client;
using System.Text.Json;
using System;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;

namespace BotCore
{
	public sealed class MarketHubClient : IAsyncDisposable
	{
		private readonly HubConnection _hub;
		private bool _wired;
		private int _quotes = 0, _trades = 0;
		public event Action<decimal>? OnLastQuote;
		public event Action<decimal, int>? OnTrade;
		private readonly string _jwt;
private static readonly ConcurrentDictionary<string, string> _contractCache = new();
private string? _lastResolvedId;

		public MarketHubClient(string jwt)
		{
			_jwt = jwt;
			_hub = new HubConnectionBuilder()
				.WithUrl("https://rtc.topstepx.com/hubs/market", o =>
				{
					o.AccessTokenProvider = () => Task.FromResult<string?>(jwt);
					// Allow negotiation and fallback to LongPolling if WebSockets is blocked
					o.Transports = Microsoft.AspNetCore.Http.Connections.HttpTransportType.WebSockets
					              | Microsoft.AspNetCore.Http.Connections.HttpTransportType.LongPolling;
					// Do not skip negotiation; server may require it
					// o.SkipNegotiation = true;
				})
				.WithAutomaticReconnect()
				.Build();
		}

		public async Task StartAsync(string contractIdOrSymbol, CancellationToken ct)
		{
			await _hub.StartAsync(ct);
			await WaitForConnectedAsync(_hub, TimeSpan.FromSeconds(15));
			if (_wired) return; _wired = true;

			_hub.Reconnecting += error =>
			{
				Console.WriteLine($"[MarketHub] reconnecting: {error?.GetType().Name} {error?.Message}");
				return Task.CompletedTask;
			};
			_hub.Reconnected += async _ =>
			{
				Console.WriteLine("[MarketHub] reconnected — resubscribing");
				var id = _lastResolvedId;
				if (!string.IsNullOrWhiteSpace(id))
				{
					try { await SubscribeAllAsync(id!, CancellationToken.None); }
					catch (Exception ex) { Console.WriteLine($"[MarketHub] resubscribe error: {ex.GetType().Name} {ex.Message}"); }
				}
			};
			_hub.Closed += error =>
			{
				Console.WriteLine($"[MarketHub] closed: {error?.GetType().Name} {error?.Message}");
				return Task.CompletedTask;
			};

			_hub.On<string, JsonElement>("GatewayQuote", (cid, data) =>
			{
				_quotes++;
				if (data.TryGetProperty("lastPrice", out var lp))
				{
					var last = lp.GetDecimal();
					OnLastQuote?.Invoke(last);
				}
			});
			_hub.On<string, JsonElement>("GatewayTrade", (cid, data) =>
			{
				_trades++;
				var price = data.GetProperty("price").GetDecimal();
				var vol = data.TryGetProperty("volume", out var v) ? v.GetInt32() : 1;
				OnTrade?.Invoke(price, vol);
				Console.WriteLine($"trade {price} x {vol}");
			});
			_hub.On<string, JsonElement>("GatewayDepth", (cid, data) =>
			{
				// Optionally handle market depth
			});

			string resolvedId;
			if (!string.IsNullOrWhiteSpace(contractIdOrSymbol) && contractIdOrSymbol.StartsWith("CON."))
			{
				resolvedId = contractIdOrSymbol;
			}
			else
			{
				resolvedId = await ResolveContractIdAsync(contractIdOrSymbol, ct);
			}
			_lastResolvedId = resolvedId;
			await SubscribeAllAsync(resolvedId, ct);
		}

		private async Task<string> ResolveContractIdAsync(string input, CancellationToken ct)
		{
			// If already a GUID-like ID, pass through
			if (Guid.TryParse(input, out _)) return input;
			if (_contractCache.TryGetValue(input, out var cached)) return cached;

			using var http = new HttpClient { BaseAddress = new Uri("https://api.topstepx.com") };
			http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _jwt);

			var body = new { live = true, searchText = input };
			var resp = await http.PostAsJsonAsync("/api/Contract/search", body, ct);
			if (!resp.IsSuccessStatusCode)
			{
				var text = await resp.Content.ReadAsStringAsync(ct);
				throw new InvalidOperationException($"Contract search failed ({(int)resp.StatusCode}): {text}");
			}
			var json = await resp.Content.ReadAsStringAsync(ct);
			using var doc = JsonDocument.Parse(json);
			if (!doc.RootElement.TryGetProperty("contracts", out var arr) || arr.GetArrayLength() == 0)
			{
				throw new InvalidOperationException($"No contract found for '{input}'.");
			}
			// Try exact match by symbol first
			string? id = null;
			for (int i = 0; i < arr.GetArrayLength(); i++)
			{
				var el = arr[i];
				if (el.TryGetProperty("symbol", out var sym) && string.Equals(sym.GetString(), input, StringComparison.OrdinalIgnoreCase))
				{
					id = el.GetProperty("id").GetString();
					break;
				}
			}
			id ??= arr[0].GetProperty("id").GetString();
			if (string.IsNullOrWhiteSpace(id))
				throw new InvalidOperationException($"Contract id missing for '{input}'.");

			_contractCache[input] = id!;
			Console.WriteLine($"Resolved '{input}' => contractId {id}");
			return id!;
		}

		private static async Task WaitForConnectedAsync(HubConnection hub, TimeSpan timeout)
		{
			var start = DateTime.UtcNow;
			while (hub.State != HubConnectionState.Connected)
			{
				if (DateTime.UtcNow - start > timeout)
					throw new TimeoutException($"Hub did not reach Connected. Current={hub.State}");
				await Task.Delay(100);
			}
		}

		private static async Task<bool> RetryAsync(Func<Task> action, int attempts = 5, int delayMs = 800)
		{
			for (int i = 1; i <= attempts; i++)
			{
				try { await action(); return true; }
				catch (Exception ex)
				{
					Console.WriteLine($"[MarketHub] Retry {i}/{attempts}: {ex.GetType().Name} {ex.Message}");
					await Task.Delay(delayMs);
				}
			}
			return false;
		}

		private async Task SubscribeAllAsync(string contractId, CancellationToken ct)
		{
			await RetryAsync(() => _hub.InvokeAsync("SubscribeContractQuotes", contractId, ct));
			await RetryAsync(() => _hub.InvokeAsync("SubscribeContractTrades", contractId, ct));
			await RetryAsync(() => _hub.InvokeAsync("SubscribeContractMarketDepth", contractId, ct));
		}

		public void PrintSmokeCounters()
		{
			Console.WriteLine($"quotes={_quotes} trades={_trades}");
		}

		public async ValueTask DisposeAsync() => await _hub.DisposeAsync();

		// Seed bar builder with historical bars
		public static async Task<string> FetchHistoricalBarsAsync(string jwt, string contractId, DateTime start, DateTime end, CancellationToken ct)
		{
			var req = new
			{
				contractId = contractId,
				live = false,
				startTime = start,
				endTime = end,
				unit = 2,
				unitNumber = 1,
				limit = 500,
				includePartialBar = true
			};
			using var http = new HttpClient { BaseAddress = new Uri("https://api.topstepx.com") };
			http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt);
			var resp = await http.PostAsJsonAsync("/api/History/retrieveBars", req, ct);
			resp.EnsureSuccessStatusCode();
			return await resp.Content.ReadAsStringAsync();
		}
	}
}
