using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;



// ----------------------- Auth Helpers -----------------------
static class Auth
{
	public static async Task<string> EnsureSessionTokenAsync()
	{
		var token = Environment.GetEnvironmentVariable("PROJECTX_TOKEN");
		if (!string.IsNullOrWhiteSpace(token))
		{
			var refreshed = await ValidateAndRefreshAsync(token);
			if (!string.IsNullOrEmpty(refreshed)) {
				Environment.SetEnvironmentVariable("PROJECTX_TOKEN", refreshed);
				return refreshed;
			}
			// if validate didn’t return newToken, keep the old one
			return token;
		}

		var user = Environment.GetEnvironmentVariable("PROJECTX_USERNAME")
				   ?? Environment.GetEnvironmentVariable("TSX_USERNAME");
		var key  = Environment.GetEnvironmentVariable("PROJECTX_APIKEY")
				   ?? Environment.GetEnvironmentVariable("TSX_API_KEY");
		if (string.IsNullOrWhiteSpace(user) || string.IsNullOrWhiteSpace(key))
			throw new Exception("Set either PROJECTX_TOKEN (session JWT) or both PROJECTX_USERNAME/TSX_USERNAME and PROJECTX_APIKEY/TSX_API_KEY.");

		var newToken = await LoginWithApiKeyAsync(user, key);
		Environment.SetEnvironmentVariable("PROJECTX_TOKEN", newToken);
		return newToken;
	}

	public static async Task<string> LoginWithApiKeyAsync(string userName, string apiKey)
	{
		using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(15) };
		var body = new { userName, apiKey };
		var r = await http.PostAsJsonAsync("https://api.topstepx.com/api/Auth/loginKey", body);
		var txt = await r.Content.ReadAsStringAsync();
		if (!r.IsSuccessStatusCode)
		{
			Console.WriteLine($"[Auth/loginKey] {(int)r.StatusCode} {r.ReasonPhrase} {txt}");
			r.EnsureSuccessStatusCode();
		}
		using var doc = JsonDocument.Parse(txt);
		var tok = doc.RootElement.GetProperty("token").GetString();
		if (string.IsNullOrWhiteSpace(tok)) throw new Exception("loginKey returned no token.");
		return tok!;
	}

	public static async Task<string?> ValidateAndRefreshAsync(string token)
	{
		using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
		http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);
		// empty JSON body per docs
		using var r = await http.PostAsync("https://api.topstepx.com/api/Auth/validate",
										   new StringContent("{}", System.Text.Encoding.UTF8, "application/json"));
		var txt = await r.Content.ReadAsStringAsync();
		if (!r.IsSuccessStatusCode)
		{
			Console.WriteLine($"[Auth/validate] {(int)r.StatusCode} {r.ReasonPhrase} {txt}");
			return null;
		}
		using var doc = JsonDocument.Parse(txt);
		if (doc.RootElement.TryGetProperty("newToken", out var nt) && nt.ValueKind == JsonValueKind.String)
			return nt.GetString(); // refreshed token
		return token; // still valid, no newToken provided
	}
}

static class CFG
{
	// ---- Endpoints (Topstep production) ----
	public const string API        = "https://api.topstepx.com";
	public const string MARKET_HUB = "https://rtc.topstepx.com/hubs/market"; // token-only URL

	// ---- Session / Behavior ----
	public static readonly string[] SYMBOLS = new[] { "ES", "NQ" }; // change as needed
	public const bool LIVE = true;                 // live trading mode
	public static bool FORCE_BUY = false;          // set true to force BUY order for testing
	public const int  UNIT = 2;                    // 1=Sec, 2=Min, 3=Hour, 4=Day, 5=Week, 6=Month
	public const int  UNITNUM = 5;                 // 5-minute bars
	public const int  HISTORY_DAYS = 7;            // bootstrap window
	public const int  LIMIT_FALLBACK = 4000;       // used when date-range returns empty
	public const int  LIMIT_MAX = 20000;           // server-side cap safety
	public const bool VERBOSE_STREAM_LOG = true;   // flip off if chatty
}

// ----------------------- REST API Client -----------------------
sealed class ApiClient : IDisposable
{
	private readonly HttpClient _http;

	public ApiClient(string bearer)
	{
		_http = new HttpClient();
		_http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", bearer);
		_http.Timeout = TimeSpan.FromSeconds(30);
	}

	public void Dispose() => _http.Dispose();

	public async Task<int> GetTradableAccountId()
	{
		var body = new { onlyActiveAccounts = true };
		var r = await _http.PostAsJsonAsync($"{CFG.API}/api/Account/search", body);
		r.EnsureSuccessStatusCode();
		using var doc = JsonDocument.Parse(await r.Content.ReadAsStringAsync());
		foreach (var a in doc.RootElement.GetProperty("accounts").EnumerateArray())
		{
			if (a.GetProperty("canTrade").GetBoolean())
			{
				int id = a.GetProperty("id").GetInt32();
				string name = a.GetProperty("name").GetString() ?? "";
				Console.WriteLine($"[Account] Using {id} ({name})");
				return id;
			}
		}
		throw new Exception("No tradable account found (canTrade=false).");
	}

	public async Task<string> ResolveContractId(string search)
	{
		var body = new { live = CFG.LIVE, searchText = search };
		var r = await _http.PostAsJsonAsync($"{CFG.API}/api/Contract/search", body);
		r.EnsureSuccessStatusCode();
		using var doc = JsonDocument.Parse(await r.Content.ReadAsStringAsync());
		var arr = doc.RootElement.GetProperty("contracts");
		if (arr.GetArrayLength() == 0) throw new Exception($"No contract found for '{search}'.");
		string id = arr[0].GetProperty("id").GetString()!;
		Console.WriteLine($"[Contract] {search} -> {id}");
		return id;
	}

	// Remove old RetrieveBars. Use HistoryApi for robust bar retrieval.

	// Optional: place/cancel helpers (you’ll call later from your strategy)
	public async Task<long?> PlaceLimit(int accountId, string contractId, int sideBuy0Sell1, int size, decimal limitPrice)
	{
		var body = new {
			accountId, contractId,
			type = 1,             // 1 = Limit
			side = sideBuy0Sell1, // 0=Buy,1=Sell
			size,
			limitPrice
		};
		var r = await _http.PostAsJsonAsync($"{CFG.API}/api/Order/place", body);
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

	public async Task CancelOrder(int accountId, long orderId)
	{
		var body = new { accountId, orderId };
		var r = await _http.PostAsJsonAsync($"{CFG.API}/api/Order/cancel", body);
		Console.WriteLine("[Order/cancel] " + await r.Content.ReadAsStringAsync());
	}
}

// ----------------------- Robust Market Hub Client -----------------------

public sealed class MarketHubClient : IAsyncDisposable
{
	private readonly HubConnection _conn;
	private readonly Func<Task<string>> _getTokenAsync;
	private readonly SemaphoreSlim _sendLock = new(1, 1);
	private readonly HashSet<string> _subs = new(StringComparer.OrdinalIgnoreCase);
	private volatile bool _disposed;

	public MarketHubClient(Func<Task<string>> getTokenAsync)
	{
		_getTokenAsync = getTokenAsync;

		_conn = new HubConnectionBuilder()
			.WithUrl("https://rtc.topstepx.com/hubs/market", opts =>
			{
				opts.AccessTokenProvider = async () => await _getTokenAsync();
			})
			.WithAutomaticReconnect(new[]
			{
				TimeSpan.Zero, TimeSpan.FromSeconds(2),
				TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(10)
			})
			.Build();

		_conn.ServerTimeout = TimeSpan.FromSeconds(60);
		_conn.KeepAliveInterval = TimeSpan.FromSeconds(15);

		_conn.Reconnecting += ex =>
		{
			Console.WriteLine($"[MarketHub RECONNECTING] {ex?.Message ?? "(no reason)"}");
			return Task.CompletedTask;
		};

		_conn.Reconnected += async newConnId =>
		{
			Console.WriteLine($"[MarketHub RECONNECTED] id={newConnId}");
			await ResubscribeAsync();
		};

		_conn.Closed += ex =>
		{
			Console.WriteLine($"[MarketHub CLOSED] {ex?.Message ?? "(server closed)"}");
			if (_disposed)
			{
				Console.WriteLine("[MarketHub CLOSED] not restarting because client is disposing.");
				return Task.CompletedTask;
			}
			_ = Task.Run(async () =>
			{
				try
				{
					await TryStartLoopAsync(default);
					Console.WriteLine("[MarketHub CLOSED] restart attempt completed.");
				}
				catch (Exception rex)
				{
					Console.WriteLine("[MarketHub CLOSED] restart failed: " + rex.Message);
				}
			});
			return Task.CompletedTask;
		};

		_conn.On<string>("Heartbeat", hb => Console.WriteLine($"[HB] {hb}"));
		// Add more handlers as needed
	}

	public void OnHeartbeat(Action<string> handler)
	{
		_conn.On<string>("Heartbeat", handler);
	}

	public async Task StartAsync(CancellationToken ct = default)
	{
		await TryStartLoopAsync(ct);
		Console.WriteLine("[MarketHub] connected.");
	}

	public async Task SubscribeAllAsync(IEnumerable<string> contractIds, CancellationToken ct = default)
	{
		foreach (var id in contractIds) _subs.Add(id);
		await InvokeSafeAsync("SubscribeInstruments", new object?[] { _subs.ToArray() }, ct);
		Console.WriteLine($"[MarketHub] subscribed: {string.Join(", ", _subs)}");
	}

	public void OnGatewayTrade(Action<string, System.Text.Json.JsonElement> handler)
	{
		_conn.On<string, System.Text.Json.JsonElement>("GatewayTrade", handler);
	}

	public async Task SubscribeTradesAsync(string contractId, CancellationToken ct = default)
	{
		await InvokeSafeAsync("SubscribeContractTrades", new object?[] { contractId }, ct);
		Console.WriteLine($"[MarketHub] SubscribeContractTrades -> {contractId}");
	}

	private async Task ResubscribeAsync(CancellationToken ct = default)
	{
		if (_subs.Count == 0) return;
		await InvokeSafeAsync("SubscribeInstruments", new object?[] { _subs.ToArray() }, ct);
		Console.WriteLine("[MarketHub] resubscribed after reconnect.");
	}

	private async Task TryStartLoopAsync(CancellationToken ct = default)
	{
		while (_conn.State != HubConnectionState.Connected && !ct.IsCancellationRequested)
		{
			try
			{
				Console.WriteLine("[MarketHub START] attempting StartAsync...");
				await _conn.StartAsync(ct);
				Console.WriteLine("[MarketHub START] StartAsync successful.");
			}
			catch (Exception ex)
			{
				Console.WriteLine($"[MarketHub START RETRY] {ex.Message}");
				await Task.Delay(1000, ct);
				continue;
			}
		}
	}

	private async Task InvokeSafeAsync(string method, object?[] args, CancellationToken ct)
	{
		await EnsureConnectedAsync(ct);
		await _sendLock.WaitAsync(ct);
		try
		{
			await _conn.InvokeCoreAsync(method, typeof(void), args, ct);
		}
		catch (InvalidOperationException ioe) when (_conn.State != HubConnectionState.Connected)
		{
			Console.WriteLine($"[MarketHub INVOKE RACE] {ioe.Message}");
			await EnsureConnectedAsync(ct);
			await _conn.InvokeCoreAsync(method, typeof(void), args, ct);
		}
		finally
		{
			_sendLock.Release();
		}
	}

	private async Task EnsureConnectedAsync(CancellationToken ct)
	{
		if (_conn.State == HubConnectionState.Connected) return;
		await TryStartLoopAsync(ct);
	}

	public async Task StopAsync()
	{
		_disposed = true;
		try
		{
			await _conn.StopAsync();
		}
		catch { /* ignore */ }
	}

	public async ValueTask DisposeAsync()
	{
		_disposed = true;
		_sendLock.Dispose();
		await _conn.DisposeAsync();
	}
}



public sealed partial class Program
{
	// ==== AUTO-EXECUTE STATE ====
	private static volatile bool MarketHubUp;
	private static volatile bool UserHubUp;
	private static long BarsSeen;

	private static bool EnvTrue(string n)
	{
		var v = (Environment.GetEnvironmentVariable(n) ?? "").Trim().ToLowerInvariant();
		return v is "1" or "true" or "yes";
	}

	private static bool ReadyToArm(bool canTrade, string? contractId)
		=> canTrade
		   && !string.IsNullOrWhiteSpace(contractId)
		   && MarketHubUp && UserHubUp
		   && BarsSeen >= 10; // saw at least 10 trade ticks → data flowing

	private static void Heartbeat(Func<decimal> getLastClose)
	{
		_ = Task.Run(async () =>
		{
			while (true)
			{
				await Task.Delay(TimeSpan.FromSeconds(10));
				var bars = System.Threading.Interlocked.Read(ref BarsSeen);
				Console.WriteLine($"heartbeat: bars={bars} last={getLastClose():0.00}");
			}
		});
	}
}

// ----------------------- Program Entry -----------------------
// Only one Program class should exist at the end of the file. All entry logic is inside it.
public sealed partial class Program
{
	public static async Task Main(string[] args)
	{
		Console.WriteLine("[Boot] TopstepX connectivity check starting...");
		LoadEnvFromFile(".env.local");

		// Set up long-running process until Ctrl+C or process exit
		var runTcs = new TaskCompletionSource<object?>();
		Console.CancelKeyPress += (s, e) => { e.Cancel = true; runTcs.TrySetResult(null); };
		AppDomain.CurrentDomain.ProcessExit += (s, e) => runTcs.TrySetResult(null);
		try
		{
			Console.WriteLine("[Bot] Starting authentication...");
			var token = await Auth.EnsureSessionTokenAsync();
			Console.WriteLine("[Auth] Session token acquired.");

			Console.WriteLine("[Bot] Probing MarketHub negotiate endpoint...");
			await ProbeMarketHubAsync(token);

			Console.WriteLine("[Bot] Preparing authorized clients...");
			using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
			http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
			using var api = new ApiClient(token);

			Console.WriteLine("[Bot] Verifying account access...");
			int accountId = await api.GetTradableAccountId();
			Console.WriteLine($"[Account] Connected. Tradable accountId={accountId}");

			Console.WriteLine("[Bot] Resolving contract and fetching history...");
			string esId = await api.ResolveContractId("ES");
			var history = new HistoryApi(http);
			var bars = await history.RetrieveBarsAsync(esId, daysBack: 1, unit: CFG.UNIT, unitNumber: CFG.UNITNUM, live: CFG.LIVE);
			Console.WriteLine($"[History] {bars.Count} bars received for ES.");

			Console.WriteLine("[Bot] Starting MarketHub client and subscribing...");
			await using var hub = new MarketHubClient(Auth.EnsureSessionTokenAsync);
			await hub.StartAsync();
			Console.WriteLine("[MarketHub] Connected.");
			// Small delay to avoid immediate server-side close races before subscribing
			await Task.Delay(1000);
			// Attempt to subscribe with brief retries in case of early close/reconnect.
			var subOk = false;
			for (int i = 0; i < 5 && !subOk; i++)
			{
				try
				{
					await hub.SubscribeAllAsync(new[] { esId });
					Console.WriteLine("[MarketHub] SubscribeInstruments invoked.");

					// Live trade -> bar aggregation wiring
					decimal last = 0m;
					var barAgg = new BarAggregator(TimeSpan.FromMinutes(1));
					barAgg.OnBar += b =>
					{
						// ...existing code...
						last = b.Close;
						Console.WriteLine($"BAR {b.Start:HH:mm} O={b.Open:0.00} H={b.High:0.00} L={b.Low:0.00} C={b.Close:0.00} V={b.Volume}");
					};

					hub.OnGatewayTrade((cid, j) =>
					{
						try
						{
							if (!string.Equals(cid, esId, StringComparison.OrdinalIgnoreCase)) return;
							var ts = DateTime.Parse(j.GetProperty("timestamp").GetString()!, null, System.Globalization.DateTimeStyles.AdjustToUniversal);
							var price = j.GetProperty("price").GetDecimal();
							var vol = j.GetProperty("volume").GetInt32();
							barAgg.OnTrade(ts, price, vol);
							if (CFG.VERBOSE_STREAM_LOG) Console.WriteLine($"[Trade] {cid} {ts:HH:mm:ss} {price} x{vol}");
						}
						catch (Exception ex)
						{
							Console.WriteLine("[GatewayTrade parse error] " + ex.Message);
						}
					});

					hub.OnHeartbeat(hb => Console.WriteLine($"[HB] {hb} bars={bars} last={last:0.00}"));

					await hub.SubscribeTradesAsync(esId);
					subOk = true;
				}
				catch (Exception sx)
				{
					Console.WriteLine("[MarketHub] subscribe retry " + (i + 1) + "/5: " + sx.Message);
					await Task.Delay(1000);
				}
			}

			// Keep running until user cancels, so the hub stays connected
			Console.WriteLine("[Run] MarketHub running. Press Ctrl+C to exit.");

			// --- LIVE STRATEGY LOOP ---
			var risk = new TopstepFullStrat.RiskEngine();
			var levels = new TopstepFullStrat.Levels();
			var env = new TopstepFullStrat.Env();
			// List<TopstepFullStrat.Bar> bars = new();
			string symbol = "ES";
			string contractId = esId;

			while (!runTcs.Task.IsCompleted)
			{
				// Fetch latest bars (last 120)
				var rawBars = await history.RetrieveBarsAsync(contractId, daysBack: 1, unit: CFG.UNIT, unitNumber: CFG.UNITNUM, live: CFG.LIVE);
				var stratBars = rawBars.Select(b => new TopstepFullStrat.Bar {
					Ts = b.t.ToUnixTimeSeconds(),
					Open = b.o,
					High = b.h,
					Low = b.l,
					Close = b.c,
					Volume = b.v
				}).ToList();
							var logPath = System.IO.Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "bot.log");
							void Log(string msg)
							{
								Console.WriteLine(msg);
								try
								{
									using (var sw = new System.IO.StreamWriter(logPath, true))
									{
										sw.WriteLine(msg);
									}
								}
								catch (Exception ex)
								{
									Console.WriteLine($"[LogError] {ex.Message}");
								}
							}

				if (stratBars.Count == 0) { await Task.Delay(5000); continue; }

				// Populate env/levels from TopstepFullStrat.Bar
				env.atr = 6m; env.bb_bw = 15m; env.don_hi = stratBars.Max(b => b.High); env.don_lo = stratBars.Min(b => b.Low);
				env.ema9 = stratBars.Average(b => b.Close); env.ema21 = stratBars.Average(b => b.Close); env.ema20 = stratBars.Average(b => b.Close);
				env.vwap = stratBars.Average(b => b.Close); env.volz = 0.5m;
				levels.ib_done = true; levels.ibh = stratBars.Max(b => b.High); levels.ibl = stratBars.Min(b => b.Low);
				levels.onh = stratBars.Max(b => b.High); levels.onl = stratBars.Min(b => b.Low); levels.on_mid = stratBars.Average(b => b.Close);
				levels.pdh = stratBars.Max(b => b.High); levels.pdl = stratBars.Min(b => b.Low);

				// Generate strategy candidates
				Console.WriteLine($"[Bot] Looking for trades at {DateTimeOffset.UtcNow:yyyy-MM-dd HH:mm:ss}...");
				var candidates = TopstepFullStrat.AllStrategies.generate_candidates(symbol, env, levels, stratBars, risk);
				Console.WriteLine($"[Bot] {candidates.Count} candidates found.");
				if (candidates.Count > 0)
				{
					foreach (var c in candidates)
					{
						Console.WriteLine($"[Candidate] {c.strategy_id} {c.symbol} {c.side} qty {c.qty} entry {c.entry} stop {c.stop} t1 {c.t1} R~{c.expR:F2}");
					}
				}
				else
				{
					Console.WriteLine("[Bot] No strategy candidates generated this cycle.");
				}

				if (CFG.FORCE_BUY)
				{
					var forceBuy = new TopstepFullStrat.Candidate {
						strategy_id = "FORCE_BUY_TEST",
						symbol = symbol,
						side = TopstepFullStrat.Side.BUY,
						entry = stratBars.Last().Close,
						stop = stratBars.Last().Close - 10,
						t1 = stratBars.Last().Close + 20,
						expR = 1.0m,
						qty = 1
					};
					int side = 0; // BUY
					Console.WriteLine($"[Bot] Forcing BUY order: qty={forceBuy.qty} entry={forceBuy.entry} stop={forceBuy.stop} t1={forceBuy.t1}");
					var orderId = await api.PlaceLimit(accountId, contractId, side, forceBuy.qty, forceBuy.entry);
					Console.WriteLine($"[TRADE] Forced BUY {forceBuy.qty} @ {forceBuy.entry} (strat={forceBuy.strategy_id}) orderId={orderId}");
				}
				else
				{
					var best = candidates.OrderByDescending(c => c.expR).FirstOrDefault();
					if (best != null)
					{
						Console.WriteLine($"[Bot] Best candidate: {best.strategy_id} side={best.side} qty={best.qty} entry={best.entry} stop={best.stop} t1={best.t1} R~{best.expR:F2}");
						if (risk.CanOpen())
						{
							int side = best.side == TopstepFullStrat.Side.BUY ? 0 : 1;
							var orderId = await api.PlaceLimit(accountId, contractId, side, best.qty, best.entry);
							Console.WriteLine($"[TRADE] Placed {best.side} {best.qty} @ {best.entry} (strat={best.strategy_id}) orderId={orderId}");
						}
						else
						{
							Console.WriteLine("[Bot] Risk engine rejected trade: max open positions or daily loss limit reached.");
						}
					}
					else
					{
						Console.WriteLine("[Bot] No valid candidate for order placement this cycle.");
					}
				}

				await Task.Delay(10000); // 10s loop
			}

			Console.WriteLine("[Shutdown] Stopping MarketHub...");
			await hub.StopAsync();
			Console.WriteLine("[Shutdown] Stopped.");
		}
		catch (Exception ex)
		{
			Console.WriteLine("[Error] " + ex.Message);
		}
	}

	private static async Task ProbeMarketHubAsync(string token)
	{
		using var http = new HttpClient();
		var req = new HttpRequestMessage(HttpMethod.Post,
			"https://rtc.topstepx.com/hubs/market/negotiate?negotiateVersion=1");
		req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
		req.Content = new StringContent("");
		var resp = await http.SendAsync(req);
		var body = await resp.Content.ReadAsStringAsync();
		Console.WriteLine($"[Negotiate] {resp.StatusCode} {body}");
	}

	private static void LoadEnvFromFile(string path)
	{
		try
		{
			if (!System.IO.File.Exists(path)) return;
			foreach (var line in System.IO.File.ReadAllLines(path))
			{
				if (string.IsNullOrWhiteSpace(line)) continue;
				var trimmed = line.Trim();
				if (trimmed.StartsWith("#")) continue;
				var idx = trimmed.IndexOf('=');
				if (idx <= 0) continue;
				var key = trimmed.Substring(0, idx).Trim();
				var value = trimmed.Substring(idx + 1).Trim();
				Environment.SetEnvironmentVariable(key, value);
			}
			Console.WriteLine("[Env] .env.local loaded.");
		}
		catch (Exception ex)
		{
			Console.WriteLine("[Env] load failed: " + ex.Message);
		}
	}
}
E