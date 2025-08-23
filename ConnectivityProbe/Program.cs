using Microsoft.AspNetCore.Http.Connections;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.SignalR.Client;

static async Task<int> Main(string[] args)
{
	// 1) PASTE your raw JWT here (eyJhbGciOi...) – value of jwtResponse.token
	var jwt = "<PASTE RAW JWT HERE>";

	var hubUser = "https://rtc.topstepx.com/hubs/user";

	Console.WriteLine("== NEGOTIATE ==");
	try
	{
		using var http = new HttpClient();
		var req = new HttpRequestMessage(HttpMethod.Post, $"{hubUser}/negotiate?negotiateVersion=1");
		req.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt);
		var resp = await http.SendAsync(req);
		Console.WriteLine($"HTTP {(int)resp.StatusCode} {resp.StatusCode}");
		var body = await resp.Content.ReadAsStringAsync();
		Console.WriteLine($"Body: {body}");
		if (!resp.IsSuccessStatusCode) return 1;

		try
		{
			using var doc = JsonDocument.Parse(body);
			if (doc.RootElement.TryGetProperty("availableTransports", out var t))
			{
				Console.WriteLine("availableTransports:");
				foreach (var tr in t.EnumerateArray())
					Console.WriteLine(" - " + tr.GetProperty("transport").GetString());
			}
		}
		catch { /* non-fatal */ }
	}
	catch (Exception ex)
	{
		Console.WriteLine("Negotiate EX: " + ex);
		return 1;
	}

	// 2) Try LongPolling (if this stays connected → WS likely blocked by network)
	Console.WriteLine("\n== CONNECT LongPolling ==");
	var okLP = await ConnectTest(hubUser, jwt, HttpTransportType.LongPolling);

	// 3) Try WebSockets (if this fails/insta-closes while LP works → WS blocked/policy)
	Console.WriteLine("\n== CONNECT WebSockets ==");
	var okWS = await ConnectTest(hubUser, jwt, HttpTransportType.WebSockets);

	Console.WriteLine($"\nSummary: LongPolling={(okLP ? "OK" : "FAIL")}  WebSockets={(okWS ? "OK" : "FAIL")}");
	return 0;
}

static async Task<bool> ConnectTest(string url, string jwt, HttpTransportType transport)
{
	var hub = new HubConnectionBuilder()
		.WithUrl(url, o =>
		{
			o.AccessTokenProvider = () => Task.FromResult(jwt);
			o.Transports = transport;            // force the transport under test
			// Uncomment to bypass system proxy while testing:
			// o.HttpMessageHandlerFactory = _ => new SocketsHttpHandler { UseProxy = false };
		})
		.WithAutomaticReconnect()
		.Build();

	hub.Closed += ex =>
	{
		Console.WriteLine($"[Closed:{transport}]\nTYPE: {ex?.GetType().FullName ?? "<null>"}\nMSG : {ex?.Message ?? "<null>"}");
		if (ex is HttpRequestException hre) Console.WriteLine($"HTTP StatusCode: {hre.StatusCode}");
		if (ex?.InnerException is Exception ie) Console.WriteLine($"INNER: {ie.GetType().FullName} | {ie.Message}");
		return Task.CompletedTask;
	};

	try
	{
		await hub.StartAsync();
		Console.WriteLine($"Started with {transport}. State={hub.State}");
		// wait a bit; if server policy kills, it will show up
		await Task.Delay(TimeSpan.FromSeconds(5));
		Console.WriteLine($"After 5s: State={hub.State}");
		await hub.DisposeAsync();
		return hub.State == HubConnectionState.Disconnected; // we disposed cleanly
	}
	catch (Exception ex)
	{
		Console.WriteLine($"StartAsync EX ({transport}): {ex.GetType().Name}: {ex.Message}");
		return false;
	}
}
