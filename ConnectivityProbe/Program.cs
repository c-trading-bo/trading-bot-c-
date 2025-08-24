using Microsoft.AspNetCore.Http.Connections;
using Microsoft.AspNetCore.SignalR.Client;
using System.Net.Http;
using System.Text.Json;

static async Task<int> Main(string[] args)
{
    string rtcBase = Environment.GetEnvironmentVariable("TOPSTEPX_RTC_BASE") ?? "https://rtc.topstepx.com";
    string hubUser = $"{rtcBase.TrimEnd('/')}/hubs/user";

    string? jwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
    if (string.IsNullOrWhiteSpace(jwt))
    {
        var user = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
        var key  = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");
        if (!string.IsNullOrWhiteSpace(user) && !string.IsNullOrWhiteSpace(key))
        {
            try
            {
                using var http = new HttpClient { BaseAddress = new Uri(Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com") };
                var auth = new TopstepAuthAgent(http);
                Console.WriteLine($"Fetching JWT using login key for {user}…");
                jwt = await auth.GetJwtAsync(user!, key!, CancellationToken.None);
                Console.WriteLine("Obtained JWT via loginKey.");
            }
            catch (Exception ex)
            {
                Console.WriteLine("Failed to obtain JWT: " + ex.Message);
            }
        }
    }

    if (string.IsNullOrWhiteSpace(jwt))
    {
        Console.WriteLine("Missing JWT. Set TOPSTEPX_JWT or provide TOPSTEPX_USERNAME and TOPSTEPX_API_KEY.");
        return 2;
    }

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
    }

    Console.WriteLine("\n== CONNECT LongPolling ==");
    var okLP = await ConnectTest(hubUser, jwt!, HttpTransportType.LongPolling);

    Console.WriteLine("\n== CONNECT WebSockets ==");
    var okWS = await ConnectTest(hubUser, jwt!, HttpTransportType.WebSockets);

    Console.WriteLine($"\nSummary: LongPolling={(okLP ? "OK" : "FAIL")}  WebSockets={(okWS ? "OK" : "FAIL")}");
    return (okLP || okWS) ? 0 : 1;
}

static async Task<bool> ConnectTest(string url, string jwt, HttpTransportType transport)
{
    var hub = new HubConnectionBuilder()
        .WithUrl(url, o =>
        {
            o.AccessTokenProvider = () => Task.FromResult<string?>(jwt);
            o.Transports = transport; // force under test
        })
        .WithAutomaticReconnect()
        .Build();

    hub.Closed += ex =>
    {
        Console.WriteLine($"[Closed:{transport}] {ex?.GetType().Name}: {ex?.Message}");
        return Task.CompletedTask;
    };

    try
    {
        await hub.StartAsync();
        Console.WriteLine($"Started with {transport}. State={hub.State}");
        await Task.Delay(TimeSpan.FromSeconds(5));
        Console.WriteLine($"After 5s: State={hub.State}");
        await hub.DisposeAsync();
        return true;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"StartAsync EX ({transport}): {ex.GetType().Name}: {ex.Message}");
        try { await hub.DisposeAsync(); } catch { }
        return false;
    }
}
