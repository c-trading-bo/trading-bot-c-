// PlaceOrderAndWatchFill.cs
// .NET 8+ single-file test for TopstepX order placement and fill event
// Usage: dotnet run --project PlaceOrderAndWatchFill.csproj

using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;

class PlaceOrderAndWatchFill
{

    public static async Task RunFillTest(string token)
    {
        if (string.IsNullOrWhiteSpace(token))
        {
            Console.WriteLine("Set TOPSTEPX_JWT=your_token");
            return;
        }

        using var http = new HttpClient { BaseAddress = new Uri("https://api.topstepx.com") };
        http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);

        // Validate token
        var validate = await http.PostAsJsonAsync("/api/Auth/validate", new { });
        Console.WriteLine($"Auth/validate => {(int)validate.StatusCode}");

        // Get account
        var acctResp = await http.PostAsJsonAsync("/api/Account/search", new { onlyActiveAccounts = true });
        var acctJson = await acctResp.Content.ReadAsStringAsync();
        using var doc = JsonDocument.Parse(acctJson);
        long accountId = doc.RootElement.TryGetProperty("accounts", out var arr) && arr.GetArrayLength() > 0
            ? arr[0].GetProperty("id").GetInt64()
            : 0;
        if (accountId == 0) { Console.WriteLine("No tradable account found."); return; }
        Console.WriteLine($"AccountId: {accountId}");

        // Get contractId for ES and NQ
        var esResp = await http.PostAsJsonAsync("/api/Contract/search", new { live = true, searchText = "ES" });
        var nqResp = await http.PostAsJsonAsync("/api/Contract/search", new { live = true, searchText = "NQ" });
        var esJson = await esResp.Content.ReadAsStringAsync();
        var nqJson = await nqResp.Content.ReadAsStringAsync();
        using var esDoc = JsonDocument.Parse(esJson);
        using var nqDoc = JsonDocument.Parse(nqJson);
        var esContracts = esDoc.RootElement.GetProperty("contracts");
        var nqContracts = nqDoc.RootElement.GetProperty("contracts");

        string? contractId = null;
        string symbol = "";
        if (esContracts.GetArrayLength() > 0)
        {
            contractId = esContracts[0].GetProperty("id").GetString();
            symbol = "ES";
        }
        else if (nqContracts.GetArrayLength() > 0)
        {
            contractId = nqContracts[0].GetProperty("id").GetString();
            symbol = "NQ";
        }
        else
        {
            Console.WriteLine("No ES or NQ contracts found. Cannot trade.");
            return;
        }
        Console.WriteLine($"Selected contract: {symbol} ({contractId})");

        // --- Dynamic trade decision logic ---
        // Example: alternate buy/sell each run (can be replaced with strategy output)
        bool isLong = DateTime.UtcNow.Second % 2 == 0; // stub: even seconds = buy, odd = sell
        int side = isLong ? 0 : 1; // 0 = Buy, 1 = Sell
        decimal entry = symbol == "ES" ? 4500.00m : 16000.00m;
        decimal stop = isLong ? entry - 10 : entry + 10; // 10 points risk
        decimal takeProfit = isLong ? entry + 20 : entry - 20; // 20 points reward
        decimal tick = symbol == "ES" ? 0.25m : 0.25m; // ES/NQ tick size
        entry = Math.Round(entry / tick, 0) * tick;
        stop = Math.Round(stop / tick, 0) * tick;
        takeProfit = Math.Round(takeProfit / tick, 0) * tick;
        decimal risk = Math.Abs(entry - stop);
        decimal reward = Math.Abs(takeProfit - entry);
        decimal rMultiple = risk > 0 ? reward / risk : 0;
        if (risk <= 0 || rMultiple < 0.5m)
        {
            Console.WriteLine($"Risk invalid: entry={entry}, stop={stop}, takeProfit={takeProfit}, R={rMultiple:F2}");
            return;
        }
        int size = 1; // stub: always 1 contract
        string customTag = $"TEST-{symbol}-{(isLong ? "LONG" : "SHORT")}-{DateTime.UtcNow:yyyyMMdd-HHmmss}";
        Console.WriteLine($"[{symbol}] side={(isLong ? "BUY" : "SELL")} qty={size} entry={entry:F2} stop={stop:F2} t1={takeProfit:F2} R~{rMultiple:F2} tag={customTag}");

        // Place order
        var orderReq = new {
            accountId,
            contractId,
            type = 1, // Limit
            side,
            size,
            limitPrice = entry,
            stopLossPrice = stop,
            takeProfitPrice = takeProfit,
            customTag
        };
        var orderResp = await http.PostAsJsonAsync("/api/Order/place", orderReq);
        var orderJson = await orderResp.Content.ReadAsStringAsync();
        Console.WriteLine($"Order/place => {(int)orderResp.StatusCode} {orderJson}");
        string? orderId = null;
        using (var odoc = JsonDocument.Parse(orderJson))
        {
            if (odoc.RootElement.TryGetProperty("orderId", out var oid))
                orderId = oid.GetString();
        }
        if (string.IsNullOrEmpty(orderId)) { Console.WriteLine("No orderId returned."); return; }
        Console.WriteLine($"OrderId: {orderId}");

        // Connect to User Hub and subscribe to fills
        var hub = new HubConnectionBuilder()
            .WithUrl("https://rtc.topstepx.com/hubs/user", o => o.AccessTokenProvider = () => Task.FromResult(token))
            .WithAutomaticReconnect()
            .Build();

        hub.On<object>("GatewayUserTrade", d => Console.WriteLine("TRADE => " + JsonSerializer.Serialize(d)));
        hub.On<object>("GatewayUserOrder", d => Console.WriteLine("ORDER => " + JsonSerializer.Serialize(d)));

        await hub.StartAsync();
        Console.WriteLine("User Hub started.");
        await hub.InvokeAsync("SubscribeOrders", accountId);
        await hub.InvokeAsync("SubscribeTrades", accountId);
        Console.WriteLine("Subscribed. Waiting 60s for fill events...");
        await Task.Delay(TimeSpan.FromSeconds(60));
        await hub.DisposeAsync();
    }
}
